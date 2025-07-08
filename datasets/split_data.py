import glob
import os
import random
import shutil

import numpy as np
import pandas as pd


def get_df(folder):
    files = glob.glob(os.path.join(folder, '*.png'))
    df = pd.DataFrame(files, columns=['filepath'])
    df['platename'] = df.filepath.apply(lambda x: '_'.join(os.path.basename(x).split('_')[:-1]))
    df['year'] = df.filepath.apply(lambda x: os.path.basename(x).split('_')[0])
    df['location'] = df.filepath.apply(lambda x: os.path.basename(x).split('_')[1])
    df['week'] = df.filepath.apply(lambda x: os.path.basename(x).split('_')[2])

    return df

def determine_min_class_size(class_file_dict):
        min_class_size = min([len(v) for v in class_file_dict.values()])
        return int(min_class_size / 100) * 100

def split_column_with_simulated_annealing(df, column, targets, max_iter=10000, temp=1000, cooling_rate=0.003, random_seed=None):
    """
    Splits a DataFrame into subsets based on unique values of a specified column using Simulated Annealing 
    to match target sample counts.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): Name of the column in df to base the splitting on.
    - targets (list): List of target sample counts for each subset.
    - max_iter (int): Maximum number of iterations for the simulated annealing algorithm.
    - temp (float): Initial temperature for simulated annealing.
    - cooling_rate (float): Cooling rate for simulated annealing.
    - random_seed (int): Seed for random number generator to ensure reproducibility.

    Returns:
    - df_with_subsets (pd.DataFrame): Original DataFrame with an added 'subset' column indicating subset assignment.
    - assignment_summary (pd.DataFrame): Summary of assignments, actual vs. target samples, and deviations.
    """

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Calculate samples per unique value in the specified column
    value_sample_dict = df[column].value_counts().to_dict()
    unique_values = list(value_sample_dict.keys())

    # Objective function: Minimize deviation
    def calculate_deviation(assignment):
        subset_sums = [0] * len(targets)
        for val, subset_idx in assignment.items():
            subset_sums[subset_idx] += value_sample_dict[val]
        deviation = sum(abs(subset_sum - target) for subset_sum, target in zip(subset_sums, targets))
        return deviation, subset_sums

    # Simulated Annealing Algorithm
    def simulated_annealing():
        num_subsets = len(targets)
        current_assignment = {val: random.randint(0, num_subsets - 1) for val in unique_values}
        current_deviation, _ = calculate_deviation(current_assignment)
        best_assignment = current_assignment.copy()
        best_deviation = current_deviation

        local_temp = temp

        for i in range(max_iter):
            local_temp *= (1 - cooling_rate)
            new_assignment = current_assignment.copy()
            val_to_move = random.choice(unique_values)
            new_subset = random.randint(0, num_subsets - 1)
            new_assignment[val_to_move] = new_subset

            new_deviation, _ = calculate_deviation(new_assignment)
            delta = new_deviation - current_deviation

            if delta < 0 or random.uniform(0, 1) < np.exp(-delta / local_temp):
                current_assignment = new_assignment
                current_deviation = new_deviation
                if new_deviation < best_deviation:
                    best_assignment = new_assignment
                    best_deviation = new_deviation

        _, final_sums = calculate_deviation(best_assignment)
        return best_assignment, final_sums, best_deviation

    # Run simulated annealing
    best_assignment, final_sums, best_deviation = simulated_annealing()

    # Prepare results
    result = {}
    for i in range(len(targets)):
        result[f'Subset {i+1}'] = [val for val, idx in best_assignment.items() if idx == i]

    assignment_summary = []
    for i, (subset, values_in_subset) in enumerate(result.items()):
        assignment_summary.append({
            'Subset': f'Subset {i+1}',
            'Values': ', '.join(values_in_subset),
            'Actual Samples': final_sums[i],
            'Target Samples': targets[i],
            'Deviation': abs(final_sums[i] - targets[i])
        })

    assignment_summary_df = pd.DataFrame(assignment_summary)

    # Add subset assignment to original DataFrame
    value_to_subset = {val: idx for val, idx in best_assignment.items()}
    df_with_subsets = df.copy()
    df_with_subsets['subset'] = df_with_subsets[column].map(value_to_subset)

    return df_with_subsets, assignment_summary_df

def handle_excess_samples(df_with_subsets, assignment_summary, targets, strategy="remove", random_seed=None):
    """
    Handles excess samples in subsets beyond the target counts by keeping only the required number of samples.

    Parameters:
    - df_with_subsets (pd.DataFrame): DataFrame with subset assignments.
    - assignment_summary (pd.DataFrame): Summary with actual vs. target samples.
    - targets (list): List of target sample counts for each subset.
    - strategy (str): "remove" to keep only target samples, "mark" to flag excess ones.
    - random_seed (int): Seed for reproducibility.

    Returns:
    - df_adjusted (pd.DataFrame): DataFrame with adjusted or marked excess samples.
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Create a deep copy to avoid modifying the original DataFrame
    df_adjusted = df_with_subsets.copy()

    # Ensure correct target mapping by pairing subset indices with targets
    unique_subsets = sorted(df_adjusted['subset'].unique())
    subset_target_mapping = dict(zip(unique_subsets, targets[:len(unique_subsets)]))

    # Process each subset separately
    subset_dfs = []  # Store processed subsets

    for subset_idx, target in subset_target_mapping.items():
        subset_df = df_adjusted[df_adjusted['subset'] == subset_idx].copy()  # Work on a copy

        current_count = len(subset_df)

        if current_count > target:
            excess_count = current_count - target
            #print(f"Subset {subset_idx}: Keeping {target} samples (target: {target}, current: {current_count})")

            if strategy == "remove":
                # Instead of dropping, randomly select `target` samples to KEEP
                subset_df = subset_df.sample(n=target, random_state=random_seed)

            elif strategy == "mark":
                # Mark excess samples instead of removing
                mark_indices = subset_df.sample(n=excess_count, random_state=random_seed).index
                subset_df.loc[mark_indices, 'excess_sample'] = True

            else:
                raise ValueError("Invalid strategy. Use 'remove' or 'mark'.")

        # Append cleaned subset to list
        subset_dfs.append(subset_df)

    # Recombine all cleaned subsets
    df_adjusted = pd.concat(subset_dfs, ignore_index=True)

    return df_adjusted

def get_df_subsets(df, targets, random_seed=42):
    df_with_subsets, assignment_summary = split_column_with_simulated_annealing(df, 'platename', targets, random_seed=random_seed)
    if assignment_summary['Deviation'].sum() > 0:
        return handle_excess_samples(df_with_subsets, assignment_summary, targets, strategy="remove", random_seed=random_seed), assignment_summary
    return df_with_subsets, assignment_summary

def calculate_custom_splits(total_samples, train_composition):
    """
    Calculates dataset splits based on custom rules:
    - Only 'good' and 'mislabel' samples are counted in total_samples.
    - 'other' samples are excluded from total_samples but train_composition may include them.
    - Train set has the given composition for 'good' and 'mislabel'.
    - Validation and test sets have 12.5% as many 'good' samples as the train set.
    - Validation set mimics the train ratio for 'good' and 'mislabel'; test is only 'good'.
    
    Returns:
        dict: sample counts per subset and type, with total_used (excluding 'other')
    """
    # Normalize over only 'good' and 'mislabel'
    gm_total = train_composition['good'] + train_composition['mislabel']
    good_prop = train_composition['good'] / gm_total
    mislabel_prop = train_composition['mislabel'] / gm_total

    # Calculate train size (T) based on desired val/test composition
    val_size = 0.125 / good_prop
    test_size = 0.125
    total_multiplier = 1 + val_size + test_size

    T = round(total_samples / total_multiplier)  # Train set size (good + mislabel)

    train_counts = {
        'good': round(good_prop * T),
        'mislabel': round(mislabel_prop * T),
        'other': round(mislabel_prop * T)  # set other equal to mislabel
    }

    val_good = round(0.125 * T)
    val_fraction = val_good / good_prop
    val_counts = {
        'good': val_good,
        'mislabel': round(mislabel_prop * val_fraction),
        'other': round(mislabel_prop * val_fraction)
    }

    test_counts = {'good': val_good, 'mislabel': 0, 'other': 0}

    total_used = train_counts['good'] + train_counts['mislabel'] + \
                 val_counts['good'] + val_counts['mislabel'] + \
                 test_counts['good']

    return {
        'train': train_counts,
        'validation': val_counts,
        'test': test_counts,
        'total_used': total_used
    }

def copy_files_to_dest(df_subsets, dest_dir, dest_folders):
    for i in range(len(dest_folders)):
        dest_path = os.path.join(dest_dir, dest_folders[i])
        os.makedirs(dest_path, exist_ok=True)
        df_subset_i = df_subsets[df_subsets['subset'] == i]
        df_subset_i['dest_filepath'] = df_subset_i.filepath.apply(lambda x: os.path.join(dest_path, os.path.basename(x)))

        # Copy file from filepath to dest_filepath using shutil.copy
        for _, row in df_subset_i.iterrows():
            shutil.copy2(row['filepath'], row['dest_filepath'])