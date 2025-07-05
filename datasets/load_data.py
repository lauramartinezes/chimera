import os

import pandas as pd

from torch.utils.data import DataLoader

from datasets.datasets import CustomBinaryInsectDF


def load_subset_df_classification(insect_classes, subset, method, od_method, root_dir, clean_dataset=''):
    dfs_subset = []
    for i, main_insect_class in enumerate(insect_classes):
        df_subset_path = os.path.join(
            root_dir,
            'clean' if clean_dataset=='_clean' else '', 
            f'df_{subset}_{method if method != "cleaning_benchmark" else "raw"}_{main_insect_class}{od_method}{clean_dataset}.csv'
        )
        df_subset_i = pd.read_csv(df_subset_path)
        if method == 'cleaning_benchmark':
            df_subset_i_no_noise = df_subset_i[df_subset_i['label'] == 0]
            df_subset_i_label_noise = df_subset_i[df_subset_i['mislabeled'] == True] # Comment this line if you want to reove label noise as well
            # Correct labels for training a classifier instead of an outlier detector
            df_subset_i_no_noise.label = i
            df_subset_i_label_noise.label = 1 - i
            dfs_subset.append(df_subset_i_no_noise)
            dfs_subset.append(df_subset_i_label_noise)
        else:
            # Correct labels for training a classifier instead of an outlier detector
            df_subset_i.label = i
            dfs_subset.append(df_subset_i)
    
    return pd.concat(dfs_subset, ignore_index=True)


def load_data_from_df(df, transform, seed, batch_size, num_workers, pin_memory, shuffle=False):
    dataset = CustomBinaryInsectDF(
        df, 
        transform = transform, 
        seed=seed
    )

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )