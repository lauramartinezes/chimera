import os
import glob
import pandas as pd
import yaml

from datasets.split_data import *

pd.set_option('display.max_rows', None)

def is_dir_empty(path):
    if not os.path.isdir(path):
        return True  
    return not any(os.scandir(path))

if __name__ == "__main__":
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    insect_classes = config["data_params"]["data_classes"] 
    trash_class = config["data_params"]["trash_class"]
    split = config["data_params"]["nature_split"] # good, meas noise, label noise percentages
    src_dir = config["data_params"]["original_data_dir"]
    root_dir = os.path.dirname(os.path.relpath(src_dir))

    dest_dir = os.path.join(root_dir, f'split_{split[0]}_{split[1]}_{split[2]}')
    config["data_params"]["splitted_data_dir"] = dest_dir

    # Write back full config
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    if is_dir_empty(dest_dir):
        # Get all PNG files for each insect class
        class_file_dict = {}
        for insect_class in insect_classes:
            cls_path = os.path.join(src_dir, insect_class)
            class_file_dict[insect_class] = glob.glob(os.path.join(cls_path, '*.png'))

        # Get files for the trash class
        trash_path = os.path.join(src_dir, trash_class)
        trash_files = glob.glob(os.path.join(trash_path, '*.png'))

        # Determine minimal number of files for balanced splitting
        total_samples = determine_min_class_size(class_file_dict)
        print(f'Total samples per class: {total_samples}')

        # Prepare split distribution
        train_composition = {'good': split[0]/100, 'mislabel': split[1]/100, 'other': split[2]/100}
        split_result = calculate_custom_splits(total_samples, train_composition)
        split_result['train']['other'] = split_result['train']['mislabel']
        split_result['validation']['other'] = split_result['validation']['mislabel']
        print(f'Split result: {split_result}')

        # Split each insect class
        df_class_subsets = {}
        for insect_class in insect_classes:
            df = get_df(os.path.join(src_dir, insect_class))
            df_subsets, summary = get_df_subsets(df, [
                split_result['train']['good'], 
                split_result['train']['mislabel'], 
                split_result['test']['good'], 
                split_result['validation']['good'], 
                split_result['validation']['mislabel']
            ])
            df_class_subsets[insect_class] = df_subsets
            print(f'Assignment summary for {insect_class}: {summary}')
            print(df_subsets['subset'].value_counts())

        # Split trash class
        df_trash = get_df(trash_path)
        df_trash_subsets, summary_trash = get_df_subsets(df_trash, [
            split_result['train']['other'], 
            split_result['train']['other'], 
            split_result['validation']['other'], 
            split_result['validation']['other']
        ])
        print(f'Assignment summary for {trash_class}: {summary_trash}')
        print(df_trash_subsets['subset'].value_counts())

        # Copy files for insect classes
        for i, insect_class in enumerate(insect_classes):
            other_class = insect_classes[1 - i]  # valid because we assume 2 classes
            
            dest_folders = []
            for subset in ['train', 'val']:
                dest_folders.append(f"{subset}/{insect_class}/{insect_class}_good")             # good samples
                dest_folders.append(f"{subset}/{other_class}/{insect_class}_for_{other_class}") # mislabelled samples
                if subset == 'train':
                    dest_folders.append(f"test/{insect_class}")  # test samples are not mislabelled

            copy_files_to_dest(df_class_subsets[insect_class], dest_dir, dest_folders)

        # Copy files for trash class
        trash_dest_folders = []
        for subset in ['train', 'val']:
            for insect_class in insect_classes:
                trash_dest_folders.append(f"{subset}/{insect_class}/{insect_class}_trash")

        copy_files_to_dest(df_trash_subsets, dest_dir, trash_dest_folders)
