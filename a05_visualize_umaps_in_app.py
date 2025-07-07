# Configuration parameters from the YAML-like file
import os
import random
import numpy as np
import pandas as pd
import torch
import yaml

from umaps.plot import plot_latent_space_with_tooltips
from utils import set_seed


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    random_seed = config["exp_params"]["manual_seed"]
    set_seed(random_seed)
    
    data_dir = config["data_params"]["splitted_data_dir"]
    raw_suffix = config["data_params"]["raw_suffix"]
    swap_suffix = config["data_params"]["swap_suffix"]

    # Load the test DataFrame
    df_test_path = os.path.join(data_dir, f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    # Define the insect classes
    insect_classes = ['wmv', 'wrl']
    feature_ext_methods = ['cnn', 'adbench']

    # Prepare dictionaries for dropdown-based selection
    train_features_dict, test_features_dict = {}, {}
    labels_train_dict, labels_test_dict = {}, {}
    df_train_dict, df_test_dict = {}, {}

    for feature_ext_method in feature_ext_methods:
        suffix = raw_suffix if feature_ext_method == 'adbench' else swap_suffix

        for i, main_insect_class in enumerate(insect_classes):
            mislabeled_insect_class = insect_classes[1 - i]

            df_train_path = os.path.join(data_dir, f'df_train_{suffix}_{main_insect_class}.csv')
            df_train = pd.read_csv(df_train_path)

            # UMAP file paths
            umap_folder = os.path.join(config["logging_params"]["save_dir"], 'UMAPS')
            umap_train_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_{feature_ext_method}_train.npy')
            umap_test_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_{feature_ext_method}_test.npy')
            labels_umap_train_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_{feature_ext_method}_train.npy')
            labels_umap_test_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_{feature_ext_method}_test.npy')

            # Check if files exist, then load
            if os.path.exists(umap_train_file_name) and os.path.exists(umap_test_file_name) and \
            os.path.exists(labels_umap_train_file_name) and os.path.exists(labels_umap_test_file_name):
                latents_2d_train = np.load(umap_train_file_name)
                latents_2d_test = np.load(umap_test_file_name)
                labels_noise_train = np.load(labels_umap_train_file_name)
                labels_encoding_test = np.load(labels_umap_test_file_name)

                # Populate dictionaries
                train_features_dict[f'{main_insect_class}_{feature_ext_method}'] = latents_2d_train
                test_features_dict[f'{main_insect_class}_{feature_ext_method}'] = latents_2d_test
                labels_train_dict[f'{main_insect_class}_{feature_ext_method}'] = labels_noise_train
                labels_test_dict[f'{main_insect_class}_{feature_ext_method}'] = labels_encoding_test
                df_train_dict[f'{main_insect_class}_{feature_ext_method}'] = df_train
                df_test_dict[f'{main_insect_class}_{feature_ext_method}'] = df_test
            else:
                print(f"UMAP files not found for {main_insect_class}_{feature_ext_method}. Skipping...")
                continue

    # Call the function with prepared data dictionaries
    plot_latent_space_with_tooltips(
        train_features_dict=train_features_dict,
        test_features_dict=test_features_dict,
        labels_train_dict=labels_train_dict,
        labels_test_dict=labels_test_dict,
        df_train_dict=df_train_dict,
        df_test_dict=df_test_dict,
        insect_classes=insect_classes
    )
