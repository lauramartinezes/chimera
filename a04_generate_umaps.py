import os
import numpy as np
import pandas as pd
import timm
import torch
import yaml

from datasets import set_feature_extraction_transform
from datasets.load_data import load_data_from_df
from models import extract_features
from umaps.get_train_test_umap import get_train_test_umap
from umaps.plot import plot_test_latent_space_wrt_train, plot_train_latent_space
from utils import set_seed

def init_umap_filenames(umap_folder, main_insect_class, method):
    stem = f"{main_insect_class}_{method}"
    files = {}

    for kind, prefix in {"umap": "umap_vect_", "labels": "labels_umap_vect_"}.items():
        for split in ("train", "test"):
            files[f"{kind}_{split}"] = os.path.join(umap_folder, f"{prefix}{stem}_{split}.npy")

    return files


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    random_seed = config["exp_params"]["manual_seed"]
    set_seed(random_seed)

    pin_memory = len(config['trainer_params']['gpus']) != 0
    insect_classes = config["data_params"]["data_classes"]
    data_dir = config["data_params"]["splitted_data_dir"]

    df_test_path = os.path.join(data_dir, f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    # Resnet method
    model = timm.create_model(config["model_params"]["name"], pretrained=config["model_params"]["pretrained"])
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    methods = ['cnn', 'adbench']
    subsets = ['train', 'val']

    umap_folder = os.path.join(config["logging_params"]["save_dir"], 'UMAPS')
    os.makedirs(umap_folder, exist_ok=True)

    for method in methods:
        if method == 'adbench':
            suffix = config["data_params"]["raw_suffix"]
        else:
            suffix = config["data_params"]["swap_suffix"]
        for i, main_insect_class in enumerate(insect_classes):
            mislabeled_insect_class = insect_classes[1 - i]

            dfs_subsets = []
            for subset in subsets:
                df_subset_path = os.path.join(
                    data_dir, 
                    f'df_{subset}_{suffix}_{main_insect_class}.csv'
                )
                df_subset = pd.read_csv(df_subset_path)
                dfs_subsets.append(df_subset)
            df_train_val = pd.concat(dfs_subsets, ignore_index=True)

            loader_kwargs = dict(
                transform=set_feature_extraction_transform(),
                seed=config["exp_params"]["manual_seed"],
                batch_size=config["data_params"]["batch_size"],
                num_workers=config["data_params"]["num_workers"],
                pin_memory=pin_memory,
                shuffle=False
            )
        
            train_loader = load_data_from_df(df=df_train_val, **loader_kwargs)
            test_loader = load_data_from_df(df=df_test, **loader_kwargs)

            print(f"{main_insect_class} dataset correctly loaded")

            # Extract features from encoding latents
            umap_files = init_umap_filenames(umap_folder, main_insect_class, method)
            umap_train_file_name = umap_files["umap_train"]
            umap_test_file_name = umap_files["umap_test"]
            labels_umap_train_file_name = umap_files["labels_train"]
            labels_umap_test_file_name = umap_files["labels_test"]

            if all(os.path.exists(p) for p in umap_files.values()): 
                latents_2d_train = np.load(umap_train_file_name)
                latents_2d_test = np.load(umap_test_file_name)
                labels_noise_train = np.load(labels_umap_train_file_name)
                labels_cnn_test = np.load(labels_umap_test_file_name)
            else:            
                (
                    latents_cnn_test, 
                    labels_cnn_test, 
                    measurement_noise_cnn_test, 
                    mislabeled_cnn_test
                ) = extract_features(test_loader, model)
                
                (
                    latents_cnn_train, 
                    labels_cnn_train, 
                    measurement_noise_cnn_train, 
                    mislabeled_cnn_train
                ) = extract_features(train_loader, model)

                measurement_noise_cnn_train = measurement_noise_cnn_train.astype(int)*2
                mislabeled_cnn_train = mislabeled_cnn_train.astype(int)
                labels_noise_train = measurement_noise_cnn_train + mislabeled_cnn_train

                # Reduce dimensions to 2D for visualization
                latents_2d_train, latents_2d_test = get_train_test_umap(
                    latents_cnn_train, 
                    latents_cnn_test, 
                    n_components=2
                )
                
                np.save(umap_train_file_name, latents_2d_train)
                np.save(umap_test_file_name, latents_2d_test)
                np.save(labels_umap_train_file_name, labels_noise_train)
                np.save(labels_umap_test_file_name, labels_cnn_test)

            plot_test_latent_space_wrt_train(
                main_insect_class, 
                mislabeled_insect_class, 
                latents_2d_train,
                latents_2d_test, 
                labels_noise_train, 
                labels_cnn_test, 
                insect_classes,
                f'{method}_{main_insect_class}_test', 
                config["logging_params"]["save_dir"]
            )

            plot_train_latent_space(
                latents_2d_train, 
                labels_noise_train, 
                f'{method}_{main_insect_class}_train', 
                config["logging_params"]["save_dir"]
            )
