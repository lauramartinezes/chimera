import os
import random
import numpy as np
import pandas as pd
import timm
import torch
import umap
import yaml

from datasets import set_feature_extraction_transform
from datasets.load_data import load_data_from_df
from models import extract_features
from umaps.plot import plot_test_latent_space_wrt_train, plot_train_latent_space

def get_train_test_umap(X_train, X_test, n_components=2):
    umap_model = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)

    # Fit UMAP on the training data with labels
    X_train_embedding = umap_model.fit_transform(X_train)

    # Transform the test data into the existing UMAP embedding
    X_test_embedding = umap_model.transform(X_test)

    return X_train_embedding, X_test_embedding


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

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
    for method in methods:
        if method == 'adbench':
            suffix = config["data_params"]["raw_suffix"]
        else:
            suffix = config["data_params"]["swap_suffix"]
        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_train_path = os.path.join(data_dir, f'df_train_{suffix}_{main_insect_class}.csv')
            df_val_path = os.path.join(data_dir, f'df_val_{suffix}_{main_insect_class}.csv')
            df_train_ = pd.read_csv(df_train_path)
            df_val_ = pd.read_csv(df_val_path)

            # Combine the training and validation dataframes
            df_train = pd.concat([df_train_, df_val_], ignore_index=True)

            train_loader = load_data_from_df(
                df_train,
                set_feature_extraction_transform(),
                config["exp_params"]["manual_seed"],
                config["data_params"][f"batch_size"],
                config["data_params"]["num_workers"],
                pin_memory,
                shuffle=False
            )

            test_loader = load_data_from_df(
                df_test,
                set_feature_extraction_transform(),
                config["exp_params"]["manual_seed"],
                config["data_params"][f"batch_size"],
                config["data_params"]["num_workers"],
                pin_memory,
                shuffle=False
            )

            print(f"{main_insect_class} dataset correctly loaded")

            # Extract features from encoding latents
            umap_folder = os.path.join(config["logging_params"]["save_dir"], 'UMAPS')
            os.makedirs(umap_folder, exist_ok=True)
            umap_train_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_{method}_train.npy')
            umap_test_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_{method}_test.npy')
            labels_umap_train_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_{method}_train.npy')
            labels_umap_test_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_{method}_test.npy')


            if os.path.exists(umap_train_file_name) and os.path.exists(umap_test_file_name) and \
                os.path.exists(labels_umap_train_file_name) and os.path.exists(labels_umap_test_file_name): 
                latents_2d_train = np.load(umap_train_file_name)
                latents_2d_test = np.load(umap_test_file_name)
                labels_noise_train = np.load(labels_umap_train_file_name)
                labels_cnn_test = np.load(labels_umap_test_file_name)
            else:            
                (
                    latents_cnn_test, 
                    labels_cnn_test, 
                    real_labels_cnn_test, 
                    measurement_noise_cnn_test, 
                    mislabeled_cnn_test
                ) = extract_features(test_loader, model)
                
                (
                    latents_cnn_train, 
                    labels_cnn_train, 
                    real_labels_cnn_train, 
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
