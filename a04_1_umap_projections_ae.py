# Configuration parameters from the YAML-like file
import os
import random
import numpy as np
import pandas as pd
import torch
import umap
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms

from b03_outliers_evaluation import extract_features_from_encoding, visualize_latent_space
from mnist_dataset import CustomBinaryInsectDF
from vq_vae import VQVAE


def get_train_test_umap(X_train, X_test, n_components=2):
    umap_model = umap.UMAP(n_components=n_components, random_state=42)

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

    df_test_path = os.path.join('data', f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    insect_classes = ['wmv', 'c']

    transform = transforms.Compose([
            transforms.Resize((config["data_params"]["patch_size"], config["data_params"]["patch_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        df_train = pd.read_csv(df_train_path)

        train_dataset_vae = CustomBinaryInsectDF(df_train, transform = transform, seed=config["exp_params"]["manual_seed"])
        test_dataset_vae = CustomBinaryInsectDF(df_test, transform = transform, seed=config["exp_params"]["manual_seed"])

        train_loader = DataLoader(
            train_dataset_vae, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset_vae, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )

        print(f"{main_insect_class} dataset correctly loaded")

        # Initialize the Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = VQVAE(
            in_channels=config["model_params"]["in_channels"],
            embedding_dim=config["model_params"]["embedding_dim"],
            num_embeddings=config["model_params"]["num_embeddings"],
            img_size=config["model_params"]["img_size"],
            beta=config["model_params"]["beta"]
        ).to(device)

        print(f"{main_insect_class} model correctly initialized")

        save_path = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{main_insect_class}.pth')
        save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{main_insect_class}_best.pth')
        os.makedirs(config["logging_params"]["save_dir"], exist_ok=True)

        if not os.path.exists(save_path_best):
            print(f"Please train a {main_insect_class} model")
            continue 
        else:
            print(f"{main_insect_class} loading best model")    
        model.load_state_dict(torch.load(save_path_best))

        # Visualization After Training
        model.eval()

        # Extract features from encoding latents
        umap_folder = os.path.join(config["logging_params"]["save_dir"], 'UMAPS')
        umap_train_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_train.npy')
        umap_test_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_test.npy')
        labels_umap_train_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_train.npy')
        labels_umap_test_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_test.npy')


        if os.path.exists(umap_train_file_name) and os.path.exists(umap_test_file_name) and \
            os.path.exists(labels_umap_train_file_name) and os.path.exists(labels_umap_test_file_name): 
            latents_2d_train = np.load(umap_train_file_name)
            latents_2d_test = np.load(umap_test_file_name)
            labels_noise_train = np.load(labels_umap_train_file_name)
            labels_encoding_test = np.load(labels_umap_test_file_name)
        else:            
            (
                raw_features_encoding_test,
                features_encoding_test,
                labels_encoding_test,
                real_labels_encoding_test,
                measurement_noise_encoding_test,
                mislabeled_encoding_test,
            ) = extract_features_from_encoding(model, test_loader, device)

            (
                raw_features_encoding_train,
                features_encoding_train,
                labels_encoding_train,
                real_labels_encoding_train,
                measurement_noise_encoding_train_,
                mislabeled_encoding_train_,
            ) = extract_features_from_encoding(model, train_loader, device)

            measurement_noise_encoding_train = measurement_noise_encoding_train_.astype(int)*2
            mislabeled_encoding_train = mislabeled_encoding_train_.astype(int)
            labels_noise_train = measurement_noise_encoding_train + mislabeled_encoding_train

            # Reduce dimensions to 2D for visualization
            reshaped_raw_features_encoding_test = raw_features_encoding_test.reshape(raw_features_encoding_test.shape[0], -1)
            reshaped_raw_features_encoding_train = raw_features_encoding_train.reshape(raw_features_encoding_train.shape[0], -1)

            latents_2d_train, latents_2d_test = get_train_test_umap(
                reshaped_raw_features_encoding_train, 
                reshaped_raw_features_encoding_test, 
                n_components=2
            )
            
            np.save(umap_train_file_name, latents_2d_train)
            np.save(umap_test_file_name, latents_2d_test)
            np.save(labels_umap_train_file_name, labels_noise_train)
            np.save(labels_umap_test_file_name, labels_encoding_test)

    print('')

