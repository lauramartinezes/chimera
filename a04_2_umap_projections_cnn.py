
import os
import random
import numpy as np
import pandas as pd
import timm
import torch
from tqdm import tqdm
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms

from a04_1_umap_projections_ae import get_train_test_umap, visualize_test_latent_space_wrt_train, visualize_train_latent_space
from mnist_dataset import CustomBinaryInsectDF


def load_data_from_df(df, transform, seed, batch_size, num_workers, pin_memory):
    dataset = CustomBinaryInsectDF(
        df, 
        transform = transform, 
        seed=seed
    )

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def extract_features_from_dataloader(dataloader, model):
    all_features = []  # To store features from all batches
    all_labels = []    # To store labels if needed
    all_real_labels = []
    all_measurement_noise = []
    all_mislabeled = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels, real_labels, measurement_noise, mislabeled, outliers in tqdm(dataloader, desc="Extracting features", total=len(dataloader)):
            # Forward pass to get the features
            features = model(images)  # Get features for the batch
            features_np = features.numpy().reshape(features.shape[0], -1)  # Flatten to 2D array
            all_features.append(features_np)
            all_labels.append(labels.numpy())  # Collect labels if needed
            all_real_labels.append(real_labels.numpy())  # Collect labels if needed
            all_measurement_noise.append(measurement_noise.numpy())  # Collect labels if needed
            all_mislabeled.append(mislabeled.numpy())  # Collect labels if needed

    # Stack all features and labels vertically
    return np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_real_labels), np.concatenate(all_measurement_noise), np.concatenate(all_mislabeled)


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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        df_train = pd.read_csv(df_train_path)

        train_dataset = CustomBinaryInsectDF(df_train, transform = transform, seed=config["exp_params"]["manual_seed"])
        test_dataset = CustomBinaryInsectDF(df_test, transform = transform, seed=config["exp_params"]["manual_seed"])

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )

        print(f"{main_insect_class} dataset correctly loaded")

        
        # Resnet method
        model = timm.create_model('resnet18', pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

        print(f"{main_insect_class} model correctly initialized")

        # Extract features from encoding latents
        umap_folder = os.path.join(config["logging_params"]["save_dir"], 'UMAPS')
        os.makedirs(umap_folder, exist_ok=True)
        umap_train_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_cnn_train.npy')
        umap_test_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_cnn_test.npy')
        labels_umap_train_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_cnn_train.npy')
        labels_umap_test_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_cnn_test.npy')


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
            ) = extract_features_from_dataloader(test_loader, model)
            
            (
                latents_cnn_train, 
                labels_cnn_train, 
                real_labels_cnn_train, 
                measurement_noise_cnn_train, 
                mislabeled_cnn_train
            ) = extract_features_from_dataloader(train_loader, model)


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

        visualize_test_latent_space_wrt_train(
            main_insect_class, 
            mislabeled_insect_class, 
            latents_2d_train,
            latents_2d_test, 
            labels_noise_train, 
            labels_cnn_test, 
            f'cnn_{main_insect_class}_test', 
            config["logging_params"]["save_dir"]
        )

        visualize_train_latent_space(
            latents_2d_train, 
            labels_noise_train, 
            f'cnn_{main_insect_class}_train', 
            config["logging_params"]["save_dir"]
        )

        print('')