
#TODO: project for both cnn and ad bench methods
#TODO: before that we need to rename the df ae to df swap or something like that

import os
import random
import numpy as np
import pandas as pd
import timm
import torch
import umap
import seaborn as sns
import yaml

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from datasets import CustomBinaryInsectDF, set_feature_extraction_transform
from models import extract_features

def get_train_test_umap(X_train, X_test, n_components=2):
    umap_model = umap.UMAP(n_components=n_components, random_state=42, n_jobs=1)

    # Fit UMAP on the training data with labels
    X_train_embedding = umap_model.fit_transform(X_train)

    # Transform the test data into the existing UMAP embedding
    X_test_embedding = umap_model.transform(X_test)

    return X_train_embedding, X_test_embedding

def visualize_test_latent_space_wrt_train(main_insect_class, mislabeled_insect_class, train_features, test_features, labels_train, labels_test, insect_classes, filename='', dirname=''):
    umap_folder = os.path.join(dirname, 'UMAPS')
    os.makedirs(umap_folder, exist_ok=True)
    
    # Dictionary to map numbers to text labels
    label_mapping_train = {0: f"Normal Sample ({main_insect_class})", 1: f"Label Noise ({mislabeled_insect_class})", 2: "Measurement Noise"}
    txt_labels_train = [label_mapping_train[label] for label in labels_train]
    
    label_mapping_test = {0: f"{insect_classes[0]}_test", 1: f"{insect_classes[1]}_test"}
    txt_labels_test = [label_mapping_test[label] for label in labels_test]

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=test_features[:, 0], y=test_features[:, 1], hue=txt_labels_test, palette=sns.color_palette("Set2", len(label_mapping_test)), marker='o', legend='full')
    sns.scatterplot(x=train_features[:, 0], y=train_features[:, 1], hue=txt_labels_train, palette=sns.color_palette("hsv", len(label_mapping_train)), marker='x', s=15, legend='full')
    plt.title("Latent Space UMAP Visualization")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.png'), format='png')
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.svg'), format='svg')

def visualize_train_latent_space(train_features, labels_train, filename='', dirname=''):
    umap_folder = os.path.join(dirname, 'UMAPS')
    os.makedirs(umap_folder, exist_ok=True)
    
    # Dictionary to map numbers to text labels
    label_mapping = {0: "Normal Sample", 1: "Label Noise", 2: "Measurement Noise"}
    txt_noise_labels = [label_mapping[label] for label in labels_train]

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=train_features[:, 0], y=train_features[:, 1], hue=txt_noise_labels, palette=sns.color_palette("hsv", 3), legend='full')
    plt.title("Latent Space UMAP Visualization")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.png'), format='png')
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.svg'), format='svg')



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
    data_dir = config["data_params"]["data_dir"]
    swap_suffix = config["data_params"]["swap_suffix"]

    df_test_path = os.path.join(data_dir, f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join(data_dir, f'df_train_{swap_suffix}_{main_insect_class}.csv')
        df_val_path = os.path.join(data_dir, f'df_val_{swap_suffix}_{main_insect_class}.csv')
        df_train_ = pd.read_csv(df_train_path)
        df_val_ = pd.read_csv(df_val_path)

        # Combine the training and validation dataframes
        df_train = pd.concat([df_train_, df_val_], ignore_index=True)

        train_dataset = CustomBinaryInsectDF(df_train, transform = set_feature_extraction_transform(), seed=config["exp_params"]["manual_seed"])
        test_dataset = CustomBinaryInsectDF(df_test, transform = set_feature_extraction_transform(), seed=config["exp_params"]["manual_seed"])

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

        visualize_test_latent_space_wrt_train(
            main_insect_class, 
            mislabeled_insect_class, 
            latents_2d_train,
            latents_2d_test, 
            labels_noise_train, 
            labels_cnn_test, 
            insect_classes,
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