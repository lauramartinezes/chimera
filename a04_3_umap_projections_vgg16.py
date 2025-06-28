
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from tqdm import tqdm
import yaml
import seaborn as sns

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

def visualize_train_latent_space_3d(train_features, labels_train, filename='', dirname=''): 
    # Dictionary to map numbers to text labels
    label_mapping = {0: "Normal Sample", 1: "Label Noise", 2: "Measurement Noise"}
    txt_noise_labels = [label_mapping[label] for label in labels_train]

    # Plot UMAP results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_features[:, 0], train_features[:, 1], train_features[:, 2], c=labels_train, cmap='viridis')
    ax.set_title("Latent Space UMAP Visualization")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.set_zlabel("UMAP dimension 3")
    #plt.show()

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

    insect_classes = config["data_params"]["data_classes"]

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        df_val_path = os.path.join('data', f'df_val_ae_{main_insect_class}.csv')
        df_train_ = pd.read_csv(df_train_path)
        df_val_ = pd.read_csv(df_val_path)

        # Combine the training and validation dataframes
        df_train = pd.concat([df_train_, df_val_], ignore_index=True)

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
        model_name = 'resnet18'
        model = timm.create_model(model_name, pretrained=False, num_classes=2)
        save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier_raw_best_.pth')#f'{model_name}_classifier{clean_dataset}_{method}_best.pth')
    
        # Load the model
        if os.path.exists(save_path_best):
            model.load_state_dict(torch.load(save_path_best))
            print("Model correctly loaded")
        else:
            print("Model not found")
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

        print(f"{main_insect_class} model correctly initialized")

        # Extract features from encoding latents
        umap_folder = os.path.join(config["logging_params"]["save_dir"], 'UMAPS')
        os.makedirs(umap_folder, exist_ok=True)
        umap_train_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_{model_name}_train.npy')
        umap_test_file_name = os.path.join(umap_folder, f'umap_vect_{main_insect_class}_{model_name}_test.npy')
        labels_umap_train_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_{model_name}_train.npy')
        labels_umap_test_file_name = os.path.join(umap_folder, f'labels_umap_vect_{main_insect_class}_{model_name}_test.npy')


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

            latents_3d_train, latents_3d_test = get_train_test_umap(
                latents_cnn_train, 
                latents_cnn_test, 
                n_components=3
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
            f'{model_name}_{main_insect_class}_test', 
            config["logging_params"]["save_dir"]
        )

        visualize_train_latent_space(
            latents_2d_train, 
            labels_noise_train, 
            f'{model_name}_{main_insect_class}_train', 
            config["logging_params"]["save_dir"]
        )

        # visualize_train_latent_space_3d(latents_3d_train, labels_noise_train, filename='', dirname='')

        print('')