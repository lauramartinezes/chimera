import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from tqdm import tqdm
import umap
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms

from PyOD import PYOD, metric
from mnist_dataset import CustomBinaryInsectDF
from vq_vae import VQVAE


def clean_df(df, model, device, config, transform, pin_memory, main_insect_class, phase="train"):
    # Load the data
    loader = load_data_from_df(
        df,
        transform,
        config["exp_params"]["manual_seed"],
        config["data_params"][f"train_batch_size"],
        config["data_params"]["num_workers"],
        pin_memory,
    )
    print(f"{phase.capitalize()} dataset correctly loaded")
    
    # Extract features from encoding latents
    (
        raw_features_encoding,
        features_encoding,
        labels_encoding,
        real_labels_encoding,
        measurement_noise_encoding,
        mislabeled_encoding,
    ) = extract_features_from_encoding(model, loader, device)
    
    # Reduce dimensions to 512D for outlier detection
    print('Starting UMAP 512D reduction')
    reshaped_raw_features_encoding = raw_features_encoding.reshape(raw_features_encoding.shape[0], -1)
    reducer_512d = umap.UMAP(n_components=512)
    latents_raw_encoding_512d = reducer_512d.fit_transform(reshaped_raw_features_encoding)
    
    y_true = (measurement_noise_encoding + mislabeled_encoding).astype(int)
    y_pred = get_outlier_predictions(
        latents_raw_encoding_512d,
        model='HBOS'
    )

    visualize_y_true_vs_y_pred_umap(
        reshaped_raw_features_encoding, 
        measurement_noise_encoding, 
        mislabeled_encoding,
        y_pred, 
        filename=f'ae_{main_insect_class}_{phase}',
        dirname=config["logging_params"]["save_dir"]
    )

    df_clean = df[y_pred == 0]
    return df_clean


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


def extract_features_from_encoding(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_features = []  # To store features from all batches
    all_labels = []    # To store labels if needed
    all_real_labels = []
    all_measurement_noise = []
    all_mislabeled = []
    all_encodings = []

    with torch.no_grad():
        for images, labels, real_labels, measurement_noise, mislabeled, outliers in tqdm(dataloader, desc="Featured extraction encoding: "):
            images = images.to(device)

            # Pass images through the encoder layers
            encoding = model.encode(images)[0]  # Raw encoder output

            # Global Average Pooling: Compute mean across spatial dimensions (H, W)
            pooled_encoding = encoding.mean(dim=[2, 3])  # Shape: [B, embedding_dim]

            # Append features to the list
            all_features.append(pooled_encoding.cpu().numpy())
            all_encodings.append(encoding.cpu().numpy())
            all_labels.append(labels.numpy())  # Collect labels if needed
            all_real_labels.append(real_labels.numpy())  # Collect labels if needed
            all_measurement_noise.append(measurement_noise.numpy())  # Collect labels if needed
            all_mislabeled.append(mislabeled.numpy())  # Collect labels if needed
    # Concatenate all features into a single array
    return np.vstack(all_encodings), np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_real_labels), np.concatenate(all_measurement_noise), np.concatenate(all_mislabeled)



def get_outlier_predictions(X_train, model='HBOS'):
    print(f'{model} outlier extraction')
    pyod_model = PYOD(seed=42, model_name=model)
    pyod_model.fit(X_train, [])
    anomaly_scores = pyod_model.predict_score(X_train)
    anomaly_binary_scores = pyod_model.predict_binary_score(X_train)

    return anomaly_binary_scores


def visualize_y_true_vs_y_pred_umap(features, measurement_noises, label_noises, y_pred, filename=None, dirname=None):
    umap_folder = os.path.join(dirname, 'True vs Pred UMAPS')
    os.makedirs(umap_folder, exist_ok=True)

    # UMAP transformation (shared latent space for both plots)
    print(f'Starting UMAP 2D reduction')
    reducer_2d = umap.UMAP(n_components=2)
    latents_2d = reducer_2d.fit_transform(features)

    measurement_noises = measurement_noises.astype(int)*2
    label_noises = label_noises.astype(int)
    noises = measurement_noises + label_noises

    # Dictionary to map numbers to text labels
    true_label_mapping = {0: "Normal Sample", 1: "Label Noise", 2: "Measurement Noise"}
    txt_true_labels = [true_label_mapping[label] for label in noises]
    
    pred_label_mapping = {0: "Normal Sample", 1: "Outlier"}
    txt_pred_labels = [pred_label_mapping[label] for label in y_pred]

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for y_true
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=txt_true_labels, palette=sns.color_palette("hsv", 3), ax=axes[0], legend='full')
    axes[0].set_title("Latent Space UMAP: True Labels")
    axes[0].set_xlabel("UMAP dimension 1")
    axes[0].set_ylabel("UMAP dimension 2")

    # Plot for y_pred
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=txt_pred_labels, palette=sns.color_palette("hsv", 2), ax=axes[1], legend='full')
    axes[1].set_title("Latent Space UMAP: Predicted Labels")
    axes[1].set_xlabel("UMAP dimension 1")
    axes[1].set_ylabel("UMAP dimension 2")

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(umap_folder, f'true_vs_pred_{filename}.png'), format='png')
    plt.savefig(os.path.join(umap_folder, f'true_vs_pred_{filename}.svg'), format='svg')


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    seed = 20
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    pin_memory = len(config['trainer_params']['gpus']) != 0

    insect_classes = ['wmv', 'c']

    transform_ae = transforms.Compose([
        transforms.Resize((config["data_params"]["patch_size"], config["data_params"]["patch_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        df_train = pd.read_csv(df_train_path)

        # AE method
        save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{main_insect_class}_best.pth')
        model_ae = VQVAE(
            in_channels=config["model_params"]["in_channels"],
            embedding_dim=config["model_params"]["embedding_dim"],
            num_embeddings=config["model_params"]["num_embeddings"],
            img_size=config["model_params"]["img_size"],
            beta=config["model_params"]["beta"]
        ).to(device)
        model_ae.load_state_dict(torch.load(save_path_best))
        model_ae.to(device)
        print("Model correctly initialized")

        df_train_clean = clean_df(df_train, model_ae, device, config, transform_ae, pin_memory, main_insect_class, phase="train")
        df_train_clean_path = os.path.join('data', f'df_train_ae_{main_insect_class}_clean.csv')
        df_train_clean.to_csv(df_train_clean_path, index=False)

        print(f'Clean {main_insect_class} training dataset available')

        print('')