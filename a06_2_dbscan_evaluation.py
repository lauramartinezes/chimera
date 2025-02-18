import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import torch
import seaborn as sns
import umap
import yaml

from torchvision import transforms

from PyOD import metric
from vq_vae import VQVAE


def DBSCAN_OD(latent_space_points, eps=0.5, min_samples=1):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust parameters as needed
    labels = dbscan.fit_predict(latent_space_points)

    # Count the number of points in each cluster (ignore noise points labeled as -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]

    predictions = np.where(labels != largest_cluster_label, 1, 0)

    return predictions

def visualize_outliers(outliers, inliers):
    # Plotting the clusters
    plt.figure(figsize=(10, 6))

    plt.scatter(outliers[:, 0], outliers[:, 1], label=f'Outliers Cluster', color='red', s=50, alpha=0.6)
    plt.scatter(inliers[:, 0], inliers[:, 1], label=f'Inliers Cluster', color='blue', s=50, alpha=0.6)

    # for label in unique_labels:
    #     if label == -1:  # Noise points
    #         cluster_points = latent_space_points[labels == label]
    #         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label='Noise', color='gray', s=30, alpha=0.3)
    #     else:
    #         cluster_points = latent_space_points[labels == label]
    #         if (label == largest_cluster_label):# | (label == second_largest_cluster_label):
    #             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Inliers Cluster {label}', color='red', s=50, alpha=0.6)
    #         else:
    #             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', s=30, alpha=0.4)

    plt.title('DBSCAN Clustering with Highlighted Largest Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_y_pred_umap(features, y_pred):
    # UMAP transformation (shared latent space for both plots)
    if features.shape[1] > 2:
        print(f'Starting UMAP 2D reduction')
        reducer_2d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        latents_2d = reducer_2d.fit_transform(features)
    else:
        latents_2d = features
    
    pred_label_mapping = {0: "Normal Sample", 1: "Outlier"}
    txt_pred_labels = [pred_label_mapping[label] for label in y_pred]

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for y_pred
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=txt_pred_labels, palette=sns.color_palette("hsv", 2), ax=axes[1], legend='full')
    axes[1].set_title("Latent Space UMAP: Predicted Labels")
    axes[1].set_xlabel("UMAP dimension 1")
    axes[1].set_ylabel("UMAP dimension 2")

    # Save the figure
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Toy example: generate random 2D data with a cluster and some outliers
    np.random.seed(42)
    cluster_1 = np.random.normal(loc=0, scale=1, size=(50, 2))  # Cluster of 50 points around (0,0)
    outliers = np.random.normal(loc=10, scale=2, size=(5, 2))   # 5 outliers far from the cluster

    y_true = np.array([0] * len(cluster_1) + [1] * len(outliers))

    # Combine them
    latent_space_points = np.vstack([cluster_1, outliers])

    # Run outlier detection
    y_pred = DBSCAN_OD(latent_space_points)

    # Visualize the results
    #visualize_y_pred_umap(latent_space_points, y_pred)

    results = metric(y_true, y_pred, pos_label=1)
    print(results)
    print('')
    # # Load the configuration
    # with open("config.yaml", "r") as file:
    #     config = yaml.safe_load(file)

    # # Set manual seed for reproducibility
    # torch.manual_seed(config["exp_params"]["manual_seed"])
    # random.seed(config["exp_params"]["manual_seed"])
    # np.random.seed(config["exp_params"]["manual_seed"])

    # pin_memory = len(config['trainer_params']['gpus']) != 0

    # insect_classes = ['wmv', 'c']
    # ae_types = [''] #, 'adv_']

    # transform = transforms.Compose([
    #     transforms.Resize((config["data_params"]["patch_size"], config["data_params"]["patch_size"])),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # results = []
    
    # for ae_type in ae_types:
    #     for i in range(len(insect_classes)):
    #         main_insect_class = insect_classes[i]
    #         mislabeled_insect_class = insect_classes[1 - i]

    #         df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
    #         df_train = pd.read_csv(df_train_path)

    #         # AE method
    #         save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{ae_type}{main_insect_class}_best.pth')
    #         model_ae = VQVAE(
    #             in_channels=config["model_params"]["in_channels"],
    #             embedding_dim=config["model_params"]["embedding_dim"],
    #             num_embeddings=config["model_params"]["num_embeddings"],
    #             img_size=config["model_params"]["img_size"],
    #             beta=config["model_params"]["beta"]
    #         ).to(device)
    #         model_ae.load_state_dict(torch.load(save_path_best))
    #         model_ae.to(device)
    #         print("Model correctly initialized")

    #         # Load the data
    #         loader = load_data_from_df(
    #             df_train,
    #             transform,
    #             config["exp_params"]["manual_seed"],
    #             config["data_params"][f"train_batch_size"],
    #             config["data_params"]["num_workers"],
    #             pin_memory,
    #         )
    #         print(f"Train dataset correctly loaded")
            
    #         # Extract features from encoding latents
    #         (
    #             raw_features_encoding,
    #             features_encoding,
    #             labels_encoding,
    #             real_labels_encoding,
    #             measurement_noise_encoding,
    #             mislabeled_encoding,
    #         ) = extract_features_from_encoding(model_ae, loader, device)
            
    #         # Reduce dimensions to 2D for visualization
    #         reshaped_raw_features_encoding = raw_features_encoding.reshape(raw_features_encoding.shape[0], -1)
    #         reducer_2_d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    #         latents_raw_encoding_2_d = reducer_2_d.fit_transform(reshaped_raw_features_encoding)
            
    #         # Reduce dimensions to 512D for outlier detection
    #         reducer_512_d = umap.UMAP(n_components=512, random_state=42, n_jobs=1)
    #         latents_raw_encoding_512_d = reducer_512_d.fit_transform(reshaped_raw_features_encoding)