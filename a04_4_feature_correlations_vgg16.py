
import os
import random
import cuml
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import timm
import torch
from tqdm import tqdm
import yaml
import seaborn as sns

from torch.utils.data import DataLoader
from torchvision import transforms

from PyOD import metric
from a04_1_umap_projections_ae import get_train_test_umap, visualize_test_latent_space_wrt_train#, visualize_train_latent_space
from a06_2_dbscan_evaluation import DBSCAN_OD
from a06_outliers_evaluation import get_outlier_methods_csv
from a07_clean_data import get_outlier_predictions, visualize_y_true_vs_y_pred_umap
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
    plt.show()

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

import numpy as np
import umap
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

def umap_permutation_importance(data, n_components=2):
    """
    Calculate UMAP feature importance by permuting each feature.
    """
    # Step 1: Fit UMAP on original data
    reducer = cuml.manifold.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components, random_state=42)    
    original_embedding = reducer.fit_transform(data)

    # Step 2: Calculate original pairwise distance matrix
    original_distances = pairwise_distances(original_embedding)

    # Step 3: Initialize an array to store importance scores
    importance_scores = np.zeros(data.shape[1])

    # Step 4: Permute each feature and recalculate UMAP
    for i in tqdm(range(data.shape[1]), desc="Calculating feature importance"):
        permuted_data = data.copy()
        np.random.shuffle(permuted_data[:, i])  # Permute feature i

        # Fit UMAP on permuted data
        permuted_embedding = reducer.fit_transform(permuted_data)
        permuted_distances = pairwise_distances(permuted_embedding)

        # Calculate the difference between the distance matrices
        distance_change = np.linalg.norm(original_distances - permuted_distances)
        importance_scores[i] = distance_change

    # Normalize importance scores
    importance_scores /= np.max(importance_scores)

    # Sort features by importance
    important_features = np.argsort(-importance_scores)

    print("Feature importance (UMAP):", importance_scores)
    print("Most important features:", important_features)

    return importance_scores, important_features


import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
import matplotlib.pyplot as plt

def find_optimal_umap_dimension(data, dims=list(range(2, 21)) + [30, 50, 100]):
    scores = []

    for d in dims:
        print(f"Fitting UMAP with {d} dimensions...")
        reducer = cuml.manifold.UMAP(n_neighbors=15, min_dist=0.1, n_components=d, random_state=42)
        embedding = reducer.fit_transform(data)

        # Calculate pairwise distances in the original space and embedding space
        original_distances = pairwise_distances(data)
        embedded_distances = pairwise_distances(embedding)

        # Calculate the preservation score (Spearman correlation of distances)
        preservation_score = np.corrcoef(original_distances.flatten(), embedded_distances.flatten())[0, 1]

        # Calculate clustering quality (Silhouette Score)
        if d > 1:  # Silhouette score needs more than 1 dimension
            db = DBSCAN(eps=0.5, min_samples=1)
            cluster_labels = db.fit_predict(embedding)
            
            # Filter out noise points (-1) for silhouette calculation
            valid_labels = cluster_labels != -1
            if np.sum(valid_labels) > 1:  # Ensure enough valid points
                sil_score = silhouette_score(embedding[valid_labels], cluster_labels[valid_labels])
            else:
                sil_score = -1  # Not enough clusters for a valid silhouette score
        else:
            sil_score = 0

        print(f"Dim: {d}, Preservation Score: {preservation_score:.4f}, Silhouette Score: {sil_score:.4f}")
        scores.append((d, preservation_score, sil_score))

    # Plotting the scores for visualization
    scores = np.array(scores)
    plt.figure(figsize=(10, 5))
    plt.plot(scores[:, 0], scores[:, 1], label="Preservation Score", marker='o')
    plt.plot(scores[:, 0], scores[:, 2], label="Silhouette Score", marker='x')
    plt.xlabel("UMAP Dimensions")
    plt.ylabel("Score")
    plt.title("Optimal UMAP Dimension Selection")
    plt.legend()
    plt.show()

    # Select the dimension with the best average score
    best_dim = int(scores[np.argmax(scores[:, 1] + scores[:, 2]), 0])
    print(f"Optimal UMAP Dimension: {best_dim}")

    return best_dim

import numpy as np
import cupy as cp
from cuml.manifold import UMAP
from cuml.cluster import DBSCAN
from cuml.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def find_optimal_umap_dbscan(main_insect_class, data, dims=list(range(2, 11)) + [20, 30, 50, 100], 
                             eps_values=np.arange(0.1, 1.1, 0.2), min_samples_values=[1, 3, 5], save_dir="umap_dbscan_plots"):
    scores = []

    for d in dims:
        print(f"\nFitting UMAP with {d} dimensions...")
        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=d, random_state=42)
        embedding = reducer.fit_transform(data)

        # Calculate pairwise distances in the original space and embedding space
        original_distances = pairwise_distances(data)
        embedded_distances = pairwise_distances(embedding)

        # Calculate the preservation score (Spearman correlation of distances)
        preservation_score = np.corrcoef(original_distances.flatten(), embedded_distances.flatten())[0, 1]

        # Loop through DBSCAN parameters
        for eps in eps_values:
            for min_samples in min_samples_values:
                # Clustering with DBSCAN
                db = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = db.fit_predict(embedding)
                cluster_labels = cluster_labels.get()

                # Check the number of clusters and noise ratio
                num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)

                # Cluster size and density analysis
                cluster_sizes = [np.sum(cluster_labels == label) for label in set(cluster_labels) if label != -1]
                
                if cluster_sizes:
                    # Density calculations for each cluster
                    cluster_densities = [
                        np.sum(cluster_labels == label) / (np.std(embedded_distances[cluster_labels == label]) + 1e-10)
                        for label in set(cluster_labels) if label != -1
                    ]
                    
                    # Find the largest cluster and its density
                    largest_cluster_size = max(cluster_sizes)
                    largest_cluster_label = set(cluster_labels) - {-1}
                    largest_cluster_density = max(cluster_densities)

                    # Calculate average density for comparison
                    avg_density = np.mean([d.get() if isinstance(d, cp.ndarray) else d for d in cluster_densities])

                    # Adaptive noise threshold based on density comparison
                    adaptive_noise_threshold = avg_density / (largest_cluster_density + 1e-10)

                    # Validity check: clusters must be sufficiently dense and noise ratio should adapt
                    valid_noise = noise_ratio < adaptive_noise_threshold
                    valid_density = largest_cluster_density > avg_density * 0.5  # Largest cluster should be denser

                    # Calculate silhouette score if clusters are valid
                    if num_clusters > 0 and valid_noise and valid_density:
                        valid_labels = cluster_labels != -1
                        valid_labels = cluster_labels != -1
                        num_valid_clusters = len(set(cluster_labels[valid_labels]))

                        if num_valid_clusters > 1:
                            sil_score = silhouette_score(embedding.get()[valid_labels], cluster_labels[valid_labels])
                        else:
                            sil_score = -1
                    else:
                        sil_score = -1
                else:
                    sil_score = -1
                    largest_cluster_density = 0

                print(f"Dim: {d}, eps: {eps}, min_samples: {min_samples}, "
                      f"Preservation: {preservation_score:.4f}, Silhouette: {sil_score:.4f}, "
                      f"Noise Ratio: {noise_ratio:.2f}, Largest Cluster Density: {largest_cluster_density:.4f}, "
                      f"Adaptive Noise Threshold: {adaptive_noise_threshold:.4f}")
                scores.append((d, eps, min_samples, preservation_score, sil_score, noise_ratio, largest_cluster_density, adaptive_noise_threshold))

    # Convert all GPU-based elements in the scores list to CPU-based
    scores = np.array([tuple(x.get() if isinstance(x, cp.ndarray) else x for x in score) for score in scores])

    # Filtering valid scores (Silhouette > 0 and dynamically calculated noise ratio)
    valid_scores = scores[(scores[:, 4] > 0)]
    if len(valid_scores) > 0:
        # Calculate a weighted score (2x Silhouette + Preservation)
        combined_score = 2 * valid_scores[:, 4] + valid_scores[:, 3]
        best_idx = np.argmax(combined_score)
        best_dim, best_eps, best_min_samples = valid_scores[best_idx, :3]
    else:
        print("No valid combination found with positive Silhouette Score. Selecting best silhouette score instead.")
        best_idx = np.argmax(scores[:, 4])  # Fallback to best silhouette score
        best_dim, best_eps, best_min_samples = scores[best_idx, :3]

    # Plotting the results
    plot_umap_dbscan_results(main_insect_class, scores, eps_values, min_samples_values, best_dim, save_dir=save_dir)

    print(f"\nOptimal UMAP Dimension: {int(best_dim)}, eps: {best_eps}, min_samples: {int(best_min_samples)}")
    return int(best_dim), best_eps, int(best_min_samples)


import os

import os

def plot_umap_dbscan_results(main_insect_class, scores, eps_values, min_samples_values, best_dim, save_dir="umap_dbscan_plots"):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    def save_plot(fig, title, main_insect_class=main_insect_class):
        fig.savefig(os.path.join(save_dir, f"{main_insect_class}_{title}.png"))
        fig.savefig(os.path.join(save_dir, f"{main_insect_class}_{title}.svg"))
        plt.close(fig)

    def plot_vertical_line(ax, best_dim):
        ax.axvline(x=best_dim, color='red', linestyle='--', label=f'Best Dim: {best_dim}')

    # Plot preservation score separately
    fig = plt.figure(figsize=(10, 6))
    preservation_scores = np.unique(scores[:, [0, 3]], axis=0)
    plt.plot(preservation_scores[:, 0], preservation_scores[:, 1], marker='o', label="Preservation Score")
    plt.title("Preservation Scores")
    plt.xlabel("UMAP Dimensions")
    plt.ylabel("Preservation Score")
    plot_vertical_line(plt, best_dim)
    plt.legend()
    plt.grid(True)
    save_plot(fig, "Preservation Scores")

    # Plot silhouette score
    fig = plt.figure(figsize=(10, 6))
    for min_samples in min_samples_values:
        subset = scores[scores[:, 2] == min_samples]
        for eps in eps_values:
            eps_subset = subset[subset[:, 1] == eps]
            if eps_subset.shape[0] > 0:
                plt.plot(eps_subset[:, 0], eps_subset[:, 4], 
                         label=f"eps={eps}, min_samples={min_samples}", marker='o')

    plt.title("Silhouette Scores")
    plt.xlabel("UMAP Dimensions")
    plt.ylabel("Silhouette Score")
    plot_vertical_line(plt, best_dim)
    plt.legend()
    plt.grid(True)
    save_plot(fig, "Silhouette Scores")

    # Plot largest cluster density
    fig = plt.figure(figsize=(10, 6))
    for min_samples in min_samples_values:
        subset = scores[scores[:, 2] == min_samples]
        for eps in eps_values:
            eps_subset = subset[subset[:, 1] == eps]
            if eps_subset.shape[0] > 0:
                plt.plot(eps_subset[:, 0], eps_subset[:, 6], 
                         label=f"eps={eps}, min_samples={min_samples}", marker='o')

    plt.title("Largest Cluster Density")
    plt.xlabel("UMAP Dimensions")
    plt.ylabel("Density")
    plot_vertical_line(plt, best_dim)
    plt.legend()
    plt.grid(True)
    save_plot(fig, "Largest Cluster Density")

    # Plot noise ratio and adaptive noise threshold as subplots in the same figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    for min_samples in min_samples_values:
        subset = scores[scores[:, 2] == min_samples]
        for eps in eps_values:
            eps_subset = subset[subset[:, 1] == eps]
            if eps_subset.shape[0] > 0:
                # Plot noise ratio
                axs[0].plot(eps_subset[:, 0], eps_subset[:, 5], 
                            label=f"eps={eps}, min_samples={min_samples}", marker='o')
                # Plot adaptive noise threshold
                axs[1].plot(eps_subset[:, 0], eps_subset[:, 7], 
                            label=f"eps={eps}, min_samples={min_samples}", marker='o')

    axs[0].set_title("Noise Ratios")
    axs[0].set_xlabel("UMAP Dimensions")
    axs[0].set_ylabel("Noise Ratio")
    plot_vertical_line(axs[0], best_dim)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title("Adaptive Noise Thresholds")
    axs[1].set_xlabel("UMAP Dimensions")
    axs[1].set_ylabel("Threshold")
    plot_vertical_line(axs[1], best_dim)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    save_plot(fig, "Noise Ratios and Adaptive Noise Thresholds")


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
        transforms.Resize((224, 224)),
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
        pretrained = True
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=2) #pretrained=False
        
        if not pretrained:
            save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier_raw_best_.pth')#f'{model_name}_classifier{clean_dataset}_{method}_best.pth')
            if os.path.exists(save_path_best):
                model.load_state_dict(torch.load(save_path_best))
                print("Model correctly loaded")
            else:
                print("Model not found")
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

        print(f"{main_insect_class} model correctly initialized")

        # Extract features from encoding latents
        (
            latents_cnn_train, 
            labels_cnn_train, 
            real_labels_cnn_train, 
            measurement_noise_cnn_train, 
            mislabeled_cnn_train
        ) = extract_features_from_dataloader(train_loader, model)

        # Find optimal umap dim
        # optimal_dim = find_optimal_umap_dimension(latents_cnn_train)
        optimal_dim, optimal_eps, optimal_min_samples = find_optimal_umap_dbscan(
            main_insect_class,
            cp.asarray(latents_cnn_train), 
            save_dir=os.path.join(config["logging_params"]["save_dir"], "umap_dbscan_analysis")
        )
        print(f"\nRecommended UMAP Dimension: {optimal_dim}, eps: {optimal_eps}, min_samples: {optimal_min_samples}")

        print(f"Recommended UMAP Dimension: {optimal_dim}")

        latents_cnn_train_relevant = cuml.manifold.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=optimal_dim,
            random_state=42
        ).fit_transform(latents_cnn_train)

        # # Apply PCA
        # pca = PCA(n_components=512, random_state=42)
        # pca.fit(latents_cnn_train)

        # # Identify features that contribute the least
        # explained_variance = pca.explained_variance_ratio_
        # high_variance_features = np.where(explained_variance >= np.mean(explained_variance))[0]
        # print(f'High variance features: {high_variance_features}')

        # # Filter the features
        # latents_cnn_train_relevant = latents_cnn_train[:, high_variance_features]

        # # Filter according to UMAP importance
        # importance_scores, important_features = umap_permutation_importance(latents_cnn_train_relevant)
        # threshold = 0.5  # Keep features with importance > 0.5
        # significant_features = np.where(importance_scores > threshold)[0]

        # # Filter the data
        # latents_cnn_train_relevant = latents_cnn_train[:, significant_features]
        # print("Filtered data shape:", latents_cnn_train_relevant.shape)

        # Plot UMAP 2D before and after filtering
        reducer_2_d = cuml.manifold.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)    
        latents_2d_train = reducer_2_d.fit_transform(latents_cnn_train)
        if latents_cnn_train_relevant.shape[1] > 2:
            latents_2d_train_relevant = reducer_2_d.fit_transform(latents_cnn_train_relevant)
        else:
            latents_2d_train_relevant = latents_cnn_train_relevant

        measurement_noise_cnn_train_ = measurement_noise_cnn_train.astype(int)*2
        mislabeled_cnn_train_ = mislabeled_cnn_train.astype(int)
        labels_noise_train = measurement_noise_cnn_train_ + mislabeled_cnn_train_
 
        visualize_train_latent_space(
            latents_2d_train, 
            labels_noise_train, 
            f'{model_name}_{main_insect_class}_train', 
            config["logging_params"]["save_dir"]
        )

        visualize_train_latent_space(
            latents_2d_train_relevant, 
            labels_noise_train, 
            f'{model_name}_{main_insect_class}_train', 
            config["logging_params"]["save_dir"]
        )

        # get_outlier_methods_csv(
        #     latents_cnn_train_relevant,
        #     measurement_noise_cnn_train.astype(int),
        #     mislabeled_cnn_train.astype(int),
        #     f'reduced_resnet_512d_{main_insect_class}',
        #     dirname=config["logging_params"]["save_dir"]
        # )

        y_true = (measurement_noise_cnn_train + mislabeled_cnn_train).astype(int)
        od_methods = ['DBSCAN', 'MCD']#, 'MCD']
        for od_method in od_methods:
            metrics = {}
            if od_method != 'DBSCAN':
                y_pred, metrics = get_outlier_predictions(
                    latents_cnn_train_relevant,
                    y_true,
                    model=od_method,
                    # contamination= 0.2
                )
            elif od_method == 'DBSCAN':
                y_pred = DBSCAN_OD(latents_cnn_train_relevant, eps=optimal_eps, min_samples=optimal_min_samples)
                metrics = metric(y_true=y_true, y_score=y_pred, pos_label=1)      

            visualize_y_true_vs_y_pred_umap(
                latents_cnn_train, 
                measurement_noise_cnn_train, 
                mislabeled_cnn_train,
                y_pred, 
                filename=f'reduced_resnet_{main_insect_class}_{od_method}',
                dirname=config["logging_params"]["save_dir"]
            )
            
        print('')