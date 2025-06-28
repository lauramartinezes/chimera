
import os
import random
import hdbscan
import numpy as np
import pandas as pd
import timm
import torch
import yaml
import seaborn as sns

from umap import UMAP
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from CustomBinaryInsectDF import CustomBinaryInsectDF


def extract_features_from_dataloader(dataloader, model, device):
    all_features = []  # To store features from all batches
    all_labels = []    # To store labels if needed
    all_real_labels = []
    all_measurement_noise = []
    all_mislabeled = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels, real_labels, measurement_noise, mislabeled, outliers in tqdm(dataloader, desc="Extracting features", total=len(dataloader)):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass to get the features
            features = model(images)  # Get features for the batch
            features_np = features.cpu().numpy().reshape(features.shape[0], -1)  # Flatten to 2D array
            all_features.append(features_np)
            all_labels.append(labels.cpu().numpy())  # Collect labels if needed
            all_real_labels.append(real_labels.numpy())  # Collect labels if needed
            all_measurement_noise.append(measurement_noise.numpy())  # Collect labels if needed
            all_mislabeled.append(mislabeled.numpy())  # Collect labels if needed

    # Stack all features and labels vertically
    return np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_real_labels), np.concatenate(all_measurement_noise), np.concatenate(all_mislabeled)


def find_optimal_umap_hdbscan(main_insect_class, data, dims=[2 ** i for i in range(1, 9)],
                               min_cluster_size_values=[5, 10, 20], min_samples_values=[1, 5, 10],
                               save_dir=None):
    scores = []

    for d in dims:
        print(f"\nFitting UMAP with {d} dimensions...")
        reducer = UMAP(n_components=d, random_state=42, n_jobs=1)
        #reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=d, random_state=42)
        embedding = reducer.fit_transform(data)

        # Preservation score
        original_distances = pairwise_distances(data)
        embedded_distances = pairwise_distances(embedding)
        preservation_score = spearmanr(original_distances.flatten(), embedded_distances.flatten()).correlation

        for min_cluster_size in min_cluster_size_values:
            for min_samples in min_samples_values:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                            min_samples=min_samples, gen_min_span_tree=True)
                cluster_labels = clusterer.fit_predict(embedding)

                if hasattr(clusterer, 'relative_validity_'):
                    rv_score = clusterer.relative_validity_
                else:
                    rv_score = -1

                noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
                scores.append((d, min_cluster_size, min_samples, preservation_score, rv_score, noise_ratio))

                print(f"Dim: {d}, min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, "
                      f"Relative Validity: {rv_score:.4f}, Preservation: {preservation_score:.4f}, "
                      f"Noise Ratio: {noise_ratio:.2f}")

    scores = np.array(scores)

    # Filter out invalid scores (e.g., rv_score = -1)
    valid_scores = scores[scores[:, 4] > 0]

    if len(valid_scores) > 0:
        max_rv = np.max(valid_scores[:, 4])
        top_candidates = valid_scores[valid_scores[:, 4] == max_rv]
        best_idx = np.argmin(top_candidates[:, 0])  # Choose lowest dimension
        best_dim, best_mcs, best_ms = top_candidates[best_idx, :3]
    else:
        print("No valid clustering found. Using highest relative validity among all scores as fallback.")
        best_idx = np.argmax(scores[:, 4])
        best_dim, best_mcs, best_ms = scores[best_idx, :3]

    if save_dir is not None:
        plot_umap_hdbscan_results(main_insect_class, scores, min_cluster_size_values, min_samples_values,
                                  best_dim, save_dir=save_dir)
        plot_final_umap_clusters(main_insect_class, data, best_dim, best_mcs, best_ms, save_dir=save_dir)

    print(f"\nOptimal UMAP Dimension: {int(best_dim)}, min_cluster_size: {int(best_mcs)}, min_samples: {int(best_ms)}")
    return int(best_dim), int(best_mcs), int(best_ms)


def plot_umap_hdbscan_results(main_insect_class, scores, min_cluster_size_values, min_samples_values,
                               best_dim, save_dir="umap_hdbscan_plots"):
    os.makedirs(save_dir, exist_ok=True)

    def save_plot(fig, title):
        fig.savefig(os.path.join(save_dir, f"{main_insect_class}_{title}.png"))
        fig.savefig(os.path.join(save_dir, f"{main_insect_class}_{title}.svg"))
        plt.close(fig)

    def plot_vertical_line(ax, best_dim):
        ax.axvline(x=best_dim, color='red', linestyle='--', label=f'Best Dim: {best_dim}')

    scores = np.array(scores)
    dims = scores[:, 0]
    mcs_vals = scores[:, 1]
    ms_vals = scores[:, 2]
    preservation_scores = scores[:, 3]
    rv_scores = scores[:, 4]
    noise_ratios = scores[:, 5]

    # Plot relative validity
    fig, ax = plt.subplots(figsize=(10, 6))
    for ms in min_samples_values:
        for mcs in min_cluster_size_values:
            subset = scores[(scores[:, 1] == mcs) & (scores[:, 2] == ms)]
            if subset.shape[0] > 0:
                ax.plot(subset[:, 0], subset[:, 4],
                        label=f"mcs={int(mcs)}, ms={int(ms)}", marker='o')
    ax.set_title("Relative Validity Scores")
    ax.set_xlabel("UMAP Dimensions")
    ax.set_ylabel("Relative Validity")
    plot_vertical_line(ax, best_dim)
    ax.legend()
    ax.grid(True)
    save_plot(fig, "Relative_Validity")

    # Plot noise ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    for ms in min_samples_values:
        for mcs in min_cluster_size_values:
            subset = scores[(scores[:, 1] == mcs) & (scores[:, 2] == ms)]
            if subset.shape[0] > 0:
                ax.plot(subset[:, 0], subset[:, 5],
                        label=f"mcs={int(mcs)}, ms={int(ms)}", marker='o')
    ax.set_title("Noise Ratios")
    ax.set_xlabel("UMAP Dimensions")
    ax.set_ylabel("Noise Ratio")
    plot_vertical_line(ax, best_dim)
    ax.legend()
    ax.grid(True)
    save_plot(fig, "Noise_Ratio")

    # Plot preservation score
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_dims = sorted(np.unique(dims))
    avg_pres_by_dim = [np.mean(preservation_scores[dims == d]) for d in unique_dims]
    ax.plot(unique_dims, avg_pres_by_dim, marker='o', label="Preservation Score")
    ax.set_title("Average Preservation Score per UMAP Dimension")
    ax.set_xlabel("UMAP Dimensions")
    ax.set_ylabel("Preservation Score (Spearman)")
    plot_vertical_line(ax, best_dim)
    ax.legend()
    ax.grid(True)
    save_plot(fig, "Preservation_Scores")


def plot_final_umap_clusters(main_insect_class, data, best_dim, best_mcs, best_ms, save_dir=None):
    """
    Projects data to 2D UMAP space and visualizes final HDBSCAN clustering using the best parameters.
   
    Parameters:
    - main_insect_class: str, name for labeling/saving plots
    - data: input array (CuPy or NumPy)
    - best_dim: int, optimal UMAP embedding dimension for clustering
    - best_mcs: int, min_cluster_size for HDBSCAN
    - best_ms: int, min_samples for HDBSCAN
    - save_dir: optional path to save plots
    """
    print(f"Running HDBSCAN on {best_dim}D embedding (min_cluster_size={best_mcs}, min_samples={best_ms})...")
    umap_opt = UMAP(n_components=int(best_dim), random_state=42, n_jobs=1)
    #umap_opt = UMAP(n_neighbors=15, min_dist=0.1, n_components=int(best_dim), random_state=42)
    embedding_opt = umap_opt.fit_transform(data)

    print("\nProjecting to 2D UMAP for visualization...")
    if best_dim!=2:
        umap_2d = UMAP(n_components=2, random_state=42, n_jobs=1)
        #umap_2d = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = umap_2d.fit_transform(data)
    else:
        embedding_2d = embedding_opt

    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(best_mcs), min_samples=int(best_ms))
    labels = clusterer.fit_predict(embedding_opt)

    # Color palette and noise handling
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    palette = sns.color_palette('hsv', n_clusters)
    colors = [palette[label] if label != -1 else (0.6, 0.6, 0.6) for label in labels]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=10, alpha=0.7)
    ax.set_title(f"HDBSCAN Clustering in 2D UMAP Space\n(Dim={best_dim}, min_cluster_size={best_mcs}, min_samples={best_ms})")
    ax.axis('off')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{main_insect_class}_HDBSCAN_2D_UMAP.png"))
        fig.savefig(os.path.join(save_dir, f"{main_insect_class}_HDBSCAN_2D_UMAP.svg"))
        plt.close(fig)
    else:
        plt.show()


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
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )

        print(f"{main_insect_class} dataset correctly loaded")

        
        # Resnet method
        model_name = 'resnet18'
        pretrained = True
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=2) #pretrained=False
        model.to(device)
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
        ) = extract_features_from_dataloader(train_loader, model, device)

        N = len(latents_cnn_train)
        min_cluster_size_values = [max(5, int(p * N)) for p in [0.005, 0.01, 0.02, 0.05]]
        min_samples_values = [max(5, int(p * N)) for p in [0.002, 0.005, 0.01]]

        optimal_dim, optimal_min_cluster_size, optimal_min_samples = find_optimal_umap_hdbscan(
            main_insect_class, 
            latents_cnn_train, 
            min_cluster_size_values=min_cluster_size_values,
            min_samples_values=min_samples_values,
            save_dir=os.path.join(config["logging_params"]["save_dir"], "umap_hdbscan_analysis_")
        )
        
        print(f"\nRecommended UMAP Dimension: {optimal_dim}, min_cluster_size: {optimal_min_cluster_size}, min_samples: {optimal_min_samples}")
