import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
import umap
import yaml

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import set_feature_extraction_transform
from outlier_detectors import UmapHdbscanOD, PYOD, metric
from a04_2_umap_projections_cnn import extract_features_from_dataloader
from datasets import CustomBinaryInsectDF


def clean_df(df, model, device, config, transform, pin_memory, main_insect_class, phase="train", method='ae', od_method='UmapHdbscanOD'):
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

    # Extract features from the dataloader
    (
        latents, 
        labels_cnn, 
        real_labels_cnn, 
        measurement_noise, 
        mislabel_noise 
    ) = extract_features_from_dataloader(loader, model)

    latents_all_dims = latents.copy()

    umap_hdbscan_od = UmapHdbscanOD(
        main_class_name=main_insect_class, 
        save_dir=os.path.join(config["logging_params"]["save_dir"], "umap_hdbscan_analysis", method)
    )

    N = len(latents)
    default_min_cluster_size = max(5, int(0.05 * N))  # Default value for HDBSCAN
    default_min_samples = max(5, int(0.01 * N))  # Default value for HDBSCAN

    if method == 'adbench':
        optimal_min_cluster_size = default_min_cluster_size
        optimal_min_samples = default_min_samples
    elif method == 'adbench_2d':
        optimal_dim = 2
        optimal_min_cluster_size = default_min_cluster_size
        optimal_min_samples = default_min_samples
    else:
        min_cluster_size_values = [max(5, int(p * N)) for p in [0.005, 0.01, 0.02, 0.05]]
        min_samples_values = [max(5, int(p * N)) for p in [0.002, 0.005, 0.01]]

        optimal_dim, optimal_min_cluster_size, optimal_min_samples = umap_hdbscan_od.find_optimal_params(
            latents, 
            dims=[2 ** i for i in range(1, 9)],
            min_cluster_size_values=min_cluster_size_values,
            min_samples_values=min_samples_values,
        )

    y_true = (measurement_noise + mislabel_noise).astype(int)
    metrics = {}
    if od_method != 'UmapHdbscanOD':
        if method !='adbench':
            reducer_optimd = umap.UMAP(n_components=optimal_dim, random_state=42, n_jobs=1)
            latents = reducer_optimd.fit_transform(latents)
        y_pred, metrics = get_outlier_predictions(
            latents,
            y_true,
            model=od_method,
            # contamination= 0.2
        )

    elif od_method == 'UmapHdbscanOD':
        y_pred = umap_hdbscan_od.predict_outliers(
            latents, 
            optimal_dim,
            optimal_min_cluster_size, 
            optimal_min_samples
        )
        metrics = metric(y_true=y_true, y_score=y_pred, pos_label=1)      

    visualize_y_true_vs_y_pred_umap(
        latents_all_dims, 
        measurement_noise, 
        mislabel_noise,
        y_pred, 
        filename=f'{method}_{main_insect_class}_{phase}_{od_method}',
        dirname=config["logging_params"]["save_dir"]
    )

    get_confusion_matrix(
        y_true, 
        y_pred, 
        filename=f'{method}_{main_insect_class}_{phase}_{od_method}', 
        dirname=config["logging_params"]["save_dir"]
    )

    # Mark outliers in the dataframe and transform them from integer to boolean
    df['outlier_detected'] = y_pred
    df['outlier_detected'] = df['outlier_detected'].astype(bool)

    # Filter the dataframe to have only the clean samples
    df_clean = df[y_pred == 0]
    if f'noisy_label_classification' in df_clean.columns:
        reference_label = 0 if main_insect_class == 'wmv' else 1
        df_clean = df_clean[df_clean[f'noisy_label_classification'] == reference_label]
    
    return df_clean, df, metrics


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


def get_outlier_predictions(X_train, y_train, model='MCD', contamination=0.1):
    print(f'{model} outlier extraction')
    pyod_model = PYOD(seed=42, model_name=model, contamination=contamination)
    pyod_model.fit(X_train, [])
    anomaly_scores = pyod_model.predict_score(X_train)
    metrics = metric(y_true=y_train, y_score=anomaly_scores, pos_label=1)
    anomaly_binary_scores = pyod_model.predict_binary_score(X_train)

    return anomaly_binary_scores, metrics


def visualize_y_true_vs_y_pred_umap(features, measurement_noises, label_noises, y_pred, filename=None, dirname=None, detected_label_noises=None):
    umap_folder = os.path.join(dirname, 'True vs Pred UMAPS')
    os.makedirs(umap_folder, exist_ok=True)

    # UMAP transformation (shared latent space for both plots)
    print(f'Starting UMAP 2D reduction')
    reducer_2d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    #reducer_2d = cuml.manifold.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    latents_2d = reducer_2d.fit_transform(features)

    measurement_noises = measurement_noises.astype(int)*2
    label_noises = label_noises.astype(int)
    if detected_label_noises is not None:
        detected_label_noises = detected_label_noises.astype(int)*3
        noises = measurement_noises + label_noises + detected_label_noises
    else:
        noises = measurement_noises + label_noises

    # Dictionary to map numbers to text labels
    true_label_mapping = {0: "Normal Sample", 1: "Label Noise", 2: "Measurement Noise"}
    if detected_label_noises is not None:
        true_label_mapping[3] = "Detected Label Noise"
    txt_true_labels = [true_label_mapping[label] for label in noises]
    
    pred_label_mapping = {0: "Normal Sample", 1: "Outlier"}
    txt_pred_labels = [pred_label_mapping[label] for label in y_pred]

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for y_true
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=txt_true_labels, palette=sns.color_palette("hsv", len(true_label_mapping)), ax=axes[0], legend='full')
    axes[0].set_title("Latent Space UMAP: True Labels")
    axes[0].set_xlabel("UMAP dimension 1")
    axes[0].set_ylabel("UMAP dimension 2")

    # Plot for y_pred
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=txt_pred_labels, palette=sns.color_palette("hsv", 2), ax=axes[1], legend='full')
    axes[1].set_title("Latent Space UMAP: Predicted Labels")
    axes[1].set_xlabel("UMAP dimension 1")
    axes[1].set_ylabel("UMAP dimension 2")

    # Save the figure
    fig.suptitle(filename, fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(umap_folder, f'true_vs_pred_{filename}.png'), format='png')
    plt.savefig(os.path.join(umap_folder, f'true_vs_pred_{filename}.svg'), format='svg')


def get_confusion_matrix(y_true, y_pred, filename=None, dirname=None):
    matrices_folder = os.path.join(dirname, 'True vs Pred Confusion Matrices')
    os.makedirs(matrices_folder, exist_ok=True)

    class_names = ['Inlier', 'Outlier']  # 0: Inlier, 1: Outlier
    cm = confusion_matrix(y_true, y_pred)

    # Normalize by row (true class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set consistent color scale
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', ax=ax, xticklabels=class_names, 
                yticklabels=class_names, vmin=0, vmax=1)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix (Normalized Colors)')
    
    plt.savefig(os.path.join(matrices_folder, f'{filename}_confusion_matrix.png'), format='png')
    plt.savefig(os.path.join(matrices_folder, f'{filename}_confusion_matrix.svg'), format='svg')
    # plt.show()

    # Save the confusion matrix as a numpy array
    np.save(os.path.join(matrices_folder, f"{filename}_confusion_matrix.npy"), cm)



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
    model_name = 'resnet18'
    cnn_types = ['cnn']#'adbench_2d']# 'cnn']  # 'adbench' is used for the AdBench method 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    subsets = ['train', 'val']

    results = []

    for cnn_type in cnn_types:
        transform_cnn = set_feature_extraction_transform()

        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_subsets = []
            for subset in subsets:
                if 'adbench' in cnn_type:
                    df_subset_path = os.path.join('data', f'df_{subset}_raw_{main_insect_class}.csv')
                else:
                    df_subset_path = os.path.join('data', f'df_{subset}_ae_{main_insect_class}.csv')
                df_subset = pd.read_csv(df_subset_path)
                df_subsets.append(df_subset)
            df_train_val = pd.concat(df_subsets, ignore_index=True)

            # CNN model
            model_cnn = timm.create_model('resnet18', pretrained=True)
            model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
            od_methods = ['UmapHdbscanOD', 'MCD'] if cnn_type == 'cnn' else ['OCSVM']
                    
            model_cnn.eval()
            print("Model correctly initialized")

            for od_method in od_methods:
                df_train_val_clean, df_train_val_outliers, metrics = clean_df(
                    df_train_val, 
                    model_cnn, 
                    device, 
                    config, 
                    transform_cnn, 
                    pin_memory, 
                    main_insect_class, 
                    phase='train_val', 
                    method=cnn_type, 
                    od_method=od_method
                )
                
                os.makedirs(os.path.join('data', 'clean'), exist_ok=True)
                os.makedirs(os.path.join('data', 'outliers'), exist_ok=True)
                
                for subset in subsets:
                    if subset == 'train':
                        df_subset_clean = df_train_val_clean[df_train_val_clean.filepath.str.contains('train')]
                        df_subset_outliers = df_train_val_outliers[df_train_val_outliers.filepath.str.contains('train')]
                    elif subset == 'val':
                        df_subset_clean = df_train_val_clean[df_train_val_clean.filepath.str.contains('val')]
                        df_subset_outliers = df_train_val_outliers[df_train_val_outliers.filepath.str.contains('val')]
                    df_subset_clean_path = os.path.join('data', 'clean', f'df_{subset}_{cnn_type}_{main_insect_class}_{od_method}_clean.csv')
                    df_subset_clean.to_csv(df_subset_clean_path, index=False)
                    print(f'Clean {main_insect_class} cnn {subset} dataset available')

                    df_subset_outliers_path = os.path.join('data', 'outliers', f'df_{subset}_{cnn_type}_{main_insect_class}_{od_method}_outliers.csv')
                    df_subset_outliers.to_csv(df_subset_outliers_path, index=False)
                    print(f'Outliers {main_insect_class} cnn {subset} dataset available')

                # Add more information to the metrics dictionary
                metrics['method'] = cnn_type
                metrics['od_method'] = od_method
                metrics['main_insect_class'] = main_insect_class

                results.append(metrics)
                df_results = pd.DataFrame(results)
                df_results.to_csv(os.path.join(config["logging_params"]["save_dir"],f'df_best_od_evaluation.csv'), mode='a', index=False, header=False)
                print('metrics: ', metrics)
            print('')