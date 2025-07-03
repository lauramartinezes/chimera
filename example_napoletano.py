import os
import random
import numpy as np
import pandas as pd
import timm
import torch
import umap
import yaml

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CustomBinaryInsectDF
from models import extract_features
    
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
        ) = extract_features(train_loader, model, device)
        
        # === Step 1: Load ResNet18 feature vectors ===
        # Replace with your actual file path
        X = latents_cnn_train  # shape: (n_samples, 512)

        # Optional: load labels for coloring the scatter plot
        # y = np.load('labels.npy')  # shape: (n_samples,)
        y = None  # Set to None if you don't have labels

        # === Step 2: PCA to retain 95% of the variance ===
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X)
        print("Size of pca components:", pca.n_components_)

        print(f"Original dimension: {X.shape[1]}")
        print(f"Reduced dimension after PCA (M): {X_pca.shape[1]}")

        # === Step 3: Choose visualization method (PCA or UMAP) ===
        # Set this flag to 'pca' or 'umap'
        visualization_method = 'pca'  # options: 'pca' or 'umap'

        if visualization_method == 'pca':
            pca_vis = PCA(n_components=2)
            X_vis = pca_vis.fit_transform(X_pca)
            title = "2D PCA Projection of ResNet18 Features"
        elif visualization_method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_vis = reducer.fit_transform(X_pca)
            title = "2D UMAP Projection of ResNet18 Features"
        else:
            raise ValueError("Invalid visualization method. Choose 'pca' or 'umap'.")

        # === Step 4: Plot the projection ===
        plt.figure(figsize=(8, 6))
        if y is not None:
            scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap='tab10', alpha=0.7, edgecolor='k')
            plt.legend(*scatter.legend_elements(), title="Classes")
        else:
            plt.scatter(X_vis[:, 0], X_vis[:, 1], c='blue', alpha=0.6, edgecolor='k')

        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
