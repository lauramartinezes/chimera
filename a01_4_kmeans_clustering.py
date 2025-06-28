import os
import random
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from mnist_dataset import CustomBinaryInsectDF
from a01_1_train_test_classifier import SimpleCNN, compute_accuracy


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

def update_mislabeled_flags(row, mislabeled_col, wmv_value, c_value):
    if row[mislabeled_col]:
        row['mislabeled_wmv'] = wmv_value
        row['mislabeled_c'] = not wmv_value
    else:
        row['mislabeled_wmv'] = False
        row['mislabeled_c'] = False
    return row

def compute_predictions(model, data_loader, device):
    all_predictions = []
    all_actuals = []
    all_probs = []

    for images, labels, real_label, measurement_noise, label_noise, outlier  in tqdm.tqdm(data_loader, desc="Computing predictions", total=len(data_loader)):
            
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        all_predictions.extend(predicted_labels.cpu().numpy())
        all_actuals.extend(labels.cpu().numpy())
        all_probs.extend(probas.cpu().numpy())

    return all_predictions, all_actuals, all_probs


def extract_features_from_dataloader(dataloader, model):
    all_features = []  # To store features from all batches
    all_labels = []    # To store labels if needed
    all_real_labels = []
    all_measurement_noise = []
    all_mislabeled = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels, real_labels, measurement_noise, mislabeled, outliers in tqdm.tqdm(dataloader, desc="Extracting features", total=len(dataloader)):
            images = images.to(DEVICE)
            # Forward pass to get the features
            features = model(images)  # Get features for the batch
            features = features.cpu()
            features_np = features.numpy().reshape(features.shape[0], -1)  # Flatten to 2D array
            all_features.append(features_np)
            all_labels.append(labels.numpy())  # Collect labels if needed
            all_real_labels.append(real_labels.numpy())  # Collect labels if needed
            all_measurement_noise.append(measurement_noise.numpy())  # Collect labels if needed
            all_mislabeled.append(mislabeled.numpy())  # Collect labels if needed

    # Stack all features and labels vertically
    return np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_real_labels), np.concatenate(all_measurement_noise), np.concatenate(all_mislabeled)


##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 512
NUM_EPOCHS = 20

# Architecture
NUM_CLASSES = 2

# Other
DEVICE = "cuda:0"
GRAYSCALE = True


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0

    ##########################
    ### BINARY CLASS INSECT DATASET
    ##########################
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    insect_classes = config["data_params"]["data_classes"]
    clean_dataset = ''
    method = 'ae' 
    subsets = ['train', 'val']

    ##########################
    ### RESNET-18 MODEL
    ##########################
    torch.manual_seed(RANDOM_SEED)
    model_name = 'vgg16'  
    #model = SimpleCNN()
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier_raw_best__.pth')#f'{model_name}_classifier{clean_dataset}_{method}_best.pth')
    
    # Load the model
    if os.path.exists(save_path_best):
        model.load_state_dict(torch.load(save_path_best))
        print("Model correctly loaded")
    else:
        print("Model not found")
    model.to(DEVICE)

    dfs_insects = []
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]
        dfs_subset = []
        for subset in subsets:
            df_subset_path = os.path.join('data', f'df_{subset}_raw_{main_insect_class}.csv')
            df_subset_i = pd.read_csv(df_subset_path)
            df_subset_i['noisy_outlier_label_original'] = df_subset_i.label
            # Correct labels for training a classifier instead of an outlier detector
            if main_insect_class == 'wmv':
                df_subset_i.label = 0
                df_subset_i['mislabeled_wmv'] = df_subset_i['mislabeled']
                df_subset_i['mislabeled_c'] = False
            if main_insect_class == 'c':
                df_subset_i.label = 1
                df_subset_i['mislabeled_c'] = df_subset_i['mislabeled']
                df_subset_i['mislabeled_wmv'] = False
            dfs_subset.append(df_subset_i)
        
        df_subset = pd.concat(dfs_subset, ignore_index=True)

        df_subset['directory'] = df_subset['filepath'].apply(os.path.dirname)
        dfs_insects.append(df_subset)

    df_insects = pd.concat(dfs_insects, ignore_index=True)
    # Prepare Dataset
    dataset = CustomBinaryInsectDF(df_insects, transform = transform, seed=config["exp_params"]["manual_seed"])
    
    loader = DataLoader(
        dataset, 
        batch_size=config["data_params"]["train_batch_size"], 
        shuffle=False,
        num_workers=config["data_params"]["num_workers"],
        pin_memory=pin_memory
    )

    ##########################
    ### INFERENCE
    ##########################
    model.eval()
    # Extract features
    features, labels, real_labels, measurement_noise, mislabeled = extract_features_from_dataloader(loader, model)

    import umap
    features_2d = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(features)

    # Perform kmeans clustering
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(features_2d)
    # kmeans_labels = kmeans.labels_

    # plot the kmeans clusters in 2D umap space
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # plt.figure(figsize=(10, 10))
    # sns.scatterplot(
    #     x=features_2d[:,0], y=features_2d[:,1],
    #     hue=kmeans_labels,
    #     palette=sns.color_palette("hsv", 2),
    #     legend="full",
    #     alpha=0.3
    # )
    # plt.title(f'UMAP projection of the {model_name} features {subset} {main_insect_class}')
    # plt.show()
    # print(f'UMAP projection of the {model_name} features')

    #plot real labels
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=features_2d[:,0], y=features_2d[:,1],
        hue=mislabeled.astype(int),
        palette=sns.color_palette("hsv", 2),
        legend="full",
        alpha=0.3
    )
    plt.title(f'UMAP projection of the {model_name} real labels {subset} {main_insect_class}')
    plt.show()
    print()