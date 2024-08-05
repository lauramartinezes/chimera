import glob
import os
import pickle
import random
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import umap
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config.config import SAVE_DIR
from kernels import AutoLaplacianKernel, AutoRbfKernel, LinKernel, PolyKernel, RbfKernel
from kmrcd import kMRCD
from utils import load_checkpoint

setting = 'fuji'
batch_size = 256

# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)

class InsectDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_source = 'inaturalist' if 'inaturalist' in img_path.lower() else setting

        if self.transform:
            image = self.transform(image)

        return image, idx, image_source
    
def extract_features(dataloader, model):
    features = []
    model.eval()
    with torch.no_grad():
        for images, _, _ in tqdm(dataloader, 'Extracting features', unit='batch', leave=False):
            images = images.to('cuda')
            outputs = model(images)
            features.append(outputs)

    # Concatenate features from different batches
    return torch.cat(features, dim=0)#.cpu().numpy()


def apply_umap(data, n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=None):
    """
    Apply UMAP dimensionality reduction to the data.


    Parameters:
        data (array-like): The input data to be transformed.
        n_components (int): The number of dimensions for the transformed data (default is 2).
        n_neighbors (int): The number of nearest neighbors to consider in the UMAP algorithm (default is 15).
        min_dist (float): The minimum distance between points in the UMAP embedding (default is 0.1).
        metric (str): The distance metric to use (default is 'euclidean').
        random_state (int or None): The random state for reproducibility (default is None).


    Returns:
        array-like: The transformed data with reduced dimensions.
    """
    # Create a UMAP instance
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)


    # Fit the UMAP model to the data and transform it to the specified number of components
    print("UMAP Fitting")
    return umap_model.fit_transform(data)

def plot_3d_umap(*umaps, labels=None, title='UMAP', caption=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if caption is None:
        caption='Number of samples:\n'

    for i, umap_data in enumerate(umaps):
        label = None if labels is None else labels[i]
        ax.scatter(umap_data[:, 0], umap_data[:, 1], umap_data[:, 2], label=label)
        caption=caption + f' - {label}: {len(umap_data)}\n'

    # Set labels for each axis
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')

    # Set title and legend
    ax.set_title(title)
    if labels is not None:
        ax.legend()
    if caption is not None:
        fig.text(0.1, 0.1, caption)
    plt.show()

def plot_umap(*umaps, labels=None, title='UMAP', caption=None):
    if caption is None:
        caption = 'Number of samples:\n'

    # Detect the dimensionality of the first UMAP dataset
    is_3d = umaps[0].shape[1] == 3

    if is_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots()

    for i, umap_data in enumerate(umaps):
        label = None if labels is None else labels[i]
        ax.scatter(*umap_data.T, label=label)  # Unpack the columns for scatter plot
        caption += f' - {label}: {len(umap_data)}\n'

    # Set labels for each axis
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    if is_3d:
        ax.set_zlabel('UMAP Dimension 3')

    # Set title and legend
    ax.set_title(title)
    if labels is not None:
        ax.legend()
    if caption is not None:
        fig.text(0.1, 0.01 if not is_3d else 0.1, caption, va='bottom')
    plt.show()

def get_nearest_neighbours(reference, target, threshold_distance = 0.5):
    # Calculate Euclidean distance between each point in reference and target
    distances_target_to_reference = np.linalg.norm(target[:, np.newaxis] - reference, axis=2)

    # Find indices of points close to any umap_sticky point
    indices_target_close_to_reference = np.any(distances_target_to_reference < threshold_distance, axis=1)

    # Filter based on the indices
    return target[indices_target_close_to_reference]

def filter_umap(umap_data, threshold=2.5): 
    # Compute z-scores for each dimension
    z_scores = (umap_data - umap_data.mean(axis=0)) / umap_data.std(axis=0)
    # Identify outliers
    outliers_mask = np.abs(z_scores) > threshold
    outliers_indices = np.any(outliers_mask, axis=1)
    # Filter outliers from the original dataset
    return umap_data[~outliers_indices], umap_data[outliers_indices], outliers_indices

# Set Number of Components for Dimensionality Reduction
n_components = 2

output_folder = 'output'
umaps_folder = os.path.join(output_folder, 'umaps')
solution_outlier_detector_folder = os.path.join(output_folder, 'outlier_detections')
outlier_folder = os.path.join(output_folder, 'output_dataset', 'outliers')
inlier_folder = os.path.join(output_folder, 'output_dataset', 'inliers')

os.makedirs(umaps_folder, exist_ok=True)
os.makedirs(solution_outlier_detector_folder, exist_ok=True)
os.makedirs(outlier_folder, exist_ok=True)
os.makedirs(inlier_folder, exist_ok=True)

# Read images that were used to train model
df_sticky_dataset_train_big = pd.read_parquet(f"{SAVE_DIR}/df_train_{setting}.parquet")
df_sticky_dataset_val_big = pd.read_parquet(f"{SAVE_DIR}/df_val_{setting}.parquet")
df_sticky_dataset_test_big = pd.read_parquet(f"{SAVE_DIR}/df_test_{setting}.parquet")
topclasses = df_sticky_dataset_train_big['txt_label'].unique().tolist()

original_fuji_folder = r'D:\All_sticky_plate_images\created_data\fuji_tile_exports'
train_paths = df_sticky_dataset_train_big.filename.to_list()
val_paths = df_sticky_dataset_val_big.filename.to_list()
test_paths = df_sticky_dataset_test_big.filename.to_list()

base_names_val = set(os.path.basename(path) for path in val_paths)
base_names_test = set(os.path.basename(path) for path in test_paths)

# Get subset for single insect species
for species in topclasses:
    if species!='wmv': 
        continue

    df_sticky_dataset_single_species = df_sticky_dataset_train_big[df_sticky_dataset_train_big.txt_label == species]
    sticky_dataset_train_filepaths = df_sticky_dataset_single_species.filename.to_list()

    # original_fuji_species = glob.glob(os.path.join(original_fuji_folder, '*', '*'))
    # sticky_dataset_train_filepaths = [path for path in original_fuji_species if os.path.basename(path) not in base_names_val and os.path.basename(path) not in base_names_test]
    # print(len(original_fuji_species), len(sticky_dataset_train_filepaths))

    umap_file_name = os.path.join(umaps_folder, f'umap_{species}.npy')
    solution_outlier_detector_file_name = os.path.join(solution_outlier_detector_folder, f'solution_outlier_detector_{species}.pkl')
    
    outlier_folder_species = os.path.join(outlier_folder, species)
    inlier_folder_species = os.path.join(inlier_folder, species)

    os.makedirs(outlier_folder_species, exist_ok=True)
    os.makedirs(inlier_folder_species, exist_ok=True)

    if os.path.exists(umap_file_name): 
        umap_sticky = np.load(umap_file_name)
    else:
        # Load pretrained model
        model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True, num_classes=len(topclasses))
        train_on_gpu = torch.cuda.is_available()
        print(f'Train on gpu: {train_on_gpu}')  # Number of gpus
        model = model.to('cuda', dtype=torch.float)

        # Load Checkpoint from WP1 Imagenet+All
        saved_model = 'fuji_tf_efficientnetv2_m.in21k_ft_in1k_pretrained_imagenet_None_seed_x_augmented'
        class_sample_count = np.unique(df_sticky_dataset_train_big.label, return_counts=True)[1]
        weight = 1. / class_sample_count
        criterion = nn.CrossEntropyLoss(
            label_smoothing=.15, weight=torch.Tensor(weight).cuda())
        optimizer = optim.AdamW(model.parameters(), lr=.003, weight_decay=0.01)
        model, optimizer = load_checkpoint(
            f'models/{saved_model}_best.pth.tar', model, optimizer)

        # Modify model architecture to remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model = model.to('cuda')

        # Freeze the parameters of the model
        for param in model.parameters():
            param.requires_grad = False

        # Forward pass the dataset through the modified model to extract features
        sticky_dataset_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize(size=(150, 150))])

        dataset_sticky = InsectDataset(file_paths=sticky_dataset_train_filepaths, transform=sticky_dataset_transforms)
        dataloader_sticky = DataLoader(dataset_sticky, batch_size=batch_size, shuffle=False)
        features_sticky = extract_features(dataloader_sticky, model).cpu().numpy()

        # Get UMAP
        umap_sticky = apply_umap(features_sticky, n_components)
        np.save(umap_file_name, umap_sticky)

    # Remove outliers from sticky dataset
    #umap_sticky_inliers, umap_sticky_outliers, outliers_indices = filter_umap(umap_sticky, 2.5)

    # Detect outliers using kmrcd
    if os.path.exists(solution_outlier_detector_file_name):
        with open(solution_outlier_detector_file_name, 'rb') as file:
            solution_outlier_detector = pickle.load(file)
    else:
        alpha = 0.85
        kernel = AutoRbfKernel
        kModel = kernel(umap_sticky)
        poc = kMRCD(kModel)
        solution_outlier_detector = poc.runAlgorithm(umap_sticky, alpha)
        with open(solution_outlier_detector_file_name, 'wb') as file:
            pickle.dump(solution_outlier_detector, file)

    outlier_indices = solution_outlier_detector['flaggedOutlierIndices']
    inlier_indices = list(set(np.arange(umap_sticky.shape[0])) - set(solution_outlier_detector['flaggedOutlierIndices']))

    umap_sticky_outliers = umap_sticky[outlier_indices]
    umap_sticky_inliers = umap_sticky[inlier_indices]

    # Plot umaps
    plot_umap(
        umap_sticky,  
        labels=[f'{setting.capitalize()} {species}'],
        title=f'UMAP Scatter Plot {species}'
    )

    plot_umap(
        umap_sticky_inliers,  
        umap_sticky_outliers,
        labels=['umap_sticky_inliers', 'umap_sticky_outliers'],
        title='UMAP Scatter Plot'
    )

    #outlier_indices = np.where(outliers_indices)[0]
    outlier_filepaths = [sticky_dataset_train_filepaths[i] for i in outlier_indices]
    inlier_filepaths = [sticky_dataset_train_filepaths[i] for i in inlier_indices]

    # Copy each image to the destination folder
    for file_path in outlier_filepaths:
        shutil.copy(file_path, os.path.join(outlier_folder_species, os.path.basename(file_path)))

    for file_path in inlier_filepaths:
        shutil.copy(file_path, os.path.join(inlier_folder_species, os.path.basename(file_path)))

    print('')
