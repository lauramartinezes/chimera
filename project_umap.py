import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import umap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config.config import SAVE_DIR

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

def filter_umap(umap_data, threshold=3): 
    # Compute z-scores for each dimension
    z_scores = (umap_data - umap_data.mean(axis=0)) / umap_data.std(axis=0)
    # Identify outliers
    outliers_mask = np.abs(z_scores) > threshold
    outliers_indices = np.any(outliers_mask, axis=1)
    # Filter outliers from the original dataset
    return umap_data[~outliers_indices], umap_data[outliers_indices]

# Set Number of Components for Dimensionality Reduction
n_components = 2

# Read images that were used to train model
df_sticky_dataset_train_big = pd.read_parquet(f"{SAVE_DIR}/df_train_{setting}.parquet")

# Get subset for fast experiments
df_sticky_dataset_train = df_sticky_dataset_train_big.sample(n=1000, random_state=42)

sticky_dataset_train_filepaths = df_sticky_dataset_train.filename.to_list()

# Load pretrained model
model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True)

# TODO: Load Checkpoint from WP1 Imagenet+All

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

# Remove outliers from sticky dataset
umap_sticky_inliers, umap_sticky_outliers = filter_umap(umap_sticky, 2.5)

# Plot umaps
plot_umap(
    umap_sticky,  
    labels=[setting.capitalize()],
    title='UMAP Scatter Plot'
)

print('')
