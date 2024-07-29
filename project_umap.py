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

# Read a subset of Imagenet
imagenet_path = '/home/u0159868/Documents/data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/test'   #'/home/u0159868/Documents/data/imagenet-mini/val'
imagenet_filepaths = glob.glob(imagenet_path + '/**/*.JPEG', recursive=True)
len_imagenet_21k = 14197122 # number taken from wikipedia

# Read images that were used to train subsequent networks
df_inaturalist_train_big = pd.read_parquet('/home/u0159868/Documents/repos/stickybugs-warmstart/data/df_train.parquet')
df_sticky_dataset_train_big = pd.read_parquet(f"{SAVE_DIR}/df_train_{setting}.parquet")

# Get subsets of the size of the smallest set
datasets_size = min(len(imagenet_filepaths), len(df_inaturalist_train_big), len(df_sticky_dataset_train_big))
imagenet_filepaths = random.sample(imagenet_filepaths, datasets_size)
df_inaturalist_train = df_inaturalist_train_big.sample(n=datasets_size, random_state=42)
df_sticky_dataset_train = df_sticky_dataset_train_big.sample(n=datasets_size, random_state=42)

inaturalist_train_file_paths = df_inaturalist_train.filepath.to_list()
sticky_dataset_train_filepaths = df_sticky_dataset_train.filename.to_list()

# Load pretrained model
model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True)

# Modify model architecture to remove the final classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to('cuda')

# Freeze the parameters of the model
for param in model.parameters():
    param.requires_grad = False

# Forward pass your dataset through the modified model to extract features
inaturalist_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Resize(150, antialias=True)])

sticky_dataset_transforms = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize(size=(150, 150))])

dataset_inaturalist = InsectDataset(file_paths=inaturalist_train_file_paths, transform=inaturalist_transforms)
dataset_sticky = InsectDataset(file_paths=sticky_dataset_train_filepaths, transform=sticky_dataset_transforms)
dataset_imagenet = InsectDataset(file_paths=imagenet_filepaths, transform=sticky_dataset_transforms)

dataloader_inaturalist = DataLoader(dataset_inaturalist, batch_size=batch_size, shuffle=False)
dataloader_sticky = DataLoader(dataset_sticky, batch_size=batch_size, shuffle=False)
dataloader_imagenet = DataLoader(dataset_imagenet, batch_size=batch_size, shuffle=False)

features_inaturalist = extract_features(dataloader_inaturalist, model)
features_sticky = extract_features(dataloader_sticky, model)
features_imagenet = extract_features(dataloader_imagenet, model)

# Create dataset identifiers
dataset_identifier_inaturalist = np.zeros((len(features_inaturalist), 1))  # 0 indicates inaturalist
dataset_identifier_sticky = np.ones((len(features_sticky), 1))   # 1 indicates sticky
dataset_identifier_imagenet = np.ones((len(features_imagenet), 1)) * 2   # 2 indicates imagenet

features_all = torch.cat((features_inaturalist, features_sticky, features_imagenet), dim=0).cpu().numpy()
dataset_identifiers_all = np.concatenate((dataset_identifier_inaturalist, dataset_identifier_sticky, dataset_identifier_imagenet), axis=0)

# Permute the features to avoid bias given by order of vectors
random_permutation = np.random.permutation(len(features_all))
features_all_shuffled = features_all[random_permutation]
dataset_identifiers_all_shuffled = dataset_identifiers_all[random_permutation]

# Get UMAP
umap_all = apply_umap(features_all_shuffled)

umap_inaturalist = umap_all[dataset_identifiers_all_shuffled[:,0] == 0]
umap_sticky = umap_all[dataset_identifiers_all_shuffled[:,0] == 1]
umap_imagenet = umap_all[dataset_identifiers_all_shuffled[:,0] == 2]

# Get proportional umaps
percent_inaturalist = len(df_inaturalist_train_big)/ len_imagenet_21k
percent_sticky_dataset = len(df_sticky_dataset_train_big)/ len_imagenet_21k

len_proportional_umap_inaturalist = round(percent_inaturalist * len(umap_imagenet))
len_proportional_umap_sticky_dataset = round(percent_sticky_dataset * len(umap_imagenet))

umap_inaturalist_proportional = umap_inaturalist[np.random.choice(umap_inaturalist.shape[0], size=len_proportional_umap_inaturalist, replace=False)]
umap_sticky_proportional = umap_sticky[np.random.choice(umap_sticky.shape[0], size=len_proportional_umap_sticky_dataset, replace=False)]

# Remove outliers from sticky dataset for a more efficient filtering of inaturalist and imagenet
umap_sticky_inliers, umap_sticky_outliers = filter_umap(umap_sticky, 2.5)
umap_sticky_proportional_inliers, umap_sticky_proportional_outliers = filter_umap(umap_sticky_proportional, 2.5)

# Filter umaps based on closeness to umap_sticky
umap_inaturalist_filtered = get_nearest_neighbours(umap_sticky_inliers, umap_inaturalist)
umap_imagenet_filtered = get_nearest_neighbours(umap_sticky_inliers, umap_imagenet)

umap_inaturalist_proportional_filtered = get_nearest_neighbours(umap_sticky_proportional_inliers, umap_inaturalist_proportional)
umap_imagenet_proportional_filtered = get_nearest_neighbours(umap_sticky_proportional_inliers, umap_imagenet)

# Plot umaps
plot_3d_umap(
    umap_inaturalist, 
    umap_sticky, 
    umap_imagenet, 
    labels=['iNaturalist', setting.capitalize(), 'ImageNet'],
    title='UMAP Scatter Plot'
)

plot_3d_umap( 
    umap_sticky_outliers, 
    umap_sticky_inliers, 
    labels=[f'{setting.capitalize()} outliers', f'{setting.capitalize()} inliers'],
    title=f'UMAP {setting.capitalize()} Inliers and Outliers Scatter Plot'
)

plot_3d_umap( 
    umap_inaturalist_filtered, 
    umap_sticky_inliers, 
    umap_imagenet_filtered, 
    labels=['iNaturalist Filtered', f'{setting.capitalize()} Inliers', 'ImageNet Filtered'],
    title='UMAP filtered Scatter Plot',
)

# Proportional samples
plot_3d_umap(
    umap_inaturalist_proportional, 
    umap_sticky_proportional, 
    umap_imagenet, 
    labels=['iNaturalist', setting.capitalize(), 'ImageNet'],
    title='UMAP Scatter Plot - Proportional'
)

plot_3d_umap( 
    umap_sticky_proportional_outliers, 
    umap_sticky_proportional_inliers, 
    labels=[f'{setting.capitalize()} outliers', f'{setting.capitalize()} inliers'],
    title=f'UMAP {setting.capitalize()} Inliers and Outliers Scatter Plot - Proportional'
)

plot_3d_umap( 
    umap_inaturalist_proportional_filtered, 
    umap_sticky_proportional_inliers, 
    umap_imagenet_proportional_filtered, 
    labels=['iNaturalist Filtered', f'{setting.capitalize()} Inliers', 'ImageNet Filtered'],
    title='UMAP filtered Scatter Plot - Proportional',
)

print('')
