# Import necessary libraries
import numpy as np
import torch
import umap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import argparse
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
from category_encoders import OrdinalEncoder
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config.config import INSECT_LABELS_MAP, SAVE_DIR
from datasets.datasets import InsectImgDataset
from datasets.transforms import set_train_transform
from detect_outliers import InsectDataset, extract_features
from kernels import AutoLaplacianKernel, AutoRbfKernel, LinKernel, PolyKernel, RbfKernel
from kmrcd import kMRCD
from utils import load_checkpoint

saved_model = 'fuji_tf_efficientnetv2_m.in21k_ft_in1k_pretrained_imagenet_None_seed_x_augmented'
saved_model_path = f'models/{saved_model}_best.pth.tar'
setting = 'fuji'
batch_size = 256

# Read images that were used to train model
df_sticky_dataset_train_big = pd.read_parquet(f"{SAVE_DIR}/df_train_{setting}.parquet")
df_sticky_dataset_val_big = pd.read_parquet(f"{SAVE_DIR}/df_val_{setting}.parquet")
df_sticky_dataset_test_big = pd.read_parquet(f"{SAVE_DIR}/df_test_{setting}.parquet")

val_plates = df_sticky_dataset_val_big.platename.unique().tolist()
test_plates = df_sticky_dataset_test_big.platename.unique().tolist()

non_raw_plates = val_plates + test_plates

df_raw_dataset = pd.read_parquet(f"{SAVE_DIR}/df_preparation_fuji_raw.parquet")
oe = OrdinalEncoder(cols=['label'], mapping=[{'col': 'label', 'mapping': {
                    'bl': 0, 'wswl': 1, 'sp': 2, 't': 3, 'sw': 4, 'k': 5, 'm': 6, 'c': 7, 'v': 8, 'wmv': 9, 'wrl': 10, 'other': 11}}])
df_raw_dataset['txt_label'] = df_raw_dataset['label']
df_raw_dataset['label'] = oe.fit_transform(df_raw_dataset.label)

df_raw_dataset_filtered = df_raw_dataset[~df_raw_dataset['platename'].isin(non_raw_plates)]


classes = df_sticky_dataset_train_big['txt_label'].unique().tolist()

# Load my data
raw_dataset = r'D:\All_sticky_plate_images\created_data\fuji_tile_exports'
cleaner_dataset = r'C:\Users\u0159868\Documents\data\photobox_vs_fuji\fuji'

# get filepaths
raw_dataset_filepaths = glob.glob(os.path.join(raw_dataset,'*','*.png'))
cleaner_dataset_filepaths = df_sticky_dataset_train_big.filename.to_list()

if os.path.exists('features_cleaner.npy') and os.path.exists('labels_cleaner.pkl') and os.path.exists('features_raw.npy') and os.path.exists('labels_raw.pkl'):
    with open('labels_cleaner.pkl', 'rb') as f:
        labels_cleaner = pickle.load(f)
    with open('labels_raw.pkl', 'rb') as f:
        labels_raw = pickle.load(f)

    features_cleaner = np.load('features_cleaner.npy')
    features_raw = np.load('features_raw.npy')

else:
    # Create UMAP
    model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True, num_classes=len(classes))
    train_on_gpu = torch.cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')  # Number of gpus
    model = model.to('cuda', dtype=torch.float)

    # Load Checkpoint from WP1 Imagenet+All
    class_sample_count = np.unique(df_sticky_dataset_train_big.label, return_counts=True)[1]
    weight = 1. / class_sample_count
    criterion = nn.CrossEntropyLoss(
        label_smoothing=.15, weight=torch.Tensor(weight).cuda())
    optimizer = optim.AdamW(model.parameters(), lr=.003, weight_decay=0.01)
    model, optimizer = load_checkpoint(saved_model_path, model, optimizer)

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

    dataset_cleaner_sticky = InsectImgDataset(df=df_sticky_dataset_train_big.reset_index(drop=True),transform=set_train_transform())
    #dataset_cleaner_sticky = InsectDataset(file_paths=cleaner_dataset_filepaths, transform=sticky_dataset_transforms)
    dataloader_cleaner_sticky = DataLoader(dataset_cleaner_sticky, batch_size=batch_size, shuffle=False)
    features_cleaner, labels_cleaner = extract_features(dataloader_cleaner_sticky, model)

    dataset_raw_sticky = InsectImgDataset(df=df_raw_dataset.reset_index(drop=True),transform=set_train_transform())
    #dataset_raw_sticky = InsectDataset(file_paths=raw_dataset_filepaths, transform=sticky_dataset_transforms)
    dataloader_raw_sticky = DataLoader(dataset_raw_sticky, batch_size=batch_size, shuffle=False)
    features_raw, labels_raw = extract_features(dataloader_raw_sticky, model)

    np.save('features_cleaner.npy', features_cleaner)
    with open('labels_cleaner.pkl', 'wb') as f:
        pickle.dump(labels_cleaner, f)

    np.save('features_raw.npy', features_raw)
    with open('labels_raw.pkl', 'wb') as f:
        pickle.dump(labels_raw, f)

#labels_cleaner = np.array([INSECT_LABELS_MAP[name] for name in labels_cleaner_txt])
#labels_raw = np.array([INSECT_LABELS_MAP[name] for name in labels_raw_txt])

# Get UMAP
#umap_sticky = apply_umap(features_sticky, labels, n_components)
#np.save(umap_file_name, umap_sticky)

for label in np.unique(labels_cleaner):
    X_train = features_cleaner[labels_cleaner == label]
    X_test = features_raw[labels_raw == label]
    y_train = labels_cleaner[labels_cleaner == label]
    y_test = labels_raw[labels_raw == label]
    # X_train, X_test, y_train, y_test = features_cleaner, features_raw, labels_cleaner, labels_raw

    # Create a UMAP instance with supervised learning
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # Fit UMAP on the training data with labels
    X_train_embedding = umap_model.fit_transform(X_train, y=y_train)

    # Transform the test data into the existing UMAP embedding
    X_test_embedding = umap_model.transform(X_test)

    # UMAP for raw dataset without influence of train
    X_test_embedding_ = umap_model.fit_transform(X_test, y=y_test)

    # Plot the training data embedding
    plt.scatter(X_train_embedding[:, 0], X_train_embedding[:, 1], c=y_train, cmap='Spectral', s=50, alpha=0.7, edgecolor='k', label='Train')

    # Plot the test data embedding
    plt.scatter(X_test_embedding[:, 0], X_test_embedding[:, 1], c=y_test, cmap='Spectral', s=50, alpha=0.7, marker='X', edgecolor='k', label='Test')

    #plt.scatter(X_test_embedding_[:, 0], X_test_embedding_[:, 1], c=y_test, cmap='Spectral', s=50, alpha=0.7, marker='X', edgecolor='k', label='Test')

    plt.legend()
    plt.title('Supervised UMAP Embedding of Training and Test Data')
    plt.show()
