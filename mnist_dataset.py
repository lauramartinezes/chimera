import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomBinaryInsectDF(Dataset):
    def __init__(self, df, transform_percent=0, transform=None, seed=None, outlier_indices=None):
        # Set the seed for reproducibility
        if seed is not None:
            self.seed = seed
            self._set_seed(self.seed)

        # Use the provided DataFrame directly
        self.real_labels = df['label'].values
        self.mislabeled = df['mislabeled'].values  # Use the mislabeled column directly
        self.measurement_noise = df['measurement_noise'].values  # Use the mislabeled column directly
        self.labels = self.real_labels.copy()
        self.image_paths = df['filepath'].values
        self.transform = transform

        # Initialize the outlier variable to False for all samples
        self.outliers = np.full(len(self.labels), False)
        
        # Set the outlier flag to True for specified outlier indices
        if outlier_indices is not None:
            self.outliers[outlier_indices] = True

    def _set_seed(self, seed):
        random.seed(seed)  # Set the seed for Python's random module
        np.random.seed(seed)  # Set the seed for NumPy's random module
        torch.manual_seed(seed)  # Set the seed for PyTorch
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Set the seed for all CUDA devices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        real_label = self.real_labels[idx]
        mislabeled = self.mislabeled[idx]  # Get mislabeled status from the column
        outlier = self.outliers[idx] 
        measurement_noise = self.measurement_noise[idx]
        image = Image.open(image_path)

        # If any transforms are provided (like tensor conversion or normalization), apply them
        if self.transform:
            image = self.transform(image)

        return image, label, real_label, measurement_noise, mislabeled, outlier  # Return mislabeled as well
