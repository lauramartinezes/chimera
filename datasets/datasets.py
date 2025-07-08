import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class NoisyImageDataset(Dataset):
    def __init__(self, df, transform=None, seed=None):
        # Set the seed for reproducibility
        if seed is not None:
            self.seed = seed
            self._set_seed(self.seed)

        # Use the provided DataFrame directly
        self.labels = df['label'].values
        self.mislabeled = df['mislabeled'].values  
        self.measurement_noise = df['measurement_noise'].values  
        self.image_paths = df['filepath'].values
        self.transform = transform

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
        mislabeled = self.mislabeled[idx]  
        measurement_noise = self.measurement_noise[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label, measurement_noise, mislabeled  
