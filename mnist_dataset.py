import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import random
from torchvision import transforms
import matplotlib.pyplot as plt

class CustomMNISTDF(Dataset):
    def __init__(self, df, transform_percent=0, wrong_label_percent=0, transform=None, seed=None):
        # Set the seed for reproducibility
        if seed is not None:
            self.seed = seed
            self._set_seed(self.seed)

        # Use the provided DataFrame directly
        self.labels = df['label'].values
        self.images = df.drop('label', axis=1).values.reshape(-1, 28, 28).astype(np.uint8)
        self.transform = transform

        # Calculate how many samples to modify
        self.transform_percent = transform_percent
        self.wrong_label_percent = wrong_label_percent

        # Prepare indices per class
        self._prepare_indices_per_class()

        # Precompute transformations for the transform_indices
        self.precomputed_transforms = {}
        self._precompute_transforms()

        # Corrupt the labels
        self._corrupt_labels()

    def _set_seed(self, seed):
        random.seed(seed)  # Set the seed for Python's random module
        np.random.seed(seed)  # Set the seed for NumPy's random module
        torch.manual_seed(seed)  # Set the seed for PyTorch
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # Set the seed for all CUDA devices

    def _prepare_indices_per_class(self):
        # Dictionary to store indices for each class
        class_indices = {}
        for class_label in np.unique(self.labels):
            class_indices[class_label] = np.where(self.labels == class_label)[0]

        self.transform_indices = []
        self.wrong_label_indices = []

        # For each class, select a proportional number of samples for transformations and wrong labels
        for class_label, indices in class_indices.items():
            # Calculate how many transformations and wrong labels to apply for this class
            num_transform_class = int(self.transform_percent * len(indices))
            num_wrong_label_class = int(self.wrong_label_percent * len(indices))

            # Randomly select indices for transformations and wrong labels
            class_transform_indices = random.sample(list(indices), num_transform_class)
            transform_set = set(class_transform_indices)
            available_class_indices = list(set(indices) - transform_set)
            class_wrong_label_indices = random.sample(available_class_indices, num_wrong_label_class)

            # Add to global lists
            self.transform_indices.extend(class_transform_indices)
            self.wrong_label_indices.extend(class_wrong_label_indices)

    def _precompute_transforms(self):
        # Define strong transformations using PIL functions
        transform_list = [
            lambda img: img.filter(ImageFilter.GaussianBlur(radius=3)),  # Strong blur with large radius
            lambda img: self._add_noise(img, scale=100)                  # Very strong noise
        ]
        
        # Assign a random transform for each transform index
        for idx in self.transform_indices:
            self.precomputed_transforms[idx] = random.choice(transform_list)

    def _corrupt_labels(self):
        # Change the label of the selected wrong_label_indices
        for idx in self.wrong_label_indices:
            original_label = self.labels[idx]
            new_label = random.choice([i for i in range(10) if i != original_label])  # Avoid the same label
            self.labels[idx] = new_label

    def _apply_precomputed_transform(self, image, idx):
        # Convert numpy array to PIL image
        pil_image = Image.fromarray(image)

        # Apply the precomputed transformation for this index
        if idx in self.precomputed_transforms:
            transformed_image = self.precomputed_transforms[idx](pil_image)
            return np.array(transformed_image)
        return image

    def _add_noise(self, img, scale=25):
        # Convert PIL image to numpy array
        img_np = np.array(img)
        # Generate very strong Gaussian noise
        noise = np.random.normal(loc=0, scale=scale, size=img_np.shape)
        noisy_image = img_np + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Ensure values are in valid range
        return Image.fromarray(noisy_image)  # Convert back to PIL image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Flags for measurement noise and label change
        measurement_noise = False
        label_noise = False

        # Apply precomputed transformations to the selected indices
        if idx in self.transform_indices:
            image = self._apply_precomputed_transform(image, idx)
            measurement_noise = True

        # Check if the label has been noise
        if idx in self.wrong_label_indices:
            label_noise = True  # The label has already been noise in _corrupt_labels

        # If any transforms are provided (like tensor conversion or normalization), apply them
        if self.transform:
            image = self.transform(image)

        return image, label, measurement_noise, label_noise  # Return the flags as well


class ClassSubset(Dataset):
    def __init__(self, original_dataset, class_label):
        # Get indices of all samples that belong to the target class
        self.class_indices = np.where(original_dataset.labels == class_label)[0]

        # Filter the images, labels, and flags from the original dataset
        self.images = original_dataset.images[self.class_indices]
        self.labels = original_dataset.labels[self.class_indices]
        self.transform_indices = [i for i in self.class_indices if i in original_dataset.transform_indices]
        self.wrong_label_indices = [i for i in self.class_indices if i in original_dataset.wrong_label_indices]

        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Map idx in class_a_dataset back to the corresponding index in the original dataset
        original_idx = self.class_indices[idx]

        # Get image, label, and flags from the original dataset
        image, label, measurement_noise, label_noise = self.original_dataset[original_idx]

        return image, label, measurement_noise, label_noise

def plot_images(images, labels, measurement_noise, label_noise, num_images=16):
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        image = images[i].squeeze(0).numpy()  # Remove channel dimension and convert to numpy
        label = labels[i].item()
        measurement_noise_ = measurement_noise[i].item()
        label_noise_ = label_noise[i]  # Use the label change flag
        plt.imshow(image, cmap="gray")
        plt.title(f'Label: {label} | Label Noise: {label_noise_}')
        plt.axis("off")
    plt.show()
    
def plot_comparison(original_dataset, class_a_dataset, class_indices, num_images=16):
    plt.figure(figsize=(12, 24))
    
    for i in range(num_images):
        # Index in the class_a_dataset
        class_a_idx = i
        
        # Corresponding index in the original dataset
        original_idx = class_indices[class_a_idx]
        
        # Get image, label, and flags from the original dataset
        original_image, original_label, original_transform_flag, original_label_flag = original_dataset[original_idx]
        
        # Get image, label, and flags from the class_a_dataset
        class_a_image, class_a_label, class_a_transform_flag, class_a_label_flag = class_a_dataset[class_a_idx]
        
        # Plot image from original dataset
        plt.subplot(num_images, 2, i * 2 + 1)
        plt.imshow(original_image, cmap="gray")
        plt.title(f'Original | Label: {original_label} | Transformed: {original_transform_flag} | Wrong Label: {original_label_flag}')
        plt.axis("off")
        
        # Plot image from class_a_dataset
        plt.subplot(num_images, 2, i * 2 + 2)
        plt.imshow(class_a_image, cmap="gray")
        plt.title(f'Class A | Label: {class_a_label} | Transformed: {class_a_transform_flag} | Wrong Label: {class_a_label_flag}')
        plt.axis("off")
    
    plt.show()

# Example usage:
# Preprocessing transformation (convert image to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Optional normalization
])

if __name__ == '__main__':
    df = pd.read_csv(os.path.join('archive', 'fashion-mnist_train.csv'))

    print(f'Number of samples: {len(df)}')
    # Split the DataFrame into train and test sets
    train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
    test_df = df.drop(train_df.index)  # Remaining 20% for testing

    # Create the custom dataset instances
    train_dataset = CustomMNISTDF(train_df, transform_percent=0.25, wrong_label_percent=0.10, seed=42)
    test_dataset = CustomMNISTDF(test_df, seed=42)
    # Get a subset of the dataset with only class 'A'
    class_a_dataset =  ClassSubset(train_dataset, class_label=1)

    # Test by loading a batch from the training dataset
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    class_a_loader = DataLoader(class_a_dataset, batch_size=64, shuffle=True)

    # Check a batch of data
    images, labels, measurement_noise, label_noise = next(iter(train_loader))
    plot_images(images, labels, measurement_noise, label_noise, num_images=16)

    images_class, labels_class, measurement_noise_class, label_noise_class = next(iter(class_a_loader))
    plot_images(images_class, labels_class, measurement_noise_class, label_noise_class, num_images=16)
    
    # Get indices of class A samples in the original dataset
    class_indices = np.where(train_dataset.labels == 1)[0]  # Assuming class A is label '1'

    # Plot comparison between original dataset and class A subset
    plot_comparison(train_dataset, class_a_dataset, class_indices, num_images=8)
    print('')
