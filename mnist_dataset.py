import pandas as pd
import torch
from torch.utils.data import Dataset
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

class CustomMNISTCSV(Dataset):
    def __init__(self, csv_file, transform_percent=0, wrong_label_percent=0, transform=None):
        data = pd.read_csv(csv_file)
        
        # Split into labels and images
        self.labels = data['label'].values
        self.images = data.drop('label', axis=1).values.reshape(-1, 28, 28).astype(np.uint8)
        self.transform = transform

        # Calculate how many samples to modify
        self.num_transform = int(transform_percent * len(self.images))  # Number of images to apply random transforms
        self.num_wrong_label = int(wrong_label_percent * len(self.labels))  # Number of labels to corrupt

        # Randomly select indices for transformation and wrong labels
        self.transform_indices = random.sample(range(len(self.images)), self.num_transform)
        self.wrong_label_indices = random.sample(range(len(self.labels)), self.num_wrong_label)
        
        # Corrupt the labels
        self._corrupt_labels()

    def _corrupt_labels(self):
        # Change the label of the selected wrong_label_indices
        for idx in self.wrong_label_indices:
            original_label = self.labels[idx]
            new_label = random.choice([i for i in range(10) if i != original_label])  # Avoid the same label
            self.labels[idx] = new_label

    def _apply_random_transform(self, image):
        # Convert numpy array to PIL image
        pil_image = Image.fromarray(image)

        # Define strong transformations using PIL functions
        transform_list = [
            lambda img: img.filter(ImageFilter.GaussianBlur(radius=3)),                     # Strong blur with large radius
            lambda img: ImageEnhance.Brightness(img).enhance(10),                          # Very dark image (low brightness)
            lambda img: ImageOps.equalize(img),                                             # Histogram equalization
            lambda img: self._add_noise(img, scale=100)                                     # Very strong noise
        ]
        
        # Apply a random transformation from the list
        random_transform = random.choice(transform_list)
        transformed_image = random_transform(pil_image)

        return np.array(transformed_image)  # Convert back to numpy array

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

        # Apply random non-geometric transforms to the selected indices
        if idx in self.transform_indices:
            image = self._apply_random_transform(image)
            measurement_noise = True

        # Check if the label has been noise
        if idx in self.wrong_label_indices:
            label_noise = True  # The label has already been noise in _corrupt_labels

        # If any transforms are provided (like tensor conversion or normalization), apply them
        if self.transform:
            image = self.transform(image)

        return image, label, measurement_noise, label_noise  # Return the flags as well

    # Add a function to plot a batch of images
    def plot_images(self, images, labels, measurement_noise, label_noise, num_images=16):
        plt.figure(figsize=(12, 12))
        for i in range(num_images):
            plt.subplot(4, 4, i + 1)
            image = images[i].squeeze(0).numpy()  # Remove channel dimension and convert to numpy
            label = labels[i].item()
            measurement_noise_ = measurement_noise[i].item()
            label_noise_ = label_noise[i]  # Use the label change flag
            plt.imshow(image, cmap="gray")
            plt.title(f'Label: {label}\n Measurement Noise: {measurement_noise_} | Label Noise: {label_noise_}')
            plt.axis("off")
        plt.show()


# Example usage:
# Preprocessing transformation (convert image to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Optional normalization
])

if __name__ == '__main__':
    # Create the custom dataset with 25% of images transformed with very strong non-geometric transforms and 10% wrong labels
    dataset = CustomMNISTCSV(csv_file=os.path.join('digit-recognizer','train.csv'), transform_percent=0.25, wrong_label_percent=0.1, transform=transform)

    # Test by loading a batch from the dataset
    from torch.utils.data import DataLoader

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Check a batch of data
    images, labels, measurement_noise, label_noise = next(iter(train_loader))

    # Plot the first 16 images
    dataset.plot_images(images, labels, measurement_noise, label_noise, num_images=16)
