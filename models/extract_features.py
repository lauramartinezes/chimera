import numpy as np
import torch
from tqdm import tqdm


def extract_features(dataloader, model):
    all_features, all_labels = [], []
    all_measurement_noise, all_mislabeled = [], []

    with torch.no_grad():  
        for images, labels, measurement_noise, mislabeled in tqdm(dataloader, desc="Extracting features", total=len(dataloader)):
            features = model(images)  
            features_np = features.numpy().reshape(features.shape[0], -1)  # Flatten to 2D array
            all_features.append(features_np)
            all_labels.append(labels.numpy())  
            all_measurement_noise.append(measurement_noise.numpy())  
            all_mislabeled.append(mislabeled.numpy())  

    return (
        np.vstack(all_features), 
        np.concatenate(all_labels), 
        np.concatenate(all_measurement_noise), 
        np.concatenate(all_mislabeled)
    )
