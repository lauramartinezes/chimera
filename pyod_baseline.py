import os
import numpy as np
import pandas as pd
import timm
import torch
from PyOD import PYOD, metric

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from mnist_dataset import CustomBinaryMNISTDF

class ToRGB:
    """Convert a single-channel image to three-channel (RGB) image."""
    def __call__(self, img):
        # Convert single-channel (grayscale) image to three-channel (RGB)
        return img.convert('RGB')
    
def extract_features_from_dataloader(dataloader, model):
    all_features = []  # To store features from all batches
    all_labels = []    # To store labels if needed
    all_real_labels = []
    all_measurement_noise = []
    all_mislabeled = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels, real_labels, measurement_noise, mislabeled, outliers in tqdm(dataloader, desc="Extracting features", total=len(dataloader)):
            # Forward pass to get the features
            features = model(images)  # Get features for the batch
            features_np = features.numpy().reshape(features.shape[0], -1)  # Flatten to 2D array
            all_features.append(features_np)
            all_labels.append(labels.numpy())  # Collect labels if needed
            all_real_labels.append(real_labels.numpy())  # Collect labels if needed
            all_measurement_noise.append(measurement_noise.numpy())  # Collect labels if needed
            all_mislabeled.append(mislabeled.numpy())  # Collect labels if needed

    # Stack all features and labels vertically
    return np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_real_labels), np.concatenate(all_measurement_noise), np.concatenate(all_mislabeled)

######################## DATA PREPARATION ###########################
train_df = pd.read_csv(os.path.join('archive', 'fashion-mnist_train.csv'))
reduced_train_df = train_df[(train_df.label == 0) | (train_df.label == 1)]
reduced_train_df['mislabeled'] = False

# Get 10% of rows for each label using value_counts
n_switch = {label: int(0.1 * len(reduced_train_df[reduced_train_df.label == label])) for label in [0, 1]}

# Loop over labels 0 and 1, switching labels and marking as mislabeled
for label in [0, 1]:
    available_indices = reduced_train_df[(reduced_train_df.label == label) & (~reduced_train_df.mislabeled)].index
    indices_to_switch = np.random.choice(available_indices, n_switch[label], replace=False)
    reduced_train_df.loc[indices_to_switch, ['label', 'mislabeled']] = [1 - label, True]

class_0_train_df = reduced_train_df[reduced_train_df.label == 0]
batch_size = 1024
transform = transforms.Compose([
    ToRGB(),                        # Convert grayscale to RGB
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Resize(224),  # Resize the image to 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])
train_dataset = CustomBinaryMNISTDF(class_0_train_df, transform_percent=0.1, transform = transform, seed=42)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

######################## MODEL ###########################
feature_extractor = timm.create_model('resnet18', pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
feature_extractor.eval()

features, labels, real_labels, measurement_noise, mislabeled = extract_features_from_dataloader(train_loader, feature_extractor)

X_train = features

measurement_noises = measurement_noise.astype(int)
label_noises = mislabeled.astype(int)
y_train = measurement_noises + label_noises

models = ['IForest', 'OCSVM', 'ABOD', 'CBLOF', 'COF', #'AOM',
          'COPOD', 'ECOD',  'FeatureBagging', 'HBOS', 'KNN',
          'LMDD', 'LODA', 'LOF', 'LOCI', #'LSCP', 'MAD',
          'MCD', 'PCA', 'ROD', 'SOD', 'SOS', 'DeepSVDD', #'VAE','AutoEncoder','XGBOD'
           'SOGAAL', 'MOGAAL']

metrics_list = []

for model in models:
    pyod_model = PYOD(seed=42, model_name=model)
    pyod_model.fit(X_train, [])
    anomaly_scores = pyod_model.predict_score(X_train)
    metrics = metric(y_true=y_train, y_score=anomaly_scores, pos_label=1)
    print(f'{model}: {metrics}')

    metrics_list.append({'Model': model, 'aucroc': metrics['aucroc'], 'aucpr': metrics['aucpr']})

# Create a DataFrame from the metrics list
metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

print()