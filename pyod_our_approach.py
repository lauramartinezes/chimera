import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import umap
from PyOD import PYOD, metric

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from mnist_autoencoder import VAE, get_latent_vectors, train_vae, visualize_latent_space, visualize_samples
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

def show_images_side_by_side(image1, image2):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Display the first image
    axs[0].imshow(image1.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    axs[0].axis('off')

    # Display the second image
    axs[1].imshow(image2.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    axs[1].axis('off')

    plt.show()

# Fix the seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

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
transform_resnet = transforms.Compose([
    ToRGB(),                        # Convert grayscale to RGB
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Resize(224),  # Resize the image to 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])
transform_vae = transforms.Compose([transforms.ToTensor()])

train_dataset_vae = CustomBinaryMNISTDF(class_0_train_df, transform_percent=0.1, transform = transform_vae, seed=42)
train_loader_vae = DataLoader(train_dataset_vae, batch_size=batch_size, shuffle=True)
test_loader_vae = DataLoader(train_dataset_vae, batch_size=batch_size, shuffle=False)

train_dataset_resnet = CustomBinaryMNISTDF(class_0_train_df, transform_percent=0.1, transform = transform_resnet, seed=42)
test_loader_resnet = DataLoader(train_dataset_resnet, batch_size=batch_size, shuffle=False)

# Get a single batch from the VAE DataLoader
vae_images, _, _, _, _, _ = next(iter(test_loader_vae))

# Get a single batch from the ResNet DataLoader
resnet_images, _, _, _, _, _ = next(iter(test_loader_resnet))

show_images_side_by_side(vae_images[0], resnet_images[0])

######################## MODEL ###########################
# VAE
beta = 0.00005  # Set the beta value for Beta-VAE
num_epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = BetaVAE(latent_dim=latent_dim, beta=beta).to(device)
vae_model = VAE().to(device)
vae_model_path = os.path.join('outputs', 'vae_model.pth')

if os.path.exists(vae_model_path):
    vae_model.load_state_dict(torch.load(vae_model_path))
else:
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=1e-4, weight_decay=1e-4) #= optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_vae(vae_model, train_loader_vae, optimizer, device, num_epochs=num_epochs, beta=beta)
    torch.save(vae_model.state_dict(), vae_model_path)

latents_vae, labels_vae, measurement_noise_vae, mislabeled_vae = get_latent_vectors(vae_model, test_loader_vae, device)

# Resnet
feature_extractor = timm.create_model('resnet18', pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
feature_extractor.eval()

#latents_resnet, labels_resnet, real_labels_resnet, measurement_noise_resnet, mislabeled_resnet = extract_features_from_dataloader(test_loader_resnet, feature_extractor)

visualize_samples(vae_model, test_loader_vae, device)
# Perform UMAP on latent space
reducer = umap.UMAP(n_components=2)

latents_vae_2d = reducer.fit_transform(latents_vae)
visualize_latent_space(latents_vae_2d, labels_vae, measurement_noise_vae, mislabeled_vae)

#latents_resnet_2d = reducer.fit_transform(latents_resnet)
#visualize_latent_space(latents_resnet_2d, labels_resnet, measurement_noise_resnet, mislabeled_resnet)

models = ['SOD', 'SOS', 'DeepSVDD', #'VAE','AutoEncoder','XGBOD'
           'SOGAAL', 'MOGAAL','IForest', 'OCSVM', 'ABOD', 'CBLOF', 'COF', #'AOM',
          'COPOD', 'ECOD',  'FeatureBagging', 'HBOS', 'KNN',
          'LMDD', 'LODA', 'LOF', #'LOCI', #'LSCP', 'MAD',
          'MCD', 'PCA']#,'ROD']

def get_outlier_methods_csv(X_train, measurement_noises,  label_noises, name=None):
    y_train = measurement_noises + label_noises

    metrics_list = []

    for model in models:
        pyod_model = PYOD(seed=42, model_name=model)
        pyod_model.fit(X_train, [])
        anomaly_scores = pyod_model.predict_score(X_train)
        metrics = metric(y_true=y_train, y_score=anomaly_scores, pos_label=1)
        print(f'{model}: {metrics}')

        metrics_list.append({'Model': model, 'aucroc': metrics['aucroc'], 'aucpr': metrics['aucpr']})
        temp_metrics_df = pd.DataFrame(metrics_list)
        temp_metrics_df.to_csv(f'temp_{name}_metrics.csv', index=False)

get_outlier_methods_csv(latents_vae, measurement_noise_vae.astype(int),  mislabeled_vae.astype(int), 'vae')
#get_outlier_methods_csv(latents_resnet, measurement_noise_resnet.astype(int),  mislabeled_resnet.astype(int), 'resnet')

print('Done')

print()