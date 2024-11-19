import glob
import os

import pandas as pd

import timm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import umap

from mnist_autoencoder import VAE, get_latent_vectors, train_vae, visualize_latent_space, visualize_samples
from mnist_dataset import CustomBinaryInsectDF, CustomBinaryMNISTDF
from pyod_our_approach import ToRGB, extract_features_from_dataloader, get_outlier_methods_csv, show_images_side_by_side

good_samples_folder = os.path.join('data', 'wmv_good')
mismeasure_samples_folder = os.path.join('data', 'wmv_trash')
mislabel_samples_folder = os.path.join('data', 'sp_good')

good_samples = glob.glob(os.path.join(good_samples_folder, '*.png'))
mismeasure_samples = glob.glob(os.path.join(mismeasure_samples_folder, '*.png'))
mislabel_samples = glob.glob(os.path.join(mislabel_samples_folder, '*.png'))

df_good_samples = pd.DataFrame({
    'label': 0,
    'mislabeled': False,
    'measurement_noise': False, 
    'filepath': good_samples
})

df_mismeasure_samples = pd.DataFrame({
    'label': 1,
    'mislabeled': False,
    'measurement_noise': True, 
    'filepath': mismeasure_samples
})

df_mislabel_samples = pd.DataFrame({
    'label': 1,
    'mislabeled': True,
    'measurement_noise': False, 
    'filepath': mislabel_samples
})

df_all_samples = pd.concat([df_good_samples, df_mismeasure_samples, df_mislabel_samples], ignore_index=True)

batch_size = 32
transform_resnet = transforms.Compose([
    ToRGB(),                        # Convert grayscale to RGB
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])
transform_vae = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((150, 150))
])

train_dataset_vae = CustomBinaryInsectDF(df_all_samples, transform_percent=0.1, transform = transform_vae, seed=42)
train_loader_vae = DataLoader(train_dataset_vae, batch_size=batch_size, shuffle=True)
test_loader_vae = DataLoader(train_dataset_vae, batch_size=batch_size, shuffle=False)

train_dataset_resnet = CustomBinaryInsectDF(df_all_samples, transform_percent=0.1, transform = transform_resnet, seed=42)
test_loader_resnet = DataLoader(train_dataset_resnet, batch_size=batch_size, shuffle=False)

# Get a single batch from the VAE DataLoader
vae_images, _, _, _, _, _ = next(iter(test_loader_vae))

# Get a single batch from the ResNet DataLoader
resnet_images, _, _, _, _, _ = next(iter(test_loader_resnet))

show_images_side_by_side(vae_images[0], resnet_images[0])

######################## MODEL ###########################
# VAE
latent_dim = 100
beta = 0 #0.00005  # Set the beta value for Beta-VAE
num_epochs = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = BetaVAE(latent_dim=latent_dim, beta=beta).to(device)
vae_model = VAE(in_channels=3, input_size=150, latent_dim=latent_dim).to(device)
vae_model_path = os.path.join('outputs', 'vae_insect_model.pth')

if os.path.exists(vae_model_path):
    vae_model.load_state_dict(torch.load(vae_model_path))
else:
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=1e-4, weight_decay=1e-4) #= optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_vae(vae_model, train_loader_vae, optimizer, device, num_epochs=num_epochs, beta=beta)
    torch.save(vae_model.state_dict(), vae_model_path)

# latents_vae, labels_vae, measurement_noise_vae, mislabeled_vae = get_latent_vectors(vae_model, test_loader_vae, device)

# Resnet
feature_extractor = timm.create_model('resnet18', pretrained=True)
feature_extractor = torch.nn.Sequential(*(list(feature_extractor.children())[:-1]))
feature_extractor.eval()

latents_resnet, labels_resnet, real_labels_resnet, measurement_noise_resnet, mislabeled_resnet = extract_features_from_dataloader(test_loader_resnet, feature_extractor)

visualize_samples(vae_model, test_loader_vae, device)
# Perform UMAP on latent space
reducer = umap.UMAP(n_components=2)

# latents_vae_2d = reducer.fit_transform(latents_vae)
# visualize_latent_space(latents_vae_2d, labels_vae, measurement_noise_vae, mislabeled_vae)

latents_resnet_2d = reducer.fit_transform(latents_resnet)
visualize_latent_space(latents_resnet_2d, labels_resnet, measurement_noise_resnet, mislabeled_resnet)

#get_outlier_methods_csv(latents_vae, measurement_noise_vae.astype(int),  mislabeled_vae.astype(int), 'vae_insect')
get_outlier_methods_csv(latents_resnet, measurement_noise_resnet.astype(int),  mislabeled_resnet.astype(int), 'resnet')

print('Done')

print()