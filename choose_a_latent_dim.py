# Fix the seed for reproducibility
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from mnist_autoencoder import VAE, train_vae
from mnist_dataset import CustomBinaryMNISTDF

def plot_losses(latent_dims, losses):
    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims, losses, marker='o', linestyle='-', color='b', label='Loss')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Loss')
    plt.title('Loss vs. Latent Dimension')
    plt.grid(True)
    plt.legend()

    # Highlighting the minimum loss
    min_loss_dim = latent_dims[losses.index(min(losses))]
    min_loss_value = min(losses)
    plt.annotate(f'Lowest Loss\n({min_loss_dim}, {min_loss_value:.2f})', 
                xy=(min_loss_dim, min_loss_value), 
                xytext=(min_loss_dim, min_loss_value + 0.05 * min(losses)),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=10)

    plt.show()

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

transform_vae = transforms.Compose([transforms.ToTensor()])

train_dataset_vae = CustomBinaryMNISTDF(class_0_train_df, transform_percent=0.1, transform = transform_vae, seed=42)
train_loader_vae = DataLoader(train_dataset_vae, batch_size=batch_size, shuffle=True)
test_loader_vae = DataLoader(train_dataset_vae, batch_size=batch_size, shuffle=False)


######################## MODEL ###########################
# VAE
beta = 1  # Set the beta value for Beta-VAE
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dims = [5, 10, 20, 50, 100, 200, 400]
losses = []
recon_losses = []
kl_losses = []

for latent_dim in latent_dims:
    vae_model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=1e-4, weight_decay=1e-4) #= optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    avg_loss, avg_recon_loss, avg_kl_loss = train_vae(vae_model, train_loader_vae, optimizer, device, num_epochs=num_epochs, beta=beta)
    losses.append(avg_loss)
    recon_losses.append(avg_recon_loss)
    kl_losses.append(avg_kl_loss)

# Plot
plot_losses(latent_dims, losses)
plot_losses(latent_dims, recon_losses)
plot_losses(latent_dims, kl_losses)

print('')