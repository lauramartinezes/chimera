import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

# Beta-VAE model
class BetaVAE(nn.Module):
    def __init__(self, input_dims=(1, 28, 28), latent_dim=5, beta=1.0):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input (1, 28, 28), Output (32, 28, 28)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # Output (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # Output (64, 14, 14)
            nn.ReLU(),
            nn.Flatten()  # Flatten the output for dense layers
        )

        # Fully connected layers to get mean (mu) and log variance (logvar)
        self.fc_mu = nn.Linear(64 * 14 * 14, latent_dim)  # mu layer
        self.fc_logvar = nn.Linear(64 * 14 * 14, latent_dim)  # logvar layer

        # Fully connected layer for decoding
        self.fc_decode = nn.Linear(latent_dim, 64 * 14 * 14)

        # Decoder (Deconvolution / Transposed Convolution)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output (64, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),  # Output (64, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output (32, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # Output (1, 28, 28)
            nn.Sigmoid()  # Use Sigmoid to ensure output is in [0, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation from log variance
        eps = torch.randn_like(std)  # Sample from a standard normal distribution
        return mu + eps * std  # Apply the reparameterization trick

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(-1, 64, 14, 14)  # Reshape to match input dimensions for the decoder
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)  # Get the mean and log variance from the encoder
        z = self.reparameterize(mu, logvar)  # Sample from the latent space
        recon_x = self.decode(z)  # Reconstruct the image from the latent space
        return recon_x, mu, logvar

# Custom loss function for Beta-VAE
def beta_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss (weighted by beta)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

batch_size = 1024
num_epochs=10

# Data loading (MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and device
latent_dim = 5  # Size of latent space
beta = 0.00005  # Set the beta value for Beta-VAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BetaVAE(latent_dim=latent_dim, beta=beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train_vae(model, train_loader, optimizer, num_epochs=10, beta=4.0):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for data, _ in train_loader:
            data = data.to(device)

            # Forward pass
            optimizer.zero_grad()
            recon_data, mu, logvar = model(data)
            
            # Compute loss
            loss, recon_loss, kl_loss = beta_vae_loss(recon_data, data, mu, logvar, beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}')

# Train the Beta-VAE model
train_vae(model, train_loader, optimizer, num_epochs=num_epochs, beta=beta)

# Visualization Functions

def visualize_samples(model, test_loader):
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        for batch, _ in test_loader:
            data = batch.to(device)
            recon_data, _, _ = model(data)
            data = data.cpu()
            recon_data = recon_data.cpu()
            break

    fig, axes = plt.subplots(2, 10, figsize=(18, 4))

    for i in range(10):
        # Original images
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        
        # Reconstructed images
        axes[1, i].imshow(recon_data[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")
        
    plt.show()

# t-SNE Latent Space Visualization

def visualize_latent_space(model, test_loader, num_samples=1000):
    model.eval()
    latents, labels = [], []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            _, mu, _ = model(data)
            latents.append(mu.cpu())
            labels.append(label)
            if len(latents) * len(data) >= num_samples:
                break

    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)

    # Perform t-SNE on latent space
    tsne = TSNE(n_components=2)
    latents_2d = tsne.fit_transform(latents)

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=labels, palette=sns.color_palette("hsv", 10), legend='full')
    plt.title("Latent Space t-SNE Visualization")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.show()

import umap

# Existing code from previous implementation (BetaVAE model, loss function, etc.)

# UMAP Latent Space Visualization
def visualize_latent_space_umap(model, test_loader, num_samples=1000):
    model.eval()
    latents, labels = [], []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            _, mu, _ = model(data)
            latents.append(mu.cpu())
            labels.append(label)
            if len(latents) * len(data) >= num_samples:
                break

    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)

    # Perform UMAP on latent space
    reducer = umap.UMAP(n_components=2)
    latents_2d = reducer.fit_transform(latents)

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=labels, palette=sns.color_palette("hsv", 10), legend='full')
    plt.title("Latent Space UMAP Visualization")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.show()

# Visualize t-SNE Latent Space
visualize_latent_space(model, test_loader)

# Visualize UMAP Latent Space
visualize_latent_space_umap(model, test_loader)

# Visualize Original vs. Reconstructed Images
visualize_samples(model, test_loader)

print('')
   
