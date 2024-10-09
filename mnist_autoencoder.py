import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import umap
import pandas as pd
import os

from mnist_dataset import CustomMNISTDF, ClassSubset

latent_dim = 20  # Size of latent space

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
    #recon_loss = F.l1_loss(recon_x, x, reduction='sum') # mae Loss (Less Sensitive to outliers)

    # KL Divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss (weighted by beta)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


###################################################


# Step 3: VAE Structure remains the same (from the previous implementation)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(12544, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 12544)
        self.reshape = nn.Unflatten(1, (64, 14, 14))
        self.conv_trans1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans4 = nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1)
        self.conv_trans5 = nn.ConvTranspose2d(128, 1, kernel_size=3, padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = self.reshape(x)
        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = F.relu(self.conv_trans3(x))
        x = F.relu(self.conv_trans4(x))
        x = torch.sigmoid(self.conv_trans5(x))
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# Step 4: Define loss function (reconstruction + KL divergence)
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD



###################################################


# Function to filter the dataset by the desired classes
def filter_dataset_by_classes(dataset, keep_classes):
    # Get the indices of the samples whose labels match the desired classes
    indices = np.where(np.isin(dataset.targets, keep_classes))[0]
    # Return a subset of the dataset containing only the desired classes
    return Subset(dataset, indices)

# Training loop
def train_vae(model, train_loader, optimizer, num_epochs=10, beta=4.0):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for data, _, _, _, _, _ in train_loader: 
            data = data.to(device)

            # Forward pass
            optimizer.zero_grad()
            recon_data, mu, logvar = model(data)
            
            # Compute loss
            loss, recon_loss, kl_loss = beta_vae_loss(recon_data, data, mu, logvar, beta)
            #loss, recon_loss, kl_loss = vae_loss(recon_data, data, mu, logvar)

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


# Visualization Functions

def visualize_samples(model, test_loader, normal_class=None):
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        for batch, _, _, _, _, _ in test_loader:
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
        
    #plt.show()
    plt.savefig(os.path.join('outputs', f'samples_{normal_class}.png'), format='png')


def get_latent_vectors(model, test_loader):
    model.eval()
    latents, labels, measurement_noises,  label_noises = [], [], [], []
    
    with torch.no_grad():
        for data, label, real_label, measurement_noise, label_noise, outlier in test_loader:
            data = data.to(device)
            _, mu, _ = model(data)
            latents.append(mu.cpu())
            labels.append(label)
            measurement_noises.append(measurement_noise)
            label_noises.append(label_noise)

    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)
    measurement_noises = torch.cat(measurement_noises, dim=0)
    label_noises = torch.cat(label_noises, dim=0)

    latents = np.array(latents)
    labels = np.array(labels)
    measurement_noises = np.array(measurement_noises)
    label_noises = np.array(label_noises)

    return latents, labels, measurement_noises,  label_noises

# UMAP Latent Space Visualization
def visualize_latent_space(latents_2d, labels, measurement_noises, label_noises, normal_class=None):
    measurement_noises = measurement_noises.astype(int)*2
    label_noises = label_noises.astype(int)
    noises = measurement_noises + label_noises
    
    # Dictionary to map numbers to text labels
    label_mapping = {0: "Normal Sample", 1: "Label Noise", 2: "Measurement Noise"}
    txt_noise_labels = [label_mapping[label] for label in noises]

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=txt_noise_labels, palette=sns.color_palette("hsv", 3), legend='full')
    plt.title("Latent Space UMAP Visualization")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    #plt.show()
    plt.savefig(os.path.join('outputs', f'umap_plot_{normal_class}.png'), format='png')

def detect_outliers(latent_space_points):
    dbscan = DBSCAN(eps=0.5, min_samples=1)  # Adjust parameters as needed
    labels = dbscan.fit_predict(latent_space_points)

    # Count the number of points in each cluster (ignore noise points labeled as -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]

    # second_largest_cluster_label = unique_labels[np.argsort(counts)[-2]]
    # smallest_cluster_label = unique_labels[np.argmin(counts)]
    # second_smallest_cluster_label = unique_labels[np.argsort(counts)[1]]

    # Step 3: Mark the points in the smallest cluster as outliers
    outliers = latent_space_points[labels != largest_cluster_label]
    inliers = latent_space_points[labels == largest_cluster_label]
    return outliers, inliers

def visualize_outliers(outliers, inliers):
    # Plotting the clusters
    plt.figure(figsize=(10, 6))

    plt.scatter(outliers[:, 0], outliers[:, 1], label=f'Outliers Cluster', color='red', s=50, alpha=0.6)
    plt.scatter(inliers[:, 0], inliers[:, 1], label=f'Inliers Cluster', color='blue', s=50, alpha=0.6)

    # for label in unique_labels:
    #     if label == -1:  # Noise points
    #         cluster_points = latent_space_points[labels == label]
    #         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label='Noise', color='gray', s=30, alpha=0.3)
    #     else:
    #         cluster_points = latent_space_points[labels == label]
    #         if (label == largest_cluster_label):# | (label == second_largest_cluster_label):
    #             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Inliers Cluster {label}', color='red', s=50, alpha=0.6)
    #         else:
    #             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', s=30, alpha=0.4)

    plt.title('DBSCAN Clustering with Highlighted Largest Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    

batch_size = 1024
transform = transforms.Compose([transforms.ToTensor()])

train_df = pd.read_csv(os.path.join('archive', 'fashion-mnist_train.csv'))
test_df = pd.read_csv(os.path.join('archive', 'fashion-mnist_test.csv'))

# Create the custom dataset instances
train_dataset = CustomMNISTDF(train_df, transform_percent=0.25, wrong_label_percent=0.25, transform = transform, seed=42)
test_dataset = CustomMNISTDF(test_df, transform_percent=0.25, wrong_label_percent=0.25, transform = transform, seed=42)

classes = train_df.label.unique().tolist()
classes.sort()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and device
beta = 0.00005  # Set the beta value for Beta-VAE
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = BetaVAE(latent_dim=latent_dim, beta=beta).to(device)
model = VAE().to(device)
model_path = os.path.join('outputs', 'model.pth')

if os.path.exists(model_path):
    model.load_state_dict(torch.load())
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) #= optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_vae(model, train_loader, optimizer, num_epochs=num_epochs, beta=beta)
    torch.save(model.state_dict(), os.path.join('outputs', 'model.pth'))

for normal_class in classes:
    # Get a subset of the dataset with only class 'A'
    class_train_dataset =  ClassSubset(train_dataset, class_label=normal_class)
    class_test_dataset =  ClassSubset(test_dataset, class_label=normal_class)

    class_train_loader = DataLoader(class_train_dataset, batch_size=batch_size, shuffle=True)
    class_test_loader = DataLoader(class_test_dataset, batch_size=batch_size, shuffle=False)

    latents, labels, measurement_noises, label_noises = get_latent_vectors(model, class_train_loader)

    # Perform UMAP on latent space
    reducer = umap.UMAP(n_components=2)
    latents_2d = reducer.fit_transform(latents)

    outliers, inliers = detect_outliers(latent_space_points)

    # Visualize UMAP Latent Space
    visualize_latent_space(latents_2d, labels, measurement_noises, label_noises, normal_class=normal_class)
    visualize_outliers(outliers, inliers)

    #np.save(os.path.join('outputs', f'full_{normal_class}.npy'), latents)
    #np.save(os.path.join('outputs', f'umap_{normal_class}.npy'), latents_2d)

    # Visualize Original vs. Reconstructed Images
    visualize_samples(model, class_train_loader, normal_class=normal_class)

    print('')

print('')
   
