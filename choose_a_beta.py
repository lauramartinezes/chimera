# Fix the seed for reproducibility
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from PyOD import PYOD, metric
from mnist_autoencoder import VAE, get_latent_vectors, train_vae
from mnist_dataset import CustomBinaryMNISTDF
from pyod_our_approach import get_outlier_methods_csv

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
    # plt.annotate(f'Lowest Loss\n({min_loss_dim}, {min_loss_value:.2f})', 
    #             xy=(min_loss_dim, min_loss_value), 
    #             xytext=(min_loss_dim, min_loss_value + 0.05 * min(losses)),
    #             arrowprops=dict(facecolor='black', arrowstyle='->'),
    #             fontsize=10)

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
betas = [0.1, 1, 10, 100, 1000, 10000, 0.0001, 0.001, 0.01, ]  # Set the beta value for Beta-VAE
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dim = 10
losses = []
recon_losses = []
kl_losses = []

models = ['SOD', 'SOS', 'DeepSVDD', #'VAE','AutoEncoder','XGBOD'
           'SOGAAL', 'MOGAAL','IForest', 'OCSVM', 'ABOD', 'CBLOF', 'COF', #'AOM',
          'COPOD', 'ECOD',  'FeatureBagging', 'HBOS', 'KNN',
          'LMDD', 'LODA', 'LOF', #'LOCI', #'LSCP', 'MAD',
          'MCD', 'PCA']#,'ROD']

results = []
beta_data = {}

for beta in betas:
    vae_model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(vae_model.parameters(), lr=1e-4, weight_decay=1e-4) #= optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    avg_loss, avg_recon_loss, avg_kl_loss = train_vae(vae_model, train_loader_vae, optimizer, device, num_epochs=num_epochs, beta=beta)
    losses.append(avg_loss)
    recon_losses.append(avg_recon_loss)
    kl_losses.append(avg_kl_loss)
    latents_vae, labels_vae, measurement_noise_vae, mislabeled_vae = get_latent_vectors(vae_model, test_loader_vae, device)
    
    y_train = measurement_noise_vae.astype(int) + mislabeled_vae.astype(int)

    metrics_list = []

    for model in models:
        pyod_model = PYOD(seed=42, model_name=model)
        pyod_model.fit(latents_vae, [])
        anomaly_scores = pyod_model.predict_score(latents_vae)
        metrics = metric(y_true=y_train, y_score=anomaly_scores, pos_label=1)
        print(f'{model}: {metrics}')

        metrics_list.append({'Model': model, 'aucroc': metrics['aucroc'], 'aucpr': metrics['aucpr']})
        outlier_metrics_df = pd.DataFrame(metrics_list)

    beta_data[beta] = outlier_metrics_df
    beta_filename = os.path.join('outputs', 'beta', f"beta_{beta}_metrics.csv")
    outlier_metrics_df.to_csv(beta_filename, index=False)

    threshold = 0.7  # Example threshold for AUC ROC and AUC PR
    filtered_df = outlier_metrics_df[(outlier_metrics_df['aucroc'] > threshold)]

    # Calculate metrics
    mean_auc_roc = filtered_df['aucroc'].mean()
    mean_auc_pr = filtered_df['aucpr'].mean()
    median_auc_roc = filtered_df['aucroc'].median()
    median_auc_pr = filtered_df['aucpr'].median()
    iqr_auc_roc = filtered_df['aucroc'].quantile(0.75) - filtered_df['aucroc'].quantile(0.25)
    iqr_auc_pr = filtered_df['aucpr'].quantile(0.75) - filtered_df['aucpr'].quantile(0.25)

    results.append({
        'beta': beta,
        'mean_aucroc': mean_auc_roc,
        'mean_aucpr': mean_auc_pr,
        'median_aucroc': median_auc_roc,
        'median_aucpr': median_auc_pr,
        'iqr_aucroc': iqr_auc_roc,
        'iqr_aucpr': iqr_auc_pr
    })

latent_dim_metrics_df = pd.DataFrame(results)
print(latent_dim_metrics_df)
latent_dim_metrics_df.to_csv(os.path.join('outputs', 'beta', f'beta_metrics.csv'), index=False)

# Plot
plot_losses(betas, losses)
plot_losses(betas, recon_losses)
plot_losses(betas, kl_losses)

print('')