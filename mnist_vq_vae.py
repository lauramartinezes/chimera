import glob
import os
import pandas as pd
import torch
import torchvision.utils as vutils
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import umap
from a00_prepare_data import get_df_test, get_df_train
from mnist_autoencoder import visualize_latent_space
from mnist_dataset import CustomBinaryInsectDF
from pyod_our_approach import get_outlier_methods_csv
from vq_vae import VQVAE 
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

def extract_features_from_quantized(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_features = []  # To store features from all batches
    all_labels = []    # To store labels if needed
    all_real_labels = []
    all_measurement_noise = []
    all_mislabeled = []
    all_quantized = []

    with torch.no_grad():
        for images, labels, real_labels, measurement_noise, mislabeled, outliers in tqdm(dataloader, desc="Featured extraction quantized: "):
            images = images.to(device)

            # Pass images through the encoder and vector quantization layers
            encoding = model.encode(images)[0]  # Raw encoder output
            quantized_inputs, _ = model.vq_layer(encoding)  # Quantized representation

            # Global Average Pooling: Compute mean across spatial dimensions (H, W)
            pooled_quantized = quantized_inputs.mean(dim=[2, 3])  # Shape: [B, embedding_dim]

            # Append features to the list
            all_features.append(pooled_quantized.cpu().numpy())
            all_quantized.append(quantized_inputs.cpu().numpy())
            all_labels.append(labels.numpy())  # Collect labels if needed
            all_real_labels.append(real_labels.numpy())  # Collect labels if needed
            all_measurement_noise.append(measurement_noise.numpy())  # Collect labels if needed
            all_mislabeled.append(mislabeled.numpy())  # Collect labels if needed


    # Concatenate all features into a single array
    return np.vstack(all_quantized), np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_real_labels), np.concatenate(all_measurement_noise), np.concatenate(all_mislabeled)


def extract_features_from_encoding(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_features = []  # To store features from all batches
    all_labels = []    # To store labels if needed
    all_real_labels = []
    all_measurement_noise = []
    all_mislabeled = []
    all_encodings = []

    with torch.no_grad():
        for images, labels, real_labels, measurement_noise, mislabeled, outliers in tqdm(dataloader, desc="Featured extraction encoding: "):
            images = images.to(device)

            # Pass images through the encoder layers
            encoding = model.encode(images)[0]  # Raw encoder output

            # Global Average Pooling: Compute mean across spatial dimensions (H, W)
            pooled_encoding = encoding.mean(dim=[2, 3])  # Shape: [B, embedding_dim]

            # Append features to the list
            all_features.append(pooled_encoding.cpu().numpy())
            all_encodings.append(encoding.cpu().numpy())
            all_labels.append(labels.numpy())  # Collect labels if needed
            all_real_labels.append(real_labels.numpy())  # Collect labels if needed
            all_measurement_noise.append(measurement_noise.numpy())  # Collect labels if needed
            all_mislabeled.append(mislabeled.numpy())  # Collect labels if needed


    # Concatenate all features into a single array
    return np.vstack(all_encodings), np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_real_labels), np.concatenate(all_measurement_noise), np.concatenate(all_mislabeled)

# Configuration parameters from the YAML-like file
config = {
    "model_params": {
        "name": "VQVAE",
        "in_channels": 3,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "img_size": 148,
        "beta": 0.25
    },
    "data_params": {
        "data_path": "Data/",
        "train_batch_size": 64,
        "val_batch_size": 64,
        "patch_size": 148,
        "num_workers": 4
    },
    "exp_params": {
        "LR": 0.001,
        "weight_decay": 0.0,
        "scheduler_gamma": 0.9,
        "kld_weight": 0.00025,
        "manual_seed": 1265
    },
    "trainer_params": {
        "gpus": [1],
        "max_epochs": 100
    },
    "logging_params": {
        "save_dir": "logs/",
        "name": "VQVAE"
    }
}


if __name__ == '__main__':
    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0

    main_insect_class = 'c'
    mislabeled_insect_class = 'wmv'

    df_train = get_df_train(main_insect_class, mislabeled_insect_class)
    df_test = get_df_test(main_insect_class, mislabeled_insect_class)

    df_train.to_csv(os.path.join('data', f'df_train_{main_insect_class}.csv'))
    df_test.to_csv(os.path.join('data', f'df_test.csv'))

    good_samples_folder_test = os.path.join('data', f'{main_insect_class}_for_{mislabeled_insect_class}')
    mismeasure_samples_folder_test = os.path.join('data', f'{mislabeled_insect_class}_trash')
    mislabel_samples_folder_test = os.path.join('data', f'{mislabeled_insect_class}_good')
   
    good_samples_test = glob.glob(os.path.join(good_samples_folder_test, '*.png'))
    mismeasure_samples_test = glob.glob(os.path.join(mismeasure_samples_folder_test, '*.png'))
    mislabel_samples_test = glob.glob(os.path.join(mislabel_samples_folder_test, '*.png'))
    mislabel_samples_test = random.sample(mislabel_samples_test, len(good_samples_test))

    df_good_samples_test = pd.DataFrame({
        'label': 0,
        'mislabeled': False,
        'measurement_noise': False, 
        'filepath': good_samples_test
    })

    df_mismeasure_samples_test = pd.DataFrame({
        'label': 1,
        'mislabeled': False,
        'measurement_noise': True, 
        'filepath': mismeasure_samples_test
    })

    df_mislabel_samples_test = pd.DataFrame({
        'label': 1,
        'mislabeled': True,
        'measurement_noise': False, 
        'filepath': mislabel_samples_test
    })

    df_test = pd.concat([df_good_samples_test, df_mismeasure_samples_test, df_mislabel_samples_test], ignore_index=True)


    # Prepare Dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        #transforms.CenterCrop(config["data_params"]["patch_size"]),
        transforms.Resize((config["data_params"]["patch_size"], config["data_params"]["patch_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset_vae = CustomBinaryInsectDF(df_train, transform_percent=0.1, transform = transform, seed=42)
    test_dataset_vae = CustomBinaryInsectDF(df_test, transform_percent=0.1, transform = transform, seed=42)

    train_loader = DataLoader(
        train_dataset_vae, 
        batch_size=config["data_params"]["train_batch_size"], 
        shuffle=True,
        num_workers=config["data_params"]["num_workers"],
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset_vae, 
        batch_size=config["data_params"]["train_batch_size"], 
        shuffle=True,
        num_workers=config["data_params"]["num_workers"],
        pin_memory=pin_memory
    )

    print("Dataset correctly loaded")

    # Initialize the Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VQVAE(
        in_channels=config["model_params"]["in_channels"],
        embedding_dim=config["model_params"]["embedding_dim"],
        num_embeddings=config["model_params"]["num_embeddings"],
        img_size=config["model_params"]["img_size"],
        beta=config["model_params"]["beta"]
    ).to(device)

    print("Model correctly initialized")

    # Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["exp_params"]["LR"], weight_decay=config["exp_params"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["exp_params"]["scheduler_gamma"])

    # Training Loop
    epochs = config["trainer_params"]["max_epochs"]
    save_path = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{main_insect_class}.pth')
    save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{main_insect_class}_best.pth')
    os.makedirs(config["logging_params"]["save_dir"], exist_ok=True)

    if os.path.exists(save_path_best):
        model.load_state_dict(torch.load(save_path_best))
    else:
        print("Start Training")
        best_loss = float('inf')  # Initialize the best loss as infinity
        patience = 5  # Set the patience parameter
        patience_counter = 0  # Counter for epochs without improvement
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_recons_loss = 0.0
            train_vq_loss = 0.0
            for images, _, _, _, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(device)

                # Forward pass
                recons, inputs, vq_loss = model(images)
                loss_dict = model.loss_function(recons, inputs, vq_loss)
                loss = loss_dict['loss']

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_recons_loss += loss_dict['Reconstruction_Loss'].item()
                train_vq_loss += loss_dict['VQ_Loss'].item()

            # Compute average losses for logging
            avg_train_loss = train_loss / len(train_loader)
            avg_recons_loss = train_recons_loss / len(train_loader)
            avg_vq_loss = train_vq_loss / len(train_loader)

            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, "
                f"Reconstruction Loss: {avg_recons_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}")

            # Check for improvement in training loss
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path_best)  # Save the best model
                print(f"New best model saved at {save_path_best} with Train Loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience counter: {patience_counter}/{patience}")
            
            # Step the scheduler
            scheduler.step()
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # Early stopping
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch+1}. Best Train Loss: {best_loss:.4f}")
                break
        # Save the trained model
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")
        model.load_state_dict(torch.load(save_path_best))

    # # Visualization After Training
    model.eval()
    test_images, _, _, _, _, _ = next(iter(train_loader))
    test_images = test_images.to(device)

    with torch.no_grad():
        recons, _, _ = model(test_images)

    vutils.save_image(recons.data,
                    os.path.join(config["logging_params"]["save_dir"] , 
                                "Reconstructions", 
                                f"recons{main_insect_class}.png"),
                    normalize=True,
                    nrow=12)

    # # Denormalize and convert for visualization
    test_images = test_images.cpu() * 0.5 + 0.5  # Undo normalization
    recons = recons.cpu() * 0.5 + 0.5

    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for i in range(8):
        axes[0, i].imshow(test_images[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[1, i].imshow(recons[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.show()

    # Assuming 'model' is your trained VQ-VAE and 'dataloader' is your data loader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Extract features from quantized latents
    # raw_features_quantized, features_quatized, labels_quantized, real_labels_quantized, measurement_noise_quantized, mislabeled_quantized = extract_features_from_quantized(model, train_loader, device)
    raw_features_encoding, features_encoding, labels_encoding, real_labels_quantized, measurement_noise_encoding, mislabeled_encoding  = extract_features_from_encoding(model, train_loader, device)
    
    reducer = umap.UMAP(n_components=2)
    
    # latents_quantized_2d = reducer.fit_transform(features_quatized)
    # visualize_latent_space(latents_quantized_2d, labels_quantized, measurement_noise_quantized, mislabeled_quantized)
    
    # latents_encoding_2d = reducer.fit_transform(features_encoding)
    # visualize_latent_space(latents_encoding_2d, labels_encoding, measurement_noise_encoding, mislabeled_encoding)

    # reshaped_raw_features_quantized = raw_features_quantized.reshape(raw_features_quantized.shape[0], -1)
    # latents_raw_quantized_2d = reducer.fit_transform(reshaped_raw_features_quantized)
    # visualize_latent_space(latents_raw_quantized_2d, labels_quantized, measurement_noise_quantized, mislabeled_quantized)
    
    reshaped_raw_features_encoding = raw_features_encoding.reshape(raw_features_encoding.shape[0], -1)
    latents_raw_encoding_2d = reducer.fit_transform(reshaped_raw_features_encoding)
    visualize_latent_space(latents_raw_encoding_2d, labels_encoding, measurement_noise_encoding, mislabeled_encoding)

    reducer_512d = umap.UMAP(n_components=512)
    latents_raw_encoding_512d = reducer_512d.fit_transform(reshaped_raw_features_encoding)
    get_outlier_methods_csv(latents_raw_encoding_512d, measurement_noise_encoding.astype(int),  mislabeled_encoding.astype(int), f'vqvae_512d_{main_insect_class}')
    print('')