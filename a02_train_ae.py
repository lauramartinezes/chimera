# Configuration parameters from the YAML-like file
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils

from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

from mnist_dataset import CustomBinaryInsectDF
from vq_vae import VQVAE


def train_vae(model, train_loader, device, epochs, optimizer, scheduler, patience, save_path_best):
    best_loss = float('inf')  # Initialize the best loss as infinity
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


def plot_original_vs_reconstructed_images(images, reconstructions):
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for i in range(8):
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.show()


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0

    df_test_path = os.path.join('data', f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    insect_classes = ['wmv', 'c']
    
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        df_train = pd.read_csv(df_train_path)

        # Prepare Dataset
        transform = transforms.Compose([
            transforms.Resize((config["data_params"]["patch_size"], config["data_params"]["patch_size"])),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset_vae = CustomBinaryInsectDF(df_train, transform = transform, seed=config["exp_params"]["manual_seed"])
        test_dataset_vae = CustomBinaryInsectDF(df_test, transform = transform, seed=config["exp_params"]["manual_seed"])

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

        print(f"{main_insect_class} dataset correctly loaded")

        # Initialize the Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = VQVAE(
            in_channels=config["model_params"]["in_channels"],
            embedding_dim=config["model_params"]["embedding_dim"],
            num_embeddings=config["model_params"]["num_embeddings"],
            img_size=config["model_params"]["img_size"],
            beta=config["model_params"]["beta"]
        ).to(device)

        print(f"{main_insect_class} model correctly initialized")

        # Optimizer and Scheduler
        optimizer = optim.Adam(model.parameters(), lr=config["exp_params"]["LR"], weight_decay=config["exp_params"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["exp_params"]["scheduler_gamma"])

        # Training Loop
        epochs = config["trainer_params"]["max_epochs"]
        patience = config["trainer_params"]["patience"]
        save_path = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{main_insect_class}.pth')
        save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{main_insect_class}_best.pth')
        os.makedirs(config["logging_params"]["save_dir"], exist_ok=True)

        if not os.path.exists(save_path_best):
            print(f"{main_insect_class} start Training")
            train_vae(model, train_loader, device, epochs, optimizer, scheduler, patience, save_path_best)
        else:
            print(f"{main_insect_class} loading best model")    
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
        plot_original_vs_reconstructed_images(test_images, recons)

