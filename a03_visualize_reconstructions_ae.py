import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml

from mnist_dataset import CustomBinaryInsectDF
from vq_vae import VQVAE



def plot_original_vs_reconstructed_images(images, reconstructions, filename=None, dirname=None):
    range_len = 8 if len(images)>8 else len(images)
    fig, axes = plt.subplots(2, range_len, figsize=(12, 4))
    for i in range(range_len):
        axes[0, i].imshow(images[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).numpy())
        axes[1, i].axis('off')

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")
    plt.savefig(os.path.join(dirname, f'original_vs_reconstruction_{filename}.png'), format='png')
    plt.savefig(os.path.join(dirname, f'original_vs_reconstruction_{filename}.svg'), format='svg')


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

    insect_classes = config["data_params"]["data_classes"]
    vq_vae_types = ['', 'adv_'] 
    
    for vq_vae_type in vq_vae_types:
        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
            df_val_path = os.path.join('data', f'df_val_ae_{main_insect_class}.csv')
            df_train_ = pd.read_csv(df_train_path)
            df_val_ = pd.read_csv(df_val_path)

            # Combine the training and validation dataframes
            df_train = pd.concat([df_train_, df_val_], ignore_index=True)

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
                shuffle=False,
                num_workers=config["data_params"]["num_workers"],
                pin_memory=pin_memory
            )
            test_loader = DataLoader(
                test_dataset_vae, 
                batch_size=config["data_params"]["train_batch_size"], 
                shuffle=False,
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

            # Training Loop
            save_path = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{vq_vae_type}{main_insect_class}.pth')
            save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{vq_vae_type}{main_insect_class}_best.pth')
            os.makedirs(config["logging_params"]["save_dir"], exist_ok=True)

            if not os.path.exists(save_path_best):
                print(f"Please train a {main_insect_class} model")
                continue 
            
            print(f"{main_insect_class} loading best model")    
            model.load_state_dict(torch.load(save_path_best))

            # Visualization After Training
            model.eval()
            batch_counter = 0

            for test_images, labels, _, _, _, _ in tqdm(test_loader):
                test_images = test_images.to(device)

                with torch.no_grad():
                    recons, _, _ = model(test_images)

                reconstructions_dir = os.path.join(config["logging_params"]["save_dir"] , f"Reconstructions{vq_vae_type}")
                os.makedirs(reconstructions_dir, exist_ok=True)
                vutils.save_image(recons.data,
                                os.path.join(reconstructions_dir, f"recons_{vq_vae_type}{main_insect_class}_{batch_counter}.png"),
                                normalize=True,
                                nrow=12)

                # Denormalize and convert for visualization
                test_images = test_images.cpu() * 0.5 + 0.5  # Undo normalization
                recons = recons.cpu() * 0.5 + 0.5

                # Plot original and reconstructed images
                plot_original_vs_reconstructed_images(
                    test_images, 
                    recons, 
                    filename=f'{main_insect_class}_{batch_counter}', 
                    dirname=os.path.join(config["logging_params"]["save_dir"], f"Reconstructions{vq_vae_type}")
                )
                batch_counter = batch_counter + 1
