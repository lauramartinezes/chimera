import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vq_vae import VQVAE 
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration parameters from the YAML-like file
config = {
    "model_params": {
        "name": "VQVAE",
        "in_channels": 3,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "img_size": 64,
        "beta": 0.25
    },
    "data_params": {
        "data_path": "Data/",
        "train_batch_size": 64,
        "val_batch_size": 64,
        "patch_size": 64,
        "num_workers": 4
    },
    "exp_params": {
        "LR": 0.005,
        "weight_decay": 0.0,
        "scheduler_gamma": 0.9,
        "kld_weight": 0.00025,
        "manual_seed": 1265
    },
    "trainer_params": {
        "gpus": [1],
        "max_epochs": 10
    },
    "logging_params": {
        "save_dir": "logs/",
        "name": "VQVAE"
    }
}

if __name__ == '__main__':
    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])

    # Prepare Dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        #transforms.CenterCrop(config["data_params"]["patch_size"]),
        transforms.Resize(config["data_params"]["patch_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load CIFAR-10 dataset
    data_path = config["data_params"]["data_path"]
    train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)

    pin_memory = len(config['trainer_params']['gpus']) != 0

    train_loader = DataLoader(train_dataset,
                            batch_size=config["data_params"]["train_batch_size"],
                            shuffle=True,
                            num_workers=config["data_params"]["num_workers"],
                            )

    val_loader = DataLoader(val_dataset,
                            batch_size=config["data_params"]["val_batch_size"],
                            shuffle=False,
                            num_workers=config["data_params"]["num_workers"],
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
    save_path = os.path.join(config["logging_params"]["save_dir"], config["logging_params"]["name"] + ".pth")
    os.makedirs(config["logging_params"]["save_dir"], exist_ok=True)

    print("Start Training")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_recons_loss = 0.0
        train_vq_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, _ = batch
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

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, _ = batch
                images = images.to(device)
                recons, inputs, vq_loss = model(images)
                loss_dict = model.loss_function(recons, inputs, vq_loss)
                val_loss += loss_dict['loss'].item()

        print(f"Epoch [{epoch+1}/{epochs}] - Validation Loss: {val_loss / len(val_loader):.4f}")

        # Step the scheduler
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    # Visualization After Training
    model.eval()
    test_images, _ = next(iter(val_loader))
    test_images = test_images.to(device)

    with torch.no_grad():
        recons, _, _ = model(test_images)

    # Denormalize and convert for visualization
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
