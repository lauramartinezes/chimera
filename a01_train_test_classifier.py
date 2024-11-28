import os
import random
import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms

from mnist_dataset import CustomBinaryInsectDF


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0

    insect_classes = ['wmv', 'c']
    dfs_train = []
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        df_train_i = pd.read_csv(df_train_path)
        dfs_train.append(df_train_i)
    df_train = pd.concat(dfs_train, ignore_index=True)
    
    df_test_path = os.path.join('data', f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    # Prepare Dataset
    transform = transforms.Compose([
        transforms.Resize((config["data_params"]["patch_size"], config["data_params"]["patch_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CustomBinaryInsectDF(df_train, transform = transform, seed=config["exp_params"]["manual_seed"])
    test_dataset = CustomBinaryInsectDF(df_test, transform = transform, seed=config["exp_params"]["manual_seed"])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data_params"]["train_batch_size"], 
        shuffle=True,
        num_workers=config["data_params"]["num_workers"],
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["data_params"]["train_batch_size"], 
        shuffle=False,
        num_workers=config["data_params"]["num_workers"],
        pin_memory=pin_memory
    )