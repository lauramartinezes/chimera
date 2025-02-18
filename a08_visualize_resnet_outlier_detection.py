import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from tqdm import tqdm
import umap
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms

from PyOD import PYOD, metric
from a07_clean_data import clean_df
from mnist_dataset import CustomBinaryInsectDF
from vq_vae import VQVAE


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

    transform_cnn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    subsets = ['train', 'val']

    results = []

    for subset in subsets:
        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            # df_train_path = os.path.join('/home/u0159868/Documents/repos/stickybugs-outliers/data/02 before adding class maps', f'df_train_ae_{main_insect_class}.csv')
            # df_train = pd.read_csv(df_train_path)
            df_subset_path = os.path.join('data', f'df_{subset}_raw_{main_insect_class}.csv') 
            df_subset = pd.read_csv(df_subset_path)

            # CNN method
            model_cnn = timm.create_model('resnet18', pretrained=True)
            model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
            model_cnn.eval()
            print("Model correctly initialized")

            df_subset_clean, df_outliers, metrics = clean_df(df_subset, model_cnn, device, config, transform_cnn, pin_memory, main_insect_class, phase=subset, method='adbench')
            
            os.makedirs(os.path.join('data', 'clean'), exist_ok=True)
            os.makedirs(os.path.join('data', 'outliers'), exist_ok=True)
            
            df_subset_clean_path = os.path.join('data', 'clean', f'df_{subset}_adbench_{main_insect_class}_clean.csv')
            df_subset_clean.to_csv(df_subset_clean_path, index=False)
            print(f'Clean {main_insect_class} cnn {subset} dataset available')

            df_outliers_path = os.path.join('data', 'outliers', f'df_{subset}_adbench_{main_insect_class}_outliers.csv')
            df_outliers.to_csv(df_outliers_path, index=False)
            print(f'Outliers {main_insect_class} cnn {subset} dataset available')

            # Add more information to the metrics dictionary
            metrics['method'] = 'adbench'
            metrics['od_method'] = 'ocsvm'
            metrics['main_insect_class'] = main_insect_class

            results.append(metrics)
            df_results = pd.DataFrame(results)
            df_results.to_csv(os.path.join(config["logging_params"]["save_dir"],f'df_best_od_evaluation.csv'), mode='a', index=False, header=False)
            print('metrics: ', metrics)
            print('')