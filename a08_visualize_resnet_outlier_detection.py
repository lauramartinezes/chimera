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
    model_name = 'resnet18'
    cnn_types = ['cnn', 'adbench', model_name]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    subsets = ['train', 'val']

    results = []

    for cnn_type in cnn_types:
        
        if cnn_type == model_name:
            img_size = 150
        else:
            img_size = 224
        transform_cnn = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            # df_train_path = os.path.join('/home/u0159868/Documents/repos/stickybugs-outliers/data/02 before adding class maps', f'df_train_ae_{main_insect_class}.csv')
            # df_train = pd.read_csv(df_train_path)
            df_subsets = []
            for subset in subsets:
                if cnn_type == 'adbench':
                    df_subset_path = os.path.join('data', f'df_{subset}_raw_{main_insect_class}.csv')
                else:
                    df_subset_path = os.path.join('data', f'df_{subset}_ae_{main_insect_class}.csv')
                df_subset = pd.read_csv(df_subset_path)
                df_subsets.append(df_subset)
            df_train_val = pd.concat(df_subsets, ignore_index=True)

            # CNN method
            if cnn_type == 'cnn' or cnn_type == 'adbench':
                model_cnn = timm.create_model('resnet18', pretrained=True)
                model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
                od_methods = ['LODA']
            elif cnn_type == model_name:
                model_cnn = timm.create_model(model_name, pretrained=False, num_classes=2)
                save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier_raw_best_.pth')#f'{model_name}_classifier{clean_dataset}_{method}_best.pth')
            
                # Load the model
                if os.path.exists(save_path_best):
                    model_cnn.load_state_dict(torch.load(save_path_best))
                    print("Model correctly loaded")
                else:
                    print("Model not found")
                model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
                od_methods = ['OCSVM'] #['DBSCAN', 'MCD']
                    
            model_cnn.eval()
            print("Model correctly initialized")

            for od_method in od_methods:
                df_train_val_clean, df_train_val_outliers, metrics = clean_df(
                    df_train_val, 
                    model_cnn, 
                    device, 
                    config, 
                    transform_cnn, 
                    pin_memory, 
                    main_insect_class, 
                    phase='train_val', 
                    method=cnn_type, 
                    od_method=od_method
                )
                
                os.makedirs(os.path.join('data', 'clean'), exist_ok=True)
                os.makedirs(os.path.join('data', 'outliers'), exist_ok=True)
                
                for subset in subsets:
                    if subset == 'train':
                        df_subset_clean = df_train_val_clean[df_train_val_clean.filepath.str.contains('train')]
                        df_subset_outliers = df_train_val_outliers[df_train_val_outliers.filepath.str.contains('train')]
                    elif subset == 'val':
                        df_subset_clean = df_train_val_clean[df_train_val_clean.filepath.str.contains('val')]
                        df_subset_outliers = df_train_val_outliers[df_train_val_outliers.filepath.str.contains('val')]
                    df_subset_clean_path = os.path.join('data', 'clean', f'df_{subset}_{cnn_type}_{main_insect_class}_{od_method}_clean.csv')
                    df_subset_clean.to_csv(df_subset_clean_path, index=False)
                    print(f'Clean {main_insect_class} cnn {subset} dataset available')

                    df_subset_outliers_path = os.path.join('data', 'outliers', f'df_{subset}_{cnn_type}_{main_insect_class}_{od_method}_outliers.csv')
                    df_subset_outliers.to_csv(df_subset_outliers_path, index=False)
                    print(f'Outliers {main_insect_class} cnn {subset} dataset available')

                # Add more information to the metrics dictionary
                metrics['method'] = cnn_type
                metrics['od_method'] = od_method
                metrics['main_insect_class'] = main_insect_class

                results.append(metrics)
                df_results = pd.DataFrame(results)
                df_results.to_csv(os.path.join(config["logging_params"]["save_dir"],f'df_best_od_evaluation.csv'), mode='a', index=False, header=False)
                print('metrics: ', metrics)
            print('')