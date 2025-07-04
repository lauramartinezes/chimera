import os
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import timm
import torch
from tqdm import tqdm
import umap
import yaml

from torch.utils.data import DataLoader

from datasets import set_feature_extraction_transform
from outlier_detectors import get_outlier_methods_csv
from datasets import CustomBinaryInsectDF
from models import extract_features


def process_data_cnn(df, model, device, config, transform, main_insect_class, phase="train", cnn_type='cnn'):
    loader = load_data_from_df(
        df,
        transform,
        config["exp_params"]["manual_seed"],
        config["data_params"][f"train_batch_size"],
        config["data_params"]["num_workers"],
        pin_memory,
    )
    latents_cnn, labels_cnn, real_labels_cnn, measurement_noise_cnn, mislabeled_cnn = extract_features(loader, model)

    if '2d' in cnn_type:
        reducer_2_d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        latents_cnn = reducer_2_d.fit_transform(latents_cnn)

    if 'napoletano' in cnn_type:
        pca = PCA(n_components=0.95)
        latents_cnn = pca.fit_transform(latents_cnn)

    get_outlier_methods_csv(
        latents_cnn,
        measurement_noise_cnn.astype(int),
        mislabeled_cnn.astype(int),
        f'{cnn_type}_512d_{main_insect_class}_{phase}',
        dirname=config["logging_params"]["save_dir"]
    )


def load_data_from_df(df, transform, seed, batch_size, num_workers, pin_memory):
    dataset = CustomBinaryInsectDF(
        df, 
        transform = transform, 
        seed=seed
    )

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0
    data_dir = config["data_params"]["data_dir"]
    insect_classes = config["data_params"]["data_classes"]
    model_name = config["model_params"]["name"]

    df_test_path = os.path.join(data_dir, f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    transform_cnn = set_feature_extraction_transform()

    device = config["trainer_params"]["device"]
    raw_suffix = config["data_params"]["raw_suffix"]
    swap_suffix = config["data_params"]["swap_suffix"]
    
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        model_cnn = timm.create_model(model_name, pretrained=True)
        model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
        model_cnn.eval()

        # Resnet method
        df_train_path = os.path.join(data_dir, f'df_train_{swap_suffix}_{main_insect_class}.csv')
        df_val_path = os.path.join(data_dir, f'df_val_{swap_suffix}_{main_insect_class}.csv')
        df_train_ = pd.read_csv(df_train_path)
        df_val_ = pd.read_csv(df_val_path)
        df_train = pd.concat([df_train_, df_val_], ignore_index=True)       

        process_data_cnn(
            df_train, 
            model_cnn, 
            device, 
            config, 
            transform_cnn, 
            main_insect_class, 
            phase="train", 
            cnn_type='cnn_2d'
        )   

        process_data_cnn(
            df_train, 
            model_cnn, 
            device, 
            config, 
            transform_cnn, 
            main_insect_class, 
            phase="train", 
            cnn_type='cnn'
        )  

        print('Metrics for CNN are available') 

        # Adbench method
        df_train_path = os.path.join(data_dir, f'df_train_{raw_suffix}_{main_insect_class}.csv') 
        df_val_path = os.path.join(data_dir, f'df_val_{raw_suffix}_{main_insect_class}.csv')
        df_train_ = pd.read_csv(df_train_path)
        df_val_ = pd.read_csv(df_val_path)
        df_train = pd.concat([df_train_, df_val_], ignore_index=True)

        process_data_cnn(
            df_train, 
            model_cnn, 
            device, 
            config, 
            transform_cnn, 
            main_insect_class, 
            phase="train", 
            cnn_type='adbench_2d'
        ) 

        process_data_cnn(
            df_train, 
            model_cnn, 
            device, 
            config, 
            transform_cnn, 
            main_insect_class, 
            phase="train", 
            cnn_type='adbench'
        )  

        process_data_cnn(
            df_train, 
            model_cnn, 
            device, 
            config, 
            transform_cnn, 
            main_insect_class, 
            phase="train", 
            cnn_type='adbench_napoletano'
        )  

        print('Metrics for AdBench are available')
