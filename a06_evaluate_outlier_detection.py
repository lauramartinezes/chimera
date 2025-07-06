import os
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import timm
import torch
import umap
import yaml

from datasets import set_feature_extraction_transform
from datasets.load_data import load_data_from_df
from outlier_detectors import get_outlier_methods_csv
from models import extract_features


def preprocess_latents_for_outlier_detection(latents_cnn, cnn_type):
    if '2d' in cnn_type:
        reducer_2_d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        return reducer_2_d.fit_transform(latents_cnn)

    elif 'napoletano' in cnn_type:
        pca = PCA(n_components=0.95)
        return pca.fit_transform(latents_cnn)
    else:
        return latents_cnn

if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0
    data_dir = config["data_params"]["splitted_data_dir"]
    insect_classes = config["data_params"]["data_classes"]

    df_test_path = os.path.join(data_dir, f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    cnn_types = ['cnn', 'cnn_2d', 'adbench', 'adbench_2d', 'adbench_napoletano']
    
    for i, main_insect_class in enumerate(insect_classes):
        mislabeled_insect_class = insect_classes[1 - i]

        model_cnn = timm.create_model(config["model_params"]["name"], pretrained=config["model_params"]["pretrained"])
        model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
        model_cnn.eval()

        for cnn_type in cnn_types:
            if 'adbench' in cnn_type:
                suffix = config["data_params"]["raw_suffix"]
            else:
                suffix = config["data_params"]["swap_suffix"]
            
            df_train_path = os.path.join(data_dir, f'df_train_{suffix}_{main_insect_class}.csv')
            df_val_path = os.path.join(data_dir, f'df_val_{suffix}_{main_insect_class}.csv')
            df_train_ = pd.read_csv(df_train_path)
            df_val_ = pd.read_csv(df_val_path)
            df_train = pd.concat([df_train_, df_val_], ignore_index=True)   

            loader = load_data_from_df(
                df_train,
                set_feature_extraction_transform(),
                config["exp_params"]["manual_seed"],
                config["data_params"][f"batch_size"],
                config["data_params"]["num_workers"],
                pin_memory,
                shuffle=False
            )
            latents_cnn, labels_cnn, real_labels_cnn, measurement_noise_cnn, mislabeled_cnn = extract_features(loader, model_cnn) 

            latents_cnn = preprocess_latents_for_outlier_detection(latents_cnn, cnn_type)

            get_outlier_methods_csv(
                latents_cnn,
                measurement_noise_cnn.astype(int),
                mislabeled_cnn.astype(int),
                f'{cnn_type}_512d_{main_insect_class}_train',
                dirname=config["logging_params"]["save_dir"]
            )
 
            print(f'Metrics for {cnn_type} are available')  

