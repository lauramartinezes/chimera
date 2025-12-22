import os
import numpy as np
import pandas as pd
import timm
import torch

from datasets import set_feature_extraction_transform
from datasets.load_data import load_data_from_df
from models import extract_features
from outlier_detectors import get_outlier_detection_metrics, preprocess_latents_for_outlier_detection
from utils import load_config, set_seed


def select_best_outlier_detection_model(df_outliers):
    eps = 1e-12  # avoid divide-by-zero
    vals = df_outliers.select_dtypes(include="number").to_numpy()   # grabs aucroc/aucpr columns automatically
    df_outliers[("overall", "harmonic_mean")] = vals.shape[1] / np.sum(1.0 / (vals + eps), axis=1)

    model_col = df_outliers.select_dtypes(exclude="number").columns[0]
    best_idx = df_outliers[("overall", "harmonic_mean")].idxmax()
    best_model = df_outliers.loc[best_idx, model_col]
    return best_model


if __name__ == '__main__':
    config = load_config("config.yaml")

    # Set manual seed for reproducibility
    random_seed = config["exp_params"]["manual_seed"]
    set_seed(random_seed)

    pin_memory = len(config['trainer_params']['gpus']) != 0
    data_dir = config["data_params"]["splitted_data_dir"]
    insect_classes = config["data_params"]["data_classes"]

    df_test_path = os.path.join(data_dir, f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    cnn_types = ['adbench_2d']#['cnn', 'adbench', 'adbench_2d']
    subsets = ['train', 'val']
    
    model_cnn = timm.create_model(
        config["model_params"]["name"], 
        pretrained=config["model_params"]["pretrained"]
    )
    model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
    model_cnn.eval()

    csv_folder = os.path.join(config["logging_params"]["save_dir"], 'OUTLIERS_CSVS')
    os.makedirs(csv_folder, exist_ok=True)
    
    for cnn_type in cnn_types:
        if 'adbench' in cnn_type:
            suffix = config["data_params"]["raw_suffix"]
        else:
            suffix = config["data_params"]["swap_suffix"]
        
        dfs_outliers = []
        for i, main_insect_class in enumerate(insect_classes):
            mislabeled_insect_class = insect_classes[1 - i]
            dfs_subsets = []
            for subset in subsets:
                df_subset_path = os.path.join(
                    data_dir, 
                    f'df_{subset}_{suffix}_{main_insect_class}.csv'
                )
                df_subset = pd.read_csv(df_subset_path)
                dfs_subsets.append(df_subset)
            df_train_val = pd.concat(dfs_subsets, ignore_index=True) 

            loader = load_data_from_df(
                df_train_val,
                set_feature_extraction_transform(),
                config["exp_params"]["manual_seed"],
                config["data_params"][f"batch_size"],
                config["data_params"]["num_workers"],
                pin_memory,
                shuffle=False
            )
            (
                latents_cnn, 
                labels_cnn, 
                measurement_noise_cnn, 
                mislabeled_cnn
            ) = extract_features(loader, model_cnn) 

            latents_cnn = preprocess_latents_for_outlier_detection(latents_cnn, cnn_type)

            df_outliers_class = get_outlier_detection_metrics(
                latents_cnn,
                measurement_noise_cnn.astype(int),
                mislabeled_cnn.astype(int),
                seed=42
            )

            dfs_outliers.append(df_outliers_class)

        df_outliers = pd.concat(dfs_outliers, axis=1, keys=insect_classes)
        best_model = select_best_outlier_detection_model(df_outliers)
        
        df_outliers.to_csv(os.path.join(csv_folder, f'{cnn_type}_outlier_metrics.csv'), index=False)
        print(f'Metrics for {cnn_type} are available')  

