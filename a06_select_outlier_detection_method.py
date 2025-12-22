import os

import pandas as pd
import timm
import torch

from datasets import set_feature_extraction_transform
from datasets.load_data import load_data_from_df
from models import extract_features
from outlier_detectors import get_outlier_detection_metrics, preprocess_latents_for_outlier_detection, select_best_outlier_detection_model
from utils import load_config, save_config, set_seed


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

    cleaning_strategies = config['cleaning_params']['strategies']
    subsets = config['cleaning_params']['subsets']
    
    model = timm.create_model(
        config["model_params"]["name"], 
        pretrained=config["model_params"]["pretrained"]
    )
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    csv_folder = os.path.join(config["logging_params"]["save_dir"], 'OUTLIERS_CSVS')
    os.makedirs(csv_folder, exist_ok=True)
    
    for strategy in cleaning_strategies:
        # First assign od method to those where it is fixed
        if strategy == 'cnn_no_od':
            config.setdefault("cleaning_params", {})
            config["cleaning_params"].setdefault(strategy, {})
            config['cleaning_params'][strategy]['best_outlier_detection'] = 'no_od'
            save_config(config, "config.yaml")
            continue

        if strategy in ['cnn', 'cnn_corrected_mislabels', 'adbench_xd_hdbscan']:
            config.setdefault("cleaning_params", {})
            config["cleaning_params"].setdefault(strategy, {})
            config['cleaning_params'][strategy]['best_outlier_detection'] = 'UmapHdbscanOD'
            save_config(config, "config.yaml")
            continue

        if 'adbench' in strategy:
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
            ) = extract_features(loader, model) 

            latents_cnn = preprocess_latents_for_outlier_detection(latents_cnn, strategy)

            df_outliers_class = get_outlier_detection_metrics(
                latents_cnn,
                measurement_noise_cnn.astype(int),
                mislabeled_cnn.astype(int),
                seed=42
            )
            dfs_outliers.append(df_outliers_class)

        df_outliers = pd.concat(dfs_outliers, axis=1, keys=insect_classes)
        best_od_method = select_best_outlier_detection_model(df_outliers)
        
        config.setdefault("cleaning_params", {})
        config["cleaning_params"].setdefault(strategy, {})
        config['cleaning_params'][strategy]['outlier_detection_method']=best_od_method
        save_config(config, "config.yaml")
        
        df_outliers.to_csv(os.path.join(csv_folder, f'{strategy}_outlier_metrics.csv'), index=False)
        print(f'Metrics for {strategy} are available')  

