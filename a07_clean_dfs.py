import os
import random
import numpy as np
import pandas as pd
import timm
import torch
import umap
import yaml

from datasets import set_feature_extraction_transform
from datasets.load_data import load_data_from_df
from models import extract_features
from outlier_detectors import UmapHdbscanOD, PYOD, metric
from outlier_detectors.plot import plot_y_true_vs_y_od_pred_umap
from utils import set_seed


def clean_df(df, model, config, transform, pin_memory, main_insect_class, phase="train", method='ae', od_method='UmapHdbscanOD'):
    # Load the data
    loader = load_data_from_df(
        df,
        transform,
        config["exp_params"]["manual_seed"],
        config["data_params"][f"batch_size"],
        config["data_params"]["num_workers"],
        pin_memory,
        shuffle=False
    )
    print(f"{phase.capitalize()} dataset correctly loaded")

    # Extract features from the dataloader
    (
        latents, 
        labels_cnn, 
        measurement_noise, 
        mislabel_noise 
    ) = extract_features(loader, model)

    latents_all_dims = latents.copy()

    umap_hdbscan_od = UmapHdbscanOD(
        main_class_name=main_insect_class, 
        save_dir=os.path.join(config["logging_params"]["save_dir"], "UMAP_HDBSCAN_Analysis", method),
        seed=42
    )

    N = len(latents)
    default_min_cluster_size = max(5, int(0.05 * N))  # Default value for HDBSCAN
    default_min_samples = max(5, int(0.01 * N))  # Default value for HDBSCAN

    if method == 'adbench':
        optimal_min_cluster_size = default_min_cluster_size
        optimal_min_samples = default_min_samples
    elif method == 'adbench_2d':
        optimal_dim = 2
        optimal_min_cluster_size = default_min_cluster_size
        optimal_min_samples = default_min_samples
    else:
        min_cluster_size_values = [max(5, int(p * N)) for p in [0.005, 0.01, 0.02, 0.05]]
        min_samples_values = [max(5, int(p * N)) for p in [0.002, 0.005, 0.01]]

        optimal_dim, optimal_min_cluster_size, optimal_min_samples = umap_hdbscan_od.find_optimal_params(
            latents, 
            dims=[2 ** i for i in range(1, 9)],
            min_cluster_size_values=min_cluster_size_values,
            min_samples_values=min_samples_values,
        )

    y_true = (measurement_noise + mislabel_noise).astype(int)
    metrics = {}
    if od_method != 'UmapHdbscanOD':
        if method !='adbench':
            reducer_optimd = umap.UMAP(n_components=optimal_dim, random_state=42, n_jobs=1)
            latents = reducer_optimd.fit_transform(latents)
        y_pred, metrics = get_outlier_predictions(
            latents,
            y_true,
            model=od_method,
            # contamination= 0.2
        )

    elif od_method == 'UmapHdbscanOD':
        y_pred = umap_hdbscan_od.predict_outliers(
            latents, 
            optimal_dim,
            optimal_min_cluster_size, 
            optimal_min_samples
        )
        metrics = metric(y_true=y_true, y_score=y_pred, pos_label=1)      

    plot_y_true_vs_y_od_pred_umap(
        latents_all_dims, 
        measurement_noise, 
        mislabel_noise,
        y_pred, 
        filename=f'{method}_{main_insect_class}_{phase}_{od_method}',
        dirname=config["logging_params"]["save_dir"]
    )

    # Mark outliers in the dataframe and transform them from integer to boolean
    df['outlier_detected'] = y_pred
    df['outlier_detected'] = df['outlier_detected'].astype(bool)

    # Filter the dataframe to have only the clean samples
    df_clean = df[y_pred == 0]
    if f'noisy_label_classification' in df_clean.columns:
        reference_label = 0 if main_insect_class == 'wmv' else 1
        df_clean = df_clean[df_clean[f'noisy_label_classification'] == reference_label]
    
    return df_clean, df, metrics


def get_outlier_predictions(X_train, y_train, model='MCD', contamination=0.1):
    print(f'{model} outlier extraction')
    pyod_model = PYOD(seed=42, model_name=model, contamination=contamination)
    pyod_model.fit(X_train, [])
    anomaly_scores = pyod_model.predict_score(X_train)
    metrics = metric(y_true=y_train, y_score=anomaly_scores, pos_label=1)
    anomaly_binary_scores = pyod_model.predict_binary_score(X_train)

    return anomaly_binary_scores, metrics


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    random_seed = config["exp_params"]["manual_seed"]
    set_seed(random_seed)

    pin_memory = len(config['trainer_params']['gpus']) != 0

    insect_classes = config["data_params"]["data_classes"]
    data_dir = config["data_params"]["splitted_data_dir"]
    cnn_types = ['cnn', 'adbench', 'adbench_2d']# 'cnn']  # 'adbench' is used for the AdBench method 

    device = config["trainer_params"]["device"]
    raw_suffix = config["data_params"]["raw_suffix"]
    swap_suffix = config["data_params"]["swap_suffix"]

    subsets = ['train', 'val']

    results = []

    # CNN model
    model_cnn = timm.create_model(config["model_params"]["name"], pretrained=config["model_params"]["pretrained"])
    model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
    model_cnn.eval()
    print("Model correctly initialized")

    transform_cnn = set_feature_extraction_transform()

    for cnn_type in cnn_types:
        for i, main_insect_class in enumerate(insect_classes):
            mislabeled_insect_class = insect_classes[1 - i]

            df_subsets = []
            for subset in subsets:
                if 'adbench' in cnn_type:
                    df_subset_path = os.path.join(data_dir, f'df_{subset}_{raw_suffix}_{main_insect_class}.csv')
                else:
                    df_subset_path = os.path.join(data_dir, f'df_{subset}_{swap_suffix}_{main_insect_class}.csv')
                df_subset = pd.read_csv(df_subset_path)
                df_subsets.append(df_subset)
            df_train_val = pd.concat(df_subsets, ignore_index=True)

            od_methods = ['UmapHdbscanOD', 'MCD'] if cnn_type == 'cnn' else ['OCSVM']
            
            for od_method in od_methods:
                df_train_val_clean, df_train_val_outliers, metrics = clean_df(
                    df_train_val, 
                    model_cnn, 
                    config, 
                    transform_cnn, 
                    pin_memory, 
                    main_insect_class, 
                    phase='train_val', 
                    method=cnn_type, 
                    od_method=od_method
                )
                
                os.makedirs(os.path.join(data_dir, 'clean'), exist_ok=True)
                os.makedirs(os.path.join(data_dir, 'outliers'), exist_ok=True)
                
                for subset in subsets:
                    df_subset_clean = df_train_val_clean[df_train_val_clean.filepath.str.contains(subset)]
                    df_subset_outliers = df_train_val_outliers[df_train_val_outliers.filepath.str.contains(subset)]

                    df_subset_clean_path = os.path.join(data_dir, 'clean', f'df_{subset}_{cnn_type}_{main_insect_class}_{od_method}_clean.csv')
                    df_subset_clean.to_csv(df_subset_clean_path, index=False)
                    print(f'Clean {main_insect_class} {cnn_type} {subset} dataset available')

                    df_subset_outliers_path = os.path.join(data_dir, 'outliers', f'df_{subset}_{cnn_type}_{main_insect_class}_{od_method}_outliers.csv')
                    df_subset_outliers.to_csv(df_subset_outliers_path, index=False)
                    print(f'Outliers {main_insect_class} {cnn_type} {subset} dataset available')

                # Add more information to the metrics dictionary
                metrics['method'] = cnn_type
                metrics['od_method'] = od_method
                metrics['main_insect_class'] = main_insect_class

                results.append(metrics)
                df_results = pd.DataFrame(results)
                df_results.to_csv(os.path.join(config["logging_params"]["save_dir"],f'df_best_od_evaluation.csv'), mode='a', index=False, header=False)
                print('metrics: ', metrics)
            print('')