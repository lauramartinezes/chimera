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
from outlier_detectors import PYOD, metric
from datasets import CustomBinaryInsectDF
from outlier_detectors import UmapHdbscanOD


def process_data_cnn(df, model, device, config, transform, main_insect_class, phase="train", cnn_type='cnn'):
    loader = load_data_from_df(
        df,
        transform,
        config["exp_params"]["manual_seed"],
        config["data_params"][f"train_batch_size"],
        config["data_params"]["num_workers"],
        pin_memory,
    )
    latents_cnn, labels_cnn, real_labels_cnn, measurement_noise_cnn, mislabeled_cnn = extract_features_from_dataloader(loader, model)

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


def get_outlier_methods_csv(X_train, measurement_noises,  label_noises, filename=None, dirname=None):
    csv_folder = os.path.join(dirname, 'OUTLIERS_CSVS')
    os.makedirs(csv_folder, exist_ok=True)

    y_train = measurement_noises + label_noises

    metrics_list = []
    models = ['UmapHdbscanOD','SOD', 'DeepSVDD', 'MOGAAL','IForest', 'OCSVM', #'SOGAAL', 'SOS', 
              'ABOD', 'COF', 'COPOD', 'ECOD',  'FeatureBagging', 'HBOS', #, 'CBLOF'
              'KNN', 'LMDD', 'LODA', 'LOF', 'MCD']#, 'PCA']
    
    for model in models:
        if model!='UmapHdbscanOD':
            pyod_model = PYOD(seed=42, model_name=model)
            pyod_model.fit(X_train, [])
            anomaly_scores = pyod_model.predict_score(X_train)
        else:
            umap_hdbscan_od = UmapHdbscanOD()
            best_dim, best_mcs, best_ms = umap_hdbscan_od.find_optimal_params(X_train)
            anomaly_scores = umap_hdbscan_od.predict_outliers(X_train, best_dim, best_mcs, best_ms)
                
        metrics = metric(y_true=y_train, y_score=anomaly_scores, pos_label=1)
        print(f'{model}: {metrics}')

        metrics_list.append({'Model': model, 'aucroc': metrics['aucroc'], 'aucpr': metrics['aucpr']})
        temp_metrics_df = pd.DataFrame(metrics_list)
        temp_metrics_df.to_csv(os.path.join(csv_folder, f'{filename}_outlier_metrics.csv'), index=False)


def extract_features_from_dataloader(dataloader, model):
    all_features = []  # To store features from all batches
    all_labels = []    # To store labels if needed
    all_real_labels = []
    all_measurement_noise = []
    all_mislabeled = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels, real_labels, measurement_noise, mislabeled, outliers in tqdm(dataloader, desc="Extracting features", total=len(dataloader)):
            # Forward pass to get the features
            features = model(images)  # Get features for the batch
            features_np = features.numpy().reshape(features.shape[0], -1)  # Flatten to 2D array
            all_features.append(features_np)
            all_labels.append(labels.numpy())  # Collect labels if needed
            all_real_labels.append(real_labels.numpy())  # Collect labels if needed
            all_measurement_noise.append(measurement_noise.numpy())  # Collect labels if needed
            all_mislabeled.append(mislabeled.numpy())  # Collect labels if needed

    # Stack all features and labels vertically
    return np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_real_labels), np.concatenate(all_measurement_noise), np.concatenate(all_mislabeled)


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

    df_test_path = os.path.join(data_dir, f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    transform_cnn = set_feature_extraction_transform()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        model_cnn = timm.create_model('resnet18', pretrained=True)
        model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
        model_cnn.eval()

        # Resnet method
        df_train_path = os.path.join(data_dir, f'df_train_ae_{main_insect_class}.csv')
        df_val_path = os.path.join(data_dir, f'df_val_ae_{main_insect_class}.csv')
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
        df_train_path = os.path.join(data_dir, f'df_train_raw_{main_insect_class}.csv') 
        df_val_path = os.path.join(data_dir, f'df_val_raw_{main_insect_class}.csv')
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
