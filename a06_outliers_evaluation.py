import os
import random
import cuml
import numpy as np
import pandas as pd
import timm
import torch
from tqdm import tqdm
import umap
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms

from PyOD import PYOD, metric
from a06_2_dbscan_evaluation import DBSCAN_OD
from mnist_dataset import CustomBinaryInsectDF
from vq_vae import VQVAE


def process_data_ae(df, model, device, config, transform, pin_memory, main_insect_class, phase="train", n_dim_reduction=512, ae_type=''):
    # Load the data
    loader = load_data_from_df(
        df,
        transform,
        config["exp_params"]["manual_seed"],
        config["data_params"][f"train_batch_size"],
        config["data_params"]["num_workers"],
        pin_memory,
    )
    print(f"{phase.capitalize()} dataset correctly loaded")
    
    # Extract features from encoding latents
    (
        raw_features_encoding,
        features_encoding,
        labels_encoding,
        real_labels_encoding,
        measurement_noise_encoding,
        mislabeled_encoding,
    ) = extract_features_from_encoding(model, loader, device)
    
    # Reduce dimensions to 2D for visualization
    reshaped_raw_features_encoding = raw_features_encoding.reshape(raw_features_encoding.shape[0], -1)
    
    # Reduce dimensions to 512D for outlier detection
    reducer_n_d = umap.UMAP(n_components=n_dim_reduction, random_state=42, n_jobs=1)
    #reducer_n_d = cuml.manifold.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_dim_reduction, random_state=42)

    latents_raw_encoding_n_d = reducer_n_d.fit_transform(reshaped_raw_features_encoding)
    
    get_outlier_methods_csv(
        latents_raw_encoding_n_d,
        measurement_noise_encoding.astype(int),
        mislabeled_encoding.astype(int),
        f'{ae_type}ae_{n_dim_reduction}d_{main_insect_class}_{phase}',
        dirname=config["logging_params"]["save_dir"]
    )


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
        #reducer_2_d = cuml.manifold.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        latents_cnn = reducer_2_d.fit_transform(latents_cnn)

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


def extract_features_from_encoding(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_features = []  # To store features from all batches
    all_labels = []    # To store labels if needed
    all_real_labels = []
    all_measurement_noise = []
    all_mislabeled = []
    all_encodings = []

    with torch.no_grad():
        for images, labels, real_labels, measurement_noise, mislabeled, outliers in tqdm(dataloader, desc="Featured extraction encoding: "):
            images = images.to(device)

            # Pass images through the encoder layers
            encoding = model.encode(images)[0]  # Raw encoder output

            # Global Average Pooling: Compute mean across spatial dimensions (H, W)
            pooled_encoding = encoding.mean(dim=[2, 3])  # Shape: [B, embedding_dim]

            # Append features to the list
            all_features.append(pooled_encoding.cpu().numpy())
            all_encodings.append(encoding.cpu().numpy())
            all_labels.append(labels.numpy())  # Collect labels if needed
            all_real_labels.append(real_labels.numpy())  # Collect labels if needed
            all_measurement_noise.append(measurement_noise.numpy())  # Collect labels if needed
            all_mislabeled.append(mislabeled.numpy())  # Collect labels if needed
    # Concatenate all features into a single array
    return np.vstack(all_encodings), np.vstack(all_features), np.concatenate(all_labels), np.concatenate(all_real_labels), np.concatenate(all_measurement_noise), np.concatenate(all_mislabeled)


def get_outlier_methods_csv(X_train, measurement_noises,  label_noises, filename=None, dirname=None):
    csv_folder = os.path.join(dirname, 'OUTLIERS_CSVS')
    os.makedirs(csv_folder, exist_ok=True)

    y_train = measurement_noises + label_noises

    metrics_list = []
    models = ['DBSCAN_OD','SOD', 'DeepSVDD', 'MOGAAL','IForest', 'OCSVM', #'SOGAAL', 'SOS', 
              'ABOD', 'COF', 'COPOD', 'ECOD',  'FeatureBagging', 'HBOS', #, 'CBLOF'
              'KNN', 'LMDD', 'LODA', 'LOF', 'MCD']#, 'PCA']
    
    for model in models:
        if model!='DBSCAN_OD':
            pyod_model = PYOD(seed=42, model_name=model)
            pyod_model.fit(X_train, [])
            anomaly_scores = pyod_model.predict_score(X_train)
        else:
            anomaly_scores = DBSCAN_OD(X_train, eps=0.5)
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

    df_test_path = os.path.join('data', f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    insect_classes = config["data_params"]["data_classes"]

    transform_ae = transforms.Compose([
        transforms.Resize((config["data_params"]["patch_size"], config["data_params"]["patch_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_cnn = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        df_val_path = os.path.join('data', f'df_val_ae_{main_insect_class}.csv')
        df_train_ = pd.read_csv(df_train_path)
        df_val_ = pd.read_csv(df_val_path)

        # Combine the training and validation dataframes
        df_train = pd.concat([df_train_, df_val_], ignore_index=True)

        # AE method
        # ae_types = ['', 'adv_']
        # for ae_type in ae_types:
        #     save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{config["logging_params"]["name"]}_{ae_type}{main_insect_class}_best.pth')
        #     model_ae = VQVAE(
        #         in_channels=config["model_params"]["in_channels"],
        #         embedding_dim=config["model_params"]["embedding_dim"],
        #         num_embeddings=config["model_params"]["num_embeddings"],
        #         img_size=config["model_params"]["img_size"],
        #         beta=config["model_params"]["beta"]
        #     ).to(device)
        #     model_ae.load_state_dict(torch.load(save_path_best))
        #     model_ae.to(device)
        #     print("Model correctly initialized")

        #     process_data_ae(
        #         df_train, 
        #         model_ae, 
        #         device, 
        #         config, 
        #         transform_ae, 
        #         pin_memory, 
        #         main_insect_class, 
        #         phase="train", 
        #         n_dim_reduction=512, 
        #         ae_type=ae_type
        #     )
        #     print(f'Metrics for AE {ae_type} are available')

        # Resnet method
        model_cnn = timm.create_model('resnet18', pretrained=True)
        model_cnn = torch.nn.Sequential(*(list(model_cnn.children())[:-1]))
        model_cnn.eval()

        # process_data_cnn(
        #     df_train, 
        #     model_cnn, 
        #     device, 
        #     config, 
        #     transform_cnn, 
        #     main_insect_class, 
        #     phase="train", 
        #     cnn_type='cnn'
        # )     
        # print('Metrics for CNN are available') 

        # Adbench method
        df_train_path = os.path.join('data', f'df_train_raw_{main_insect_class}.csv') 
        df_val_path = os.path.join('data', f'df_val_raw_{main_insect_class}.csv')
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
        print('Metrics for AdBench are available')

        # # VGG16 method
        # model_name = 'resnet18'
        # model_vgg16 = timm.create_model(model_name, pretrained=False, num_classes=2)
        # save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier_raw_best_.pth')#f'{model_name}_classifier{clean_dataset}_{method}_best.pth')
    
        # # Load the model
        # if os.path.exists(save_path_best):
        #     model_vgg16.load_state_dict(torch.load(save_path_best))
        #     print("Model correctly loaded")
        # else:
        #     print("Model not found")
        # model_vgg16 = torch.nn.Sequential(*(list(model_vgg16.children())[:-1]))
        # model_vgg16.eval()

        # process_data_cnn(
        #     df_train, 
        #     model_vgg16, 
        #     device, 
        #     config, 
        #     transform_cnn, 
        #     main_insect_class, 
        #     phase="train", 
        #     cnn_type=model_name
        # ) 
        # print('Metrics for VGG16 are available')

        print('')