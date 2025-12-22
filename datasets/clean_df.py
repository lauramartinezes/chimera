import os
import pandas as pd
import umap
from datasets.load_data import load_data_from_df
from models import extract_features
from outlier_detectors.PyOD import PYOD
from outlier_detectors.UmapHdbscanOD import UmapHdbscanOD
from outlier_detectors.plot import plot_y_true_vs_y_od_pred_umap
from outlier_detectors.utils import metric


def clean_df(df, model, config, transform, pin_memory, main_insect_class, insect_classes, phase="train", method='cnn', od_method='UmapHdbscanOD', seed=42):
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
        seed=seed
    )

    N = len(latents)
    default_min_cluster_size = max(5, int(0.05 * N))  # Default value for HDBSCAN
    default_min_samples = max(5, int(0.01 * N))  # Default value for HDBSCAN

    if method == 'adbench':
        optimal_min_cluster_size = default_min_cluster_size
        optimal_min_samples = default_min_samples
    elif 'adbench_2d' in method:
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
            reducer_optimd = umap.UMAP(n_components=optimal_dim, random_state=seed, n_jobs=1)
            latents = reducer_optimd.fit_transform(latents)
        y_pred, metrics = get_outlier_predictions(
            latents,
            y_true,
            model=od_method,
            seed=seed
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
    # Remove outliers
    df_no_outliers = df[y_pred == 0]

    # Remove samples whose predicted label does not coincide with the original one
    if f'noisy_label_classification' in df_no_outliers.columns:
        reference_label = insect_classes.index(main_insect_class)
        df_good = df_no_outliers[df_no_outliers[f'noisy_label_classification'] == reference_label]
        df_mislabels = df_no_outliers[df_no_outliers[f'noisy_label_classification'] != reference_label]
        df_mislabels[f'noisy_label_classification'] = reference_label

        if 'corrected_mislabels' in method:
            df_clean = pd.concat([df_good, df_mislabels], ignore_index=True)
        else:
            df_clean = df_good
    else:
        df_clean = df_no_outliers
        
    return df_clean, df, metrics

def clean_df_no_od(df, main_insect_class, insect_classes):
    df['outlier_detected'] = False
    reference_label = insect_classes.index(main_insect_class)
    df_good = df[df[f'noisy_label_classification'] == reference_label]

    return df_good, df, {}


def get_outlier_predictions(X_train, y_train, model='MCD', seed=42):
    print(f'{model} outlier extraction')
    pyod_model = PYOD(seed=seed, model_name=model)
    pyod_model.fit(X_train, [])
    anomaly_scores = pyod_model.predict_score(X_train)
    metrics = metric(y_true=y_train, y_score=anomaly_scores, pos_label=1)
    anomaly_binary_scores = pyod_model.predict_binary_score(X_train)

    return anomaly_binary_scores, metrics