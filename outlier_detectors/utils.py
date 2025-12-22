import random
import numpy as np
import torch
import umap

from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed):
    # basic seed
    np.random.seed(seed)
    random.seed(seed)

    # pytorch seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric(y_true, y_score, pos_label=1):
    aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
    aucpr = average_precision_score(y_true=y_true, y_score=y_score, pos_label=pos_label)

    return {'aucroc':aucroc, 'aucpr':aucpr}


def preprocess_latents_for_outlier_detection(latents_cnn, cnn_type):
    if '2d' in cnn_type:
        reducer_2_d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        return reducer_2_d.fit_transform(latents_cnn)
    else:
        return latents_cnn
    

def select_best_outlier_detection_model(df_outliers):
    eps = 1e-12  # avoid divide-by-zero
    vals = df_outliers.select_dtypes(include="number").to_numpy()   # grabs aucroc/aucpr columns automatically
    df_outliers[("overall", "harmonic_mean")] = vals.shape[1] / np.sum(1.0 / (vals + eps), axis=1)

    model_col = df_outliers.select_dtypes(exclude="number").columns[0]
    best_idx = df_outliers[("overall", "harmonic_mean")].idxmax()
    best_model = df_outliers.loc[best_idx, model_col]
    return best_model