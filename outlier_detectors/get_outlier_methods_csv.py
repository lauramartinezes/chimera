import os

import pandas as pd

from outlier_detectors.PyOD import PYOD
from outlier_detectors.UmapHdbscanOD import UmapHdbscanOD
from outlier_detectors.utils import metric


def get_outlier_methods_csv(X_train, measurement_noises,  label_noises, seed=42):
    y_train = measurement_noises + label_noises

    metrics_list = []
    
    models = ['SOS', 'IForest', 'OCSVM', 'ABOD', 'CBLOF', 'COF', #'AOM', aom considers scores from ensemble 
        'COPOD', 'ECOD',  'FeatureBagging', 'HBOS', 'KNN',
        'LMDD', 'LODA', 'LOF', #'LOCI', 'LSCP', 'MAD', loci takes a lifetime, lscp needs a detector list, MAD algorithm is just for univariate data. Got Data with 2 Dimensions
        'MCD', 'PCA', 'SOD', 'DeepSVDD', #'VAE','AutoEncoder','XGBOD' xgbod also needs an estimator list for unsupervised
        'MOGAAL'] #'ROD', 'SOGAAL' rod blocks the pc, sogaal there is a bug related to mismatching batch sizes in the lib
    
    for model in models:
        if model!='UmapHdbscanOD':
            pyod_model = PYOD(seed=seed, model_name=model)
            pyod_model.fit(X_train, [])
            anomaly_scores = pyod_model.predict_score(X_train)
        else:
            umap_hdbscan_od = UmapHdbscanOD(seed=seed)
            best_dim, best_mcs, best_ms = umap_hdbscan_od.find_optimal_params(X_train)
            anomaly_scores = umap_hdbscan_od.predict_outliers(X_train, best_dim, best_mcs, best_ms)
                
        metrics = metric(y_true=y_train, y_score=anomaly_scores, pos_label=1)
        print(f'{model}: {metrics}')

        metrics_list.append({'Model': model, 'aucroc': metrics['aucroc'], 'aucpr': metrics['aucpr']})
        temp_metrics_df = pd.DataFrame(metrics_list)
    return temp_metrics_df