import os

import pandas as pd

from outlier_detectors.PyOD import PYOD
from outlier_detectors.UmapHdbscanOD import UmapHdbscanOD
from outlier_detectors.utils import metric


def get_outlier_methods_csv(X_train, measurement_noises,  label_noises, filename=None, dirname=None, seed=42):
    csv_folder = os.path.join(dirname, 'OUTLIERS_CSVS')
    os.makedirs(csv_folder, exist_ok=True)

    y_train = measurement_noises + label_noises

    metrics_list = []
    models = ['UmapHdbscanOD','SOD', 'DeepSVDD', 'MOGAAL','IForest', 'OCSVM', #'SOGAAL', 'SOS', 
              'ABOD', 'COF', 'COPOD', 'ECOD',  'FeatureBagging', 'HBOS', #, 'CBLOF'
              'KNN', 'LMDD', 'LODA', 'LOF', 'MCD']#, 'PCA']
    
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
        temp_metrics_df.to_csv(os.path.join(csv_folder, f'{filename}_outlier_metrics.csv'), index=False)