import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

def plot_conf_matrix(row, title, method, path):
    """Plots a confusion matrix with row-wise color scaling."""
    conf_matrix = pd.DataFrame([
        [row['good_pred_good'], row['good_pred_mislabels'], row['good_pred_meas_noise']],  
        [row['mislabels_pred_good'], row['mislabels_pred_mislabels'], row['mislabels_pred_meas_noise']],  
        [row['meas_noise_pred_good'], row['meas_noise_pred_mislabels'], row['meas_noise_pred_meas_noise']]  
    ], index=['Actual Good', 'Actual Mislabels', 'Actual Meas. Noise'],
       columns=['Pred Good', 'Pred Mislabels', 'Pred Meas. Noise'])

    # Normalize each row independently
    conf_matrix_normalized = conf_matrix.div(conf_matrix.max(axis=1), axis=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_normalized, annot=conf_matrix, cmap="Blues", fmt=".0f")
    plt.title(f'Confusion Matrix - {method} - {title}')
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    # plt.show()
    plt.savefig(os.path.join(path, f'confusion_matrix_{method}_{title}.png'))
    plt.savefig(os.path.join(path, f'confusion_matrix_{method}_{title}.svg'))


if __name__ == '__main__':  
    pd.set_option('display.max_columns', None)  
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    insect_classes = config["data_params"]["data_classes"]
    method_datasets = ['adbench', 'cnn'] #'ae', 'adv_ae', 'resnet18' 'adbench_2d'
    subsets = ['train', 'val']


    for method in method_datasets:
        # Generate a df with rows for each insect class
        # df_analysis = pd.DataFrame({'insect_class': insect_classes})
        df_analysis = pd.DataFrame(index=insect_classes)

        if method == 'adbench':
            od_methods = ['OCSVM']
        elif method == 'cnn':
            od_methods = ['DBSCAN', 'MCD']
        elif method == 'adbench_2d':
            od_methods = ['MCD']

        print(f'Analyzing {method} method')
        for od_method in od_methods:
            for i in range(len(insect_classes)):
                main_insect_class = insect_classes[i]
                mislabeled_insect_class = insect_classes[1 - i]

                df_subsets = []
                for subset in subsets:
                    df_subset_path = os.path.join('data', 'outliers', f'df_{subset}_{method}_{main_insect_class}_{od_method}_outliers.csv')
                    df_subset = pd.read_csv(df_subset_path)
                    df_subsets.append(df_subset)
                df = pd.concat(df_subsets)

                if 'directory' not in df.columns:
                    df['directory'] = df['filepath'].apply(os.path.dirname)

                good_original = df[df['directory'].str.contains(f'{os.sep}{main_insect_class}_good', case=False, na=False)]
                good_pred_good = good_original[good_original['outlier_detected'] == False].shape[0]
                good_pred_meas_noise = good_original[good_original['outlier_detected'] == True].shape[0]
                good_pred_mislabels = 0 #these moved to the other class

                meas_noise_original = df[df['directory'].str.contains(f'{os.sep}{main_insect_class}_trash', case=False, na=False)]
                meas_noise_pred_good = meas_noise_original[meas_noise_original['outlier_detected'] == False].shape[0]
                meas_noise_pred_meas_noise = meas_noise_original[meas_noise_original['outlier_detected'] == True].shape[0]
                meas_noise_pred_mislabels = 0 #these moved to the other class

                mislabels_original = df[df['directory'].str.contains(f'{os.sep}{mislabeled_insect_class}_for_{main_insect_class}', case=False, na=False)]
                mislabels_pred_good = mislabels_original[mislabels_original['outlier_detected'] == False].shape[0]
                mislabels_pred_meas_noise = mislabels_original[mislabels_original['outlier_detected'] == True].shape[0]
                mislabels_pred_mislabels = 0 #these moved to the other class

                df_analysis.loc[main_insect_class, 'good_pred_good'] = good_pred_good
                df_analysis.loc[main_insect_class, 'good_pred_mislabels'] = good_pred_mislabels #these moved to the other class
                df_analysis.loc[main_insect_class, 'good_pred_meas_noise'] = good_pred_meas_noise #these moved to the other class
                df_analysis.loc[main_insect_class, 'meas_noise_pred_meas_noise'] = meas_noise_pred_meas_noise
                df_analysis.loc[main_insect_class, 'meas_noise_pred_good'] = meas_noise_pred_good
                df_analysis.loc[main_insect_class, 'meas_noise_pred_mislabels'] = meas_noise_pred_mislabels
                df_analysis.loc[main_insect_class, 'mislabels_pred_mislabels'] = mislabels_pred_mislabels
                df_analysis.loc[main_insect_class, 'mislabels_pred_good'] = mislabels_pred_good
                df_analysis.loc[main_insect_class, 'mislabels_pred_meas_noise'] = mislabels_pred_meas_noise
            
            for i in range(len(insect_classes)):
                main_insect_class = insect_classes[i]
                mislabeled_insect_class = insect_classes[1 - i]  
                df_subsets = []
                for subset in subsets:
                    df_subset_path = os.path.join('data', 'outliers', f'df_{subset}_{method}_{main_insect_class}_{od_method}_outliers.csv')
                    df_subset = pd.read_csv(df_subset_path)
                    df_subsets.append(df_subset)
                df = pd.concat(df_subsets)

                if 'directory' not in df.columns:
                    df['directory'] = df['filepath'].apply(os.path.dirname)

                mislabels_transferred = df[df['directory'].str.contains(f'{os.sep}{main_insect_class}_for_{mislabeled_insect_class}', case=False, na=False)]
                mislabels_transferred_pred_mislabels = mislabels_transferred[mislabels_transferred['outlier_detected'] == False].shape[0]
                mislabels_transferred_pred_meas_noise = mislabels_transferred[mislabels_transferred['outlier_detected'] == True].shape[0]

                good_transferred = df[df['directory'].str.contains(f'{os.sep}{mislabeled_insect_class}_good', case=False, na=False)]
                good_transferred_pred_mislabels = good_transferred[good_transferred['outlier_detected'] == False].shape[0]
                good_transferred_pred_meas_noise = good_transferred[good_transferred['outlier_detected'] == True].shape[0]
                
                meas_noise_transferred = df[df['directory'].str.contains(f'{os.sep}{mislabeled_insect_class}_trash', case=False, na=False)]
                meas_noise_transferred_pred_mislabels = meas_noise_transferred[meas_noise_transferred['outlier_detected'] == False].shape[0]
                meas_noise_transferred_pred_meas_noise = meas_noise_transferred[meas_noise_transferred['outlier_detected'] == True].shape[0]

                df_analysis.loc[mislabeled_insect_class, 'mislabels_pred_mislabels'] += mislabels_transferred_pred_mislabels
                df_analysis.loc[mislabeled_insect_class, 'mislabels_pred_meas_noise'] += mislabels_transferred_pred_meas_noise
                df_analysis.loc[mislabeled_insect_class, 'good_pred_mislabels'] += good_transferred_pred_mislabels
                df_analysis.loc[mislabeled_insect_class, 'good_pred_meas_noise'] += good_transferred_pred_meas_noise
                df_analysis.loc[mislabeled_insect_class, 'meas_noise_pred_mislabels'] += meas_noise_transferred_pred_mislabels
                df_analysis.loc[mislabeled_insect_class, 'meas_noise_pred_meas_noise'] += meas_noise_transferred_pred_meas_noise

            data_analysis_path = os.path.join(config["logging_params"]["save_dir"], 'data_analysis')
            os.makedirs(data_analysis_path, exist_ok=True)
            df_analysis.to_csv(os.path.join(data_analysis_path, f'df_analysis_{method}_{od_method}_train_val.csv'))

            for index, row in df_analysis.iterrows():
                plot_conf_matrix(row, index + f'_{od_method}_train_val', method, data_analysis_path)
            
            print('')


                    