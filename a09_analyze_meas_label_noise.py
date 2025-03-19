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
        [row['meas_noise_pred_good'], row['meas_noise_pred_mislabels'], row['meas_noise_detected']]  
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

    insect_classes = ['wmv', 'c']
    method_datasets = ['ae', 'adv_ae', 'cnn']


    for method in method_datasets:
        # Generate a df with rows for each insect class
        # df_analysis = pd.DataFrame({'insect_class': insect_classes})
        df_analysis = pd.DataFrame(index=insect_classes)

        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_path = os.path.join('data', 'outliers', f'df_train_{method}_{main_insect_class}_outliers.csv')
            df = pd.read_csv(df_path)

            
            mislabels_pred_mislabels = len(df[df[f'mislabeled_{main_insect_class}'] == True])
            mislabels_pred_good = len(df[(df['outlier_detected'] == False) & (df[f'mislabeled_{mislabeled_insect_class}'] == True) & (df[f'measurement_noise'] == False)])
            mislabels_pred_meas_noise = len(df[(df['outlier_detected'] == True) & (df[f'mislabeled_{mislabeled_insect_class}'] == True) & (df[f'measurement_noise'] == False)])
            
            good_pred_good = ((df['noisy_label_classification']==i) & (df.noisy_outlier_label_original==0) & (df.outlier_detected==False)).value_counts().get(True, 0)
            good_pred_meas_noise = ((df['noisy_label_classification']==i) & (df.noisy_outlier_label_original==0)& (df.outlier_detected==True)).value_counts().get(True, 0)
            good_pred_mislabels = 4200 - good_pred_good - good_pred_meas_noise

            meas_noise_detected = ((df['noisy_label_classification']==i) & (df.noisy_outlier_label_original==1) & (df.measurement_noise==True) & (df['outlier_detected'] == True)).value_counts().get(True, 0)
            meas_noise_pred_good = ((df['noisy_label_classification']==i) & (df.noisy_outlier_label_original==1) & (df.measurement_noise==True) & (df['outlier_detected'] == False)).value_counts().get(True, 0)
            meas_noise_pred_mislabels = ((df['noisy_label_classification']==1-i) & (df.noisy_outlier_label_original==1) & (df.measurement_noise==True)).value_counts().get(True, 0)

            df_analysis.loc[main_insect_class, 'good_pred_good'] = good_pred_good
            df_analysis.loc[main_insect_class, 'good_pred_mislabels'] = good_pred_mislabels #these moved to the other class
            df_analysis.loc[main_insect_class, 'good_pred_meas_noise'] = good_pred_meas_noise #these moved to the other class
            df_analysis.loc[main_insect_class, 'meas_noise_detected'] = meas_noise_detected
            df_analysis.loc[main_insect_class, 'meas_noise_pred_good'] = meas_noise_pred_good
            df_analysis.loc[main_insect_class, 'meas_noise_pred_mislabels'] = meas_noise_pred_mislabels
            df_analysis.loc[mislabeled_insect_class, 'mislabels_pred_mislabels'] = mislabels_pred_mislabels
            df_analysis.loc[main_insect_class, 'mislabels_pred_good'] = mislabels_pred_good
            df_analysis.loc[main_insect_class, 'mislabels_pred_meas_noise'] = mislabels_pred_meas_noise

        data_analysis_path = os.path.join(config["logging_params"]["save_dir"], 'data_analysis')
        os.makedirs(data_analysis_path, exist_ok=True)
        df_analysis.to_csv(os.path.join(data_analysis_path, f'df_analysis_{method}.csv'))

        for index, row in df_analysis.iterrows():
            plot_conf_matrix(row, index, method, data_analysis_path)
        
        # Sum both rows to create an overall confusion matrix
        df_analysis_overall = df_analysis.sum()

        # Plot the confusion matrix for the summed data
        plot_conf_matrix(df_analysis_overall, "all_data", method, data_analysis_path)
        print('')

# P

            