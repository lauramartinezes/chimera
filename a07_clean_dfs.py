import os
import pandas as pd
import timm
import torch

from datasets import set_feature_extraction_transform
from datasets.clean_df import clean_df, clean_df_no_od
from utils import load_config, set_seed


if __name__ == '__main__':
    config = load_config("config.yaml")
    set_seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0
    device = config["trainer_params"]["device"]
    raw_suffix = config["data_params"]["raw_suffix"]
    swap_suffix = config["data_params"]["swap_suffix"]

    insect_classes = config["data_params"]["data_classes"]
    data_dir = config["data_params"]["splitted_data_dir"]
    
    cleaning_strategies = config['cleaning_params']['strategies']
    subsets = config['cleaning_params']['subsets']

    results = []

    # CNN model
    model = timm.create_model(
        config["model_params"]["name"], 
        pretrained=config["model_params"]["pretrained"]
    )
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    print("Model correctly initialized")

    os.makedirs(os.path.join(data_dir, 'clean'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'outliers'), exist_ok=True)

    for strategy in cleaning_strategies:
        od_method = config['cleaning_params'][strategy]['outlier_detection_method']
        for i, main_insect_class in enumerate(insect_classes):
            mislabeled_insect_class = insect_classes[1 - i]

            df_subsets = []
            for subset in subsets:
                if 'adbench' in strategy:
                    df_subset_path = os.path.join(data_dir, f'df_{subset}_{raw_suffix}_{main_insect_class}.csv')
                else:
                    df_subset_path = os.path.join(data_dir, f'df_{subset}_{swap_suffix}_{main_insect_class}.csv')
                df_subset = pd.read_csv(df_subset_path)
                df_subsets.append(df_subset)
            df_train_val = pd.concat(df_subsets, ignore_index=True)
            
            if od_method == 'no_od':
                df_train_val_clean, df_train_val_outliers, metrics = clean_df_no_od(
                    df_train_val, 
                    main_insect_class
                    )
            else:
                df_train_val_clean, df_train_val_outliers, metrics = clean_df(
                    df_train_val, 
                    model, 
                    config, 
                    set_feature_extraction_transform(), 
                    pin_memory, 
                    main_insect_class, 
                    phase='train_val', 
                    method=strategy, 
                    od_method=od_method
                )
            
            for subset in subsets:
                df_subset_clean = df_train_val_clean[df_train_val_clean.filepath.str.contains(subset)]
                df_subset_outliers = df_train_val_outliers[df_train_val_outliers.filepath.str.contains(subset)]

                df_subset_clean_path = os.path.join(
                    data_dir, 
                    'clean', 
                    f'df_{subset}_{strategy}_{main_insect_class}_{od_method}_clean.csv'
                )
                df_subset_clean.to_csv(df_subset_clean_path, index=False)
                print(f'Clean {main_insect_class} {strategy} {subset} dataset available')

                df_subset_outliers_path = os.path.join(
                    data_dir, 
                    'outliers', 
                    f'df_{subset}_{strategy}_{main_insect_class}_{od_method}_outliers.csv'
                )
                df_subset_outliers.to_csv(df_subset_outliers_path, index=False)
                print(f'Outliers {main_insect_class} {strategy} {subset} dataset available')

            # Add more information to the metrics dictionary
            metrics['method'] = strategy
            metrics['od_method'] = od_method
            metrics['main_insect_class'] = main_insect_class

            results.append(metrics)
            df_results = pd.DataFrame(results)
            results_path = os.path.join(config["logging_params"]["save_dir"], 'df_best_od_evaluation.csv')
            df_results.to_csv(results_path, mode='a', index=False, header=False)
            print('metrics: ', metrics)
            print('')