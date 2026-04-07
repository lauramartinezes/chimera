import os
import pandas as pd
import timm
import torch

from datasets import load_data_from_df, set_test_transform, plot_conf_matrix_after_swap
from models import compute_accuracy, compute_predictions
from utils import load_config, set_seed


if __name__ == '__main__':
    config = load_config("config.yaml")

    # Set manual seed for reproducibility
    random_seed = config["exp_params"]["manual_seed"]
    set_seed(random_seed)

    pin_memory = len(config['trainer_params']['gpus']) != 0

    raw_suffix = config["data_params"]["raw_suffix"]
    swap_suffix = config["data_params"]["swap_suffix"]

    ##########################
    ### BINARY CLASS INSECT DATASET
    ##########################    
    insect_classes = config["data_params"]["data_classes"]
    num_classes = len(insect_classes)
    data_dir = config["data_params"]["splitted_data_dir"]
    model_name = config["model_params"]["name"] 
    subsets = ['train', 'val']
    device = config["trainer_params"]["device"]

    ##########################
    ### MODEL LOADING
    ##########################
    save_path_best = os.path.join(
        config["logging_params"]["save_dir"], 
        f'{model_name}_classifier_{raw_suffix}_best_initial.pth'
    )
    
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.to(device)
    
    if os.path.exists(save_path_best):
        model.load_state_dict(torch.load(save_path_best))
        print("Model correctly loaded")
    else:
        print("Model not found")

    dfs = []
    for subset in subsets:
        ##########################
        ### DATA LOADING
        ##########################
        dfs_subset = []
        for i, main_insect_class in enumerate(insect_classes):
            mislabeled_insect_class = insect_classes[1 - i]

            df_subset_path = os.path.join(data_dir, f'df_{subset}_{raw_suffix}_{main_insect_class}.csv')
            df_subset_i = pd.read_csv(df_subset_path)
            df_subset_i['noisy_outlier_label_original'] = df_subset_i.label
            
            # Correct labels for training a classifier instead of an outlier detector
            df_subset_i.label = i
            df_subset_i[f'mislabeled_{main_insect_class}'] = df_subset_i['mislabeled']
            df_subset_i[f'mislabeled_{mislabeled_insect_class}'] = False

            dfs_subset.append(df_subset_i)
        df_subset = pd.concat(dfs_subset, ignore_index=True)
        df_subset['directory'] = df_subset['filepath'].apply(os.path.dirname)

        loader = load_data_from_df(
            df_subset,
            set_test_transform(image_size=config["data_params"]["image_size"]),
            config["exp_params"]["manual_seed"],
            config["data_params"][f"batch_size"],
            config["data_params"]["num_workers"],
            pin_memory,
            shuffle=False
        )
        
        ##########################
        ### INFERENCE
        ##########################
        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            all_subset_predictions, all_subset_actuals, all_subset_probs = compute_predictions(model, loader, device=device)
            accuracy = compute_accuracy(model, loader, device=device)
            print(f"Prediction: {all_subset_predictions[0]}\nActual: {all_subset_actuals[0]}\nProbabilities: {all_subset_probs[0]}")
            print(f"Accuracy: {accuracy[0]}") 
        
        df_subset['pred_label'] = all_subset_predictions
        df_subset['pred_probas'] = all_subset_probs

        dfs.append(df_subset)

        for i, main_insect_class in enumerate(insect_classes):
            mislabeled_insect_class = insect_classes[1 - i]

            df_out = df_subset[df_subset["pred_label"] == i].copy()

            # update mislabeled column
            df_out["mislabeled"] = df_out["directory"].isin([
                os.path.join(data_dir, subset, main_insect_class, f"{mislabeled_insect_class}_for_{main_insect_class}"),
                os.path.join(data_dir, subset, mislabeled_insect_class, f"{mislabeled_insect_class}_good"),
            ])

            # keep original label, then overwrite label with outlier/inlier
            df_out["noisy_label_classification"] = df_out["label"]
            df_out["label"] = (df_out["measurement_noise"] | df_out["mislabeled"]).astype(int)

            # save
            out_path = os.path.join(data_dir, f"df_{subset}_{swap_suffix}_{main_insect_class}.csv")
            df_out.to_csv(out_path, index=False)
    
    # Get total confusion matrix
    conf_matrix_path = os.path.join('logs', 'Conf_Mat_after_swap')
    os.makedirs(conf_matrix_path, exist_ok=True)

    df_all = pd.concat(dfs, ignore_index=True)
    plot_conf_matrix_after_swap(
        df_all, 
        df_all['pred_label'].to_list(), 
        insect_classes, 
        subtitle='total', 
        path=conf_matrix_path
    )
