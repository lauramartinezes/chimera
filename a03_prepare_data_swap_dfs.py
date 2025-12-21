import os
import numpy as np
import pandas as pd
import timm
import torch
import yaml
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import set_test_transform
from datasets.load_data import load_data_from_df
from models import compute_accuracy, compute_predictions, plot_prediction_confidence_by_predicted_class, plot_probability_distribution
from utils import set_seed


def update_mislabeled_flags(row, mislabeled_col, insect_0_value, insect_classes):
    if row[mislabeled_col]:
        row[f'mislabeled_{insect_classes[0]}'] = insect_0_value
        row[f'mislabeled_{insect_classes[1]}'] = not insect_0_value
    else:
        row[f'mislabeled_{insect_classes[0]}'] = False
        row[f'mislabeled_{insect_classes[1]}'] = False
    return row


# Step 1: Define categories using df-consistent terminology
def get_category_lists(insect_classes):
    true_categories = []

    # Add 'good' and 'mislabeled' categories for each real class
    for i, cls in enumerate(insect_classes):
        other_cls = insect_classes[1 - i]  # pick the other class
        true_categories.append(f"{cls}_good")
        true_categories.append(f"{cls}_labeled_as_{other_cls}")

    # Add measurement noise categories
    for cls in insect_classes:
        true_categories.append(f"measurement_noise_labeled_as_{cls}")

    
    predicted_categories = insect_classes  # still only the predicted class names
    return true_categories, predicted_categories


# Step 2: Determine the true category of a sample   
def get_true_category(row, insect_classes):
    if row["measurement_noise"]:
        cls = insect_classes[row["label"]]  # still uses original label
        return f"measurement_noise_labeled_as_{cls}"

    cls = insect_classes[row["label"]]
    if row["mislabeled"]:
        # Swap logic for mislabeled
        if cls == insect_classes[0]:  # wmv mislabeled
            return f"{insect_classes[1]}_labeled_as_{insect_classes[0]}"
        else:  # v mislabeled
            return f"{insect_classes[0]}_labeled_as_{insect_classes[1]}"
    else:
        return f"{cls}_good"



# Step 3: Determine the predicted category based on classifier output and true nature
def get_predicted_category(true_cat, pred_label, insect_classes):
    return insect_classes[pred_label]


# Step 4: Compute confusion matrix
def build_extended_confusion_matrix(df, all_predictions, insect_classes):
    true_cats, pred_cats = get_category_lists(insect_classes)
    conf_matrix = pd.DataFrame(0, index=true_cats, columns=pred_cats)

    for idx, row in df.iterrows():
        true_cat = get_true_category(row, insect_classes)
        pred_label = all_predictions[idx]
        pred_cat = get_predicted_category(true_cat, pred_label, insect_classes)
        if pred_cat in conf_matrix.columns:
            conf_matrix.loc[true_cat, pred_cat] += 1
    
    return conf_matrix


# Step 5: Plotting
def plot_confusion_matrix(conf_matrix, subtitle=None, path='.'):
    plt.figure(figsize=(10, 8))

    # Normalize the values row-wise for color scaling
    normalized = conf_matrix.div(conf_matrix.sum(axis=1), axis=0).fillna(0)

    # Plot the heatmap with normalized colors but integer annotations
    sns.heatmap(
        normalized,
        annot=conf_matrix,          # Show actual counts
        fmt="d",                    # Format annotation as integer
        cmap="Blues",
        vmin=0, vmax=1              # Row-wise normalization is in [0,1]
    )

    plt.xticks(rotation=0)# 20, ha="right")  # Make x-axis labels easier to read
    plt.yticks(rotation=0)               # Keep y-axis labels straight

    plt.title(f"Confusion Matrix ({subtitle.title()})" if subtitle else "Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class and Nature")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'confusion_matrix_{subtitle}.svg'))



if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

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
    ### RESNET-18 MODEL
    ##########################
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.to(device)

    save_path_best = os.path.join(
        config["logging_params"]["save_dir"], 
        f'{model_name}_classifier_{raw_suffix}_best_initial.pth'
    )
    
    # Load the model
    if os.path.exists(save_path_best):
        model.load_state_dict(torch.load(save_path_best))
        print("Model correctly loaded")
    else:
        print("Model not found")

    conf_matrix_path = os.path.join('logs', 'Conf_Mat_after_swap')
    os.makedirs(conf_matrix_path, exist_ok=True)

    all_confusion_matrices = []

    for subset in subsets:
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
            set_test_transform(),
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
            print(f"Accuracy: {accuracy}") 
        
        conf_matrix = build_extended_confusion_matrix(df_subset, all_subset_predictions, insect_classes)
        all_confusion_matrices.append(conf_matrix)
        plot_confusion_matrix(conf_matrix, subtitle=subset, path=conf_matrix_path)

        df_subset['pred_label'] = all_subset_predictions
        df_subset['pred_probas'] = all_subset_probs

        df_insect_0 = df_subset[df_subset['pred_label'] == 0]
        df_insect_1 = df_subset[df_subset['pred_label'] == 1]

        # update mislabeled column
        df_insect_0['mislabeled'] = df_insect_0["directory"].isin([f"{data_dir}/{subset}/{insect_classes[0]}/{insect_classes[1]}_for_{insect_classes[0]}", f"{data_dir}/{subset}/{insect_classes[1]}/{insect_classes[1]}_good"])
        df_insect_1['mislabeled'] = df_insect_1["directory"].isin([f"{data_dir}/{subset}/{insect_classes[1]}/{insect_classes[0]}_for_{insect_classes[1]}", f"{data_dir}/{subset}/{insect_classes[0]}/{insect_classes[0]}_good"])

        # update labels column to be outlier or inlier
        df_insect_0['noisy_label_classification'] = df_insect_0['label']
        df_insect_1['noisy_label_classification'] = df_insect_1['label']

        outlier_labels_insect_0 = ((df_insect_0['measurement_noise']==True) | (df_insect_0['mislabeled'])).astype(int)
        outlier_labels_insect_1 = ((df_insect_1['measurement_noise']==True) | (df_insect_1['mislabeled'])).astype(int)

        df_insect_0['label'] = outlier_labels_insect_0
        df_insect_1['label'] = outlier_labels_insect_1

        df_insect_0_path = os.path.join(data_dir, f'df_{subset}_{swap_suffix}_{insect_classes[0]}.csv')
        df_insect_0.to_csv(df_insect_0_path)

        df_insect_1_path = os.path.join(data_dir, f'df_{subset}_{swap_suffix}_{insect_classes[1]}.csv')
        df_insect_1.to_csv(df_insect_1_path)

        print('')
    
    # Get total confusion matrix
    conf_matrix_total = sum(all_confusion_matrices)
    plot_confusion_matrix(conf_matrix_total, subtitle='total', path=conf_matrix_path)

    print('')
