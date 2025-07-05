import os
import random
import numpy as np
import pandas as pd
import timm
import torch
import yaml
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasets import CustomBinaryInsectDF, set_test_transform
from models import compute_accuracy, compute_predictions, plot_prediction_confidence_by_predicted_class, plot_probability_distribution


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

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
    plt.savefig(os.path.join(path, f'confusion_matrix_{subtitle}.png'))
    plt.savefig(os.path.join(path, f'confusion_matrix_{subtitle}.svg'))


##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0

    raw_suffix = config["data_params"]["raw_suffix"]
    swap_suffix = config["data_params"]["swap_suffix"]

    ##########################
    ### BINARY CLASS INSECT DATASET
    ##########################    
    insect_classes = config["data_params"]["data_classes"]
    num_classes = len(insect_classes)
    data_dir = config["data_params"]["data_dir"]
    model_name = config["model_params"]["name"] 
    subsets = ['train', 'val']
    device = config["trainer_params"]["device"]

    ##########################
    ### RESNET-18 MODEL
    ##########################
    torch.manual_seed(RANDOM_SEED)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.to(device)

    save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier_{raw_suffix}_best_.pth')
    
    # Load the model
    if os.path.exists(save_path_best):
        model.load_state_dict(torch.load(save_path_best))
        print("Model correctly loaded")
    else:
        print("Model not found")

    conf_matrix_path = os.path.join('logs', 'initial_confusion_matrices')
    os.makedirs(conf_matrix_path, exist_ok=True)

    all_pred_counts = []
    all_good_pred_counts = []
    all_mislabels_pred_counts = []
    all_confusion_matrices = []

    all_predictions = []
    all_actuals = []
    all_probs = []

    all_actuals_no_noise = []
    all_probs_no_noise = []
    all_predictions_no_noise = []

    for subset in subsets:
        dfs_subset = []

        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
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

        # Prepare Dataset
        dataset = CustomBinaryInsectDF(df_subset, transform = set_test_transform(), seed=config["exp_params"]["manual_seed"])
        
        loader = DataLoader(
            dataset, 
            batch_size=config["data_params"]["batch_size"], 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
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

        all_probs.extend(all_subset_probs)
        all_actuals.extend(all_subset_actuals)

        # Remove noisy samples from the predictions
        meas_noise_samples = df_subset[df_subset['measurement_noise'] == True].index
        mislabel_samples = df_subset[df_subset['mislabeled'] == True].index
        all_noisy_samples = meas_noise_samples.union(mislabel_samples) 

        all_subset_probs_no_noise = np.delete(np.array(all_subset_probs), all_noisy_samples, axis=0)
        all_subset_actuals_no_noise = np.delete(all_subset_actuals, all_noisy_samples)

        all_probs_no_noise.extend(all_subset_probs_no_noise)
        all_actuals_no_noise.extend(all_subset_actuals_no_noise)

        prob_distribution_path = os.path.join(config["logging_params"]["save_dir"], f'prob_distribution_{model_name}')
        os.makedirs(prob_distribution_path, exist_ok=True)
        # plot_probability_distribution(
        #     np.array(all_subset_probs), 
        #     np.array(all_subset_actuals), 
        #     start=0, end=1.0, step=0.1,
        #     filename_suffix=subset,
        #     save_path=prob_distribution_path,
        #     clean_dataset=clean_dataset,
        #     method=method
        #     )
        
        # plot_prediction_confidence_by_predicted_class(
        #     np.array(all_subset_probs), 
        #     start=0.5, end=1.0, step=0.1,
        #     filename_suffix=subset,
        #     save_path=prob_distribution_path,
        #     clean_dataset=clean_dataset,
        #     method=method
        #     )
        
        # do plot for no noisy samples
        # plot_probability_distribution(
        #     np.array(all_subset_probs_no_noise),
        #     np.array(all_subset_actuals_no_noise),
        #     start=0, end=1.0, step=0.1,
        #     filename_suffix=f'{subset}_no_noise',
        #     save_path=prob_distribution_path,
        #     clean_dataset=clean_dataset,
        #     method=method
        #     )
        # plot_prediction_confidence_by_predicted_class(
        #     np.array(all_subset_probs_no_noise),
        #     start=0.5, end=1.0, step=0.1,
        #     filename_suffix=f'{subset}_no_noise',
        #     save_path=prob_distribution_path,
        #     clean_dataset=clean_dataset,
        #     method=method
        #     )
        
        conf_matrix = build_extended_confusion_matrix(df_subset, all_subset_predictions, insect_classes)
        all_confusion_matrices.append(conf_matrix)
        plot_confusion_matrix(conf_matrix, subtitle=subset, path=conf_matrix_path)

        df_subset['pred_label'] = all_subset_predictions
        df_subset['pred_probas'] = all_subset_probs

        df_insect_0 = df_subset[df_subset['pred_label'] == 0]
        df_insect_1 = df_subset[df_subset['pred_label'] == 1]

        df_insect_0_directory_counts = df_insect_0.directory.value_counts()
        pred_good_samples_insect_0 = df_insect_0_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[0]}_good|{os.sep}{insect_classes[0]}_for_{insect_classes[1]}')].sum()
        pred_good_samples_orig_good_insect_0 = df_insect_0_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[0]}_good')].sum()
        pred_good_samples_orig_mislabeled_insect_0 = df_insect_0_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[0]}_for_{insect_classes[1]}')].sum()
        pred_mislabels_insect_0 = df_insect_0_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[1]}_good|{os.sep}{insect_classes[1]}_for_{insect_classes[0]}')].sum()
        pred_mislabels_orig_good_insect_0 = df_insect_0_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[1]}_good')].sum()
        pred_mislabels_orig_mislabeled_insect_0 = df_insect_0_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[1]}_for_{insect_classes[0]}')].sum()
        pred_measurement_noise_insect_0 = df_insect_0_directory_counts.loc[lambda x: x.index.str.contains('trash')].sum()

        df_insect_1_directory_counts = df_insect_1.directory.value_counts()
        pred_good_samples_insect_1 = df_insect_1_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[1]}_good|{os.sep}{insect_classes[1]}_for_{insect_classes[0]}')].sum()
        pred_good_samples_orig_good_insect_1 = df_insect_1_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[1]}_good')].sum()
        pred_good_samples_orig_mislabeled_insect_1 = df_insect_1_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[1]}_for_{insect_classes[0]}')].sum()
        pred_mislabels_insect_1 = df_insect_1_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[0]}_good|{os.sep}{insect_classes[0]}_for_{insect_classes[1]}')].sum()
        pred_mislabels_orig_good_insect_1 = df_insect_1_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[0]}_good')].sum()
        pred_mislabels_orig_mislabeled_insect_1 = df_insect_1_directory_counts.loc[lambda x: x.index.str.contains(f'{os.sep}{insect_classes[0]}_for_{insect_classes[1]}')].sum()
        pred_measurement_noise_insect_1 = df_insect_1_directory_counts.loc[lambda x: x.index.str.contains('trash')].sum()
        

        # Generate df with rows good samples, mislabels and measurement noise and columns wmv and c
        df_pred_counts = pd.DataFrame({
            'predicted good_samples': [pred_good_samples_insect_0, pred_good_samples_insect_1],
            'predicted mislabels': [pred_mislabels_insect_0, pred_mislabels_insect_1],
            'predicted measurement_noise': [pred_measurement_noise_insect_0, pred_measurement_noise_insect_1]
        }, index=[insect_classes[0], insect_classes[1]])

        print(df_pred_counts)
        all_pred_counts.append(df_pred_counts)

        # Generate df that shows proportion of good samples that were originally good and mislabeled
        df_pred_good = pd.DataFrame({
            'orig_good_samples': [pred_good_samples_orig_good_insect_0, pred_good_samples_orig_good_insect_1],
            'orig_mislabels': [pred_good_samples_orig_mislabeled_insect_0, pred_good_samples_orig_mislabeled_insect_1],
            'total': [pred_good_samples_insect_0, pred_good_samples_insect_1]
        }, index=[insect_classes[0], insect_classes[1]])
        all_good_pred_counts.append(df_pred_good)

        df_pred_mislabels = pd.DataFrame({
            'orig_good_samples': [pred_mislabels_orig_good_insect_0, pred_mislabels_orig_good_insect_1],
            'orig_mislabels': [pred_mislabels_orig_mislabeled_insect_0, pred_mislabels_orig_mislabeled_insect_1],
            'total': [pred_mislabels_insect_0, pred_mislabels_insect_1]
        }, index=[insect_classes[0], insect_classes[1]])
        all_mislabels_pred_counts.append(df_pred_mislabels)

        print('Predicted Good:\n',df_pred_good)
        print('Predicted Mislabels: \n' ,df_pred_mislabels)

        ########################## Probas counts
        df_insect_0_pred_good_orig_good_samples = df_insect_0[df_insect_0['directory'].str.contains(f'{os.sep}{insect_classes[0]}_good', case=False, na=False)]
        insect_0_pred_good_orig_good_probas = df_insect_0_pred_good_orig_good_samples.pred_probas.apply(lambda x: np.round(x, 1))
        insect_0_pred_good_probas_orig_good_counts = insect_0_pred_good_orig_good_probas.apply(lambda x: list(x)).value_counts()
        
        df_insect_0_pred_good_orig_mislabels_samples = df_insect_0[df_insect_0['directory'].str.contains(f'{os.sep}{insect_classes[0]}_for_{insect_classes[1]}', case=False, na=False)]
        insect_0_pred_good_orig_mislabels_probas = df_insect_0_pred_good_orig_mislabels_samples.pred_probas.apply(lambda x: np.round(x, 1))
        insect_0_pred_good_probas_orig_mislabels_counts = insect_0_pred_good_orig_mislabels_probas.apply(lambda x: list(x)).value_counts()


        df_insect_0_pred_mislabels_orig_good = df_insect_0[df_insect_0['directory'].str.contains(f'{os.sep}{insect_classes[1]}_good', case=False, na=False)]
        insect_0_pred_mislabels_probas_orig_good = df_insect_0_pred_mislabels_orig_good.pred_probas.apply(lambda x: np.round(x, 1))
        insect_0_pred_mislabels_probas_orig_good_counts = insect_0_pred_mislabels_probas_orig_good.apply(lambda x: list(x)).value_counts()

        df_insect_0_pred_mislabels_orig_mislabels = df_insect_0[df_insect_0['directory'].str.contains(f'{os.sep}{insect_classes[1]}_for_{insect_classes[0]}', case=False, na=False)]
        insect_0_pred_mislabels_probas_orig_mislabels = df_insect_0_pred_mislabels_orig_mislabels.pred_probas.apply(lambda x: np.round(x, 1))
        insect_0_pred_mislabels_probas_orig_mislabels_counts = insect_0_pred_mislabels_probas_orig_mislabels.apply(lambda x: list(x)).value_counts()

        df_insect_1_pred_good_orig_good_samples = df_insect_1[df_insect_1['directory'].str.contains(f'{os.sep}{insect_classes[1]}_good', case=False, na=False)]
        insect_1_pred_good_orig_good_probas = df_insect_1_pred_good_orig_good_samples.pred_probas.apply(lambda x: np.round(x, 1))
        insect_1_pred_good_probas_orig_good_counts = insect_1_pred_good_orig_good_probas.apply(lambda x: list(x)).value_counts()

        df_insect_1_pred_good_orig_mislabels = df_insect_1[df_insect_1['directory'].str.contains(f'{os.sep}{insect_classes[1]}_for_{insect_classes[0]}', case=False, na=False)]
        insect_1_pred_good_orig_mislabels_probas = df_insect_1_pred_good_orig_mislabels.pred_probas.apply(lambda x: np.round(x, 1))
        insect_1_pred_good_probas_orig_mislabels_counts = insect_1_pred_good_orig_mislabels_probas.apply(lambda x: list(x)).value_counts()

        df_insect_1_pred_mislabels_orig_good = df_insect_1[df_insect_1['directory'].str.contains(f'{os.sep}{insect_classes[0]}_good', case=False, na=False)]
        insect_1_pred_mislabels_orig_good_probas = df_insect_1_pred_mislabels_orig_good.pred_probas.apply(lambda x: np.round(x, 1))
        insect_1_pred_mislabels_orig_good_probas_counts = insect_1_pred_mislabels_orig_good_probas.apply(lambda x: list(x)).value_counts()

        df_insect_1_pred_mislabels_orig_mislabels= df_insect_1[df_insect_1['directory'].str.contains(f'{os.sep}{insect_classes[0]}_for_{insect_classes[1]}', case=False, na=False)]
        insect_1_pred_mislabels_orig_mislabels_probas = df_insect_1_pred_mislabels_orig_mislabels.pred_probas.apply(lambda x: np.round(x, 1))
        insect_1_pred_mislabels_orig_mislabels_probas_counts = insect_1_pred_mislabels_orig_mislabels_probas.apply(lambda x: list(x)).value_counts()

        print(f"{insect_classes[0]}_pred_good_probas_orig_good_counts: \n", insect_0_pred_good_probas_orig_good_counts)
        print(f"{insect_classes[0]}_pred_good_probas_orig_mislabel_counts: \n", insect_0_pred_good_probas_orig_mislabels_counts)
        print(f"{insect_classes[0]}_pred_mislabels_probas_orig_good_counts: \n", insect_0_pred_mislabels_probas_orig_mislabels_counts)
        print(f"{insect_classes[0]}_pred_mislabels_probas_orig_mislabels_counts: \n", insect_0_pred_mislabels_probas_orig_mislabels_counts)
        print(f"{insect_classes[1]}_pred_good_probas_orig_good_counts: \n", insect_1_pred_good_probas_orig_good_counts)
        print(f"{insect_classes[1]}_pred_good_probas_orig_mislabels_counts: \n", insect_1_pred_good_probas_orig_mislabels_counts)
        print(f"{insect_classes[1]}_pred_mislabels_orig_good_probas_counts: \n", insect_1_pred_mislabels_orig_good_probas_counts)
        print(f"{insect_classes[1]}_pred_mislabels_orig_mislabels_probas_counts: \n", insect_1_pred_mislabels_orig_mislabels_probas_counts)

        # update mislabeled column
        df_insect_0['mislabeled'] = df_insect_0["directory"].isin([f"data/{subset}/{insect_classes[0]}/{insect_classes[1]}_for_{insect_classes[0]}", f"data/{subset}/{insect_classes[1]}/{insect_classes[1]}_good"])

        df_insect_1['mislabeled'] = df_insect_1["directory"].isin([f"data/{subset}/{insect_classes[1]}/{insect_classes[0]}_for_{insect_classes[1]}", f"data/{subset}/{insect_classes[0]}/{insect_classes[0]}_good"])

        # add Best Alternative Class (BAC) column
        df_insect_0['bac'] = 1 - df_insect_0.label
        df_insect_1['bac'] = 1 - df_insect_1.label

        # add Probability of Alternative Class (PAC) column
        df_insect_0['pac'] = df_insect_0.apply(lambda row: row['pred_probas'][int(row['bac'])], axis=1)
        df_insect_1['pac'] = df_insect_1.apply(lambda row: row['pred_probas'][int(row['bac'])], axis=1)

        # update labels column to be outlier or inlier
        df_insect_0['noisy_label_classification'] = df_insect_0['label']
        df_insect_1['noisy_label_classification'] = df_insect_1['label']

        outlier_labels_insect_0 = ((df_insect_0['measurement_noise']==True) | (df_insect_0['mislabeled'])).astype(int)
        outlier_labels_insect_1 = ((df_insect_1['measurement_noise']==True) | (df_insect_1['mislabeled'])).astype(int)

        df_insect_0['label'] = outlier_labels_insect_0
        df_insect_1['label'] = outlier_labels_insect_1

        df_insect_0_path = os.path.join('data', f'df_{subset}_{swap_suffix}_{insect_classes[0]}.csv')
        df_insect_0.to_csv(df_insect_0_path)

        df_insect_1_path = os.path.join('data', f'df_{subset}_{swap_suffix}_{insect_classes[1]}.csv')
        df_insect_1.to_csv(df_insect_1_path)

        print('')
    
    df_all_pred_counts = all_pred_counts[0] + all_pred_counts[1]
    print(df_all_pred_counts)
    df_all_pred_counts.to_csv(os.path.join('logs', f'df_{model_name}_all_pred_counts_{model_name}_{raw_suffix}.csv'))

    df_all_good_pred_counts = all_good_pred_counts[0] + all_good_pred_counts[1]
    print(df_all_good_pred_counts)
    df_all_good_pred_counts.to_csv(os.path.join('logs', f'df_{model_name}_all_good_pred_counts_{model_name}_{raw_suffix}.csv'))

    df_all_mislabels_pred_counts = all_mislabels_pred_counts[0] + all_mislabels_pred_counts[1]
    print(df_all_mislabels_pred_counts)
    df_all_mislabels_pred_counts.to_csv(os.path.join('logs', f'df_{model_name}_all_mislabels_pred_counts_{model_name}_{raw_suffix}.csv'))
    
    # Get total confusion matrix
    conf_matrix_total = sum(all_confusion_matrices)
    plot_confusion_matrix(conf_matrix_total, subtitle='total', path=conf_matrix_path)

    # Get total probability distribution
    # plot_probability_distribution(
    #     np.array(all_probs), 
    #     np.array(all_actuals), 
    #     start=0, end=1.0, step=0.1,
    #     filename_suffix='train_val',
    #     save_path=prob_distribution_path,
    #     clean_dataset=clean_dataset,
    #     method=method
    #     )
    
    # plot_prediction_confidence_by_predicted_class(
    #     np.array(all_probs), 
    #     start=0.5, end=1.0, step=0.1,
    #     filename_suffix='train_val',
    #     save_path=prob_distribution_path,
    #     clean_dataset=clean_dataset,
    #     method=method
    #     )
    
    # # do plot for no noisy samples
    # plot_probability_distribution(
    #     np.array(all_probs_no_noise),
    #     np.array(all_actuals_no_noise),
    #     start=0, end=1.0, step=0.1,
    #     filename_suffix='train_val_no_noise',
    #     save_path=prob_distribution_path,
    #     clean_dataset=clean_dataset,
    #     method=method
    #     )
    # plot_prediction_confidence_by_predicted_class(
    #     np.array(all_probs_no_noise),
    #     start=0.5, end=1.0, step=0.1,
    #     filename_suffix='train_val_no_noise',
    #     save_path=prob_distribution_path,
    #     clean_dataset=clean_dataset,
    #     method=method
    #     )

    print('')
