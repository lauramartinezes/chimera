import os
import random
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from mnist_dataset import CustomBinaryInsectDF
from a01_1_train_test_classifier import SimpleCNN, compute_accuracy


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

def compute_predictions(model, data_loader, device):
    all_predictions = []
    all_actuals = []
    all_probs = []

    for images, labels, real_label, measurement_noise, label_noise, outlier  in tqdm.tqdm(data_loader, desc="Computing predictions", total=len(data_loader)):
            
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        all_predictions.extend(predicted_labels.cpu().numpy())
        all_actuals.extend(labels.cpu().numpy())
        all_probs.extend(probas.cpu().numpy())

    return all_predictions, all_actuals, all_probs


# Step 1: Define categories using df-consistent terminology
def get_category_lists(insect_classes):
    natures = ["good", "mislabeled", "meas_noise"]
    true_categories = [f"{cls}_{nature}" for nature in natures for cls in insect_classes]
    predicted_categories = [f"{cls}_{nature}" for nature in natures for cls in insect_classes]
    return true_categories, predicted_categories


# Step 2: Determine the true category of a sample
def get_true_category(row, insect_classes):
    cls = insect_classes[row["label"]]
    if row["measurement_noise"]:
        return f"{cls}_meas_noise"
    elif row["mislabeled"]:
        return f"{cls}_mislabeled"
    else:
        return f"{cls}_good"


# Step 3: Determine the predicted category based on classifier output and true nature
def get_predicted_category(true_cat, pred_label, insect_classes):
    # Extract components from true_cat
    true_class, nature = true_cat.split("_", 1)
    pred_class = insect_classes[pred_label]

    # Define a lookup table of (nature, predicted == true?) → predicted nature
    transition = {
        "good":              {True: "good", False: "mislabeled"},
        "mislabeled":        {True: "mislabeled", False: "good"},
        "meas_noise": {True: "meas_noise", False: "meas_noise"}
    }

    is_correct = (pred_class == true_class)
    predicted_nature = transition.get(nature, {}).get(is_correct)

    if predicted_nature is None:
        return "UNKNOWN"

    return f"{pred_class}_{predicted_nature}"


# Step 4: Compute confusion matrix
def build_extended_confusion_matrix(df, all_predictions, insect_classes):
    true_cats, pred_cats = get_category_lists(insect_classes)
    conf_matrix = pd.DataFrame(0, index=true_cats, columns=pred_cats)

    for idx, row in df.iterrows():
        true_cat = get_true_category(row, insect_classes)
        pred_label = all_predictions[idx]
        pred_cat = get_predicted_category(true_cat, pred_label, insect_classes)
        if pred_cat != "UNKNOWN":
            conf_matrix.loc[true_cat, pred_cat] += 1
    
    return conf_matrix


# Step 5: Plotting
def plot_confusion_matrix(conf_matrix, subtitle=None):
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

    plt.xticks(rotation=20, ha="right")  # Make x-axis labels easier to read
    plt.yticks(rotation=0)               # Keep y-axis labels straight

    plt.title(f"Confusion Matrix ({subtitle.title()})" if subtitle else "Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.tight_layout()
    plt.show()


##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 512
NUM_EPOCHS = 20

# Architecture
NUM_CLASSES = 2

# Other
DEVICE = "cuda:0"
GRAYSCALE = True


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0

    ##########################
    ### BINARY CLASS INSECT DATASET
    ##########################
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    insect_classes = config["data_params"]["data_classes"]
    clean_dataset = ''
    method = 'ae' 
    subsets = ['train', 'val']

    ##########################
    ### RESNET-18 MODEL
    ##########################
    torch.manual_seed(RANDOM_SEED)
    model_name = 'resnet18' #'simplecnn' #'mobilenetv3_small_100' #overfittingcnn
    #model = SimpleCNN()
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier_raw_best.pth')#f'{model_name}_classifier{clean_dataset}_{method}_best.pth')
    
    # Load the model
    if os.path.exists(save_path_best):
        model.load_state_dict(torch.load(save_path_best))
        print("Model correctly loaded")
    else:
        print("Model not found")

    all_pred_counts = []
    all_good_pred_counts = []
    all_mislabels_pred_counts = []
    all_confusion_matrices = []

    for subset in subsets:
        dfs_subset = []

        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_subset_path = os.path.join('data', f'df_{subset}_raw_{main_insect_class}.csv')
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
        dataset = CustomBinaryInsectDF(df_subset, transform = transform, seed=config["exp_params"]["manual_seed"])
        
        loader = DataLoader(
            dataset, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )

        ##########################
        ### INFERENCE
        ##########################
        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            all_predictions, all_actuals, all_probs = compute_predictions(model, loader, device=DEVICE)
            accuracy = compute_accuracy(model, loader, device=DEVICE)
            print(f"Prediction: {all_predictions[0]}\nActual: {all_actuals[0]}\nProbabilities: {all_probs[0]}")
            print(f"Accuracy: {accuracy}")

        thresholded_predictions = np.array([
            1 - pred if max(probs) < 0.7 else pred
            for pred, probs in zip(all_predictions, all_probs)
        ])

        conf_matrix = build_extended_confusion_matrix(df_subset, all_predictions, insect_classes)
        all_confusion_matrices.append(conf_matrix)
        plot_confusion_matrix(conf_matrix, subtitle=subset)

        df_subset['pred_label'] = all_predictions
        df_subset['pred_probas'] = all_probs

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

        df_insect_0_path = os.path.join('data', f'df_{subset}_ae_{insect_classes[0]}.csv')
        df_insect_0.to_csv(df_insect_0_path)

        df_insect_1_path = os.path.join('data', f'df_{subset}_ae_{insect_classes[1]}.csv')
        df_insect_1.to_csv(df_insect_1_path)

        print('')
    
    df_all_pred_counts = all_pred_counts[0] + all_pred_counts[1]
    print(df_all_pred_counts)
    df_all_pred_counts.to_csv(os.path.join('logs', f'df_{model_name}_all_pred_counts_{model_name}.csv'))

    df_all_good_pred_counts = all_good_pred_counts[0] + all_good_pred_counts[1]
    print(df_all_good_pred_counts)
    df_all_good_pred_counts.to_csv(os.path.join('logs', f'df_{model_name}_all_good_pred_counts_{model_name}.csv'))

    df_all_mislabels_pred_counts = all_mislabels_pred_counts[0] + all_mislabels_pred_counts[1]
    print(df_all_mislabels_pred_counts)
    df_all_mislabels_pred_counts.to_csv(os.path.join('logs', f'df_{model_name}_all_mislabels_pred_counts_{model_name}.csv'))
    
    # Get total confusion matrix
    conf_matrix_total = sum(all_confusion_matrices)
    plot_confusion_matrix(conf_matrix_total, subtitle='Total')

    print('')
