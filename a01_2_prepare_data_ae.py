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

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from mnist_dataset import CustomBinaryInsectDF
from a01_1_train_test_classifier import SimpleCNN, compute_accuracy


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

def update_mislabeled_flags(row, mislabeled_col, wmv_value, c_value):
    if row[mislabeled_col]:
        row['mislabeled_wmv'] = wmv_value
        row['mislabeled_c'] = not wmv_value
    else:
        row['mislabeled_wmv'] = False
        row['mislabeled_c'] = False
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
    
    insect_classes = ['wmv', 'c']
    clean_dataset = ''
    method = 'ae' 
    subsets = ['train', 'val']

    ##########################
    ### RESNET-18 MODEL
    ##########################
    torch.manual_seed(RANDOM_SEED)
    model_name = 'vgg16' #'mobilenetv3_small_100' 
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

    for subset in subsets:
        dfs_subset = []

        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_subset_path = os.path.join('data', f'df_{subset}_raw_{main_insect_class}.csv')
            df_subset_i = pd.read_csv(df_subset_path)
            df_subset_i['noisy_outlier_label_original'] = df_subset_i.label
            # Correct labels for training a classifier instead of an outlier detector
            if main_insect_class == 'wmv':
                df_subset_i.label = 0
                df_subset_i['mislabeled_wmv'] = df_subset_i['mislabeled']
                df_subset_i['mislabeled_c'] = False
            if main_insect_class == 'c':
                df_subset_i.label = 1
                df_subset_i['mislabeled_c'] = df_subset_i['mislabeled']
                df_subset_i['mislabeled_wmv'] = False
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

        df_subset['pred_label'] = all_predictions
        df_subset['pred_probas'] = all_probs

        df_wmv = df_subset[df_subset['pred_label'] == 0]
        df_c = df_subset[df_subset['pred_label'] == 1]

        df_wmv_directory_counts = df_wmv.directory.value_counts()
        pred_good_samples_wmv = df_wmv_directory_counts.loc[lambda x: x.index.str.contains('wmv_good|wmv_for_c')].sum()
        pred_measurement_noise_wmv = df_wmv_directory_counts.loc[lambda x: x.index.str.contains('trash')].sum()
        pred_mislabels_wmv = df_wmv_directory_counts.loc[lambda x: x.index.str.contains('c_good|c_for_wmv')].sum()

        df_c_directory_counts = df_c.directory.value_counts()
        pred_good_samples_c = df_c_directory_counts.loc[lambda x: x.index.str.contains('c_good|c_for_wmv')].sum()
        pred_measurement_noise_c = df_c_directory_counts.loc[lambda x: x.index.str.contains('trash')].sum()
        pred_mislabels_c = df_c_directory_counts.loc[lambda x: x.index.str.contains('wmv_good|wmv_for_c')].sum()
    

        # Generate df with rows good samples, mislabels and measurement noise and columns wmv and c
        df_pred_counts = pd.DataFrame({
            'good_samples': [pred_good_samples_wmv, pred_good_samples_c],
            'mislabels': [pred_mislabels_wmv, pred_mislabels_c],
            'measurement_noise': [pred_measurement_noise_wmv, pred_measurement_noise_c]
        }, index=['wmv', 'c'])

        print(df_pred_counts)

        # This is useless code that counts the mislabels and good samples that have been transferred to the other class
        df_new_mislabels_wmv = df_wmv[((df_wmv['label'] != df_wmv['pred_label']) & 
        (df_wmv['measurement_noise'] == False) & 
        (df_wmv['mislabeled'] == False))] # here is only c_good, c_for_wmv is missing
        rounded_new_mislabels_wmv_pred_probas = df_new_mislabels_wmv.pred_probas.apply(lambda x: np.round(x, 1))
        new_mislabels_wmv_counts_probas = rounded_new_mislabels_wmv_pred_probas.apply(lambda x: list(x)).value_counts()

        df_new_good_samples_wmv = df_wmv[((df_wmv['label'] == df_wmv['pred_label']) &
        (df_wmv['measurement_noise'] == False) & 
        (df_wmv['mislabeled'] == False))] # here is only wmv_good, wmv_for_c is missing
        rounded_new_good_samples_wmv_pred_probas = df_new_good_samples_wmv.pred_probas.apply(lambda x: np.round(x, 1))
        new_good_samples_wmv_counts_probas = rounded_new_good_samples_wmv_pred_probas.apply(lambda x: list(x)).value_counts()

        df_new_mislabels_c = df_c[((df_c['label'] != df_c['pred_label']) &
        (df_c['measurement_noise'] == False) &
        (df_c['mislabeled'] == False))] # here is only wmv_good, wmv_for_c is missing
        rounded_new_mislabels_c_pred_probas = df_new_mislabels_c.pred_probas.apply(lambda x: np.round(x, 1))
        new_mislabels_c_counts_probas = rounded_new_mislabels_c_pred_probas.apply(lambda x: list(x)).value_counts()

        df_new_good_samples_c = df_c[((df_c['label'] == df_c['pred_label']) &
        (df_c['measurement_noise'] == False) &
        (df_c['mislabeled'] == False))] # here is only c_good, c_for_wmv is missing
        rounded_new_good_samples_c_pred_probas = df_new_good_samples_c.pred_probas.apply(lambda x: np.round(x, 1))
        new_good_samples_c_counts_probas = rounded_new_good_samples_c_pred_probas.apply(lambda x: list(x)).value_counts()

        # print(f'new_mislabels_wmv_counts_probas: \n', new_mislabels_wmv_counts_probas)
        # print(f'new_good_samples_wmv_counts_probas: \n', new_good_samples_wmv_counts_probas)
        # print(f'new_mislabels_c_counts_probas: \n', new_mislabels_c_counts_probas)
        # print(f'new_good_samples_c_counts_probas: \n', new_good_samples_c_counts_probas)

        # update mislabeled column
        df_wmv['mislabeled'] = df_wmv["directory"].isin([f"data/{subset}/wmv/c_for_wmv", f"data/{subset}/c/c_good"])

        df_c['mislabeled'] = df_c["directory"].isin([f"data/{subset}/c/wmv_for_c", f"data/{subset}/wmv/wmv_good"])

        # add Best Alternative Class (BAC) column
        df_wmv['bac'] = 1 - df_wmv.label
        df_c['bac'] = 1 - df_c.label

        # add Probability of Alternative Class (PAC) column
        df_wmv['pac'] = df_wmv.apply(lambda row: row['pred_probas'][int(row['bac'])], axis=1)
        df_c['pac'] = df_c.apply(lambda row: row['pred_probas'][int(row['bac'])], axis=1)

        # update labels column to be outlier or inlier
        df_wmv['noisy_label_classification'] = df_wmv['label']
        df_c['noisy_label_classification'] = df_c['label']

        outlier_labels_wmv = ((df_wmv['measurement_noise']==True) | (df_wmv['mislabeled'])).astype(int)
        outlier_labels_c = ((df_c['measurement_noise']==True) | (df_c['mislabeled'])).astype(int)

        df_wmv['label'] = outlier_labels_wmv
        df_c['label'] = outlier_labels_c

        # df_wmv_path = os.path.join('data', f'df_{subset}_ae_wmv.csv')
        # df_wmv.to_csv(df_wmv_path)

        # df_c_path = os.path.join('data', f'df_{subset}_ae_c.csv')
        # df_c.to_csv(df_c_path)

        print('')
