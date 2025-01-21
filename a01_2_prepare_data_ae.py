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
    transform_train = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    insect_classes = ['wmv', 'c']
    clean_dataset = ''
    method = 'ae' 

    dfs_train_val = []

    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_{method}_{main_insect_class}{clean_dataset}.csv')
        df_train_i = pd.read_csv(df_train_path)
        # Correct labels for training a classifier instead of an outlier detector
        if main_insect_class == 'wmv':
            df_train_i.label = 0
            df_train_i = df_train_i.apply(update_mislabeled_flags, axis=1, args=('mislabeled', True, False))
        if main_insect_class == 'c':
            df_train_i.label = 1
            df_train_i = df_train_i.apply(update_mislabeled_flags, axis=1, args=('mislabeled', False, True))
        dfs_train_val.append(df_train_i)
    
    df_train_val = pd.concat(dfs_train_val, ignore_index=True)

    # Prepare Dataset
    dataset = CustomBinaryInsectDF(df_train_val, transform = transform, seed=config["exp_params"]["manual_seed"])
    
    loader = DataLoader(
        dataset, 
        batch_size=config["data_params"]["train_batch_size"], 
        shuffle=True,
        num_workers=config["data_params"]["num_workers"],
        pin_memory=pin_memory
    )


    ##########################
    ### RESNET-18 MODEL
    ##########################
    torch.manual_seed(RANDOM_SEED)
    model_name = 'mobilenetv3_small_100' 
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    save_path = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}.pth')
    save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}_best.pth')
    
    # Load the model
    if os.path.exists(save_path_best):
        model.load_state_dict(torch.load(save_path_best))
        print("Model correctly loaded")
    else:
        print("Model not found")

    ##########################
    ### INFERENCE
    ##########################
    with torch.set_grad_enabled(False): # save memory during inference
        all_predictions, all_actuals, all_probs = compute_predictions(model, loader, device=DEVICE)
        print(f"Prediction: {all_predictions[0]}\nActual: {all_actuals[0]}\nProbabilities: {all_probs[0]}")

    df_train_val['pred_label'] = all_predictions
    df_train_val['pred_probas'] = all_probs

    df_wmv = df_train_val[df_train_val['pred_label'] == 0]
    df_c = df_train_val[df_train_val['pred_label'] == 1]

    value_counts_good_samples_wmv = ((df_wmv['label'] == df_wmv['pred_label']) & 
    (df_wmv['measurement_noise'] == False) & 
    (df_wmv['mislabeled'] == False)).value_counts()
    print(f'value_counts_good_samples_wmv: \n', value_counts_good_samples_wmv)

    value_counts_good_samples_c = ((df_c['label'] == df_c['pred_label']) & 
    (df_c['measurement_noise'] == False) & 
    (df_c['mislabeled'] == False)).value_counts()
    print(f'value_counts_good_samples_c: \n', value_counts_good_samples_c)

    print(f'c mislabeled wmv: \n', df_c.mislabeled_wmv.value_counts())
    print(f'c mislabeled c: \n', df_c.mislabeled_c.value_counts())
    print(f'c measurement noise: \n', df_c.measurement_noise.value_counts())
    
    print(f'wmv mislabeled wmv: \n', df_wmv.mislabeled_wmv.value_counts()) 
    print(f'wmv mislabeled c: \n', df_wmv.mislabeled_c.value_counts()) 
    print(f'wmv measurement noise: \n', df_wmv.measurement_noise.value_counts())

    print('')
