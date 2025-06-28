import os
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

from mnist_dataset import CustomBinaryInsectDF
from a01_1_train_test_classifier import compute_accuracy

##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.0001 #0.0001
BATCH_SIZE = 512
NUM_EPOCHS = 100

# Architecture
NUM_CLASSES = 2

# Other
DEVICE = "cuda:0"
GRAYSCALE = True

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set manual seed for reproducibility
    torch.manual_seed(config["exp_params"]["manual_seed"])
    random.seed(config["exp_params"]["manual_seed"])
    np.random.seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0

    insect_classes = ['wmv', 'c']

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    df_test_path = os.path.join('data', f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    test_dataset = CustomBinaryInsectDF(df_test, transform = transform, seed=config["exp_params"]["manual_seed"])

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["data_params"]["train_batch_size"], 
        shuffle=False,
        num_workers=config["data_params"]["num_workers"],
        pin_memory=pin_memory
    )

    model_name = 'mobilenetv3_small_100' #'vgg16' #'efficientnet_lite0' #'tf_efficientnetv2_m.in21k_ft_in1k' #'resnet18'
    model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    save_path = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier_clean_ae.pth')
    save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier_clean_ae_best.pth')

    model.load_state_dict(torch.load(save_path_best, map_location=DEVICE))
    
    model.eval()

    test_accuracy, test_wmv_accuracy, test_c_accuracy = compute_accuracy(model, test_loader, device=DEVICE)
    print('Test accuracy: %.2f%%' % test_accuracy)
    print('Test WMV Accuracy: %.2f%%' % test_wmv_accuracy)
    print('Test C Accuracy: %.2f%%' % test_c_accuracy)


