import argparse
import os

import numpy as np
import pandas as pd
import timm
import torch

from torch import optim
from config.config import SAVE_DIR
from datasets.load_data import *
from datasets.transforms import set_test_transform, set_train_transform
from model.test import test_model
from utils import load_checkpoint


setting = 'fuji'
architecture = 'tf_efficientnetv2_m.in21k_ft_in1k'
batch_size = 32
batch_size_val = 32
num_workers = 0

def get_args():
    parser = argparse.ArgumentParser(description='Test a separate iteration')
    parser.add_argument('--iteration', type=int, default=0)
    args = parser.parse_args()
    return args

def test(model_path, df_train, df_val, df_test, save_folder):
    # Get dataloader
    datasets = get_datasets(df_train, df_val, df_test, transform_train=set_train_transform(False), transform_test=set_test_transform())
    _, _, test_dataset = datasets.values()

    dataloaders = load_data(datasets, batch_size, batch_size_val, num_workers)
    _, _, test_dataloader = dataloaders.values()
    topclasses = df_train['txt_label'].unique().tolist()

    model = timm.create_model(architecture, pretrained=True, num_classes=len(topclasses))

    # Choosing whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')  # Number of gpus
    model = model.to('cuda', dtype=torch.float)

    optimizer = optim.AdamW(model.parameters(), lr=.003, weight_decay=0.01)
    model, optimizer = load_checkpoint(model_path, model, optimizer)
    model = model.to('cuda', dtype=torch.float)

    bacc, cm, y_true, y_pred, info, insect_accuracies = test_model(model, test_dataloader, test_dataset)

    insect_accuracies.to_csv(os.path.join(save_folder, f'accuracy_values.npy'), index=False)

    return bacc, cm, y_true, y_pred, info, insect_accuracies

if __name__ == '__main__':
    args = get_args()
    i = args.iteration 
    saved_model = 'fuji_tf_efficientnetv2_m.in21k_ft_in1k_pretrained_imagenet_None_seed_x_augmented'
    iteration_folder = os.path.join('output', f'iteration_{i}')
    model_i_path = os.path.join(iteration_folder, f'{saved_model}_{i}_best.pth.tar')

    df_train = pd.read_parquet(f"{SAVE_DIR}/df_train_{setting}.parquet") # the train is not precise but it is not really used in test so whatever
    df_val = pd.read_parquet(f"{SAVE_DIR}/df_val_{setting}.parquet")
    df_test = pd.read_parquet(f"{SAVE_DIR}/df_test_{setting}.parquet")

    test(model_i_path, df_train, df_val, df_test, iteration_folder)