import os

import numpy as np
import pandas as pd
import timm
import torch

from torch import nn, optim
from datasets.load_data import *
from datasets.transforms import set_test_transform, set_train_transform
from model.test import test_model
from utils import load_checkpoint


setting = 'fuji'
architecture = 'tf_efficientnetv2_m.in21k_ft_in1k'
batch_size = 32
batch_size_val = 32
num_workers = 0


def test(model_path, df_train, df_val, df_test, save_folder):
    # Get dataloader
    datasets = get_datasets(df_train, df_val, df_test, transform_train=set_train_transform(False), transform_test=set_test_transform())
    train_dataset, valid_dataset, test_dataset = datasets.values()

    dataloaders = load_data(datasets, batch_size, batch_size_val, num_workers)
    train_dataloader, valid_dataloader, test_dataloader = dataloaders.values()
    topclasses = df_train['txt_label'].unique().tolist()

    model = timm.create_model(architecture, pretrained=True, num_classes=len(topclasses))

    # Choosing whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')  # Number of gpus
    model = model.to('cuda', dtype=torch.float)

    class_sample_count = np.unique(df_train.label, return_counts=True)[1]
    weight = 1. / class_sample_count
    criterion = nn.CrossEntropyLoss(
        label_smoothing=.15, weight=torch.Tensor(weight).cuda())
    optimizer = optim.AdamW(model.parameters(), lr=.003, weight_decay=0.01)
    model, optimizer = load_checkpoint(model_path, model, optimizer)
    model = model.to('cuda', dtype=torch.float)

    bacc, cm, y_true, y_pred, info, insect_accuracies = test_model(model, test_dataloader, test_dataset)

    insect_accuracies.to_csv(os.path.join(save_folder, f'accuracy_values.npy'), index=False)

    return bacc, cm, y_true, y_pred, info, insect_accuracies
