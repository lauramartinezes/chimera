import os
import time
import itertools

import numpy as np
import pandas as pd

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
from mnist_dataset import CustomMNISTDF
from mnist_matrix import plot_result_matrix


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets, _, _) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 512
NUM_EPOCHS = 5

# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 10

# Other
DEVICE = "cuda:0"
GRAYSCALE = True

##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
# train_dataset = datasets.MNIST(root='data', 
#                                train=True, 
#                                transform=transforms.ToTensor(),
#                                download=True)

# test_dataset = datasets.MNIST(root='data', 
#                               train=False, 
#                               transform=transforms.ToTensor())

train_df = pd.read_csv(os.path.join('archive', 'fashion-mnist_train.csv'))
test_df = pd.read_csv(os.path.join('archive', 'fashion-mnist_test.csv'))

# Split the DataFrame into train and test sets
# train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
# test_df = df.drop(train_df.index)  # Remaining 20% for testing

# Create the custom dataset instances
test_dataset = CustomMNISTDF(test_df, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

noise_percent_values = [0, 0.10, 0.25]
noise_combinations = list(itertools.product(noise_percent_values, repeat=2))

results_df = pd.DataFrame(columns=['transform_percent', 'wrong_label_percent', 'test_accuracy'])

for transform_percent, wrong_label_percent in noise_combinations:
    train_dataset = CustomMNISTDF(train_df, transform_percent=transform_percent, wrong_label_percent=wrong_label_percent, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)

    # Checking the dataset
    for images, labels, _, _ in train_loader:  
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

    ##########################
    ### RESNET-18 MODEL
    ##########################
    torch.manual_seed(RANDOM_SEED)
    model = timm.create_model('resnet18', pretrained=False, num_classes=NUM_CLASSES)

    # Modify the first layer of ResNet18 to accept 1-channel 28x28 input images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Change kernel to 3x3, stride to 1
    model.maxpool = nn.Identity()  # Remove the maxpool layer

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  

    ##########################
    ### TRAIN
    ##########################
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        model.train()
        for batch_idx, (features, targets, _, _) in enumerate(train_loader):
            
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
                
            ### FORWARD AND BACK PROP
            logits = model(features)
            probas = F.softmax(logits, dim=1)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 50:
                print(transform_percent, wrong_label_percent)
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                    %(epoch+1, NUM_EPOCHS, batch_idx, 
                        len(train_loader), cost))

            

        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%%' % (
                epoch+1, NUM_EPOCHS, 
                compute_accuracy(model, train_loader, device=DEVICE)))
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    ##########################
    ### TEST
    ##########################
    with torch.set_grad_enabled(False): # save memory during inference
        test_accuracy = compute_accuracy(model, test_loader, device=DEVICE)
        print('Test accuracy: %.2f%%' % test_accuracy)

    # Append the results to the DataFrame
    results_df = results_df.append({
        'transform_percent': transform_percent,
        'wrong_label_percent': wrong_label_percent,
        'test_accuracy': test_accuracy.item()
    }, ignore_index=True)
    
    print('')

print(results_df)
plot_result_matrix(results_df)