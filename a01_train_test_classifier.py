import os
import random
import time
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms

from mnist_dataset import CustomBinaryInsectDF


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
NUM_EPOCHS = 100

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
    insect_classes = ['wmv', 'c']
    dfs_train = []
    for i in range(len(insect_classes)):
        main_insect_class = insect_classes[i]
        mislabeled_insect_class = insect_classes[1 - i]

        df_train_path = os.path.join('data', f'df_train_ae_{main_insect_class}.csv')
        df_train_i = pd.read_csv(df_train_path)
        dfs_train.append(df_train_i)
    df_train = pd.concat(dfs_train, ignore_index=True)
    
    df_test_path = os.path.join('data', f'df_test.csv')
    df_test = pd.read_csv(df_test_path)

    # Prepare Dataset
    transform = transforms.Compose([
        transforms.Resize((config["data_params"]["patch_size"], config["data_params"]["patch_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CustomBinaryInsectDF(df_train, transform = transform, seed=config["exp_params"]["manual_seed"])
    test_dataset = CustomBinaryInsectDF(df_test, transform = transform, seed=config["exp_params"]["manual_seed"])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data_params"]["train_batch_size"], 
        shuffle=True,
        num_workers=config["data_params"]["num_workers"],
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["data_params"]["train_batch_size"], 
        shuffle=False,
        num_workers=config["data_params"]["num_workers"],
        pin_memory=pin_memory
    )

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
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            loss.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                    %(epoch+1, NUM_EPOCHS, batch_idx, 
                        len(train_loader), loss))        

        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%%' % (
                epoch+1, NUM_EPOCHS, 
                compute_accuracy(model, train_loader, device=DEVICE)))
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    torch.save(model.state_dict(), save_path)

    ##########################
    ### TEST
    ##########################
    with torch.set_grad_enabled(False): # save memory during inference
        test_accuracy = compute_accuracy(model, test_loader, device=DEVICE)
        print('Test accuracy: %.2f%%' % test_accuracy)
    
    print('')
