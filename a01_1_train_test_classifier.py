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

from mnist_dataset import CustomBinaryInsectDF


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (images, labels, _, _, _, _) in enumerate(data_loader):
            
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += labels.size(0)
        correct_pred += (predicted_labels == labels).sum()
    return correct_pred.float()/num_examples * 100


def augment_train_data(num_augmentations, df_train, transform, seed):
    # Define the original dataset
    train_dataset = CustomBinaryInsectDF(
        df_train, transform=transform, seed=seed
    )

    # Wrap the dataset with data augmentation applied multiple times
    augmented_datasets = [train_dataset]  # Start with the original dataset

    # Append augmented copies of the dataset
    for _ in range(num_augmentations):  # Add 4 additional copies with transformations
        augmented_datasets.append(CustomBinaryInsectDF(
            df_train, transform=transform, seed=seed
        ))

    # Concatenate datasets to effectively increase size
    augmented_train_dataset = ConcatDataset(augmented_datasets)

    return augmented_train_dataset


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
    method_datasets = ['raw', 'ae', 'adv_ae', 'cnn', 'cleaning_benchmark']
    
    # transform_train = transforms.Compose([
    #     transforms.Resize((150, 150)),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomAutocontrast(p=0.5),
    #     transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
    #     transforms.RandomRotation(degrees=(-5, 5)),
    #     transforms.RandomPosterize(bits=7, p=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

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
    results = []

    for method in method_datasets:
        if method=='cnn' or method=='adv_ae' or method=='ae':
            clean_dataset='_clean'
        elif method=='cleaning_benchmark' or method=='raw': 
            clean_dataset==''
        dfs_train_val = []
        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_train_path = os.path.join('data', f'df_train_{method if method != "cleaning_benchmark" else "raw"}_{main_insect_class}{clean_dataset}.csv')
            df_train_i = pd.read_csv(df_train_path)
            if method == 'cleaning_benchmark':
                df_train_i = df_train_i[df_train_i['label'] == 0]
            # Correct labels for training a classifier instead of an outlier detector
            if main_insect_class == 'wmv':
                df_train_i.label = 0
            if main_insect_class == 'c':
                df_train_i.label = 1
            dfs_train_val.append(df_train_i)
        
        df_train_val = pd.concat(dfs_train_val, ignore_index=True)
        df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=42, stratify=df_train_val['label'])

        df_test_path = os.path.join('data', f'df_test.csv')
        df_test = pd.read_csv(df_test_path)

        # Prepare Dataset
        train_dataset = CustomBinaryInsectDF(df_train, transform = transform_train, seed=config["exp_params"]["manual_seed"])
        # train_dataset = augment_train_data(4, df_train, transform = transform_train, seed=config["exp_params"]["manual_seed"])
        val_dataset = CustomBinaryInsectDF(df_val, transform = transform, seed=config["exp_params"]["manual_seed"])
        test_dataset = CustomBinaryInsectDF(df_test, transform = transform, seed=config["exp_params"]["manual_seed"])

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=True,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["data_params"]["train_batch_size"], 
            shuffle=False,
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
        model_name = 'mobilenetv3_small_100' #'efficientnet_lite0' #'tf_efficientnetv2_m.in21k_ft_in1k' #'resnet18'
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
        model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["exp_params"]["scheduler_gamma"])

        save_path = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}.pth')
        save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}_best.pth')
        
        ##########################
        ### TRAIN
        ##########################
        best_val_accuracy = 0
        train_accuracies = []
        val_accuracies = []
        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch_idx, (images, labels, _, _, _, _) in enumerate(train_loader):
                
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                    
                ### FORWARD AND BACK PROP
                logits = model(images)
                probas = F.softmax(logits, dim=1)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                
                loss.backward()
                
                ### UPDATE MODEL PARAMETERS
                optimizer.step()
                
                ### LOGGING
                if not batch_idx % 50:
                    if method == 'raw':
                        case = 'raw dirty'
                    elif method == 'cleaning_benchmark':
                        case = 'cleaning benchmark'
                    else:
                        case = f'{method} cleaned'
                    print (f'Case {case} dataset: '+'Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                        %(epoch+1, NUM_EPOCHS, batch_idx, 
                            len(train_loader), loss))        

            model.eval()
            with torch.set_grad_enabled(False): # save memory during inference
                train_accuracy = compute_accuracy(model, train_loader, device=DEVICE)
                val_accuracy = compute_accuracy(model, val_loader, device=DEVICE)
                train_accuracies.append(train_accuracy.item())
                val_accuracies.append(val_accuracy.item())

                print('Epoch: %03d/%03d | Train: %.3f%% | Validation: %.3f%%' % (
                    epoch+1, NUM_EPOCHS, 
                    train_accuracy,
                    val_accuracy))
                if best_val_accuracy < val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), save_path_best)
                    print(f"New best model saved at {save_path_best} with Validation Accuracy: {best_val_accuracy:.3f}%")
                
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            # Step the scheduler
            scheduler.step()
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
        print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
        torch.save(model.state_dict(), save_path)

        # Plot the training and validation accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Train Accuracy')
        plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        # plt.show()
        plt.savefig(f'zz_train_val_acc_cleaning_{clean_dataset}_{method}_100_epochs.png')

        ##########################
        ### TEST
        ##########################
        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            test_accuracy = compute_accuracy(model, test_loader, device=DEVICE)
            print('Test accuracy: %.2f%%' % test_accuracy)
        
        results.append({
            "clean_dataset": not(clean_dataset==''),
            "best_val_accuracy": best_val_accuracy.item(),
            "test_accuracy": test_accuracy.item(),
            'method': f'{method}'
        })
        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(config["logging_params"]["save_dir"],f'df_{model_name}_results.csv'), index=False)
    print(df_results)
    print('')
