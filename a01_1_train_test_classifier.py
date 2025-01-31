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

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # Downsample to 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Apply adaptive pooling
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x




if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    correct_pred_class_0, correct_pred_class_1 = 0, 0
    num_examples_class_0, num_examples_class_1 = 0, 0
    for i, (images, labels, _, _, _, _) in enumerate(data_loader):
            
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += labels.size(0)
        num_examples_class_0 += (labels == 0).sum()
        num_examples_class_1 += (labels == 1).sum()
        correct_pred += (predicted_labels == labels).sum()
        correct_pred_class_0 += ((predicted_labels == labels) & (labels == 0)).sum()
        correct_pred_class_1 += ((predicted_labels == labels) & (labels == 1)).sum()

    return correct_pred.float()/num_examples * 100, correct_pred_class_0.float()/num_examples_class_0 * 100, correct_pred_class_1.float()/num_examples_class_1 * 100


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
LEARNING_RATE = 0.0001 #0.0001
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
    method_datasets = ['raw', 'cleaning_benchmark']#, 'ae', 'adbench', 'cnn', 'adv_ae']
    retrain_models = True
    
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

    # transform_train = transforms.Compose([
    #     transforms.Resize((150, 150)),
    #     transforms.ToTensor(),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(degrees=(-15, 15)),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    results = []

    for method in method_datasets:
        if method=='cnn' or method=='adv_ae' or method=='ae' or method=='adbench':
            clean_dataset='_clean'
        elif method=='cleaning_benchmark' or method=='raw': 
            clean_dataset=''
        dfs_train_val = []
        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_train_path = os.path.join(
                'data', 
                'clean' if clean_dataset=='_clean' else '', 
                f'df_train_{method if method != "cleaning_benchmark" else "raw"}_{main_insect_class}{clean_dataset}.csv'
            )
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

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df_train.label), y=df_train.label.to_numpy())
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        # Prepare Dataset
        train_dataset = CustomBinaryInsectDF(df_train, transform = transform, seed=config["exp_params"]["manual_seed"])
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
        model_name = 'mobilenetv3_small_100' #'vgg16' #'efficientnet_lite0' #'tf_efficientnetv2_m.in21k_ft_in1k' #'resnet18'
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
        model.to(DEVICE)

        save_path = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}.pth')
        save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}_best.pth')

        if not os.path.exists(save_path_best) or retrain_models==True:
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["exp_params"]["scheduler_gamma"])

            ##########################
            ### TRAIN
            ##########################
            best_val_accuracy = 0
            best_lowest_val_class_accuracy = 0
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
                    loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)
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
                    train_accuracy, train_wmv_accuracy, train_c_accuracy = compute_accuracy(model, train_loader, device=DEVICE)
                    val_accuracy, val_wmv_accuracy, val_c_accuracy = compute_accuracy(model, val_loader, device=DEVICE)
                    train_accuracies.append(train_accuracy.item())
                    val_accuracies.append(val_accuracy.item())
                    test_accuracy, test_wmv_accuracy, test_c_accuracy = compute_accuracy(model, test_loader, device=DEVICE)


                    print('Epoch: %03d/%03d | Train: %.3f%% | Validation: %.3f%% | Test: %.3f%%' % (
                        epoch+1, NUM_EPOCHS, 
                        train_accuracy,
                        val_accuracy,
                        test_accuracy))
                    print('Train WMV Accuracy: %.2f%%' % train_wmv_accuracy)
                    print('Train C Accuracy: %.2f%%' % train_c_accuracy)
                    print('Validation WMV Accuracy: %.2f%%' % val_wmv_accuracy)
                    print('Validation C Accuracy: %.2f%%' % val_c_accuracy)
                    print('Test WMV Accuracy: %.2f%%' % test_wmv_accuracy)
                    print('Test C Accuracy: %.2f%%' % test_c_accuracy)

                    lowest_val_class_accuracy = min(val_wmv_accuracy, val_c_accuracy)

                    if best_val_accuracy < val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_lowest_val_class_accuracy = lowest_val_class_accuracy
                        torch.save(model.state_dict(), save_path_best)
                        print(f"New best model saved at {save_path_best} with Validation Accuracy: {best_val_accuracy:.3f}% and lowest class accuracy: {best_lowest_val_class_accuracy:.3f}%")
                    
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
        # Load the best model
        model.load_state_dict(torch.load(save_path_best))
        
        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            test_accuracy, test_wmv_accuracy, test_c_accuracy = compute_accuracy(model, test_loader, device=DEVICE)
            print('Test accuracy: %.2f%%' % test_accuracy)
            print('Test WMV Accuracy: %.2f%%' % test_wmv_accuracy)
            print('Test C Accuracy: %.2f%%' % test_c_accuracy)
        
        results.append({
            "clean_dataset": not(clean_dataset==''),
            "best_val_accuracy": best_val_accuracy.item() if retrain_models==True else None,
            "test_accuracy": test_accuracy.item(),
            "test_wmv_accuracy": test_wmv_accuracy.item(),
            "test_c_accuracy": test_c_accuracy.item(),
            'method': f'{method}'
        })
        df_results = pd.DataFrame(results)
        only_test_results = '_only_test' if retrain_models==False else ''
        results_file_name = f'df_{model_name}_results{only_test_results}.csv'
        df_results.to_csv(os.path.join(config["logging_params"]["save_dir"],results_file_name), index=False)
    print(df_results)
    print('')
