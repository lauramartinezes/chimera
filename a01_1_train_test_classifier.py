import os
import random
import shutil
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

# class SimpleCNN(nn.Module):
#     def __init__(self, n_classes=2):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.AdaptiveAvgPool2d((7, 7))  # Downsample to 7x7
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, n_classes)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)  # Apply adaptive pooling
#         x = torch.flatten(x, 1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for stability
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling (halves the size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Ensures fixed size
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.4)  # Slightly higher dropout for regularization

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv -> BN -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.adaptive_pool(x)  # Adaptive pooling to ensure 7x7 output
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  # Output logits for n_classes
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

#73 vs 82
# class OverfittingCNN(nn.Module):
#     def __init__(self, n_classes=2):
#         super(OverfittingCNN, self).__init__()

#         # Use stride=2 instead of pooling to reduce feature map size
#         self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)  # 150x150 -> 75x75
#         self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 75x75 -> 38x38
#         self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 38x38 -> 19x19
#         self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1) # 19x19 -> 10x10

#         # Reduce FC layer size by processing a smaller feature map
#         self.fc1 = nn.Linear(1024 * 10 * 10, 1024)  
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc3 = nn.Linear(256, n_classes)
        
#         self.leaky_relu = nn.LeakyReLU(0.01)

#         self._init_weights()  # Apply extreme weight initialization

#     def forward(self, x):
#         x = self.leaky_relu(self.conv1(x))
#         x = self.leaky_relu(self.conv2(x))
#         x = self.leaky_relu(self.conv3(x))  
#         x = self.leaky_relu(self.conv4(x))  
#         x = torch.flatten(x, 1)  
#         x = self.leaky_relu(self.fc1(x))
#         x = self.leaky_relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#     def _init_weights(self):
#         """Applies extreme weight initialization to make the model unstable."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=2.0)  # Large std makes it overfit
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

class OverfittingCNN(nn.Module):
    def __init__(self, n_classes=2):
        super(OverfittingCNN, self).__init__()

        # Reduce feature map size gradually
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # Stride to downsample

        # Global average pooling to guarantee fixed size before FC layers
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output shape = (batch, 512, 1, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 512)  
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_classes)
        
        self.leaky_relu = nn.LeakyReLU(0.01)

        self._init_weights()  # Extreme weight initialization

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))  
        x = self.leaky_relu(self.conv4(x))  
        x = self.leaky_relu(self.conv5(x))  

        x = self.global_pool(x)  # Ensure the shape is (batch, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (batch, 512)

        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _init_weights(self):
        """Applies extreme weight initialization to make the model unstable."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=2.0)  # Large std makes it overfit
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
# class SimpleCNN(nn.Module):
#     def __init__(self, n_classes=2):
#         super(SimpleCNN, self).__init__()
#         self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Single conv layer
#         self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Smaller adaptive pooling
#         self.fc = nn.Linear(16 * 4 * 4, n_classes)  # Fully connected layer

#     def forward(self, x):
#         x = F.relu(self.conv(x))  # Apply single conv layer with ReLU
#         x = self.pool(x)  # Downsample to (4x4)
#         x = torch.flatten(x, 1)  # Flatten for FC layer
#         x = self.fc(x)  # Output logits
#         return x


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    correct_pred_class_0, correct_pred_class_1 = 0, 0
    num_examples_class_0, num_examples_class_1 = 0, 0
    num_examples_measurement_noise_class_0, num_examples_measurement_noise_class_1 = 0, 0
    num_examples_label_noise_class_0, num_examples_label_noise_class_1 = 0, 0
    num_examples_good_class_0, num_examples_good_class_1 = 0, 0
    correct_pred_measurement_noise_class_0, correct_pred_measurement_noise_class_1 = 0, 0
    correct_pred_label_noise_class_0, correct_pred_label_noise_class_1 = 0, 0
    correct_pred_good_class_0, correct_pred_good_class_1 = 0, 0

    for i, (images, labels, _, measurement_noise, label_noise, _) in enumerate(data_loader):
            
        images = images.to(device)
        labels = labels.to(device)
        measurement_noise = measurement_noise.to(device)
        label_noise = label_noise.to(device)

        logits = model(images)
        probas = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probas, 1)

        num_examples += labels.size(0)
        num_examples_class_0 += (labels == 0).sum()
        num_examples_class_1 += (labels == 1).sum()
        num_examples_measurement_noise_class_0 += ((measurement_noise == True) & (labels == 0)).sum()
        num_examples_measurement_noise_class_1 += ((measurement_noise == True) & (labels == 1)).sum()
        num_examples_label_noise_class_0 += ((label_noise == True) & (labels == 0)).sum()
        num_examples_label_noise_class_1 += ((label_noise == True) & (labels == 1)).sum()
        num_examples_good_class_0 += ((label_noise == False) & (measurement_noise == False) & (labels == 0)).sum()
        num_examples_good_class_1 += ((label_noise == False) & (measurement_noise == False) & (labels == 1)).sum()

        correct_pred += (predicted_labels == labels).sum()
        correct_pred_class_0 += ((predicted_labels == labels) & (labels == 0)).sum()
        correct_pred_class_1 += ((predicted_labels == labels) & (labels == 1)).sum()
        correct_pred_measurement_noise_class_0 += ((predicted_labels == labels) & (measurement_noise == True) & (labels == 0)).sum()
        correct_pred_measurement_noise_class_1 += ((predicted_labels == labels) & (measurement_noise == True) & (labels == 1)).sum()
        correct_pred_label_noise_class_0 += ((predicted_labels == labels) & (label_noise == True) & (labels == 0)).sum()
        correct_pred_label_noise_class_1 += ((predicted_labels == labels) & (label_noise == True) & (labels == 1)).sum()
        correct_pred_good_class_0 += ((predicted_labels == labels) & (measurement_noise == False) & (label_noise == False) & (labels == 0)).sum()
        correct_pred_good_class_1 += ((predicted_labels == labels) & (measurement_noise == False) & (label_noise == False) & (labels == 1)).sum()
    
    correct_pred_percent = correct_pred.float() / num_examples * 100
    correct_pred_class_0_percent = correct_pred_class_0.float() / num_examples_class_0 * 100
    correct_pred_class_1_percent = correct_pred_class_1.float() / num_examples_class_1 * 100
    correct_pred_measurement_noise_class_0_percent = correct_pred_measurement_noise_class_0.float() / num_examples_measurement_noise_class_0 * 100
    correct_pred_measurement_noise_class_1_percent = correct_pred_measurement_noise_class_1.float() / num_examples_measurement_noise_class_1 * 100
    correct_pred_label_noise_class_0_percent = correct_pred_label_noise_class_0.float() / num_examples_label_noise_class_0 * 100
    correct_pred_label_noise_class_1_percent = correct_pred_label_noise_class_1.float() / num_examples_label_noise_class_1 * 100
    correct_pred_good_class_0_percent = correct_pred_good_class_0.float() / num_examples_good_class_0 * 100
    correct_pred_good_class_1_percent = correct_pred_good_class_1.float() / num_examples_good_class_1 * 100

    return correct_pred_percent, correct_pred_class_0_percent, correct_pred_class_1_percent, correct_pred_measurement_noise_class_0_percent, correct_pred_measurement_noise_class_1_percent, correct_pred_label_noise_class_0_percent, correct_pred_label_noise_class_1_percent, correct_pred_good_class_0_percent, correct_pred_good_class_1_percent

def compute_loss(model, data_loader, device):
    epoch_loss = 0
    for batch_idx, (images, labels, _, _, _, _) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)  # Compute validation loss
        epoch_loss += loss.item()

    epoch_loss /= (batch_idx + 1)  # Compute average validation loss
    return epoch_loss

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


def split_data(df_train_val, test_size=0.2):
    """
    Splits the dataset into train and validation sets, ensuring no location overlap and class balance.
    
    Parameters:
    - df_train_val: pandas DataFrame containing the dataset with 'location' and 'label' columns.
    - test_size: Proportion of the dataset to include in the validation set (default is 0.2).
    
    Returns:
    - train_data: pandas DataFrame for the training set.
    - val_data: pandas DataFrame for the validation set.
    """
    # Step 1: Get unique locations
    locations = df_train_val['location'].unique()

    # Step 2: Split the locations into train/val, stratifying by the most frequent label per location
    train_locations, val_locations = train_test_split(
        locations, 
        test_size=test_size, 
        stratify=df_train_val.groupby('location')['label'].apply(lambda x: x.mode()[0])
    )

    # Step 3: Filter the original dataset based on the location split
    train_data = df_train_val[df_train_val['location'].isin(train_locations)]
    val_data = df_train_val[df_train_val['location'].isin(val_locations)]

    return train_data, val_data


##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.0001 #0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 10 #1

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pin_memory = len(config['trainer_params']['gpus']) != 0

    ##########################
    ### BINARY CLASS INSECT DATASET
    ##########################
    insect_classes = config["data_params"]["data_classes"] #['wmv', 'm']
    method_datasets = ['raw', 'cleaning_benchmark', 'ae', 'adv_ae', 'adbench']#['adv_ae', 'adbench', 'raw', 'cleaning_benchmark']#'adv_ae', 'adbench', 'raw', 'cleaning_benchmark']#, 'ae', 'adbench', 'cnn', 'adv_ae']
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

        if 'ae' in method:
            od_method = '_DBSCAN'
        elif method == 'adbench':
            od_method = '_OCSVM'
        else:   
            od_method = ''
        dfs_train_ = []
        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_train_path = os.path.join(
                'data', 
                'clean' if clean_dataset=='_clean' else '', 
                f'df_train_{method if method != "cleaning_benchmark" else "raw"}_{main_insect_class}{od_method}{clean_dataset}.csv'
            )
            df_train_i = pd.read_csv(df_train_path)
            if method == 'cleaning_benchmark':
                df_train_i = df_train_i[df_train_i['label'] == 0]
            # Correct labels for training a classifier instead of an outlier detector
            df_train_i.label = i
            dfs_train_.append(df_train_i)
        
        df_train = pd.concat(dfs_train_, ignore_index=True)

        dfs_val = []
        for i in range(len(insect_classes)):
            main_insect_class = insect_classes[i]
            mislabeled_insect_class = insect_classes[1 - i]

            df_val_path = os.path.join(
                'data', 
                'clean' if clean_dataset=='_clean' else '', 
                f'df_val_{method if method != "cleaning_benchmark" else "raw"}_{main_insect_class}{od_method}{clean_dataset}.csv'
                #f'df_val_vgg16_{main_insect_class}_{method if method != "cleaning_benchmark" else "raw"}{clean_dataset}.csv'
            )
            df_val_i = pd.read_csv(df_val_path)
            if method == 'cleaning_benchmark':
                df_val_i = df_val_i[df_val_i['label'] == 0]
            # Correct labels for training a classifier instead of an outlier detector
            df_val_i.label = i
            dfs_val.append(df_val_i)
        
        df_val = pd.concat(dfs_val, ignore_index=True)


        # df_train_val['location'] = df_train_val['filepath'].apply(lambda x: os.path.basename(x).split('_')[1])
        # df_train_val['plate'] = df_train_val['filepath'].apply(lambda x: '_'.join(os.path.basename(x).split('.')[0].split('_')[:-1]))
        # df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=42, stratify=df_train_val['label'])
        
        # df_val_good = df_val[(df_val.measurement_noise==False) & (df_val.mislabeled==False)]
        # df_train, df_val = split_data(df_train_val, test_size=0.2)
        
        # copy images into a train and val folders
        # os.makedirs(os.path.join('data', 'df_train'), exist_ok=True)
        # os.makedirs(os.path.join('data', 'df_val'), exist_ok=True)
        # df_train.filepath.apply(lambda x: os.makedirs(os.path.dirname(x).replace('train', 'df_train'), exist_ok=True))
        # df_val.filepath.apply(lambda x: os.makedirs(os.path.dirname(x).replace('train', 'df_val'), exist_ok=True))
        # df_train.filepath.apply(lambda x: shutil.copy(x, os.path.dirname(x).replace('train', 'df_train')))
        # df_val.filepath.apply(lambda x: shutil.copy(x, os.path.dirname(x).replace('train', 'df_val')))

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
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=config["data_params"]["num_workers"],
            pin_memory=pin_memory
        )

        ##########################
        ### RESNET-18 MODEL
        ##########################
        torch.manual_seed(RANDOM_SEED) # Apparently at some point I decided to change the seed to RANDOM_SEED, this is the one that matters
        model_name = 'resnet18' #'vgg16' #'efficientnet_lite0' #'tf_efficientnetv2_m.in21k_ft_in1k' #'resnet18'
        # model = OverfittingCNN(n_classes=len(insect_classes))
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
        model.to(DEVICE)

        save_path = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}.pth')
        save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}_best.pth')

        if not os.path.exists(save_path_best) or retrain_models==True:
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)  
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["exp_params"]["scheduler_gamma"])

            ##########################
            ### TRAIN
            ##########################
            best_val_accuracy = 0
            best_val_loss = 100
            best_lowest_val_class_accuracy = 0
            train_accuracies = []
            val_accuracies = []
            test_accuracies = []
            train_losses = []
            val_losses = []
            test_losses = []
            start_time = time.time()
            for epoch in range(NUM_EPOCHS):
                model.train()
                epoch_loss = 0
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
                    epoch_loss += loss.item()
                epoch_loss /= batch_idx + 1
                train_losses.append(epoch_loss)


                model.eval()
                with torch.set_grad_enabled(False): # save memory during inference
                    val_epoch_loss = compute_loss(model, val_loader, DEVICE)
                    test_epoch_loss = compute_loss(model, test_loader, DEVICE)
                    val_losses.append(val_epoch_loss)
                    test_losses.append(test_epoch_loss)
                                
                    (train_accuracy, train_insect_0_accuracy, train_insect_1_accuracy, 
                     train_insect_0_meas_noise_accuracy, train_insect_1_meas_noise_accuracy, 
                     train_insect_0_milabel_accuracy, train_insect_1_milabel_accuracy, 
                     train_insect_0_good_accuracy, train_insect_1_good_accuracy) = compute_accuracy(model, train_loader, device=DEVICE)
                    (val_accuracy, val_insect_0_accuracy, val_insect_1_accuracy, 
                     val_insect_0_meas_noise_accuracy, val_insect_1_meas_noise_accuracy, 
                     val_insect_0_milabel_accuracy, val_insect_1_milabel_accuracy,
                     val_insect_0_good_accuracy, val_insect_1_good_accuracy) = compute_accuracy(model, val_loader, device=DEVICE)
                    (test_accuracy, test_insect_0_accuracy, test_insect_1_accuracy, 
                     test_insect_0_meas_noise_accuracy, test_insect_1_meas_noise_accuracy, 
                     test_insect_0_milabel_accuracy, test_insect_1_milabel_accuracy,
                     test_insect_0_good_accuracy, test_insect_1_good_accuracy) = compute_accuracy(model, test_loader, device=DEVICE)
                    train_accuracies.append(train_accuracy.item())
                    val_accuracies.append(val_accuracy.item())
                    test_accuracies.append(test_accuracy.item())


                    print('Epoch: %03d/%03d | Train: %.3f%% | Validation: %.3f%% | Test: %.3f%%' % (
                        epoch+1, NUM_EPOCHS, 
                        train_accuracy,
                        val_accuracy,
                        test_accuracy))
                    print(f'Train {insect_classes[0]} Accuracy: %.2f%%' % train_insect_0_accuracy)
                    print(f'Train {insect_classes[1]} Accuracy: %.2f%%' % train_insect_1_accuracy)
                    print(f'Train {insect_classes[0]} Meas. Noise Accuracy: %.2f%%' % train_insect_0_meas_noise_accuracy)
                    print(f'Train {insect_classes[1]} Meas. Noise Accuracy: %.2f%%' % train_insect_1_meas_noise_accuracy)
                    print(f'Train {insect_classes[0]} Mislabel Accuracy: %.2f%%' % train_insect_0_milabel_accuracy)
                    print(f'Train {insect_classes[1]} Mislabel Accuracy: %.2f%%' % train_insect_1_milabel_accuracy)
                    print(f'Train {insect_classes[0]} Good Accuracy: %.2f%%' % train_insect_0_good_accuracy)
                    print(f'Train {insect_classes[1]} Good Accuracy: %.2f%%' % train_insect_1_good_accuracy)

                    print(f'Validation {insect_classes[0]} Accuracy: %.2f%%' % val_insect_0_accuracy)
                    print(f'Validation {insect_classes[1]} Accuracy: %.2f%%' % val_insect_1_accuracy)
                    print(f'Validation {insect_classes[0]} Meas. Noise Accuracy: %.2f%%' % val_insect_0_meas_noise_accuracy)
                    print(f'Validation {insect_classes[1]} Meas. Noise Accuracy: %.2f%%' % val_insect_1_meas_noise_accuracy)
                    print(f'Validation {insect_classes[0]} Mislabel Accuracy: %.2f%%' % val_insect_0_milabel_accuracy)
                    print(f'Validation {insect_classes[1]} Mislabel Accuracy: %.2f%%' % val_insect_1_milabel_accuracy)
                    print(f'Validation {insect_classes[0]} Good Accuracy: %.2f%%' % val_insect_0_good_accuracy)
                    print(f'Validation {insect_classes[1]} Good Accuracy: %.2f%%' % val_insect_1_good_accuracy)

                    print(f'Test {insect_classes[0]} Accuracy: %.2f%%' % test_insect_0_accuracy)
                    print(f'Test {insect_classes[1]} Accuracy: %.2f%%' % test_insect_1_accuracy)
                    # print(f'Test {insect_classes[0]} Meas. Noise Accuracy: %.2f%%' % test_insect_0_meas_noise_accuracy)
                    # print(f'Test {insect_classes[1]} Meas. Noise Accuracy: %.2f%%' % test_insect_1_meas_noise_accuracy)
                    # print(f'Test {insect_classes[0]} Mislabel Accuracy: %.2f%%' % test_insect_0_milabel_accuracy)
                    # print(f'Test {insect_classes[1]} Mislabel Accuracy: %.2f%%' % test_insect_1_milabel_accuracy)
                    # print(f'Test {insect_classes[0]} Good Accuracy: %.2f%%' % test_insect_0_good_accuracy)
                    # print(f'Test {insect_classes[1]} Good Accuracy: %.2f%%' % test_insect_1_good_accuracy)

                    lowest_val_class_accuracy = min(val_insect_0_accuracy, val_insect_1_accuracy)

                        # Check if the validation accuracy improved
                    valid_accuracy_improved = val_accuracy > best_val_accuracy

                    # Check if the validation loss improved or if it is in the 10% percentile of the validation losses and the validation accuracy improved
                    model_improved = (val_epoch_loss < best_val_loss) or (val_epoch_loss < np.percentile(val_losses, 20) and valid_accuracy_improved)

                    if best_val_accuracy < val_accuracy:
                        best_val_loss = val_epoch_loss
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
            plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label=f'Train Accuracy')
            plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label=f'Validation Accuracy')
            plt.plot(range(1, NUM_EPOCHS + 1), test_accuracies, label=f'Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Training and Validation Accuracy')
            plt.legend()
            # plt.show()
            train_curves_path = os.path.join(config["logging_params"]["save_dir"], f'training_curves')
            os.makedirs(train_curves_path, exist_ok=True)
            plt.savefig(os.path.join(train_curves_path, f'train_val_test_acc_{clean_dataset}_{method}.png'))
            plt.savefig(os.path.join(train_curves_path, f'train_val_test_acc_{clean_dataset}_{method}.svg'))

            # Plot the training and validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label=f'Train Loss')
            plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label=f'Validation Loss')
            plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label=f'Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss')
            plt.legend()
            # plt.show()
            plt.savefig(os.path.join(train_curves_path, f'train_val_test_loss_{clean_dataset}_{method}.png'))
            plt.savefig(os.path.join(train_curves_path, f'train_val_test_loss_{clean_dataset}_{method}.svg'))
        else:
            model.load_state_dict(torch.load(save_path_best))
            model.eval()
            with torch.set_grad_enabled(False): # save memory during inference
                (best_val_accuracy, _, _, _, _, _, _, _, _) = compute_accuracy(model, val_loader, device=DEVICE)
                print('Best Validation accuracy: %.2f%%' % best_val_accuracy)

        ##########################
        ### TEST
        ##########################
        # Load the best model
        model.load_state_dict(torch.load(save_path_best))
        
        model.eval()
        with torch.set_grad_enabled(False): # save memory during inference
            (test_accuracy, test_insect_0_accuracy, test_insect_1_accuracy, 
            test_insect_0_meas_noise_accuracy, test_insect_1_meas_noise_accuracy, 
            test_insect_0_milabel_accuracy, test_insect_1_milabel_accuracy,
            test_insect_0_good_accuracy, test_insect_1_good_accuracy) = compute_accuracy(model, test_loader, device=DEVICE)
            print(f'Test accuracy: %.2f%%' % test_accuracy)
            print(f'Test {insect_classes[0]} Accuracy: %.2f%%' % test_insect_0_accuracy)
            print(f'Test {insect_classes[1]} Accuracy: %.2f%%' % test_insect_1_accuracy)
        
        results.append({
            "clean_dataset": not(clean_dataset==''),
            "best_val_accuracy": best_val_accuracy.item(),# if retrain_models==True else None,
            "test_accuracy": test_accuracy.item(),
            f"test_{insect_classes[0]}accuracy": test_insect_0_accuracy.item(),
            f"test_{insect_classes[1]}_accuracy": test_insect_1_accuracy.item(),
            'method': f'{method}'
        })
        df_results = pd.DataFrame(results)
        only_test_results = '_only_test' if retrain_models==False else ''
        results_file_name = f'df_{model_name}_results{only_test_results}.csv'
        df_results.to_csv(os.path.join(config["logging_params"]["save_dir"],results_file_name), index=False)
    print(df_results)
    print('')
