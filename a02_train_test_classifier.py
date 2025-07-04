import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml

from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from datasets import CustomBinaryInsectDF, set_test_transform, set_train_transform
from models import compute_accuracy, compute_loss, compute_predictions


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def load_subset_df_classification(insect_classes, subset, method, od_method, root_dir, clean_dataset=''):
    dfs_subset = []
    for i, main_insect_class in enumerate(insect_classes):
        df_subset_path = os.path.join(
            root_dir,
            'clean' if clean_dataset=='_clean' else '', 
            f'df_{subset}_{method if method != "cleaning_benchmark" else "raw"}_{main_insect_class}{od_method}{clean_dataset}.csv'
        )
        df_subset_i = pd.read_csv(df_subset_path)
        if method == 'cleaning_benchmark':
            df_subset_i_no_noise = df_subset_i[df_subset_i['label'] == 0]
            df_subset_i_label_noise = df_subset_i[df_subset_i['mislabeled'] == True] # Comment this line if you want to reove label noise as well
            # Correct labels for training a classifier instead of an outlier detector
            df_subset_i_no_noise.label = i
            df_subset_i_label_noise.label = 1 - i
            dfs_subset.append(df_subset_i_no_noise)
            dfs_subset.append(df_subset_i_label_noise)
        else:
            # Correct labels for training a classifier instead of an outlier detector
            df_subset_i.label = i
            dfs_subset.append(df_subset_i)
    
    return pd.concat(dfs_subset, ignore_index=True)


def train_epoch(model, train_loader, optimizer, criterion, device, case, epoch, num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, labels, _, _, _, _) in enumerate(train_loader):
        
        images = images.to(device)
        labels = labels.to(device)
            
        ### FORWARD AND BACK PROP
        logits = model(images)
        probas = F.softmax(logits, dim=1)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        
        loss.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print (f'Case {case} dataset: '+'Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                %(epoch+1, num_epochs, batch_idx, 
                    len(train_loader), loss))  
        epoch_loss += loss.item()
    return epoch_loss / (batch_idx + 1)


def validate_epoch(model, train_loader, val_loader, test_loader, criterion, device, class_weights_tensor, epoch, num_epochs):
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        val_epoch_loss = compute_loss(model, val_loader, device, criterion=criterion, class_weights_tensor=class_weights_tensor)
        test_epoch_loss = compute_loss(model, test_loader, device, criterion=criterion, class_weights_tensor=class_weights_tensor)
                    
        (train_accuracy, train_insect_0_accuracy, train_insect_1_accuracy, 
        train_insect_0_meas_noise_accuracy, train_insect_1_meas_noise_accuracy, 
        train_insect_0_milabel_accuracy, train_insect_1_milabel_accuracy, 
        train_insect_0_good_accuracy, train_insect_1_good_accuracy) = compute_accuracy(model, train_loader, device=device)

        (val_accuracy, val_insect_0_accuracy, val_insect_1_accuracy, 
        val_insect_0_meas_noise_accuracy, val_insect_1_meas_noise_accuracy, 
        val_insect_0_milabel_accuracy, val_insect_1_milabel_accuracy,
        val_insect_0_good_accuracy, val_insect_1_good_accuracy) = compute_accuracy(model, val_loader, device=device)

        (test_accuracy, test_insect_0_accuracy, test_insect_1_accuracy, 
        test_insect_0_meas_noise_accuracy, test_insect_1_meas_noise_accuracy, 
        test_insect_0_milabel_accuracy, test_insect_1_milabel_accuracy,
        test_insect_0_good_accuracy, test_insect_1_good_accuracy) = compute_accuracy(model, test_loader, device=device)

        print('Epoch: %03d/%03d | Train: %.3f%% | Validation: %.3f%% | Test: %.3f%%' % (
            epoch+1, num_epochs, 
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

        lowest_val_class_accuracy = min(val_insect_0_accuracy, val_insect_1_accuracy)
    return val_epoch_loss, test_epoch_loss, train_accuracy.item(), val_accuracy.item(), test_accuracy.item(), lowest_val_class_accuracy


def test_model(model, test_loader, device):
    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        (test_accuracy, test_insect_0_accuracy, test_insect_1_accuracy, 
        test_insect_0_meas_noise_accuracy, test_insect_1_meas_noise_accuracy, 
        test_insect_0_milabel_accuracy, test_insect_1_milabel_accuracy,
        test_insect_0_good_accuracy, test_insect_1_good_accuracy) = compute_accuracy(model, test_loader, device)
        print(f'Test accuracy: %.2f%%' % test_accuracy)
        print(f'Test {insect_classes[0]} Accuracy: %.2f%%' % test_insect_0_accuracy)
        print(f'Test {insect_classes[1]} Accuracy: %.2f%%' % test_insect_1_accuracy)
    test_predictions, test_actuals, test_probs = compute_predictions(model, test_loader, device)
    return test_accuracy, test_insect_0_accuracy, test_insect_1_accuracy, test_predictions, test_actuals, test_probs


def save_best_model(model, save_path_best, best_val_accuracy, val_accuracy, lowest_val_class_accuracy, epochs_no_improve):
    if best_val_accuracy < val_accuracy:
        best_val_accuracy = val_accuracy
        best_lowest_val_class_accuracy = lowest_val_class_accuracy
        torch.save(model.state_dict(), save_path_best)
        print(f"New best model saved at {save_path_best} with Validation Accuracy: {best_val_accuracy:.3f}% and lowest class accuracy: {best_lowest_val_class_accuracy:.3f}%")
        epochs_no_improve = 0  # reset patience counter
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s)")
    return best_val_accuracy, epochs_no_improve


def plot_training_curves(train_vals, val_vals, test_vals, ylabel, title, filename_suffix, save_path, clean_dataset, method, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_vals, label='Train ' + ylabel)
    plt.plot(range(1, num_epochs + 1), val_vals, label='Validation ' + ylabel)
    plt.plot(range(1, num_epochs + 1), test_vals, label='Test ' + ylabel)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    png_path = os.path.join(save_path, f'{filename_suffix}_{clean_dataset}_{method}.png')
    svg_path = os.path.join(save_path, f'{filename_suffix}_{clean_dataset}_{method}.svg')
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.close()


def plot_probability_distribution(probs, true_labels, start=0.5, end=1.0, step=0.05, filename_suffix='', save_path='', clean_dataset='', method=''):
    """
    Plots the distribution of predicted probabilities for the true class.
    
    Parameters:
    - probs: array of shape (n_samples, 2), predicted class probabilities
    - true_labels: array of shape (n_samples,), true labels (0 or 1)
    - start: float, start of probability range (default 0.5)
    - end: float, end of probability range (default 1.0)
    - step: float, bin step size (default 0.05)
    """
    # 1. Predicted probability for the true class
    true_probs = probs[np.arange(len(probs)), true_labels]

    # 2. Split by true class
    true_probs_class0 = true_probs[true_labels == 0]
    true_probs_class1 = true_probs[true_labels == 1]

    # 3. Bin centers and edges
    bin_centers = np.arange(start, end + 1e-8, step)
    half_step = step / 2
    bin_edges = np.arange(start - half_step, end + half_step + 1e-8, step)

    # 4. Histogram counts
    counts_class0, _ = np.histogram(true_probs_class0, bins=bin_edges)
    counts_class1, _ = np.histogram(true_probs_class1, bins=bin_edges)

    # 5. Plot
    bin_width = step
    plt.figure(figsize=(8,6))
    plt.bar(bin_centers, counts_class0, width=bin_width, label='True Class 0', edgecolor='black', alpha=0.6)
    plt.bar(bin_centers, counts_class1, width=bin_width, label='True Class 1', edgecolor='black', alpha=0.6)
    plt.xlabel('Predicted Probability for True Class')
    plt.ylabel('Number of Samples')
    plt.title(f'Probability Distribution per True Class (step={step})')
    plt.xticks(bin_centers)
    plt.xlim(start - half_step, end + half_step)
    plt.legend()
    plt.grid(True)
    png_path = os.path.join(save_path, f'{filename_suffix}_prob_distribution_{clean_dataset}_{method}.png')
    svg_path = os.path.join(save_path, f'{filename_suffix}_prob_distribution_{clean_dataset}_{method}.svg')
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.close()


def plot_prediction_confidence_by_predicted_class(probs, start=0.5, end=1.0, step=0.05,
                                                  filename_suffix='', save_path='', clean_dataset='', method=''):
    """
    Plots the distribution of model confidence (max predicted probability),
    grouped by the predicted class.

    Parameters:
    - probs: array of shape (n_samples, 2) or list of 2-element arrays
    - start: float, min prob to plot (default 0.5)
    - end: float, max prob to plot (default 1.0)
    - step: float, bin width (default 0.05)
    - filename_suffix, save_path, clean_dataset, method: optional suffixes for saving the plot
    """
    # Convert list of arrays to 2D array if needed
    if isinstance(probs, list):
        probs = np.stack(probs)

    # 1. Model confidence (max probability)
    max_probs = probs.max(axis=1)

    # 2. Predicted class
    pred_labels = np.argmax(probs, axis=1)

    # 3. Split by predicted class
    max_probs_pred0 = max_probs[pred_labels == 0]
    max_probs_pred1 = max_probs[pred_labels == 1]

    # 4. Bin setup
    bin_centers = np.arange(start, end + 1e-8, step)
    half_step = step / 2
    bin_edges = np.arange(start - half_step, end + half_step + 1e-8, step)

    # 5. Histogram counts
    counts_pred0, _ = np.histogram(max_probs_pred0, bins=bin_edges)
    counts_pred1, _ = np.histogram(max_probs_pred1, bins=bin_edges)

    # 6. Plot
    bin_width = step
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, counts_pred0, width=bin_width, label='Predicted Class 0', edgecolor='black', alpha=0.6)
    plt.bar(bin_centers, counts_pred1, width=bin_width, label='Predicted Class 1', edgecolor='black', alpha=0.6)
    plt.xlabel('Model Confidence (Max Predicted Probability)')
    plt.ylabel('Number of Samples')
    plt.title(f'Confidence Distribution per Predicted Class (step={step})')
    plt.xticks(bin_centers)
    plt.xlim(start - half_step, end + half_step)
    plt.legend()
    plt.grid(True)

    # 7. Save if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        png_path = os.path.join(save_path, f'{filename_suffix}_confidence_by_predicted_class_{clean_dataset}_{method}.png')
        svg_path = os.path.join(save_path, f'{filename_suffix}_confidence_by_predicted_class_{clean_dataset}_{method}.svg')
        plt.savefig(png_path)
        plt.savefig(svg_path)
        print(f"Saved to {png_path} and {svg_path}")
    else:
        plt.show()


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Classifier')
    parser.add_argument('--random_seed_1', type=int, default=1)
    parser.add_argument('--random_seed_2', type=int, default=1265)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--experiments', type=str, default='noisy_vs_cleaning_benchmark', help='noisy_vs_cleaning_benchmark, all_cases')
    args = parser.parse_args()
    return args

##########################
### SETTINGS
##########################

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.00001 #0.0001 #80, 81 0.00001 
BATCH_SIZE = 64
NUM_EPOCHS = 2 #1


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    args = get_args()

    # Set manual seed for reproducibility
    set_seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0

    ##########################
    ### BINARY CLASS INSECT DATASET
    ##########################
    data_dir = config["data_params"]["data_dir"]
    insect_classes = config["data_params"]["data_classes"] 
    num_classes = len(insect_classes)
    model_name = config["model_params"]["resnet18"] 
    device = config["trainer_params"]["device"]
    method_datasets = ['adbench', 'cnn', 'raw', 'cleaning_benchmark']
    retrain_models = True

    results = []

    for method in method_datasets:
        if method=='cleaning_benchmark' or method=='raw': 
            clean_dataset=''
        else:
            clean_dataset='_clean'

        if method == 'cnn':
            od_method = '_DBSCAN'
        elif method == 'adbench':
            od_method = '_OCSVM'
        else:   
            od_method = ''
            
        ##########################
        ### DATA LOADING
        ##########################
        df_train = load_subset_df_classification(insect_classes, 'train', method, od_method, data_dir, clean_dataset)
        df_val = load_subset_df_classification(insect_classes, 'val', method, od_method, data_dir, clean_dataset)
        df_test = pd.read_csv(os.path.join(data_dir, f'df_test.csv'))

        train_dataset = CustomBinaryInsectDF(df_train, transform = set_train_transform(), seed=config["exp_params"]["manual_seed"])
        val_dataset = CustomBinaryInsectDF(df_val, transform = set_test_transform(), seed=config["exp_params"]["manual_seed"])
        test_dataset = CustomBinaryInsectDF(df_test, transform = set_test_transform(), seed=config["exp_params"]["manual_seed"])

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
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        model.to(device)

        save_path = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}.pth')
        save_path_best = os.path.join(config["logging_params"]["save_dir"], f'{model_name}_classifier{clean_dataset}_{method}_best.pth')

        if not os.path.exists(save_path_best) or retrain_models==True:
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df_train.label), y=df_train.label.to_numpy())
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)  
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
                    
            ##########################
            ### TRAIN
            ##########################
            best_val_accuracy = 0
            best_lowest_val_class_accuracy = 0
            train_accuracies = []
            val_accuracies = []
            test_accuracies = []
            train_losses = []
            val_losses = []
            test_losses = []
            start_time = time.time()

            patience = 5  
            epochs_no_improve = 0

            if method == 'raw':
                case = 'raw dirty'
            elif method == 'cleaning_benchmark':
                case = 'cleaning benchmark'
            else:
                case = f'{method} cleaned'

            for epoch in range(NUM_EPOCHS):
                train_epoch_loss = train_epoch(model, train_loader, optimizer, criterion, device, case, epoch, NUM_EPOCHS)
                train_losses.append(train_epoch_loss)

                (val_epoch_loss, test_epoch_loss, 
                 train_accuracy, val_accuracy, test_accuracy, 
                 lowest_val_class_accuracy) = validate_epoch(
                        model, 
                        train_loader, val_loader, test_loader, 
                        criterion, device, class_weights_tensor, 
                        epoch, NUM_EPOCHS
                    )
                
                val_losses.append(val_epoch_loss)
                test_losses.append(test_epoch_loss)
                
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                test_accuracies.append(test_accuracy)
                
                # Check if the validation accuracy improved
                best_val_accuracy, epochs_no_improve = save_best_model(
                    model, save_path_best, best_val_accuracy, val_accuracy, lowest_val_class_accuracy, epochs_no_improve)

                print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

                if epoch > 1:
                    # Plot the training and validation accuracy
                    train_curves_path = os.path.join(config["logging_params"]["save_dir"], f'training_curves_{model_name}')
                    os.makedirs(train_curves_path, exist_ok=True)

                    plot_training_curves(
                        train_accuracies, val_accuracies, test_accuracies,
                        ylabel='Accuracy',
                        title='Training and Validation Accuracy',
                        filename_suffix='train_val_test_acc',
                        save_path=train_curves_path,
                        clean_dataset=clean_dataset,
                        method=method,
                        num_epochs=epoch + 1
                    )

                    plot_training_curves(
                        train_losses, val_losses, test_losses,
                        ylabel='Loss',
                        title='Training and Validation Loss',
                        filename_suffix='train_val_test_loss',
                        save_path=train_curves_path,
                        clean_dataset=clean_dataset,
                        method=method,
                        num_epochs=epoch + 1
                    )

            print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
            torch.save(model.state_dict(), save_path)         

        else:
            model.load_state_dict(torch.load(save_path_best))
            model.eval()
            with torch.set_grad_enabled(False): # save memory during inference
                (best_val_accuracy, _, _, _, _, _, _, _, _) = compute_accuracy(model, val_loader, device=device)
                print('Best Validation accuracy: %.2f%%' % best_val_accuracy)

        ##########################
        ### TEST
        ##########################
        # Load the best model
        model.load_state_dict(torch.load(save_path_best))
        (test_accuracy, test_insect_0_accuracy, test_insect_1_accuracy, 
        test_predictions, test_actuals, test_probs) = test_model(model, test_loader, device)

        prob_distribution_path = os.path.join(config["logging_params"]["save_dir"], f'prob_distribution_{model_name}')
        os.makedirs(prob_distribution_path, exist_ok=True)
        plot_probability_distribution(
            np.array(test_probs), 
            np.array(test_actuals), 
            start=0, end=1.0, step=0.1,
            filename_suffix='test',
            save_path=prob_distribution_path,
            clean_dataset=clean_dataset,
            method=method
            )
        
        plot_prediction_confidence_by_predicted_class(
            np.array(test_probs), 
            start=0.5, end=1.0, step=0.1,
            filename_suffix='test',
            save_path=prob_distribution_path,
            clean_dataset=clean_dataset,
            method=method
            )

        results.append({
            "clean_dataset": not(clean_dataset==''),
            "best_val_accuracy": round(best_val_accuracy.item(), 2),
            "test_accuracy": round(test_accuracy.item(), 2),
            f"test_{insect_classes[0]}_accuracy": round(test_insect_0_accuracy.item(), 2),
            f"test_{insect_classes[1]}_accuracy": round(test_insect_1_accuracy.item(), 2),
            'method': f'{method}'
        })
        df_results = pd.DataFrame(results)
        only_test_results = '_only_test' if retrain_models==False else ''
        results_file_name = f'df_{model_name}_results{only_test_results}.csv'
        df_results.to_csv(os.path.join(config["logging_params"]["save_dir"],results_file_name), index=False)
    print(df_results)
    print('')
