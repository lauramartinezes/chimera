import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import yaml

from sklearn.utils.class_weight import compute_class_weight

from datasets import set_test_transform, set_train_transform, load_subset_df_classification
from datasets.load_data import load_data_from_df
from models.save_best_model import save_best_model
from models.metrics import compute_accuracy
from models.plot import plot_prediction_confidence_by_predicted_class, plot_probability_distribution, plot_training_curves
from models.train_val_test import test_model, train_epoch, validate_epoch


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Classifier')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--experiments', type=str, default='noisy_vs_cleaning_benchmark', help='noisy_vs_cleaning_benchmark, all_cases')
    args = parser.parse_args()
    return args

# Hyperparameters
RANDOM_SEED = 1


if __name__ == '__main__':
    # Load the configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    args = get_args()

    # Set manual seed for reproducibility
    set_seed(config["exp_params"]["manual_seed"])

    pin_memory = len(config['trainer_params']['gpus']) != 0
    data_dir = config["data_params"]["data_dir"]
    insect_classes = config["data_params"]["data_classes"] 
    num_classes = len(insect_classes)
    model_name = config["model_params"]["resnet18"] 
    device = config["trainer_params"]["device"]
    batch_size = config["data_params"]["batch_size"]
    learning_rate = config["exp_params"]["LR"]
    num_epochs = config["trainer_params"]["num_epochs"]

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

        train_loader = load_data_from_df(
            df_train,
            set_train_transform(),
            config["exp_params"]["manual_seed"],
            config["data_params"][f"batch_size"],
            config["data_params"]["num_workers"],
            pin_memory,
            shuffle=True
        )

        val_loader = load_data_from_df(
            df_val,
            set_test_transform(),
            config["exp_params"]["manual_seed"],
            config["data_params"][f"batch_size"],
            config["data_params"]["num_workers"],
            pin_memory,
            shuffle=False
        )

        test_loader = load_data_from_df(
            df_test,
            set_test_transform(),
            config["exp_params"]["manual_seed"],
            config["data_params"][f"batch_size"],
            config["data_params"]["num_workers"],
            pin_memory,
            shuffle=False
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

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)  
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

            for epoch in range(num_epochs):
                train_epoch_loss = train_epoch(model, train_loader, optimizer, criterion, device, case, epoch, num_epochs)
                train_losses.append(train_epoch_loss)

                (val_epoch_loss, test_epoch_loss, 
                 train_accuracy, val_accuracy, test_accuracy, 
                 lowest_val_class_accuracy) = validate_epoch(
                        insect_classes,
                        model, 
                        train_loader, val_loader, test_loader, 
                        criterion, device, class_weights_tensor, 
                        epoch, num_epochs
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
        test_predictions, test_actuals, test_probs) = test_model(insect_classes, model, test_loader, device)

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
