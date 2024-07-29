import argparse
from matplotlib import cm
import timm
import itertools
import os
import seaborn as sns
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch

from torch import nn, optim
from config.config import SAVE_DIR
from datasets.load_data import *
from datasets.transforms import set_test_transform, set_train_transform
from model.test import test_model
from train import freeze_model_layers
from utils import load_checkpoint
from plot.config import *

#plt.rcParams['svg.fonttype'] = 'none'


def get_args():
    parser = argparse.ArgumentParser(description='Test for Pretrained nets and scratch net comparison')
    parser.add_argument('--setting', type=str, default="fuji")
    parser.add_argument('--pretrainings', nargs='+', type=str, default=['pretrained_inaturalist', "pretrained_imagenet", "scratch"], help="scratch, pretrained_imagenet or pretrained_inaturalist")
    parser.add_argument('--dataset_sizes', nargs='+', type=str, default=['1000', '2000', '5000', '10000', '20000', 'None'])
    parser.add_argument('--architectures', nargs='+', type=str, default=['tf_efficientnetv2_s.in21k_ft_in1k'])
    parser.add_argument('--augmentations', nargs='+', type=str, default=['augmented'], help='non_augmented or augmented')
    parser.add_argument('--frozen_layers', nargs='+', type=str, default=['_lastlayer_conv_stem', '', '_4_5_6_conv_stem'], help='_5_6')
    parser.add_argument('--seeds', nargs='+', type=int, default=[156], help='482, 743')
    parser.add_argument('--experiment_name', type=str, default="ex_3_all_pretrainings_once", choices=["ex_1_architectures", "ex_2_augmentations", "ex_3_pretrainings", "ex_2_imagenet_pretraining"])

    args = parser.parse_args()

    args.dataset_sizes = [int(x) if x.isdigit() else x for x in args.dataset_sizes]
    return args

if __name__ == '__main__':
    args = get_args()
    setting = 'fuji'
    batch_size = 32
    batch_size_val = 32
    num_workers = 0
    apply_augment = False

    df_train = pd.read_parquet(f"{SAVE_DIR}/df_train_{setting}.parquet")
    df_val = pd.read_parquet(f"{SAVE_DIR}/df_val_{setting}.parquet")
    df_test = pd.read_parquet(f"{SAVE_DIR}/df_test_{setting}.parquet")

    # Get dataloader
    datasets = get_datasets(df_train, df_val, df_test, transform_train=set_train_transform(False), transform_test=set_test_transform())
    train_dataset, valid_dataset, test_dataset = datasets.values()

    dataloaders = load_data(datasets, batch_size, batch_size_val, num_workers)
    train_dataloader, valid_dataloader, test_dataloader = dataloaders.values()
    topclasses = df_train['txt_label'].unique().tolist()

    directory = f'{SAVE_DIR}/test_paper'
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(f"{directory}/df_{args.experiment_name}.csv"):
        df_results = pd.DataFrame({
            'dataset_size': [],
            'architecture': [],
            'augmentation': [],
            'pretraining': [],
            'frozen_layers': [],
            'seed': []
        })

        # Generate all combinations of the parameters
        combinations = list(itertools.product(args.dataset_sizes, args.architectures, args.augmentations, args.pretrainings, args.frozen_layers, args.seeds))

        # Create a DataFrame from the combinations
        df_results = pd.DataFrame(combinations, columns=['dataset_size_name', 'architecture', 'augmentation', 'pretraining', 'frozen_layers', 'seed'])

        # Duplicate the dataset_size column and call it dataset_size_values
        df_results['dataset_size_value'] = df_results['dataset_size_name']

        # Replace 'None' with 5000 in dataset_size_values
        df_results['dataset_size_value'] = df_results['dataset_size_value'].replace('None', len(df_train)).astype(int)

        # Reorder columns to place dataset_size_values next to dataset_size
        df_results = df_results[['dataset_size_name', 'dataset_size_value', 'architecture', 'augmentation', 'pretraining', 'frozen_layers', 'seed']]

        # Update frozen_layers based on the value of pretraining
        df_results['frozen_layers'] = df_results.assign(frozen_layers=lambda x: np.where(x['pretraining'] == 'scratch', '', x['frozen_layers']))['frozen_layers']
        df_results = df_results.drop_duplicates().reset_index(drop=True)

        # Add an accuracy column, initialized to None or any default value
        df_results['balanced_accuracy'] = np.random.rand(len(df_results)) #None

        df_results_sorted = df_results.sort_values(by=['augmentation', 'architecture', 'pretraining', 'dataset_size_value', 'seed'], ascending=[True, False, False, True, True])
        df_results_sorted.reset_index(drop=True, inplace=True)

        for index, row in df_results_sorted.iterrows():
            if row['dataset_size_name']=='None':
                saved_model = f'{setting}_{row["architecture"]}_{row["pretraining"]}_{row["dataset_size_name"]}_seed_x_{row["augmentation"]}{row["frozen_layers"]}'
            else:
                saved_model = f'{setting}_{row["architecture"]}_{row["pretraining"]}_{row["dataset_size_name"]}_seed_{row["seed"]}_{row["augmentation"]}{row["frozen_layers"]}'
            model = timm.create_model(row['architecture'], pretrained=True, num_classes=len(topclasses))
            
            # If we indicated to freeze layers do it
            if row["frozen_layers"] != '':
                if row["frozen_layers"] == '_lastlayer_conv_stem':
                    frozen_layers = [-1]
                else:
                    frozen_layers = [int(digit) for digit in row['frozen_layers'] if digit.isdigit()]
                model = freeze_model_layers(model, frozen_layers)
                
            # Choosing whether to train on a gpu
            train_on_gpu = torch.cuda.is_available()
            print(f'Train on gpu: {train_on_gpu}')  # Number of gpus
            model = model.to('cuda', dtype=torch.float)

            class_sample_count = np.unique(df_train.label, return_counts=True)[1]
            weight = 1. / class_sample_count
            criterion = nn.CrossEntropyLoss(
                label_smoothing=.15, weight=torch.Tensor(weight).cuda())
            if row['frozen_layers']=='':
                optimizer = optim.AdamW(model.parameters(), lr=.003, weight_decay=0.05)
            else:
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003, weight_decay=0.05)
            
            model, optimizer = load_checkpoint(
                f'{SAVE_DIR}/{saved_model}_best.pth.tar', model, optimizer)
            model = model.to('cuda', dtype=torch.float)

            test_bacc, _, _, _, _, _, test_accuracy_wmv, _, _ = test_model(model, test_dataloader, test_dataset)
            df_results_sorted.at[index, 'balanced_accuracy'] = test_bacc
        df_results_sorted.to_csv(f"{directory}/df_{args.experiment_name}.csv")
    else:
        df_results_sorted = pd.read_csv(f"{directory}/df_{args.experiment_name}.csv").fillna('')

    # Create a hidden Tkinter root window to get screen size
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Set seaborn style
    sns.set_style(
        "whitegrid", 
        {
            "grid.color": '.6', 
            "grid.linestyle": ":", 
            #"axes.spines.left": False, 
            "axes.spines.right": False, 
            "axes.spines.top": False, 
            #"axes.spines.bottom": False
        }
    )
    sns.set_context("talk")
    #sns.set(rc={'axes.labelcolor': 'red', 'xtick.color': 'blue', 'ytick.color': 'green'})

    plt.figure(figsize=(screen_width/100, screen_height/100))

    # Experiment 1: Comparison of Architecture Sizes
    if len(df_results_sorted['architecture'].unique()) > 1 and len(df_results_sorted['augmentation'].unique()) == 1 \
        and len(df_results_sorted['seed'].unique()) == 1 and len(df_results_sorted['pretraining'].unique()) == 1:
        # Plot lines for each architecture
        for architecture, group in df_results_sorted.groupby('architecture'):
            plt.plot(group['dataset_size_value'], group['balanced_accuracy'], label=architecture, marker='o')
        plt.title('Comparison of Architecture Sizes')

    # Experiment 2: Comparison of Layers One Pretraining
    if len(df_results_sorted['architecture'].unique()) == 1 and len(df_results_sorted['augmentation'].unique()) == 1 \
        and len(df_results_sorted['seed'].unique()) == 1 and len(df_results_sorted['pretraining'].unique()) == 1 \
            and len(df_results_sorted['frozen_layers'].unique()) > 1:
        # Plot lines for each architecture
        for frozen_layer, group in df_results_sorted.groupby('frozen_layers'):
            if frozen_layer == '_lastlayer_conv_stem':
                label = "last layer"
            if frozen_layer == '':
                label = "all layers"
            if frozen_layer =='_4_5_6_conv_stem':
                label = 'mid and last layers'
            plt.plot(group['dataset_size_value'], group['balanced_accuracy'], label=label, marker='o')
        plt.title(f'Comparison of {df_results_sorted["pretraining"].unique()[0]} Layers')

    # Experiment 3: Comparison of Layers All Pretrainings
    if len(df_results_sorted['architecture'].unique()) == 1 and len(df_results_sorted['augmentation'].unique()) == 1 \
        and len(df_results_sorted['seed'].unique()) == 1 and len(df_results_sorted['pretraining'].unique()) > 1 \
            and len(df_results_sorted['frozen_layers'].unique()) > 1:
        pretraining_colormaps = {
            'scratch': cm.bone,
            'pretrained_imagenet': cm.spring,
            'pretrained_inaturalist': cm.winter
        }

        # Define linestyles for each pretraining
        frozen_layer_linestyles = {
            '_4_5_6_conv_stem': '-',
            '_lastlayer_conv_stem': ':',
            '': '--'
        }

        # Plot lines for each combination of pretraining and frozen_layers
        for pretraining in df_results_sorted['pretraining'].unique():
            colormap = pretraining_colormaps.get(pretraining, cm.viridis)  # Default to 'viridis' if not found
            colors = colormap(np.linspace(0.3, 0.7, len(df_results_sorted['frozen_layers'].unique())))
            if 'ñ' not in pretraining:
                for (frozen_layer, group), color in zip(df_results_sorted[df_results_sorted['pretraining'] == pretraining].groupby('frozen_layers'), colors):
                    if pretraining != 'scratch':
                        linestyle = frozen_layer_linestyles.get(frozen_layer, '-')  # Default to solid line if not found
                    else:
                        linestyle='-.'
                    if frozen_layer == '_lastlayer_conv_stem':
                        label = f"{pretraining} - last layer"
                    elif frozen_layer == '':
                        label = f"{pretraining} - all layers"
                    elif frozen_layer == '_4_5_6_conv_stem':
                        label = f"{pretraining} - mid and last layers"
                    else:
                        label = f"{frozen_layer} + {pretraining}"  # Fallback in case of unexpected frozen_layer values
                    
                    plt.plot(group['dataset_size_value'], group['balanced_accuracy'], label=label, marker='o', color=colors[0], linestyle=linestyle)
        plt.title(f'Comparison of Layers for all pretrainings')

    # Experiment 2: Comparison of Augmentation vs. Not Augmentation
    elif len(df_results_sorted['architecture'].unique()) == 1 and len(df_results_sorted['augmentation'].unique()) == 2 \
        and len(df_results_sorted['seed'].unique()) == 1 and len(df_results_sorted['pretraining'].unique()) == 1:
        # Plot lines for each augmentation
        for augmentation, group in df_results_sorted.groupby('augmentation'):
            plt.plot(group['dataset_size_value'], group['balanced_accuracy'], label=augmentation, marker='o')
        plt.title('Comparison of Augmentation vs. Not Augmentation')

    # Experiment 3: Comparison of Pretraining Strategies
    elif len(df_results_sorted['architecture'].unique()) == 1 and len(df_results_sorted['augmentation'].unique()) == 1 \
        and len(df_results_sorted['seed'].unique()) >= 1 and len(df_results_sorted['pretraining'].unique()) > 1:
        # Plot error bars with lines for each pretraining
        for pretraining, group in df_results_sorted.groupby('pretraining'):
            if len(df_results_sorted['seed'].unique()) > 1:
                mean_accuracy = group.groupby('dataset_size_value')['balanced_accuracy'].mean()
                std_accuracy = group.groupby('dataset_size_value')['balanced_accuracy'].std()
                plt.errorbar(mean_accuracy.index, mean_accuracy, yerr=std_accuracy, label=pretraining, fmt='o-', capsize=5)
            else:
                plt.plot(group['dataset_size_value'], group['balanced_accuracy'], label=pretraining, marker='o')
        plt.title('Comparison of Pretraining Strategies')

    else:
        raise ValueError("Unexpected experiment configuration. Please check the conditions.")

    # Set x-axis ticks to include only dataset sizes
    plt.xticks(sorted(df_results_sorted['dataset_size_value'].unique()), rotation=-10)
    # Set y-axis ticks to the specified interval
    y_step = 0.05
    plt.yticks(np.arange(0.45 - y_step, 1 + y_step, y_step))

    # Set plot labels and legend
    plt.xlabel('Dataset Size')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

   # plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(f'{directory}/plot_{args.experiment_name}.svg', format='svg')

    plt.show()
