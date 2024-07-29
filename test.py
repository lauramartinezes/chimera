import os
import re
import matplotlib

import matplotlib.colors as mcolors
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
from utils import load_checkpoint
from plot.config import *
from plot.set_plot_format import format_axes_and_grid

plt.rcParams['svg.fonttype'] = 'none'


kul_colour = {
    'sky':'#DBF2F9',
    'lagoon': '#52BDEC',
    'turquoise': '#1D8DB0',
    'night': '#2F4D5D',
    'orange': '#E7B037',
    'lime': '#D4D842',
    'dark_lime': '#ACAF24',
    'grid': '#DAE6EC',
    'log_grid': '#B2CBD8',
    'text': '#5f93aeff'
}

kul_gradient_colours = [kul_colour['sky'], kul_colour['lagoon'], kul_colour['turquoise'], kul_colour['night']]
kul_gradient = mcolors.LinearSegmentedColormap.from_list('custom_gradient', kul_gradient_colours, N=200)
kul_line_colours = [kul_colour['turquoise'], kul_colour['orange'], kul_colour['lime']]


# Function to generate example line plots
def get_line_plot(ax, architecture, x_values, y_values, y_errors, labels, y_axis_label, y_min, fontsize=12, line_style='-'):
    for i, label in enumerate(labels):
        if label[0]=='_': label=label[1:]
        #ax.plot(x_values, y_values[:, i], label=f'{label} - {architecture}', marker='o', color=kul_line_colours[i], linewidth=2, markersize=5, linestyle=line_style)
        ax.errorbar(x_values, y_values[:, i], yerr=y_errors[:, i], label=f'{label} - {architecture}', fmt='o-', capsize=5)
    #ax.set_title(title, fontsize=14, color=kul_colour['text'])
    ax.set_xlabel('Dataset Size', fontsize=fontsize, color=kul_colour['text'])
    # Set the x-axis ticks to the actual values
    ax.set_xticks(x_values)
    # Set y-axis ticks with a specified step
    y_step = 0.05
    y_ticks = np.arange(0, 1+y_step, y_step)
    ax.set_yticks(y_ticks)
    
    ax.grid(True)

    if y_axis_label == 0:
        ax.set_ylabel('Accuracy', fontsize=fontsize, color=kul_colour['text'])

    # Set y-axis limits (force the range between 0 and 1)
    ax.set_ylim(y_min-y_step, 1)
    format_axes_and_grid(ax, KUL_PALETTE)

    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()
    # Sort handles and labels alphabetically by labels
    sorted_handles, sorted_labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))
    # Set the sorted labels for the legend
    legend = ax.legend(sorted_handles, sorted_labels, labelcolor=kul_colour['text'], markerscale=0)
    legend.get_frame().set_edgecolor(kul_colour['grid'])


def plot_results_experiments(accuracy_values, augmentations, accuracies, architectures, dataset_sizes, scenarios, y_min):
    for i in range(len(augmentations)):
        # Create the main figure with two subfigures
        fig = plt.figure(layout='constrained', figsize=(12, 8))
        subfigs = fig.subfigures(len(accuracies), 1, wspace=0.01, hspace=0.03)
        
        for j in range(len(accuracies)):
            # Create subfigure for a row
            axs = subfigs[j].subplots(1, len(architectures), sharey=True, sharex=True)
            #subfigs[j].set_facecolor('0.95')
            subfigs[j].suptitle(accuracies[j], fontsize='x-large', color=kul_colour['text'])

            # Populate the row with line plots
            for k in range(len(architectures)):
                    get_line_plot(axs[k], architecture=architectures[k], x_values=dataset_sizes,
                                    y_values=accuracy_values[i,j,k, :, :], labels=scenarios, y_axis_label=k, y_min=y_min, fontsize=12)
        # Overall figure title
        fig.suptitle(augmentations[i].capitalize(), fontsize=20, color=kul_colour['text'])

        # Display the plot
        plt.show()

def plot_results_experiments_simplified(accuracy_values, augmentations, accuracies, architectures, dataset_sizes, scenarios, y_min):
    line_styles = ['-', '--', ':', '-.'] # for simplicity we assume a maximum of 4 architectures

    pattern = re.compile(r'tf_(\w+_\w+)\.')
    architecture_titles = [pattern.search(model).group(1) for model in architectures]

    for i in range(len(augmentations)):
        # Create the main figure
        fig = plt.figure(layout='constrained', figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)

        # Populate the row with line plots
        for k in range(len(architectures)):
                get_line_plot(ax, architecture=architecture_titles[k], x_values=dataset_sizes,
                                y_values=accuracy_values[i,0,k, :, :, 0], y_errors=accuracy_values[i,0,k, :, :, 1], labels=scenarios, y_axis_label=k, y_min=y_min, fontsize=12, line_style=line_styles[k])

        # Overall figure title
        fig.suptitle(f'{augmentations[i].capitalize()} - Balanced Accuracy', fontsize=20, color=kul_colour['text'])

        # Display the plot
        plt.show()

# Generate example data
dataset_sizes = [1000, 2000, 5000, 10000, 20000, 'None']
augmentations=['non_augmented', 'augmented']
scenarios=['scratch', 'pretrained_imagenet','pretrained_inaturalist']
architectures = ['tf_efficientnetv2_m.in21k_ft_in1k']
accuracies = ["Balanced Accuracy"]#, 'Target Accuracy']
seeds = [156, 482, 743]

# Generate random accuracy values
num_augmentations = len(augmentations)
num_accuracies = len(accuracies)
num_scenarios = len(scenarios)
num_architectures = len(architectures)
num_dataset_sizes = len(dataset_sizes)

#accuracy_values = np.random.rand(num_augmentations, num_accuracies, num_architectures, num_dataset_sizes-1, num_scenarios)
#plot_results_experiments(accuracy_values, augmentations, accuracies, architectures, dataset_sizes[:-1], scenarios, y_min=np.min(accuracy_values))



setting = 'fuji'
batch_size = 32
batch_size_val = 32
num_workers = 0
apply_augment = False

df_train = pd.read_parquet(f"{SAVE_DIR}/df_train_{setting}.parquet")
df_val = pd.read_parquet(f"{SAVE_DIR}/df_val_{setting}.parquet")
df_test = pd.read_parquet(f"{SAVE_DIR}/df_test_{setting}.parquet")

if os.path.exists(f'{SAVE_DIR}/accuracy_values.npy'):
    accuracy_values = np.load(f'{SAVE_DIR}/accuracy_values.npy')
else:
    # Get dataloader
    datasets = get_datasets(df_train, df_val, df_test, transform_train=set_train_transform(False), transform_test=set_test_transform())
    train_dataset, valid_dataset, test_dataset = datasets.values()

    dataloaders = load_data(datasets, batch_size, batch_size_val, num_workers)
    train_dataloader, valid_dataloader, test_dataloader = dataloaders.values()
    topclasses = df_train['txt_label'].unique().tolist()

    accuracy_values = np.zeros([num_augmentations, num_accuracies, num_architectures, num_dataset_sizes, num_scenarios, 2]) #the 2 corresponds to the mean and std dimesion

    for j, augmentation in enumerate(augmentations):
        for k, architecture in enumerate(architectures):
            for i, dataset_size in enumerate(dataset_sizes):
                for l, scenario in enumerate(scenarios):
                    seed_accuracies=[]
                    for m, seed in enumerate(seeds):
                        # Define and test model
                        if 'tf_efficientnet_b4' in architecture and dataset_size=='None':
                            saved_model = f'{setting}_{architecture}_{scenario}_{dataset_size}_{augmentation}'
                        elif 'v2' in architecture and dataset_size=='None':
                            saved_model = f'{setting}_{architecture}_{scenario}_{dataset_size}_seed_x_{augmentation}'
                        else:
                            saved_model = f'{setting}_{architecture}_{scenario}_{dataset_size}_seed_{seed}_{augmentation}'
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
                        model, optimizer = load_checkpoint(
                            f'{SAVE_DIR}/{saved_model}_best.pth.tar', model, optimizer)
                        model = model.to('cuda', dtype=torch.float)

                        test_bacc, _, _, _, _, _, test_accuracy_wmv, _, _ = test_model(model, test_dataloader, test_dataset)
                        seed_accuracies.append(test_bacc)
                        if 'v2' in architecture and dataset_size=='None':
                            break
                    mean_accuracies = np.mean(seed_accuracies, axis=0)
                    std_accuracies = np.std(seed_accuracies, axis=0)
                    # Assign the accuracy values to the corresponding positions in the accuracy_values array
                    accuracy_values[j, 0, k, i, l, 0] = mean_accuracies  # mean Balanced Accuracy
                    accuracy_values[j, 0, k, i, l, 1] = std_accuracies  # std Balanced Accuracy

    np.save(f'{SAVE_DIR}/accuracy_values.npy', accuracy_values)

dataset_sizes[-1] = len(df_train)
plot_results_experiments_simplified(accuracy_values, augmentations, accuracies, architectures, dataset_sizes, scenarios, y_min=np.min(accuracy_values))

