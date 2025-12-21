import os

from matplotlib import pyplot as plt
import numpy as np


def plot_training_curves(train_vals, val_vals, test_vals, ylabel, title, filename_suffix, save_path, clean_dataset, method, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_vals, label='Train ' + ylabel)
    plt.plot(range(1, num_epochs + 1), val_vals, label='Validation ' + ylabel)
    plt.plot(range(1, num_epochs + 1), test_vals, label='Test ' + ylabel)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    svg_path = os.path.join(save_path, f'{filename_suffix}_{clean_dataset}_{method}.svg')
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