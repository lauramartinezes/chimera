import os
from matplotlib import pyplot as plt
import seaborn as sns
import umap


def plot_y_true_vs_y_od_pred_umap(features, measurement_noises, label_noises, y_pred, filename=None, dirname=None, detected_label_noises=None):
    umap_folder = os.path.join(dirname, 'True vs Pred UMAPS')
    os.makedirs(umap_folder, exist_ok=True)

    # UMAP transformation (shared latent space for both plots)
    print(f'Starting UMAP 2D reduction')
    reducer_2d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    latents_2d = reducer_2d.fit_transform(features)

    measurement_noises = measurement_noises.astype(int)*2
    label_noises = label_noises.astype(int)
    if detected_label_noises is not None:
        detected_label_noises = detected_label_noises.astype(int)*3
        noises = measurement_noises + label_noises + detected_label_noises
    else:
        noises = measurement_noises + label_noises

    # Dictionary to map numbers to text labels
    true_label_mapping = {0: "Normal Sample", 1: "Label Noise", 2: "Measurement Noise"}
    if detected_label_noises is not None:
        true_label_mapping[3] = "Detected Label Noise"
    txt_true_labels = [true_label_mapping[label] for label in noises]
    
    pred_label_mapping = {0: "Inlier", 1: "Outlier"}
    txt_pred_labels = [pred_label_mapping[label] for label in y_pred]

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot for y_true
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=txt_true_labels, palette=sns.color_palette("hsv", len(true_label_mapping)), ax=axes[0], legend='full')
    axes[0].set_title("Latent Space UMAP: True Labels")
    axes[0].set_xlabel("UMAP dimension 1")
    axes[0].set_ylabel("UMAP dimension 2")

    # Plot for y_pred
    sns.scatterplot(x=latents_2d[:, 0], y=latents_2d[:, 1], hue=txt_pred_labels, palette=sns.color_palette("hsv", 2), ax=axes[1], legend='full')
    axes[1].set_title("Latent Space UMAP: Predicted Labels")
    axes[1].set_xlabel("UMAP dimension 1")
    axes[1].set_ylabel("UMAP dimension 2")

    # Save the figure
    fig.suptitle(filename, fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(umap_folder, f'true_vs_pred_{filename}.png'), format='png')
    plt.savefig(os.path.join(umap_folder, f'true_vs_pred_{filename}.svg'), format='svg')