import os
import seaborn as sns

from matplotlib import pyplot as plt


def plot_test_latent_space_wrt_train(main_insect_class, mislabeled_insect_class, train_features, test_features, labels_train, labels_test, insect_classes, filename='', dirname=''):
    umap_folder = os.path.join(dirname, 'UMAPS')
    os.makedirs(umap_folder, exist_ok=True)
    
    # Dictionary to map numbers to text labels
    label_mapping_train = {0: f"Normal Sample ({main_insect_class})", 1: f"Label Noise ({mislabeled_insect_class})", 2: "Measurement Noise"}
    txt_labels_train = [label_mapping_train[label] for label in labels_train]
    
    label_mapping_test = {0: f"{insect_classes[0]}_test", 1: f"{insect_classes[1]}_test"}
    txt_labels_test = [label_mapping_test[label] for label in labels_test]

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=test_features[:, 0], y=test_features[:, 1], hue=txt_labels_test, palette=sns.color_palette("Set2", len(label_mapping_test)), marker='o', legend='full')
    sns.scatterplot(x=train_features[:, 0], y=train_features[:, 1], hue=txt_labels_train, palette=sns.color_palette("hsv", len(label_mapping_train)), marker='x', s=15, legend='full')
    plt.title("Latent Space UMAP Visualization")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.png'), format='png')
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.svg'), format='svg')


def plot_train_latent_space(train_features, labels_train, filename='', dirname=''):
    umap_folder = os.path.join(dirname, 'UMAPS')
    os.makedirs(umap_folder, exist_ok=True)
    
    # Dictionary to map numbers to text labels
    label_mapping = {0: "Normal Sample", 1: "Label Noise", 2: "Measurement Noise"}
    txt_noise_labels = [label_mapping[label] for label in labels_train]

    # Plot UMAP results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=train_features[:, 0], y=train_features[:, 1], hue=txt_noise_labels, palette=sns.color_palette("hsv", 3), legend='full')
    plt.title("Latent Space UMAP Visualization")
    plt.xlabel("UMAP dimension 1")
    plt.ylabel("UMAP dimension 2")
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.png'), format='png')
    plt.savefig(os.path.join(umap_folder, f'umap_plot_{filename}.svg'), format='svg')

