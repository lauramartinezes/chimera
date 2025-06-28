import os
import cuml
import pandas as pd
import timm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

from PyOD import PYOD
from a04_2_umap_projections_cnn import extract_features_from_dataloader
from mnist_dataset import CustomBinaryInsectDF
from tqdm import tqdm


def visualize_y_true_vs_y_pred_umap(features, measurement_noises, label_noises, y_pred, filename=None, dirname=None, detected_label_noises=None):
    # umap_folder = os.path.join(dirname, 'True vs Pred UMAPS')
    # os.makedirs(umap_folder, exist_ok=True)

    # UMAP transformation (shared latent space for both plots)
    print(f'Starting UMAP 2D reduction')
    # reducer_2d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    reducer_2d = cuml.manifold.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
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
    
    pred_label_mapping = {0: "Normal Sample", 1: "Outlier"}
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
    # plt.savefig(os.path.join(umap_folder, f'true_vs_pred_{filename}.png'), format='png')
    # plt.savefig(os.path.join(umap_folder, f'true_vs_pred_{filename}.svg'), format='svg')
    plt.show()

# Step 1: Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

insect_classes = ['wmv', 'wrl']
clean_dataset = ''
method = 'raw'
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
    )
    df_val_i = pd.read_csv(df_val_path)
    if method == 'cleaning_benchmark':
        df_val_i = df_val_i[df_val_i['label'] == 0]
    # Correct labels for training a classifier instead of an outlier detector
    df_val_i.label = i
    dfs_val.append(df_val_i)

df_val = pd.concat(dfs_val, ignore_index=True)

df_train_val = pd.concat([df_train, df_val], ignore_index=True)

# Replace this with your dataset if needed
dataset = CustomBinaryInsectDF(df_train_val, transform = transform, seed=42)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Step 2: Load pretrained ResNet-18 and remove final classification layer
# resnet = models.resnet18(pretrained=True)
# resnet.fc = torch.nn.Identity()  # remove final layer to get 512-dim features
# resnet.eval()

resnet = timm.create_model('resnet18', pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
resnet.eval()

# Extract features
features, labels, actuals, measurement_noises, label_noises = extract_features_from_dataloader(loader, resnet)

X = features  # shape: (n_samples, 512)
y = labels   # class labels: shape (n_samples,)

# Step 3: Sparse Logistic Regression for feature selection
clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
clf.fit(X, y)

# Select only informative features
selector = SelectFromModel(clf, prefit=True)
X_selected = selector.transform(X)
print(f"Reduced from {X.shape[1]} to {X_selected.shape[1]} features.")

# Step 4: Outlier detection
# clf = MCD()
# clf.fit(X_selected)
# outlier_scores = clf.decision_scores_
# outlier_labels = clf.labels_

pyod_model = PYOD(seed=42, model_name='MCD', contamination=0.2)
pyod_model.fit(X_selected, [])
anomaly_scores = pyod_model.predict_score(X_selected)
binary_scores = pyod_model.predict_binary_score(X_selected)
y_pred = binary_scores

# Step 5: (Optional) UMAP for visualization
X_umap = cuml.manifold.UMAP().fit_transform(X_selected)

visualize_y_true_vs_y_pred_umap(features, measurement_noises, label_noises, y_pred, filename=None, dirname=None, detected_label_noises=None)

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=binary_scores, cmap='coolwarm', s=10)
plt.colorbar(label='Outlier score (LOF)')
plt.title("UMAP projection colored by outlier score")
plt.show()
print()
