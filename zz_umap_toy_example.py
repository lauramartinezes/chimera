import cuml
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from time import time

# Create a toy dataset
X, _ = make_blobs(n_samples=1000, centers=5, random_state=42, n_features=50)

# UMAP using cuML (GPU)
start = time()
cuml_umap = cuml.manifold.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
X_cuml = cuml_umap.fit_transform(X)
end = time()
print(f"cUMAP (GPU) Time: {end - start:.4f} seconds")

# UMAP using UMAP-learn (CPU)
start = time()
umap_learn = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
X_umap = umap_learn.fit_transform(X)
end = time()
print(f"UMAP-learn (CPU) Time: {end - start:.4f} seconds")

# Plotting the results

# Plot cuML UMAP (GPU)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_cuml[:, 0], X_cuml[:, 1], c='blue', alpha=0.5)
plt.title("cUMAP (GPU)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

# Plot UMAP-learn (CPU)
plt.subplot(1, 2, 2)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c='green', alpha=0.5)
plt.title("UMAP-learn (CPU)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.tight_layout()
plt.show()