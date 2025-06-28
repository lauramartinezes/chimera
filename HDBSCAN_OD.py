import numpy as np
import hdbscan

def HDBSCAN_OD(latent_space_points, min_cluster_size=5, min_samples=1):
    # Fit HDBSCAN model
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, 
        min_samples=min_samples, 
        gen_min_span_tree=True
    )
    labels = hdbscan_model.fit_predict(latent_space_points)

    # Count the number of points in each cluster (ignore noise points labeled as -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]

    predictions = np.where(labels != largest_cluster_label, 1, 0)

    return predictions  
