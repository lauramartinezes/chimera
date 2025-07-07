import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from umap import UMAP


class UmapHdbscanOD:
    def __init__(self, main_class_name=None, save_dir=None, seed=42):
        self.main_class_name = main_class_name
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.seed = seed

    def find_optimal_params(self, data, dims=[2 ** i for i in range(1, 9)],
                            min_cluster_size_values=[5, 10, 20], min_samples_values=[1, 5, 10]):
        scores = []

        # Check that none of the dims are larger than the number of features
        original_dim = data.shape[1]
        dims = [d for d in dims if d <= original_dim]

        for d in dims:
            if d == original_dim:
                print(f"Using original dimension {d} without UMAP...")
                embedding = data
            else:
                print(f"\nFitting UMAP with {d} dimensions...")
                reducer = UMAP(n_components=d, random_state=self.seed, n_jobs=1)
                embedding = reducer.fit_transform(data)

            for mcs in min_cluster_size_values:
                for ms in min_samples_values:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, gen_min_span_tree=True)
                    labels = clusterer.fit_predict(embedding)
                    rv = getattr(clusterer, 'relative_validity_', -1)
                    scores.append((d, mcs, ms, rv))
                    print(f"Dim: {d}, mcs: {mcs}, ms: {ms}, RV: {rv:.4f}")

        scores = np.array(scores)
        valid_scores = scores[scores[:, 3] > 0]
        if len(valid_scores) > 0:
            max_rv = np.max(valid_scores[:, 3])
            top_candidates = valid_scores[valid_scores[:, 3] == max_rv]
            best_idx = np.argmin(top_candidates[:, 0])  # Choose lowest dimension
            best_dim, best_mcs, best_ms = [int(x) for x in top_candidates[best_idx, :3]]
        else:
            print("No valid clustering found. Using highest relative validity among all scores as fallback.")
            best_idx = np.argmax(scores[:, 3])
            best_dim, best_mcs, best_ms = [int(x) for x in scores[best_idx, :3]]

        if self.save_dir:
            self._plot_scores(scores, dims, min_cluster_size_values, min_samples_values, best_dim)
            self._plot_final_clusters(data, best_dim, best_mcs, best_ms)

        print(f"\nOptimal: dim={best_dim}, mcs={best_mcs}, ms={best_ms}")
        return best_dim, best_mcs, best_ms

    def predict_outliers(self, data, dim, mcs, ms):
        original_dim = data.shape[1]
        if dim < original_dim:
            reducer = UMAP(n_components=dim, random_state=self.seed, n_jobs=1)
            embedding = reducer.fit_transform(data)
        else:
            embedding = data
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        labels = clusterer.fit_predict(embedding)

        # Find largest cluster
        unique, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(unique) == 0:
            return np.ones(len(labels))  # all outliers if no cluster
        largest_cluster = unique[np.argmax(counts)]
        return np.where(labels != largest_cluster, 1, 0)

    def _plot_scores(self, scores, dims, mcs_values, ms_values, best_dim):
        fig, ax = plt.subplots(figsize=(10, 6))
        scores = np.array(scores)
        for ms in ms_values:
            for mcs in mcs_values:
                subset = scores[(scores[:, 1] == mcs) & (scores[:, 2] == ms)]
                ax.plot(subset[:, 0], subset[:, 3], label=f"mcs={mcs}, ms={ms}", marker='o')

        ax.axvline(best_dim, color='red', linestyle='--', label=f"Best Dim: {best_dim}")
        ax.set(title="Relative Validity", xlabel="UMAP Dimensions", ylabel="Relative Validity")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, f"{self.main_class_name}_Relative_Validity.png"))
        fig.savefig(os.path.join(self.save_dir, f"{self.main_class_name}_Relative_Validity.svg"))

    def _plot_final_clusters(self, data, best_dim, best_mcs, best_ms):
        embedding = UMAP(n_components=best_dim, random_state=self.seed, n_jobs=1).fit_transform(data)
        embedding_2d = (
            UMAP(n_components=2, random_state=self.seed, n_jobs=1).fit_transform(data)
            if best_dim != 2 else embedding
        )

        labels = hdbscan.HDBSCAN(min_cluster_size=best_mcs, min_samples=best_ms).fit_predict(embedding)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        palette = sns.color_palette('hsv', n_clusters)
        colors = [palette[label] if label != -1 else (0.6, 0.6, 0.6) for label in labels]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=10, alpha=0.7)
        ax.set_title(f"HDBSCAN in 2D UMAP (Dim={best_dim}, mcs={best_mcs}, ms={best_ms})")
        ax.axis('off')

        fig.savefig(os.path.join(self.save_dir, f"{self.main_class_name}_Final_HDBSCAN_2D.png"))
        fig.savefig(os.path.join(self.save_dir, f"{self.main_class_name}_Final_HDBSCAN_2D.svg"))


if __name__ == "__main__":
    print('Example usage of UmapHdbscanOD class')
    # Example usage:
    # umap_hdbscan_od = UmapHdbscanOD("beetle", save_dir="./results")

    # features, labels, real_labels, noise, mislabeled = umap_hdbscan_od.extract_features(
    #     dataloader=test_loader, model=model, device="cuda"
    # )

    # best_dim, best_mcs, best_ms = umap_hdbscan_od.find_optimal_params(features)

    # outliers = umap_hdbscan_od.predict_outliers(features, best_dim, best_mcs, best_ms)
