import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_conf_matrix_after_data_cleaning(row, title, method, path):
    """Plots a confusion matrix with row-wise color scaling. Drops all-zero columns (and rows)."""
    conf_matrix = pd.DataFrame(
        [
            [row['good_pred_good'],      row['good_pred_mislabels'],      row['good_pred_meas_noise']],
            [row['mislabels_pred_good'], row['mislabels_pred_mislabels'], row['mislabels_pred_meas_noise']],
            [row['meas_noise_pred_good'],row['meas_noise_pred_mislabels'],row['meas_noise_pred_meas_noise']]
        ],
        index=['Actual Good', 'Actual Mislabels', 'Actual Meas. Noise'],
        columns=['Pred Good', 'Pred Mislabels', 'Pred Meas. Noise']
    )

    # Drop all-zero columns (and rows, just in case)
    conf_matrix = conf_matrix.loc[conf_matrix.sum(axis=1).ne(0), conf_matrix.sum(axis=0).ne(0)]

    # If everything got dropped, skip gracefully
    if conf_matrix.empty:
        print(f"Skipping confusion matrix for {method} - {title}: all entries are zero.")
        return

    # Row-wise normalization (avoid divide-by-zero)
    row_max = conf_matrix.max(axis=1).replace(0, 1)
    conf_matrix_normalized = conf_matrix.div(row_max, axis=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_normalized, annot=conf_matrix, cmap="Blues", fmt=".0f")
    plt.title(f'Confusion Matrix - {method} - {title}')
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.savefig(os.path.join(path, f'confusion_matrix_{method}_{title}.svg'), bbox_inches="tight")
    plt.close()
