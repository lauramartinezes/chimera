import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Plotting Confusion Matrix after swap
def plot_conf_matrix_after_swap(df, all_predictions, insect_classes, subtitle=None, path='.'):
    conf_matrix = build_extended_confusion_matrix(df, all_predictions, insect_classes)
    plt.figure(figsize=(10, 8))

    # Normalize the values row-wise for color scaling
    normalized = conf_matrix.div(conf_matrix.sum(axis=1), axis=0).fillna(0)

    # Plot the heatmap with normalized colors but integer annotations
    sns.heatmap(
        normalized,
        annot=conf_matrix,          # Show actual counts
        fmt="d",                    # Format annotation as integer
        cmap="Blues",
        vmin=0, vmax=1              # Row-wise normalization is in [0,1]
    )

    plt.xticks(rotation=0)# 20, ha="right")  # Make x-axis labels easier to read
    plt.yticks(rotation=0)               # Keep y-axis labels straight

    plt.title(f"Confusion Matrix ({subtitle.title()})" if subtitle else "Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class and Nature")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'confusion_matrix_{subtitle}.svg'))


# Compute confusion matrix
def build_extended_confusion_matrix(df, all_predictions, insect_classes):
    true_cats, pred_cats = get_category_lists(insect_classes)
    conf_matrix = pd.DataFrame(0, index=true_cats, columns=pred_cats)

    for idx, row in df.iterrows():
        true_cat = get_true_category(row, insect_classes)
        pred_label = all_predictions[idx]
        pred_cat = get_predicted_category(true_cat, pred_label, insect_classes)
        if pred_cat in conf_matrix.columns:
            conf_matrix.loc[true_cat, pred_cat] += 1
    
    return conf_matrix


# Step 1: Define categories using df-consistent terminology
def get_category_lists(insect_classes):
    true_categories = []

    # Add 'good' and 'mislabeled' categories for each real class
    for i, cls in enumerate(insect_classes):
        other_cls = insect_classes[1 - i]  # pick the other class
        true_categories.append(f"{cls}_good")
        true_categories.append(f"{cls}_labeled_as_{other_cls}")

    # Add measurement noise categories
    for cls in insect_classes:
        true_categories.append(f"measurement_noise_labeled_as_{cls}")

    
    predicted_categories = insect_classes  # still only the predicted class names
    return true_categories, predicted_categories


# Step 2: Determine the true category of a sample   
def get_true_category(row, insect_classes):
    if row["measurement_noise"]:
        cls = insect_classes[row["label"]]  # still uses original label
        return f"measurement_noise_labeled_as_{cls}"

    cls = insect_classes[row["label"]]
    if row["mislabeled"]:
        # Swap logic for mislabeled
        if cls == insect_classes[0]:  # wmv mislabeled
            return f"{insect_classes[1]}_labeled_as_{insect_classes[0]}"
        else:  # v mislabeled
            return f"{insect_classes[0]}_labeled_as_{insect_classes[1]}"
    else:
        return f"{cls}_good"


# Step 3: Determine the predicted category based on classifier output and true nature
def get_predicted_category(true_cat, pred_label, insect_classes):
    return insect_classes[pred_label]


# Plotting Confusion Matrix after cleaning
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
