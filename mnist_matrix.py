import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_result_matrix(df):
    # Pivot the DataFrame into a 3x3 matrix
    heatmap_data = df.pivot('transform_percent', 'wrong_label_percent', 'test_accuracy')

    # Plot the heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", square=True, cbar_kws={'label': 'Test Accuracy'})
    plt.title('Test Accuracy Heatmap')
    plt.xlabel('Label Noise Percent')
    plt.ylabel('Measurement Noise Percent')
    plt.show()

if __name__ == '__main__':
    # Your DataFrame
    data = {
        'transform_percent': [0.00, 0.00, 0.00, 0.10, 0.10, 0.10, 0.25, 0.25, 0.25],
        'wrong_label_percent': [0.00, 0.10, 0.25, 0.00, 0.10, 0.25, 0.00, 0.10, 0.25],
        'test_accuracy': [98.595238, 96.285713, 94.345238, 96.369049, 96.321426, 94.047615, 95.130951, 93.142853, 88.190475]
    }

    df = pd.DataFrame(data)
    plot_result_matrix(df)
