import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from kernels import AutoRbfKernel, LinKernel
from kmrcd import halfkernel, kMRCD, Utils
from standarization import rz_scores
from scipy.io import savemat


def get_krmcd_outliers(x, alpha=0.75, kernel=AutoRbfKernel):
    """
    x: the data in a set of 2D feature vectors
    alpha: the expected amount of regular observations
    kernel:
    """
    kModel = kernel(x)
    poc = kMRCD(kModel)
    solution = poc.runAlgorithm(x, alpha)
    outliers = x[solution['flaggedOutlierIndices']]
    inliers = x[~solution['flaggedOutlierIndices']]
    return inliers, outliers, solution['flaggedOutlierIndices']

def main():
    np.random.seed(5)
    
    color_BLUE = (0, 0.6980, 0.9333)
    color_ORANGE = (0.9333, 0.4627, 0)
    color_GREEN = (0, 0.6980, 0.1)
    color_GREY = (0.25, 0.25, 0.25)
    color_RED = (0.80, 0, 0)

    # Set the expected amount of regular observations
    alpha = 0.75

    # Marker and font size
    fontSize = 10
    mSize = 4

    x = np.load(r'C:\Users\u0159868\Documents\repos\stickybugs-outliers\umap_sticky_test.npy')
    y = np.ones(len(x))
    kModel = AutoRbfKernel(x)

    # Run the kMRCD algorithm
    poc = kMRCD(kModel)

    file_name = 'solution_wmv.pkl'
    if os.path.exists(file_name):
        # Read the dictionary from the file
        with open(file_name, 'rb') as file:
            solution = pickle.load(file)
    else:
        solution = poc.runAlgorithm(x, alpha)
        # Save the dictionary to the file
        with open(file_name, 'wb') as file:
            pickle.dump(solution, file)

    print('We have the following solution:')
    for key, value in solution.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: [{value.shape} {value.dtype.name}]")
        elif isinstance(value, str):
            print(f"{key}: '{value}'")
        else:
            print(f"{key}: {round(value, 4)}")

    plt.rcParams.update({'font.size': fontSize})

    plt.plot(x[:, 0], x[:, 1], '.', color=color_GREY, markersize=mSize)
    plt.title('Input dataset with marked outliers')
    plt.savefig('images/inputdataset.png')
    plt.show()

    plt.plot(x[:, 0], x[:, 1], '.', color=color_GREY, markersize=mSize)
    plt.plot(x[solution['hsubsetIndices'], 0], x[solution['hsubsetIndices'], 1], '.', color=color_GREEN, markersize=mSize)
    plt.title('the h-subset')
    plt.savefig('images/hsubset.png')
    plt.show()

    plt.plot(x[:, 0], x[:, 1], '.', color=color_BLUE, markersize=mSize)
    plt.plot(x[solution['flaggedOutlierIndices'], 0], x[solution['flaggedOutlierIndices'], 1], '.', color=color_ORANGE, markersize=mSize)
    plt.title('Flagged outliers')
    plt.savefig('images/result.png')
    plt.show()
    print('')

if __name__ == "__main__":
    main()