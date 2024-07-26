import numpy as np
import matplotlib.pyplot as plt

from kernels import AutoRbfKernel, LinKernel
from alyz import NewDataModel, ALYZCorrelationType, ClusterContamination
from kmrcd import halfkernel, kMRCD, Utils
from standarization import rz_scores
from scipy.io import savemat

def main():
    np.random.seed(5)
    
    color_BLUE = (0, 0.6980, 0.9333)
    color_ORANGE = (0.9333, 0.4627, 0)
    color_GREEN = (0, 0.6980, 0.1)
    color_GREY = (0.25, 0.25, 0.25)
    color_RED = (0.80, 0, 0)

    # Set the example to run
    runExample = 2

    # Set the contamination degree
    epsilon = 0.2

    # Set the expected amount of regular observations
    alpha = 0.75

    # Marker and font size
    fontSize = 10
    mSize = 8

    if runExample == 2:
        N = 1000
        N1 = int(np.ceil((1 - epsilon) * N))
        N2 = N - N1
        data = halfkernel(N1, N2, -20, 20, 40, 5, 0.6)
        mask = data[:, 2].astype(bool)
        x = data[:, :2]
        y = data[:, 2]
        ind = np.random.permutation(len(x))
        x = x[ind, :]
        y = y[ind]
        savemat('x_file.mat', {'x': x})
        savemat('y_file.mat', {'y': y})
        x = rz_scores(x)
        kModel = AutoRbfKernel(x)
    else:
        ndm = NewDataModel(ALYZCorrelationType(), ClusterContamination())
        x, _, _, idxOutliers = ndm.generateDataset(1000, 2, epsilon, 20)
        y = np.ones(1000)
        y[idxOutliers] = 0
        y = y.astype(bool)
        x = rz_scores(x)
        kModel = LinKernel()

    # Run the kMRCD algorithm
    poc = kMRCD(kModel)
    solution = poc.runAlgorithm(x, alpha)

    print('We have the following solution:')
    print(solution)

    rho = solution['rho']
    scfac = solution['scfac']

    # Visualization
    rr, cc = np.meshgrid(np.arange(-5, 5.1, 0.1), np.arange(-5, 5.1, 0.1))
    yy = np.column_stack([rr.ravel(), cc.ravel()])

    Kx = kModel.compute(x[solution['hsubsetIndices'], :], x[solution['hsubsetIndices'], :])
    nx = Kx.shape[0]
    Kt = kModel.compute(yy, x[solution['hsubsetIndices'], :])
    Kc = Utils.center(Kx)
    Kt_c = Utils.center(Kx, Kt)
    Ktt_diag = np.diag(kModel.compute(yy, yy))  # Precompute
    Kxx = Ktt_diag - (2 / nx) * np.sum(Kt, axis=1) + (1 / nx ** 2) * np.sum(Kx)
    denominator = (1 - rho) * scfac * Kc + nx * rho * np.eye(nx)
    denominator_inv = np.linalg.inv(denominator)
    Kt_c_divided = np.dot(Kt_c, denominator_inv)
    smdMesh = (1 / rho) * (Kxx - (1 - rho) * scfac * np.sum((Kt_c_divided * Kt_c), axis=1))
                

    ss = y.astype(bool)

    plt.rcParams.update({'font.size': fontSize})

    fig = plt.figure(1)
    plt.contour(rr, cc, np.log(smdMesh).reshape(rr.shape), 20, cmap='bone')
    plt.plot(x[y > 0, 0], x[y > 0, 1], '.', color=color_GREY, markersize=mSize)
    plt.plot(x[y == 0, 0], x[y == 0, 1], '.', color=color_RED, markersize=mSize)
    plt.ylim([-4, 4])
    plt.title('Input dataset with marked outliers')
    plt.savefig('images/inputdataset.png')
    plt.show()

    fig = plt.figure(2)
    plt.contour(rr, cc, np.log(smdMesh).reshape(rr.shape), 20, cmap='bone')
    plt.plot(x[:, 0], x[:, 1], '.', color=color_GREY, markersize=mSize)
    plt.plot(x[solution['hsubsetIndices'], 0], x[solution['hsubsetIndices'], 1], '.', color=color_GREEN, markersize=mSize)
    plt.ylim([-4, 4])
    plt.title('the h-subset')
    plt.savefig('images/hsubset.png')
    plt.show()

    fig = plt.figure(3)
    plt.contour(rr, cc, np.log(smdMesh).reshape(rr.shape), 20, cmap='bone')
    plt.plot(x[:, 0], x[:, 1], '.', color=color_BLUE, markersize=mSize)
    plt.plot(x[solution['flaggedOutlierIndices'], 0], x[solution['flaggedOutlierIndices'], 1], '.', color=color_ORANGE, markersize=mSize)
    plt.ylim([-4, 4])
    plt.title('Flagged outliers')
    plt.savefig('images/result.png')
    plt.show()

if __name__ == "__main__":
    main()