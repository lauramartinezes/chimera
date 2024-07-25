import numpy as np

class AutoLaplacianKernel:
    def __init__(self, x):
        # Compute distances and sigma
        distances = np.linalg.norm(x[:, np.newaxis] - x, axis=2) ** 2
        self.sigma = np.sqrt(np.median(distances))
        print(f'AutoLaplacianKernel: Sigma = {self.sigma}')

    def compute(self, Xtrain, Xtest=None):
        if Xtest is None:
            Xtest = Xtrain
        
        n = Xtrain.shape[0]
        m = Xtest.shape[0]
        
        # Compute kernel matrix
        Ka = np.sum(np.abs(Xtrain), axis=1)[:, np.newaxis]  # Column vector
        Kb = np.sum(np.abs(Xtest), axis=1)  # Row vector
        K = Ka + Kb[np.newaxis, :]
        K = np.exp(-K / self.sigma)
        
        return K
