import numpy as np

class RbfKernel:
    def __init__(self, bandwidth):
        self.sigma = bandwidth

    def update_kernel(self, bandwidth):
        self.sigma = bandwidth

    def compute(self, Xtrain, Xtest=None):
        if Xtest is None:
            Xtest = Xtrain
        
        n = Xtrain.shape[0]
        m = Xtest.shape[0]
        
        # Compute the RBF kernel matrix
        Ka = np.sum(Xtrain**2, axis=1)[:, np.newaxis]
        Kb = np.sum(Xtest**2, axis=1)
        K = Ka + Kb[np.newaxis, :] - 2 * np.dot(Xtrain, Xtest.T)
        K = np.exp(-K / (2 * self.sigma**2))
        
        return K