import numpy as np
from scipy.spatial.distance import pdist, squareform

class AutoRbfKernel:
    def __init__(self, x):
        # Compute the pairwise distances and their median to set sigma
        distances = pdist(x, 'sqeuclidean')
        self.sigma = np.sqrt(np.median(distances))
        print(f"AutoRbfKernel: Sigma = {self.sigma}")

    def compute(self, Xtrain, Xtest=None):
        if Xtest is None:
            Xtest = Xtrain
        
        n = Xtrain.shape[0]
        m = Xtest.shape[0]
        
        Ka = np.tile(np.sum(Xtrain**2, axis=1, keepdims=True), (1, m)) 
        Kb = np.tile(np.sum(Xtest**2, axis=1, keepdims=True), (1, n))
        
        K = Ka + Kb.T - 2 * np.dot(Xtrain, Xtest.T)
        K = np.exp(-K / (2 * self.sigma**2))
        
        return K