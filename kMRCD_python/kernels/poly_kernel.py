import numpy as np

class PolyKernel:
    def __init__(self, degree):
        self.degree = degree

    def compute(self, Xtrain, Xtest=None):
        if Xtest is None:
            Xtest = Xtrain
        
        # Compute the polynomial kernel matrix
        K = (np.dot(Xtrain, Xtest.T) + 1) ** self.degree
        
        return K