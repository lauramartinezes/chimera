import numpy as np

class LinKernel:
    def compute(self, Xtrain, Xtest=None):
        if Xtest is None:
            Xtest = Xtrain
        
        K = np.dot(Xtrain, Xtest.T)
        
        assert K.shape[0] == Xtrain.shape[0], "Kernel matrix row count mismatch with Xtrain"
        assert K.shape[1] == Xtest.shape[0], "Kernel matrix column count mismatch with Xtest"
        
        return K