import numpy as np
from numpy.linalg import svd, inv
from scipy.stats import multivariate_normal

class NewDataModel:
    def __init__(self, CorrelationType, ContaminationType):
        self.CorrelationType = CorrelationType
        self.ContaminationType = ContaminationType

    def generateA09cormatrix(self, d, correl=0.9):
        # generates A09 correlation matrix; correl = 0.9 is the default choice
        columns = np.tile(np.arange(1, d + 1), (d, 1))
        rows = np.tile(np.arange(1, d + 1).reshape(-1, 1), (1, d))
        Sigma = -correl * np.ones((d, d))
        Sigma = Sigma ** np.abs(columns - rows)
        return Sigma

    def generateDataset(self, n, p, eps, k):
        contaminationDegree = int(np.floor(n * eps))

        # Generate correlation matrix
        tCorrelation = self.CorrelationType.generateCorr(p)
        tLocation = self.CorrelationType.generateLocation(p)

        # Generate multivariate normal data with location zero and correlation
        samples = multivariate_normal.rvs(mean=tLocation, cov=tCorrelation, size=n)

        # Generate contamination
        U, _, _ = svd(tCorrelation)
        replacement = U[:, -1]

        delta = replacement - tLocation
        smd = np.sqrt(np.dot(delta.T, np.dot(inv(tCorrelation), delta)))
        replacement = replacement * (k / smd)

        # Sigma contamination
        # Sigma_outlier = k * self.generateA09cormatrix(p, 0.95)  # A09
        # replacement = tLocation
        Sigma_outlier = tCorrelation

        contamination = self.ContaminationType.generateContamination(
            contaminationDegree, p, replacement, tLocation, Sigma_outlier
        )

        cindices = np.random.permutation(n)[:contaminationDegree]
        samples[cindices, :] = contamination

        return samples, tCorrelation, tLocation, cindices