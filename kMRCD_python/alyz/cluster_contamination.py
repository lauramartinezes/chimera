import numpy as np
from scipy.stats import multivariate_normal

class ClusterContamination:
    def generateContamination(self, m, p, r, tLoc, tCor):
        # Generate contaminated data using multivariate normal distribution
        return multivariate_normal.rvs(mean=r + tLoc, cov=tCor, size=m)