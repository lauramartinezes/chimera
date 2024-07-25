import numpy as np

class ALYZCorrelationType:
    def covariance2correlationInPlace(self, covariance):
        D = np.diag(np.diag(covariance)**(-1/2))
        cor = np.dot(np.dot(D, covariance), D)
        return cor

    def generateCorr(self, p):
        conditionNumber = 100
        HI = conditionNumber
        LO = 1.0
        aRandomVector = np.sort(LO + (HI - LO) * np.random.rand(p - 2))
        L = np.diag(np.concatenate(([1], aRandomVector, [conditionNumber])))

        Y = np.random.randn(p, p)
        U, _, _ = np.linalg.svd(np.dot(Y.T, Y))

        corEstimation = np.dot(np.dot(U, L), U.T)
        for iteration in range(100):
            corEstimation = self.covariance2correlationInPlace(corEstimation)
            U, S, _ = np.linalg.svd(corEstimation)
            L = S
            oldConditionNumber = L[0] / L[-1]

            if abs(conditionNumber - oldConditionNumber) < 1e-10:
                break

            L[0] = conditionNumber * L[-1]
            corEstimation = np.dot(np.dot(U, np.diag(L)), U.T)

        corEstimation = self.covariance2correlationInPlace(corEstimation)
        return corEstimation

    def generateLocation(self, p):
        return np.zeros(p)