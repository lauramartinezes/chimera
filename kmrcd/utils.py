import numpy as np
from scipy.stats import chi2, gamma, norm, median_abs_deviation
from scipy.linalg import svd

class Utils:

    @staticmethod
    def MCDcons(p, alpha):
        qalpha = chi2.ppf(alpha, p)
        caI = gamma.cdf(qalpha / 2, p / 2 + 1, scale=1) / alpha
        return 1 / caI

    @staticmethod
    def SpatialMedian(K):
        assert K.shape[0] == K.shape[1], "K must be a square matrix"
        n = K.shape[0]
        assert n > 0, "Matrix K must be non-empty"
        gamma = np.ones(n) / n
        nn = np.ones(n)
        for _ in range(10):
            # Assuming K, gamma, and nn are already defined
            diag_K = np.diag(K)  # Get the diagonal of K
            K_gamma = np.dot(K.T, gamma)  # K' * gamma
            gamma_K_gamma = np.dot(gamma.T, np.dot(K, gamma))  # gamma' * K * gamma

            # Compute the denominator
            denominator = diag_K - 2 * K_gamma + gamma_K_gamma

            # Compute weights
            w = nn / np.sqrt(denominator)
            gamma = w / np.sum(w)
        return gamma.reshape(-1, 1)

    @staticmethod
    def W_scale(x):
        n, p = x.shape
        def Wc(x): return ((1 - (x / 4.5) ** 2) ** 2 * (np.abs(x) < 4.5))
        sigma0 = median_abs_deviation(x)
        median_x_repmat = np.tile(np.median(x, axis=0), (n, 1))
        sigma0_repmat = np.tile(sigma0, (n, 1))
        w = Wc(x - median_x_repmat) / sigma0_repmat
        loc = np.diag(x.T @ w) / np.sum(w, axis=0)

        def rc(x): return np.minimum(x ** 2, 3 ** 2)
        sigma0 = median_abs_deviation(x)
        b = 3 * norm.ppf(3 / 4)
        nes = n * (2 * ((1 - b ** 2) * norm.cdf(b) - b * norm.pdf(b) + b ** 2) - 1)
        scale = (sigma0**2 / nes) * np.sum(rc((x - np.tile(loc, (n, 1))) / np.tile(sigma0, (n, 1))))
        return np.sqrt(scale)

    @staticmethod
    def kernel_OGK(hIndices, K):
        n = K.shape[0]
        n_h = len(hIndices)
        K_h = K[np.ix_(hIndices, hIndices)]
        Kt = K[:, hIndices]

        # Calculate Covariance matrix
        K_tilde = Utils.center(K_h)
        U, S_F, _ = svd(K_tilde)
        mask = S_F > 1000 * np.finfo(float).eps
        U = U[:, mask]
        S_F = S_F[mask]
        U_scaled = U / np.tile(np.sqrt(S_F.T), (U.shape[0], 1))
        
        # Step 1: Compute E and B
        o = np.ones((n_h, 1))
        gamma = o / n_h
        K_Phi_PhiTilde = Kt - np.outer(Kt @ gamma, o)
        B_F = K_Phi_PhiTilde @ U_scaled
        lambda_F = Utils.W_scale(B_F)

        # Step 2: Estimate the center
        K_Adapted = K_Phi_PhiTilde @ U_scaled @ np.diag(1 / lambda_F) @ U_scaled.T @ K_Phi_PhiTilde.T
        gamma_c = Utils.SpatialMedian(K_Adapted)

        # Step 3: Calculate Mahalanobis
        on = np.ones((n, 1))
        Kt_cCov = Kt - (on @ gamma_c.T @ Kt) - (Kt @ (gamma @ o.T)) + (gamma_c.T @ Kt @ gamma) * (on @ o.T)
        
        mahal_F = np.sum((Kt_cCov @ U_scaled @ np.diag(1 / lambda_F ** 2)) * (Kt_cCov @ U_scaled), axis=1)
        return np.argsort(mahal_F)

    @staticmethod
    def SpatialMedianEstimator(K, h):
        assert K.shape[0] == K.shape[1]
        n = K.shape[0]

        gamma = Utils.SpatialMedian(K)

        dist = np.diag(K) - 2 * np.sum(K * np.tile(gamma.T, (n, 1)), axis=1) + np.dot(np.dot(gamma.T, K), gamma).item()
        hIndices = np.argsort(dist)

        return Utils.kernel_OGK(hIndices[:int(np.ceil(n * h))], K), dist, gamma

    @staticmethod
    def SSCM(K):
        assert K.shape[0] == K.shape[1]
        n = K.shape[0]
        gamma = Utils.SpatialMedian(K)

        o = np.ones(n)
        Kc = K - np.outer(o, gamma) - np.outer(gamma, o) + np.outer(gamma, gamma) * np.outer(o, o)
        D = np.diag(1 / (np.diag(K) - 2 * np.sum(K * np.tile(gamma.T, (n, 1)), axis=1) + np.dot(np.dot(gamma.T, K), gamma).flatten()))
        K_tilde = np.sqrt(D) @ Kc @ np.sqrt(D)
        U, S_F, _ = svd(K_tilde)
        mask = S_F > 1000 * np.finfo(float).eps
        U = U[:, mask]
        S_F = S_F[mask]
        U_scaled = U / np.tile(np.sqrt(S_F.T), (U.shape[0], 1))

        K_Phi_PhiTilde = (K - np.outer(gamma, o)) @ np.sqrt(D)
        B_F = K_Phi_PhiTilde @ U_scaled
        lambda_F = Utils.W_scale(B_F)

        K_Adapted = K_Phi_PhiTilde @ U_scaled @ np.diag(1 / lambda_F) @ U_scaled.T @ K_Phi_PhiTilde.T
        gamma_c = Utils.SpatialMedian(K_Adapted)

        K_cCov = K - np.outer(o, gamma_c) - np.outer(gamma, o) + np.outer(gamma_c, gamma) * np.outer(o, o)
        K_cCov = K_cCov @ np.sqrt(D)
        mahal_F = np.sum((K_cCov @ U_scaled @ np.diag(1 / lambda_F ** 2)) * (K_cCov @ U_scaled), axis=1)
        return np.argsort(mahal_F), gamma

    @staticmethod
    def SDO(K, h):
        assert K.shape[0] == K.shape[1]
        n = K.shape[0]
        gamma = np.zeros(n)
        for _ in range(500):
            rindices = np.random.choice(n, 2, replace=False)
            lambda_ = np.zeros((n))
            lambda_[rindices[0]] = 1
            lambda_[rindices[1]] = -1
            a = K @ lambda_ / np.sqrt(lambda_.T @ K @ lambda_)
            sdo = np.abs(a - np.median(a)) / np.mean(np.abs(a - np.mean(a)))
            mask = sdo > gamma
            gamma[mask] = sdo[mask]
        
        hIndices = np.argsort(gamma)
        return Utils.kernel_OGK(hIndices[:int(np.ceil(n * h))], K), gamma

    @staticmethod
    def SpatialRank(K, h):
        n = K.shape[0]
        ook = np.zeros(n)
        for k in range(n):
            tmpA = K[k, k] - np.outer(K[:, k], np.ones(n)) - np.outer(np.ones(n), K[k, :]) + K
            tmpB = np.sqrt(K[k, k] + np.diag(K) - 2 * K[k, :])
            tmpC = np.outer(tmpB, tmpB)
            mask = np.ones((n, n), dtype=bool)
            mask[k, :] = False
            mask[:, k] = False
            ook[k] = np.sum(tmpA[mask] / tmpC[mask])
        ook = (1 / n) * np.sqrt(ook)
        hIndices = np.argsort(ook)
        return Utils.kernel_OGK(hIndices[:int(np.ceil(n * h))], K), ook

    @staticmethod
    def reweightedMean(y, mask):
        ncas = len(y)
        xorig = y
        h = len(mask)
        
        initmean = np.mean(xorig[mask])
        initcov = np.var(xorig[mask], ddof=1)
        
        res = (xorig - initmean) ** 2 / initcov
        sortres = np.sort(res)
        factor = sortres[h] / chi2.ppf(h / ncas, 1)
        initcov = factor * initcov
        res = (xorig - initmean) ** 2 / initcov
        quantile = chi2.ppf(0.975, 1)
        weights = res <= quantile
        rawrd = np.sqrt(res)
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
        tmcd = np.sum(xorig * weights) / np.sum(weights)
        smcd = np.sqrt(np.sum((xorig - tmcd) ** 2 * weights) / (np.sum(weights) - 1))
        return tmcd, smcd

    @staticmethod
    def center(omega, Kt=None):
        nb_data = omega.shape[0]
        Meanvec = np.mean(omega, axis=1)
        MM = np.mean(Meanvec)
        if Kt is None:
            Kc = omega - np.outer(Meanvec, np.ones(nb_data)) - np.outer(np.ones(nb_data), Meanvec.T) + MM
        else:
            nt = Kt.shape[0]
            MeanvecT = np.mean(Kt, axis=1)
            Kc = Kt - np.outer(np.ones(nt), Meanvec.T) - np.outer(MeanvecT, np.ones(nb_data)) + MM
        return Kc
