import numpy as np
from scipy.stats import chi2

def unimcd(y, h, centered=0):
    """
    Computes the MCD estimator of a univariate data set.
    
    Parameters:
    - y: np.ndarray
        Univariate data set.
    - h: int
        Number of observations in the subset.
    - centered: int, optional
        Indicates whether the data should be centered.
        
    Returns:
    - tmcd: float
        MCD location estimate.
    - smcd: float
        MCD scale estimate.
    - weights: np.ndarray
        Weights based on the MCD.
    - initmean: float
        Initial mean estimate.
    - initcov: float
        Initial covariance estimate.
    - rawrd: np.ndarray
        Raw robust distances.
    - Hopt: np.ndarray
        Indices of the subset with the smallest variance.
    """
    ncas = len(y)
    len_y = ncas - h + 1
    
    xorig = y.copy()

    if len_y == 1:
        # Only one subset possible
        if centered == 1:
            tmcd = 0
        else:
            tmcd = np.mean(y)
        smcd = np.sqrt(np.var(y, ddof=1))
        weights = np.ones(len(y))
    else:
        if centered == 1:
            y_sorted = np.sort(np.abs(y))
            Hopt = np.argsort(np.abs(y))[:h]
            initmean = 0
            initcov = np.sum(y_sorted[:h] ** 2) / (h - 1)
        else:
            y_sorted = np.sort(y)
            ay = np.zeros(len_y)
            ay[0] = np.sum(y_sorted[:h])
            for samp in range(1, len_y):
                ay[samp] = ay[samp - 1] - y_sorted[samp - 1] + y_sorted[samp + h - 1]
            ay2 = ay ** 2 / h
            sq = np.zeros(len_y)
            sq[0] = np.sum(y_sorted[:h] ** 2) - ay2[0]
            for samp in range(1, len_y):
                sq[samp] = sq[samp - 1] - y_sorted[samp - 1] ** 2 + y_sorted[samp + h - 1] ** 2 - ay2[samp] + ay2[samp - 1]
            sqmin = np.min(sq)
            indices = np.where(sq == sqmin)[0]
            Hopt = np.argsort(y)[indices[0]:indices[0] + h]
            initmean = np.median(ay[indices]) / h
            initcov = sqmin / (h - 1)

        # Calculating consistency factor
        res = (xorig - initmean) ** 2 / initcov
        sortres = np.sort(res)
        factor = sortres[h - 1] / chi2.ppf(h / ncas, 1)
        initcov = factor * initcov
        res = (xorig - initmean) ** 2 / initcov
        quantile = chi2.ppf(0.975, 1)
        weights = res <= quantile
        rawrd = np.sqrt(res)

        # Reweighting procedure
        if weights.ndim == 1:
            weights = weights.astype(float)
        if centered == 1:
            tmcd = 0
        else:
            tmcd = np.sum(xorig * weights) / np.sum(weights)
        smcd = np.sqrt(np.sum((xorig - tmcd) ** 2 * weights) / (np.sum(weights) - 1))

    return tmcd, smcd, weights, initmean, initcov, rawrd, Hopt
