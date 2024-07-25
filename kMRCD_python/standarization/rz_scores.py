import numpy as np
from standarization.unimcd import unimcd

def rz_scores(x):
    """
    Standardizes the data using robust Z-scores.
    
    Parameters:
    - x: np.ndarray
        Data matrix where rows are samples and columns are features.
        
    Returns:
    - x: np.ndarray
        Standardized data matrix.
    - mu: np.ndarray
        Robust means for each feature.
    - sigma: np.ndarray
        Robust scales for each feature.
    """
    n, p = x.shape
    mu = np.full(p, np.nan)
    sigma = np.full(p, np.nan)

    for featureIndex in range(p):
        non_nan_values = x[~np.isnan(x[:, featureIndex]), featureIndex]
        tmcd, smcd, _, _, _, _, _ = unimcd(non_nan_values, int(np.ceil(n * 0.5)))
        mu[featureIndex] = tmcd
        sigma[featureIndex] = smcd

    mask = (sigma < 1e-12) | np.isnan(sigma)
    sigma[mask] = 1
    mu[np.isnan(mu)] = 0
    
    x = (x - np.tile(mu, (n, 1))) / np.tile(sigma, (n, 1))

    
    return x
