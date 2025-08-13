import numpy as np
from numba import njit

@njit
def logl_uncorrelated_normal(X, means, variances):
    """
    Compute log-likelihoods for a set of diagonal-covariance Gaussians.
    This is equivalent to the negative chi^2 value

    Parameters:
    - X: shape (n_samples, n_features)
    - means: shape (n_models, n_features)
    - variances: shape (n_models, n_features)

    Returns: 
    - logl: (n_models, n_samples)    
    """
    n_samples = X.shape[0]

    
    X = X[None, :, :]               # shape (1, n_samples, n_features)
    means = means[:, None, :]       # shape (n_models, 1, n_features)
    variances = variances[:, None, :]  # shape (n_models, 1, n_features)

    log_det = np.sum(np.log(variances), axis=2)  # (n_models, 1)
    quad_term = np.sum((X - means)**2 / variances, axis=2)  # (n_models, n_samples)

    logl = -0.5 * (quad_term + log_det + X.shape[2] * np.log(2 * np.pi))  # (n_models, n_samples)
    return logl/n_samples