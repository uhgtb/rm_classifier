"""
Statistical Models for Likelihood Evaluations
=============================================
Author: Johann Luca Kastner
Date: 15/09/2025
License: MIT

"""
import numpy as np
from numba import njit
import scipy.linalg as linalg


@njit
def logl_uncorrelated_normal(X, means, variances):
    """
    Compute log-likelihoods for a set of diagonal-covariance Gaussians.
    This is equivalent to the negative chi^2 value

    Args:
        X (np.ndarray): input with shape (n_samples, n_features)
        means (np.ndarray): cluster mean traces (n_models, n_features)
        variances (np.ndarray): cluster variances (n_models, n_features)

    Returns: 
        logl (np.ndarray): log-likelihoods (n_models, n_samples)
    """
    n_samples = X.shape[1]
    n_features = X.shape[1]

    
    X = X[None, :, :]               # shape (1, n_samples, n_features)
    means = means[:, None, :]       # shape (n_models, 1, n_features)
    variances = variances[:, None, :]  # shape (n_models, 1, n_features)

    log_det = np.sum(np.log(variances), axis=2)  # (n_models, 1)
    quad_term = np.sum((X - means)**2 / variances, axis=2)  # (n_models, n_samples)

    logl = -0.5 * (quad_term + log_det + X.shape[2] * np.log(2 * np.pi))  # (n_models, n_samples)
    return logl/n_features

def cholesky_factors(cov, eps=1e-8):
    """ Compute the Cholesky factors of covariance matrices, adding a small value to the diagonal if necessary.
    
    Args:
        cov (np.ndarray): Cluster covariance matrix of shape (n_features, n_features)
        eps (float): Small value to add to the diagonal for numerical stability

    Returns:
        W (np.ndarray): Whitening matrix of shape (n_features, n_features)
        log_det (float): Log determinant of the covariance matrix
    """
    d = np.shape(cov)[1]
    try:
        cov += eps * np.eye(d)*np.trace(cov)
        L = linalg.cholesky(cov, lower=True)
        W = linalg.solve(L, np.eye(L.shape[0])) # Whitening matrix
    except:
        print("Cholesky decomposition failed, adding small value to diagonal")
        cov += eps * np.eye(d)*np.trace(cov)*100
        L = linalg.cholesky(cov, lower=True)
        W = linalg.solve(L, np.eye(L.shape[0]))
    log_det = 2 * np.sum(np.log(np.diag(L)))  # log(det(Σ)) from Cholesky

    return np.array(W), np.array(log_det)

@njit
def logl_normal(X, means, L_invs, log_dets):
    """
    Fast log-likelihood evaluation, assuming a multivariate Gaussian distribution, using precomputed means and cholesky factors.
    
    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features)
        means (np.ndarray): Cluster means of shape (n_models, n_features)
        L_invs (np.ndarray): Inverses of Cholesky factors of shape (n_models, n_features, n_features)
        log_dets (np.ndarray): Log determinants of covariance matrices of shape (n_models,)
    
    Returns:
        logl (np.ndarray): Log-likelihoods of shape (n_models, n_samples)
    """

    n_models, d = means.shape
    n_samples = X.shape[0]
    n_features = X.shape[1]
    const = d * np.log(2 * np.pi)
    logl = np.empty((n_models, n_samples))

    for i in range(n_models):
        mean = means[i]
        Linv = L_invs[i]
        log_det = log_dets[i]

        for j in range(n_samples):
            x = X[j]
            diff = x - mean
            z = Linv @ diff 
            mahal = np.sum(z ** 2)
            logl[i, j] = -0.5 * (mahal + log_det + const)

    return logl/n_features

