import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from scipy.sparse.linalg import eigsh

def create_toeplitz_cov_mat(sigma_sq, first_column_except_1):
    """
    sigma_sq: scalar_like,
    first_column_except_1: array_like, 1d-array, except diagonal 1.
    return:
        2d-array with dimension (len(first_column)+1, len(first_column)+1)
    """
    first_column = np.append(1, first_column_except_1)
    cov_mat = sigma_sq * sp.linalg.toeplitz(first_column)
    return cov_mat


def ar1_cov(rho, n, sigma_sq=1):
    """
    sigma_sq: scalar
    rho: scalar, should be within -1 and 1.
    n: integer
    return: 2d-matrix of (n,n)
    """
    rho_tile = rho * np.ones([n - 1])
    first_column_except_1 = np.cumprod(rho_tile)
    cov_mat = create_toeplitz_cov_mat(sigma_sq, first_column_except_1)
    return cov_mat

def generate_data(n, phi, rho=1., sigma=1.):
    p = int(n*phi)

    if rho>0.:
        beta0 = np.random.randn(p)
#         beta0 = beta0 / np.sqrt(np.sum(beta0**2)) * rho
        beta0 = beta0 / np.sqrt(p) * rho
    else:
        beta0 = np.zeros(p)
        
    X = np.random.randn(n,p)
    Y = X@beta0[:,None]

    X_test = np.random.randn(n,p)
    Y_test = X_test@beta0[:,None]
    
    if sigma>0.:
        Y += np.random.randn(n,1)*sigma
        Y_test += np.random.randn(n,1)*sigma

    return beta0, X, Y, X_test, Y_test


def generate_data_ar1(p, phi, rho_ar1, sigma=1, misspecified=False):
    n = int(p/phi)
        
    Sigma = ar1_cov(rho_ar1, p)
    
    if sigma<np.inf:
        top_k = 5
        _, beta0 = eigsh(Sigma, k=top_k)
        beta0 = np.mean(beta0, axis=-1)
        rho2 = (1-rho_ar1**2)/(1-rho_ar1)**2/top_k
    else:
        beta0 = np.zeros(p)
        rho2 = 0.
    
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
    Y = X@beta0[:,None]

    X_test = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
    Y_test = X_test@beta0[:,None]
    
    if misspecified:
        Y += (np.mean(X**2, axis=-1) - np.trace(Sigma)/p)[:, None]
        Y_test += (np.mean(X_test**2, axis=-1) - np.trace(Sigma)/p)[:, None]
    
    if sigma>0.:
        Y += np.random.randn(n,1)*sigma
        Y_test += np.random.randn(n,1)*sigma
    else:
        sigma = 0.

    return Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma**2




def generate_data_bi(p, phi, rmax, rho=1, sigma=1.):
    n = int(p/phi)
        
    Sigma = np.identity(p)
    Sigma[:int(p/2),:int(p/2)] *= rmax
    
    if rho>0.:
        #beta0 = np.random.multivariate_normal(np.zeros(p), np.diag(1./np.diag(Sigma)), size=1)[0,:]
        #beta0 = beta0/np.linalg.norm(beta0)*rho
        beta0 = np.zeros(p)
        beta0[0] = rho
    else:
        beta0 = np.zeros(p)
    
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
    Y = X@beta0[:,None]

    X_test = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
    Y_test = X_test@beta0[:,None]
    
    if sigma>0.:
        Y += np.random.randn(n,1)*sigma
        Y_test += np.random.randn(n,1)*sigma
    else:
        sigma = 0

    return Sigma, beta0, X, Y, X_test, Y_test