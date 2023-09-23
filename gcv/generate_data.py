import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from scipy.sparse.linalg import eigsh

def create_toeplitz_cov_mat(sigma_sq, first_column_except_1):
    '''
    Parameters
    ----------
    sigma_sq: float
        scalar_like,
    first_column_except_1: array
        1d-array, except diagonal 1.
    
    Return:
    ----------
        2d-array with dimension (len(first_column)+1, len(first_column)+1)
    '''
    first_column = np.append(1, first_column_except_1)
    cov_mat = sigma_sq * sp.linalg.toeplitz(first_column)
    return cov_mat


def ar1_cov(rho, n, sigma_sq=1):
    """
    Parameters
    ----------
    sigma_sq : float
        scalar
    rho : float
        scalar, should be within -1 and 1.
    n : int
    
    Return
    ----------
        2d-matrix of (n,n)
    """
    rho_tile = rho * np.ones([n - 1])
    first_column_except_1 = np.cumprod(rho_tile)
    cov_mat = create_toeplitz_cov_mat(sigma_sq, first_column_except_1)
    return cov_mat

def generate_data(n, phi, rho=1., sigma=1., n_test=None):
    p = int(n*phi)

    if rho>0.:
        beta0 = np.random.randn(p)
        beta0 = beta0 / np.sqrt(p) * rho
    else:
        beta0 = np.zeros(p)
        
    X = np.random.randn(n,p)
    Y = X@beta0[:,None]

    if n_test is None:
        n_test = n
    X_test = np.random.randn(n_test,p)
    Y_test = X_test@beta0[:,None]
    
    if sigma>0.:
        Y += np.random.randn(n,1)*sigma
        Y_test += np.random.randn(n_test,1)*sigma

    return beta0, X, Y, X_test, Y_test


def generate_data_ar1(p, phi, rho_ar1, sigma=1, 
                      top_k=5, which='LM', misspecified=False, n_test=None):
    '''
    Parameters
    ----------
    which: str
        'LM', 'SM', and 'LSM' for k largest eigenvalues, k smallest eigenvalues, and both of them (in magnitude).
    '''
    n = int(p/phi)
        
    Sigma = ar1_cov(rho_ar1, p)
    
    if sigma<np.inf:
        rho2 = 0.
        
        if which != 'SM':
            _, beta1 = eigsh(Sigma, k=top_k, which = 'LM')
            rho2 += (1-rho_ar1**2)/(1-rho_ar1)**2/top_k
            beta1 = np.mean(beta1, axis=-1)
        else:
            beta1 = np.zeros(p)
            
        
        if which != 'LM':
            _, beta2 = eigsh(Sigma, k=top_k, which = 'SM')
            rho2 += (1-rho_ar1**2)/(1+rho_ar1)**2/top_k
            beta2 = np.mean(beta2, axis=-1)
        else:
            beta2 = np.zeros(p)
            
        beta0 = beta1 + beta2
        
    else:
        beta0 = np.zeros(p)
        rho2 = 0.
    
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
    Y = X@beta0[:,None]
    
    if n_test is None:
        n_test = n
    X_test = np.random.multivariate_normal(np.zeros(p), Sigma, size=n_test)
    Y_test = X_test@beta0[:,None]
    
    if misspecified:
        Y += (np.mean(X**2, axis=-1) - np.trace(Sigma)/p)[:, None]
        Y_test += (np.mean(X_test**2, axis=-1) - np.trace(Sigma)/p)[:, None]
    
    if sigma>0.:
        Y += np.random.randn(n,1)*sigma
        Y_test += np.random.randn(n_test,1)*sigma
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