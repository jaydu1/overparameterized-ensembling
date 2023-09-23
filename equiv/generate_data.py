import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.linalg import sqrtm


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
    if rho!=0.:
        rho_tile = rho * np.ones([n - 1])
        first_column_except_1 = np.cumprod(rho_tile)
        cov_mat = create_toeplitz_cov_mat(sigma_sq, first_column_except_1)
    else:
        cov_mat = np.identity(n)
    return cov_mat




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


def generate_data(p, phi, rho_ar1=1., sigma=1, 
                  func='quad', df=5., n_test=1000):
    n = int(p/phi)
        
    Sigma = ar1_cov(rho_ar1, p)
    
#     if cov=='ar1':
        
#     elif cov=='random':
#         s = np.diag(np.random.uniform(1., 2., size=p))
#         Q, _ = np.linalg.qr(np.random.rand(p, p))
#         Sigma = Q.T @ s @ Q
    if df==np.inf:
        Z = np.random.normal(size=(n,p))
        Z_test = np.random.normal(size=(n_test,p))
    else:
        Z = np.random.standard_t(df=df, size=(n,p)) / np.sqrt(df / (df - 2))
        Z_test = np.random.standard_t(df=df, size=(n_test,p)) / np.sqrt(df / (df - 2))
    
    Sigma_sqrt = sqrtm(Sigma)
    X = Z @ Sigma_sqrt
    X_test = Z_test @ Sigma_sqrt
    
    if sigma<np.inf:
        top_k = 5
        _, beta0 = eigsh(Sigma, k=top_k)
        rho2 = (1-rho_ar1**2)/(1-rho_ar1)**2/top_k
        beta0 = np.mean(beta0, axis=-1)        
    else:
        rho2 = 0.
        beta0 = np.zeros(p)

    Y = X@beta0[:,None]   
    Y_test = X_test@beta0[:,None]
    
    if func=='linear':
        pass
    elif func=='tanh':
        Y = np.tanh(Y)
        Y_test = np.tanh(Y_test)
    elif func=='quad':
        Y += (np.mean(X**2, axis=-1) - np.trace(Sigma)/p)[:, None]
        Y_test += (np.mean(X_test**2, axis=-1) - np.trace(Sigma)/p)[:, None]
    else:
        raise ValueError('Not implemented.')
    
    if sigma>0.:
        if df==np.inf:
            Y += np.random.normal(size=(n,1))
            Y_test += np.random.normal(size=(n_test,1))
        else:
            Y += np.random.standard_t(df=df, size=(n,1))
            Y_test += np.random.standard_t(df=df, size=(n_test,1))
        sigma = 1. if df==np.inf else np.sqrt(df / (df - 2))
    else:
        sigma = 0.

    return Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma**2


