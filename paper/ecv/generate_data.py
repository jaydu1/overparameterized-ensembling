import numpy as np
import scipy as sp
from scipy.linalg import sqrtm, toeplitz, block_diag
from scipy.sparse.linalg import eigsh


def create_toeplitz_cov_mat(sigma_sq, first_column_except_1):
    '''Compute the Toepiltz covariance matrix.

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
    cov_mat = sigma_sq * toeplitz(first_column)
    return cov_mat


def ar1_cov(rho, n, sigma_sq=1):
    """Compute the covariance matrix of an AR(1) process.

    Parameters
    ----------
    rho : float
        scalar, should be within -1 and 1.
    n : int
        scalar, size of the matrix.
    sigma_sq : float
        scalar, multiplier of the covariance matrix.
    
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



def block_ar1_cov(rhos, n):
    """Compute the covariance matrix of block AR(1) processes.

    Parameters
    ----------
    rhos : float
        array, should be within -1 and 1.
    n : int
    
    Return
    ----------
        2d-matrix of (n,n)
    """
    n_block = len(rhos)
    s_block = n//n_block
    covs = []
    for i in range(n_block):
        ns = s_block if i<n_block-1 else n-s_block*(n_block-1)
        covs.append(ar1_cov(rhos[i], ns))
    cov_mat = sp.linalg.block_diag(*covs)
    return cov_mat



def generate_data(
    n, p, coef='sorted', func='linear',
    rho_ar1=0., sigma=1, df=np.inf, n_test=1000, sigma_quad=1.
):
    '''Generate simulated data.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Number of features.
    coef : str
        Available options are 'sorted', 'uniform', 'random', 'sparse-k', 'eig-k', with k being an integer.
    func : str
        Available options are 'linear', 'quad', 'tanh'.
    rho_ar1 : float
        Autoregressive coefficient.
    sigma : float
        Noise level.
    df : float
        Degree of freedom of the t-distribution for the noise.
    n_test : int
        Sample size for the test set.
    sigma_quad : float
        Strength of the quadratic function.
    '''
        
    if np.isscalar(rho_ar1):
        Sigma = ar1_cov(rho_ar1, p)
    else:
        Sigma = block_ar1_cov(rho_ar1, p)

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
        if coef.startswith('eig'):
            top_k = int(coef.split('-')[1])
            _, beta0 = eigsh(Sigma, k=top_k)
            
            beta0 = np.mean(beta0, axis=-1)
            if np.isscalar(rho_ar1):
                rho2 = (1-rho_ar1**2)/(1-rho_ar1)**2/top_k
            else:
                rho2 = np.linalg.norm(beta0)**2
        elif coef=='sorted':
            beta0 = np.random.normal(size=(p,))
            beta0 = beta0[np.argsort(-np.abs(beta0))]
            beta0 /= np.linalg.norm(beta0)
            rho2 = 1.   
        elif coef=='uniform':
            beta0 = np.ones(p,) / np.sqrt(p)
            rho2 = 1.
        elif coef=='random':
            beta0 = np.random.normal(size=(p,))
            beta0 /= np.linalg.norm(beta0)
            rho2 = 1.
        elif coef.startswith('sparse'):
            s = int(coef.split('-')[1])
            beta0 = np.zeros(p,)
            beta0[:s] = 1/np.sqrt(s)
            rho2 = 1.
            
    else:
        rho2 = 0.
        beta0 = np.zeros(p)

    Y = X@beta0[:,None]   
    Y_test = X_test@beta0[:,None]
    
    if func=='linear':
        pass
    elif func=='quad':
        Y += sigma_quad * (np.mean(X**2, axis=-1) - np.trace(Sigma)/p)[:, None]
        Y_test += sigma_quad * (np.mean(X_test**2, axis=-1) - np.trace(Sigma)/p)[:, None]
    elif func=='tanh':
        Y = np.tanh(Y)
        Y_test = np.tanh(Y_test)
    else:
        raise ValueError('Not implemented.')

    if sigma>0.:
        if df==np.inf:
            Y = Y + sigma * np.random.normal(size=(n,1))            
            Y_test += sigma * np.random.normal(size=(n_test,1))
        else:
            Y = Y + np.random.standard_t(df=df, size=(n,1))
            Y_test += np.random.standard_t(df=df, size=(n_test,1))
            sigma = np.sqrt(df / (df - 2))
    else:
        sigma = 0.

    return Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma**2

