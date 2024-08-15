import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels
import scipy as sp
from scipy.optimize import root_scalar
from joblib import Parallel, delayed
from functools import reduce
import itertools

import warnings
warnings.filterwarnings('ignore')


class Ridgeless(object):
    def __init__(self):        
        pass
    def fit(self, X, Y):
        self.coef_ = sp.linalg.lstsq(X, Y, check_finite=False, lapack_driver='gelsy')[0].T
        return self    
    def predict(self, X_test):
        return X_test @ self.coef_.T

def fit_predict(X, Y, X_test, method, param):
    sqrt_k = np.sqrt(X.shape[0])
    if method=='tree':
        regressor = DecisionTreeRegressor(max_features=1./3, min_samples_split=5)#, splitter='random')
        regressor.fit(X, Y)
        Y_hat = regressor.predict(X_test)
    elif method=='NN':
        regressor = MLPRegressor(random_state=0, max_iter=500).fit(X, Y)
        Y_hat = regressor.predict(X_test)
    elif method=='kNN':
        regressor = KNeighborsRegressor().fit(X, Y)
        Y_hat = regressor.predict(X_test)
    elif method=='logistic':
        clf = LogisticRegression(
            random_state=0, fit_intercept=False, C=1/np.maximum(param,1e-6)
        ).fit(X, Y.astype(int))
        Y_hat = clf.predict_proba(X_test)[:,1].astype(float)
    else:
        if method=='kernelridge':
            lam, kernel = param['lam'], param['kernel']
            degree = 3 if 'degree' not in param.keys() else param['degree']
            regressor = KernelRidge(alpha=X.shape[0]/X.shape[1]*lam, 
                                    kernel=kernel, coef0=0., degree=degree)
            regressor.fit(X, Y)
            Y_hat = regressor.predict(X_test)
        else:
            lam = param
            if method=='ridge':
                regressor = Ridgeless() if lam==0 else Ridge(alpha=lam, fit_intercept=False, solver='lsqr')
                regressor.fit(X/sqrt_k, Y/sqrt_k)
            else:
                regressor = Lasso(alpha=np.maximum(lam,1e-6), fit_intercept=False)
                regressor.fit(X, Y)
            Y_hat = regressor.predict(X_test)
    if len(Y_hat.shape)<2:
        Y_hat = Y_hat[:,None]
    return Y_hat


def comp_empirical_risk(X, Y, X_test, Y_test, psi, method, param, 
                        M=2, data_val=None, replace=True, 
                        return_allM=False, return_pred_diff=False):
    n,p = X.shape
    Y_test = Y_test.reshape((-1,1))    
    if data_val is not None:
        X_val, Y_val = data_val
        Y_val = Y_val.reshape((-1,1))
        Y_hat = np.zeros((Y_test.shape[0]+Y_val.shape[0], M))
        X_eval = np.r_[X_val, X_test]
    else:
        Y_hat = np.zeros((Y_test.shape[0], M))
        X_eval = X_test
        
    if replace:
        k = int(p/psi)
        ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
    
    if k==0:
        Y_hat = np.zeros((X_eval.shape[0], M))
    else:
        with Parallel(n_jobs=8, verbose=0) as parallel:
            res = parallel(
                delayed(fit_predict)(X[ids,:], Y[ids,:], X_eval, method, param)
                for ids in ids_list
            )
        Y_hat = np.concatenate(res, axis=-1)
    #     for j in range(M):
    #         ids = ids_list[j]
    #         Y_hat[:,j:j+1] = fit_predict(X[ids,:]/np.sqrt(len(ids)), Y[ids,:]/np.sqrt(len(ids)), X_eval, param)
        
    if return_allM:
        Y_hat = np.cumsum(Y_hat, axis=1) / np.arange(1,M+1)
        idM = np.arange(M)
    else:
        Y_hat = np.mean(Y_hat, axis=1, keepdims=True)
        idM = 0
        
    if return_pred_diff:
        risk_test = (Y_hat[-Y_test.shape[0]:,:]-Y_test)[:,idM]
    else:
        risk_test = np.mean((Y_hat[-Y_test.shape[0]:,:]-Y_test)**2, axis=0)[idM]
        
    if data_val is not None:
        risk_val = np.mean((Y_hat[:-Y_test.shape[0],:]-Y_val)**2, axis=0)[idM]
        return risk_val, risk_test
    else:
        return risk_test

    

def get_dof(X, method, lam, **kwargs):
    k = X.shape[0]
    
    if method=='ridge':
        svds = np.linalg.svd(X, compute_uv=False)
        evds = svds[:k]**2
        # evds = np.r_[evds, np.zeros(np.maximum(k - len(evds), 0))]
    else:    
        # kernel = kwargs['kernel']
        # kwargs.pop('kernel', None)
        # K = pairwise_kernels(X, metric=kernel, **kwargs)
        K = X
        evds = np.linalg.svd(K, compute_uv=False)
    
    dof = np.sum(evds/(evds + lam))
    return dof

def get_estimator(X, Y, lam, method='ridge', **kwargs):
    
    if method=='ridge':
        regressor = Ridgeless() if lam==0 else Ridge(alpha=lam, fit_intercept=False, solver='lsqr')
        regressor.fit(X, Y)
        coef = regressor.coef_.T
    elif method=='kernelridge':
        kernel = 'precomputed'# kwargs['kernel']        
        regressor = KernelRidge(alpha=lam, kernel=kernel, coef0=0.)
        regressor.fit(X, Y)
        coef = regressor.dual_coef_

    dof = get_dof(X, method, lam, **kwargs)
    return coef, dof
    
# def comp_empirical_norm(X, Y, psi, beta0, method, param, M=2, replace=True):
    
#     n,p = X.shape
#     if replace:
#         k = int(p/psi)
#         ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
#     else:
#         k = np.floor(n/M)
#         assert 1 <= k <= n
#         ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
        
#     if k==0:
#         return 0
        
#     with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
#         res = parallel(
#             delayed(get_estimator)(X[ids,:], Y[ids,:], param)
#             for ids in ids_list
#         )
#     beta_hat, dof = zip(*res)
#     dof = np.mean(dof)
#     beta_hat = np.concatenate(beta_hat, axis=-1)
#     beta_hat = np.mean(beta_hat, axis=1)
#     beta_diff = beta_hat - beta0
#     return np.sum(beta_hat**2), np.sum(beta_diff**2), dof


def sample_D(n, k, M, replace):

    if isinstance(replace, bool):
        weight_int = 1        
    else:
        weight_int = weight_int
        replace = False
        
    ids_list = []
    kp0 = int(np.ceil(k/weight_int))
    kp1 = int(np.floor(k//weight_int))
    np.random.seed(0)
    for j in range(M):
        ids = np.random.choice(n, kp0, replace=replace)
        if weight_int>1:
            ids = np.r_[ids, np.tile(ids[:kp1], weight_int-1)]
        ids = np.sort(ids)
        ids_list.append(ids)
    return ids_list



def comp_empirical_beta_stat(
    X, Y, k, method, param, a_gau, a_nongau, 
    M=2, replace=False, weight=None, **kwargs):
    n = X.shape[0]
    
    if k==0:
        return np.zeros((M,8))
    
    ids_list = sample_D(n, k, M, replace)#[np.sort(np.random.choice(n,k,replace=replace)) for j in range(M)]

    if method=='ridge':        
        X /= np.sqrt(k)
        Y /= np.sqrt(k)
    else:
        # X /= np.sqrt(p)
        # Y /= np.sqrt(p)
        # param *= np.sqrt(p)
        pass

    if weight is None:
        weight = np.ones((n,1))
    else:
        weight = weight.reshape(-1,1)
    X *= weight
    Y *= weight

    with Parallel(n_jobs=8, verbose=0) as parallel:
        res = parallel(
            delayed(get_estimator)(X[ids,:] if method=='ridge' else X[ids,:][:, ids], 
                Y[ids,:], param*np.sum(weight[ids]**2)/len(ids), method, **kwargs)
            for ids in ids_list
        )
    beta_hat, dof = zip(*res)
    if method=='kernelridge':
        stat = np.zeros((M,7))
    else:
        beta_hat = np.concatenate(beta_hat, axis=-1)
        beta_hat = np.cumsum(beta_hat, axis=-1) / np.arange(1,M+1)
        stat = np.c_[
            np.min(beta_hat, axis=0),
            np.max(beta_hat, axis=0),
            np.mean(beta_hat, axis=0),
            np.median(beta_hat, axis=0),
            np.std(beta_hat, axis=0),
            beta_hat.T @ a_gau,
            beta_hat.T @ a_nongau
        ]
        
    dof = np.cumsum(dof) / np.arange(1,M+1)
    stat = np.c_[stat,dof]
    
    return stat




def comp_empirical_generalized_risk(
    X, Y, k, method, param, 
    beta0=None, Sigma=None, Sigma_out=None, X_test=None, Y_test=None,
    M=2, replace=False, **kwargs):
    
    n, p = X.shape
    ids_list = sample_D(n, k, M, replace)
        
    if method=='ridge':        
        X /= np.sqrt(k)
        Y /= np.sqrt(k)
    else:
        # X /= np.sqrt(p)
        # Y /= np.sqrt(p)
        # param *= np.sqrt(p)
        pass

    if k==0:
        beta_hat = np.zeros((p, M))
        dof = np.zeros((M,))
    else:
        with Parallel(n_jobs=8, verbose=0) as parallel:
            res = parallel(
                delayed(get_estimator)(X[ids,:] if method=='ridge' else X[ids,:][:, ids], 
                    Y[ids,:], param, method, **kwargs)
                for ids in ids_list
            )
        beta_hat, dof = zip(*res)
        dof = np.cumsum(dof) / np.arange(1,M+1)
        beta_hat = np.concatenate(beta_hat, axis=-1)
        beta_hat = np.cumsum(beta_hat, axis=-1) / np.arange(1,M+1)
    
    if X_test is None:
        beta_diff = beta_hat - beta0[:, None]

        stat = np.c_[
            np.mean(beta_diff**2, axis=0),
            np.mean((Y - X @ beta_diff)**2, axis=0),
            np.diagonal(beta_diff.T @ Sigma @ beta_diff),
            np.diagonal(beta_diff.T @ Sigma_out @ beta_diff),
            dof,
        ]
    else:
        beta0_train = Ridgeless().fit(X*np.sqrt(k)/np.sqrt(n), Y*np.sqrt(k)/np.sqrt(n)).coef_.T
        beta0_test = Ridgeless().fit(
            X_test/np.sqrt(X_test.shape[0]), Y_test/np.sqrt(X_test.shape[0])).coef_.T
        beta_diff_train = beta_hat - beta0_train
        beta_diff_test = beta_hat - beta0_test

        stat = np.c_[
            np.mean(beta_diff_train**2, axis=0),
            np.mean((Y - X @ beta_hat)**2, axis=0),
            np.mean(beta_diff_test**2, axis=0),
            np.mean((Y_test - X_test @ beta_hat)**2, axis=0),
            dof
        ]
    return stat





def get_tr(X, method, param):
    k = X.shape[0]
    sqrt_k = np.sqrt(k)
    
    if method=='ridge':
        lam = param
        svds = np.linalg.svd(X/sqrt_k, compute_uv=False)
        evds = svds[:k]**2
        evds = np.r_[evds, np.zeros(np.maximum(k - len(evds), 0))]
    else:    
        lam, kernel = param['lam'], param['kernel']
        degree = 3 if 'degree' not in param.keys() else param['degree']
        K = pairwise_kernels(X/sqrt_k, metric=param['kernel'], degree=degree, coef0=0.)
        evds = np.linalg.svd(K, compute_uv=False)
    
    tr = np.sum(1/(evds + lam)) / k    
    return tr


def get_evds(X, method, **kwargs):
    k = X.shape[0]
    if method=='ridge':
        evds = np.linalg.svd(X / np.sqrt(k), compute_uv=False)[:k]**2
        evds = np.r_[evds, np.zeros(np.maximum(k - len(evds), 0))]
    else:
        kernel = kwargs['kernel']
        degree = 3 if 'degree' not in kwargs.keys() else kwargs['degree']
        K = pairwise_kernels(X / np.sqrt(k), metric=kwargs['kernel'], degree=degree, coef0=0.)
        evds = np.linalg.svd(K, compute_uv=False)
    return evds


def est_lam(X, psi, method, param=None, M=100, replace=False, **kwargs):
    
    n,p = X.shape
    k = int(p/psi)
    ids_list = [np.sort(np.random.choice(n,k,replace=replace)) for j in range(M)]

    # with Parallel(n_jobs=16, verbose=0) as parallel:
    #     res = parallel(
    #         delayed(get_tr)(X[ids,:], method, param)
    #         for ids in ids_list
    #     )

    # tr_psi = np.mean(res)
    # svds = np.linalg.svd(X/np.sqrt(n), compute_uv=False)
    # tr_phi = lambda lam: tr_psi-(np.sum(1/(svds[:n]**2 + lam)) + np.maximum(n - len(svds), 0.)/lam)/n
    
    # sol = root_scalar(tr_phi, bracket=[param if method=='ridge' else 1e-4,1e2], method='brentq')
    # lam = sol.root

    if method=='ridge':
        evds = np.linalg.svd(X / np.sqrt(n), compute_uv=False)[:n]**2
    else:
        kernel = kwargs['kernel']
        degree = 3 if 'degree' not in kwargs.keys() else kwargs['degree']
        K = pairwise_kernels(X / np.sqrt(n), metric=kwargs['kernel'], degree=degree, coef0=0.)
        evds = np.linalg.svd(K, compute_uv=False)
    obj = np.sum(evds / (evds + param))

    with Parallel(n_jobs=16, verbose=0) as parallel:
        res = parallel(
            delayed(get_evds)(X[ids,:], method, **kwargs)
            for ids in ids_list
        )
    evds = np.concatenate(res)
    func = lambda lam:obj - k * np.mean(evds/(evds + lam))
    sol = root_scalar(func, x0=0, x1=param)
    lam = sol.root
    
    return lam


def est_k_theory(nu, G=None, X=None, lam=0.):    
    
    if X is not None:
        # design matrix
        n = X.shape[0]
        evds = np.linalg.svd(X/np.sqrt(n), compute_uv=False)[:n]**2
    elif G is not None:
        # gram matrix
        n = G.shape[0]
        evds = np.linalg.svd(G, compute_uv=False)[:n]
    else:
        raise ValueError('Both X and G are None!')
    dof = np.sum(evds / (evds + nu))

    if nu == 0 and lam == 0:
        return X.shape[0]
    else:
        k = (1 - (1 - dof/n) * (1 - lam/nu)) * n
        return k
