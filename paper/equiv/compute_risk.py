import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import pairwise_kernels
from fixed_point_sol import *
import scipy as sp
from scipy.optimize import root_scalar
from joblib import Parallel, delayed
from functools import reduce
import itertools

import warnings
warnings.filterwarnings('ignore') 


############################################################################
#
# Theoretical evaluation
#
############################################################################

def comp_theoretic_risk_M1(rho, sigma, lam, psi):
    '''
    Compute the theoretical risk for M=1.
    '''
    if lam==0.:
        if psi<1:
            B0 = 0.
            V0 = sigma**2 * psi / (1-psi)
        elif psi==1:
            B0 = 0.
            V0 = 0. if sigma==0 else np.inf
        else:
            v = v_phi_lam(psi, lam)
            B0 = rho**2 * (1 + vb_phi_lam(psi, lam, v=v)) / (1 + v)**2
            V0 = sigma**2 * psi * vv_phi_lam(psi, lam, v=v) / (1 + v)**2
        
    else:
        v = v_phi_lam(psi, lam)
        B0 = rho**2 * (1 + vb_phi_lam(psi,lam,v=v)) / (1 + v)**2
        V0 = sigma**2 * psi * vv_phi_lam(psi,lam,v=v) / (1+v)**2
    return B0, V0, sigma**2+B0+V0


def comp_theoretic_risk(rho, sigma, lam, phi, psi, M, replace=True):
    sigma2 = sigma**2
    if psi == np.inf:
        return np.full_like(M, rho**2, dtype=np.float32), np.zeros_like(M), np.full_like(M, rho**2 + sigma2, dtype=np.float32)
    else:
        v = v_phi_lam(psi,lam)
        tc = rho**2 * tc_phi_lam(psi, lam, v)
        
        if np.any(M!=1):
            tv = tv_phi_lam(phi, psi, lam, v) if replace else 0
        else:
            tv = 0
        if np.any(M!=np.inf):
            tv_s = tv_phi_lam(psi, psi, lam, v)
        else:
            tv_s = 0
            
        B = ((1 + tv_s) / M + (1 - 1/M) * (1 + tv)) * tc
        V = (tv_s / M + (1 - 1/M) * tv) * sigma2
        
        return B, V, B+V+sigma2



# def comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, psi, M, 
#                                 replace=True, v=None, tc=None, tv=None, tv_s=None):
#     if psi == np.inf:
#         rho2 = beta0.T @ Sigma @ beta0 #(1-rho_ar1**2)/(1-rho_ar1)**2/5
#         return np.full_like(M, rho2, dtype=np.float32), np.zeros_like(M), np.full_like(M, rho2 + sigma2, dtype=np.float32)
#     else:
#         if v is None:
#             v = v_general(psi, lam, Sigma)
#         if tc is None:
#             tc = tc_general(psi, lam, Sigma, beta0, v)
        
#         if np.any(M!=1):
#             if tv is None:
#                 tv = tv_general(phi, psi, lam, Sigma, v) if replace else 0
#         else:
#             tv = 0
#         if np.any(M!=np.inf):
#             if tv_s is None:
#                 tv_s = tv_general(psi, psi, lam, Sigma, v)
#         else:
#             tv_s = 0
            
#         B = ((1 + tv_s) / M + (1 - 1/M) * (1 + tv)) * tc
#         V = (tv_s / M + (1 - 1/M) * tv) * sigma2
        
#         return B, V, B+V+sigma2

def comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, psi, M, 
                                replace=True, v=None, 
                                tc=None, tc_s=None, tv=None, tv_s=None, ATA=None):
    if psi == np.inf:
        rho2 = beta0.T @ Sigma @ beta0 #(1-rho_ar1**2)/(1-rho_ar1)**2/5
        return np.full_like(M, rho2, dtype=np.float32), np.zeros_like(M), np.full_like(M, rho2 + sigma2, dtype=np.float32)
    else:
        if v is None:
            v = v_general(psi, lam, Sigma)
        
        if np.any(M!=1):
            if tv is None:
                tv = tv_general(phi, psi, lam, Sigma, v, ATA) if replace else 0
        else:
            tv = 0

        if np.any(M!=np.inf):
            if tv_s is None:
                tv_s = tv_general(psi, psi, lam, Sigma, v, ATA)
        else:
            tv_s = 0

        if tc is None:
            tc = tc_general(phi, psi, lam, Sigma, beta0, v, tv, ATA)
        if tc_s is None:
            tc_s = tc_general(psi, psi, lam, Sigma, beta0, v, tv_s, ATA)
        
        B = 1/M * tc_s + (1 - 1/M) * tc
        V = (tv_s / M + (1 - 1/M) * tv) * sigma2
        
        return B, V, B+V+sigma2



    
    
    
############################################################################
#
# Empirical evaluation
#
############################################################################
    
def compute_cov(Xs, lam):
    n,p = Xs.shape
    Sigma = Xs.T @ Xs / n
    if lam==0.:
        M = np.linalg.pinv(Sigma, hermitian=True)
        T1 = np.identity(p) - M @ Sigma
        T2 = M
    else:
        M = np.linalg.inv(Sigma + lam * np.identity(p))
        T1 = lam * M
        T2 = M @ Sigma @ M

    return Sigma, M/n, T1, T2/n

def comp_true_empirical_risk(X, Y, psi, lam, rho, sigma, beta0, M, replace=True):
    n,p = X.shape
    phi = p/n  
    
    assert replace is False or M==2
    
    if replace:
        k = int(p/psi)
        
        # randomly sample
        ids_list = [
            np.sort(np.random.choice(n,k,replace=False)) for _ in range(M)]
    else:
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)

    Sigma_list, M_list, T1_list, T2_list = zip(
        *[compute_cov(X[ids,:], lam) for ids in ids_list])
    T1_list = np.stack(T1_list, axis=0)
    T1_list = np.mean(T1_list, 0)
    
    if rho==0:
        B0 = 0
    else:
        B0 = beta0[:,None].T @ T1_list @ T1_list @ beta0[:,None]
        B0 = B0[0,0]
        
    if sigma==0:
        V0 = 0.
    else:
        V0 = sigma**2/M**2 * np.sum([np.trace(T2) for T2 in T2_list])

        if replace:
            for i in range(M):
                for j in range(i+1,M):
                    ids0 = np.intersect1d(ids_list[i],ids_list[j])
                    i0 = len(ids0)
                    if i0>0:
                        Sigma0 = X[ids0,:].T @ X[ids0,:] / i0
                        V0 += 2*sigma**2*i0/M**2 * np.trace(M_list[i] @ Sigma0 @ M_list[j])

    return B0 + V0 + sigma**2



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
#             regressor.fit(X/sqrt_k, Y/sqrt_k)
#             Y_hat = regressor.predict(X_test/sqrt_k) * sqrt_k
            regressor.fit(X, Y)
            Y_hat = regressor.predict(X_test)
        else:
            lam = param
            if method=='ridge':
                regressor = Ridgeless() if lam==0 else Ridge(alpha=lam, fit_intercept=False, solver='lsqr')
    #             regressor = Ridge(alpha=np.maximum(lam,1e-6), fit_intercept=False, solver='lsqr')
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

    

    
def get_estimator(X, Y, lam):
    sqrt_k = np.sqrt(X.shape[0])
    regressor = Ridgeless() if lam==0 else Ridge(alpha=lam, fit_intercept=False, solver='lsqr')
    regressor.fit(X/sqrt_k, Y/sqrt_k)
    return regressor.coef_.T
    
def comp_empirical_norm(X, Y, psi, beta0, method, param, M=2, replace=True):
    
    n,p = X.shape
    if replace:
        k = int(p/psi)
        ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
        
    if k==0:
        return 0
        
    with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
        res = parallel(
            delayed(get_estimator)(X[ids,:], Y[ids,:], param)
            for ids in ids_list
        )
    beta_hat = np.concatenate(res, axis=-1)
    beta_hat = np.mean(beta_hat, axis=1)
    beta_diff = beta_hat - beta0
    return np.sum(beta_hat**2), np.sum(beta_diff**2)


def comp_empirical_beta_stat(X, Y, psi, method, param, a_gau, a_nongau, M=2, replace=True):
    
    n,p = X.shape
    if replace:
        k = int(p/psi)
        ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
        
    if k==0:
        return np.zeros((M,7))

    with Parallel(n_jobs=8, verbose=0) as parallel:
        res = parallel(
            delayed(get_estimator)(X[ids,:], Y[ids,:], param)
            for ids in ids_list
        )
    beta_hat = np.concatenate(res, axis=-1)
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
    
    return stat




def comp_empirical_generalized_risk(
    X, Y, psi, method, param, 
    beta0=None, Sigma=None, Sigma_out=None, X_test=None, Y_test=None,
    M=2, replace=True):
    
    n,p = X.shape
    if replace:
        k = int(p/psi)
        ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
        
    if k==0:
        beta_hat = np.zeros((p, M))
    else:
        with Parallel(n_jobs=8, verbose=0) as parallel:
            res = parallel(
                delayed(get_estimator)(X[ids,:], Y[ids,:], param)
                for ids in ids_list
            )
        beta_hat = np.concatenate(res, axis=-1)
        beta_hat = np.cumsum(beta_hat, axis=-1) / np.arange(1,M+1)
    
    if X_test is None:
        beta_diff = beta_hat - beta0[:, None]

        stat = np.c_[
            np.mean(beta_diff**2, axis=0),
            np.mean((Y - X @ beta_diff)**2, axis=0),
            np.diagonal(beta_diff.T @ Sigma @ beta_diff),
            np.diagonal(beta_diff.T @ Sigma_out @ beta_diff)
        ]
    else:
        beta0_train = Ridgeless().fit(X/np.sqrt(n), Y/np.sqrt(n)).coef_.T
        beta0_test = Ridgeless().fit(
            X_test/np.sqrt(X_test.shape[0]), Y_test/np.sqrt(X_test.shape[0])).coef_.T
        beta_diff_train = beta_hat - beta0_train
        beta_diff_test = beta_hat - beta0_test

        stat = np.c_[
            np.mean(beta_diff_train**2, axis=0),
            np.mean((Y - X @ beta_diff_train)**2, axis=0),
            np.mean(beta_diff_test**2, axis=0),
            np.mean((Y_test - X_test @ beta_diff_test)**2, axis=0)
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
        evds = np.linalg.svd(X, compute_uv=False)
    
    tr = np.sum(1/(evds + lam)) / k    
    return tr


def est_lam(X, psi, method, param, M=100, replace=True):
    
    n,p = X.shape
    if replace:
        k = int(p/psi)
        ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)

    with Parallel(n_jobs=16, verbose=0) as parallel:
        res = parallel(
            delayed(get_tr)(X[ids,:], method, param)
            for ids in ids_list
        )

    tr_psi = np.mean(res)
    svds = np.linalg.svd(X/np.sqrt(n), compute_uv=False)
    tr_phi = lambda lam: tr_psi-(np.sum(1/(svds[:n]**2 + lam)) + np.maximum(n - len(svds), 0.)/lam)/n
    
    sol = root_scalar(tr_phi, bracket=[param if method=='ridge' else 1e-4,1e2], method='brentq')
    lam = sol.root
    return lam


