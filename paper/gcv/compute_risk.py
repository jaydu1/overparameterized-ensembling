import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from fixed_point_sol import *
import scipy as sp
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

def comp_theoretic_risk_M1(rho, sigma, lam, phi_s):
    if lam==0.:
        if phi_s<1:
            B0 = 0.
            V0 = sigma**2 * phi_s / (1-phi_s)
        elif phi_s==1:
            B0 = 0.
            V0 = 0. if sigma==0 else np.inf
        else:
            v = v_phi_lam(phi_s, lam)
            B0 = rho**2 * (1 + vb_phi_lam(phi_s, lam, v=v)) / (1 + v)**2
            V0 = sigma**2 * phi_s * vv_phi_lam(phi_s, lam, v=v) / (1 + v)**2
        
    else:
        v = v_phi_lam(phi_s, lam)
        B0 = rho**2 * (1 + vb_phi_lam(phi_s,lam,v=v)) / (1 + v)**2
        V0 = sigma**2 * phi_s * vv_phi_lam(phi_s,lam,v=v) / (1+v)**2
    return B0, V0, sigma**2+B0+V0


def comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M, replace=True):
    sigma2 = sigma**2
    if phi_s == np.inf:
        return np.full_like(M, rho**2, dtype=np.float32), np.zeros_like(M), np.full_like(M, rho**2 + sigma2, dtype=np.float32)
    else:
        v = v_phi_lam(phi_s,lam)
        tc = rho**2 * tc_phi_lam(phi_s, lam, v)
        
        if np.any(M!=1):
            tv = tv_phi_lam(phi, phi_s, lam, v) if replace else 0
        else:
            tv = 0
        if np.any(M!=np.inf):
            tv_s = tv_phi_lam(phi_s, phi_s, lam, v)
        else:
            tv_s = 0
            
        B = ((1 + tv_s) / M + (1 - 1/M) * (1 + tv)) * tc
        V = (tv_s / M + (1 - 1/M) * tv) * sigma2
        
        return B, V, B+V+sigma2



def comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, phi_s, M, 
                                replace=True, v=None, tc=None, tv=None, tv_s=None):
    if phi_s == np.inf:
        rho2 = beta0.T @ Sigma @ beta0 #(1-rho_ar1**2)/(1-rho_ar1)**2/5
        return np.full_like(M, rho2, dtype=np.float32), np.zeros_like(M), np.full_like(M, rho2 + sigma2, dtype=np.float32)
    else:
        if v is None:
            v = v_general(phi_s, lam, Sigma)
        if tc is None:
            tc = tc_general(phi_s, lam, Sigma, beta0, v)
        
        if np.any(M!=1):
            if tv is None:
                tv = tv_general(phi, phi_s, lam, Sigma, v) if replace else 0
        else:
            tv = 0
        if np.any(M!=np.inf):
            if tv_s is None:
                tv_s = tv_general(phi_s, phi_s, lam, Sigma, v)
        else:
            tv_s = 0
            
        B = ((1 + tv_s) / M + (1 - 1/M) * (1 + tv)) * tc
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

def comp_true_empirical_risk(X, Y, phi_s, lam, rho, sigma, beta0, M, replace=True):
    n,p = X.shape
    phi = p/n  
    
    assert replace is False or M==2
    
    if replace:
        k = int(p/phi_s)
        
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
        self.beta = sp.linalg.lstsq(X, Y, check_finite=False, lapack_driver='gelsy')[0]
        
    def predict(self, X_test):
        return X_test @ self.beta

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


def comp_empirical_risk(X, Y, X_test, Y_test, phi_s, method, param, 
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
        k = int(p/phi_s)
        ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
        
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



############################################################################
#
# Generalized cross-validation
#
############################################################################
def fit_predict_gcv(X, Y, X_test, lam):
    k, p = X.shape
    if lam==0:
        beta = sp.linalg.lstsq(X, Y, check_finite=False, lapack_driver='gelsy')[0]
        Y_hat = X_test @ beta
        assert len(Y_hat.shape)==2
    else:
        clf = Ridge(alpha=lam, fit_intercept=False, solver='lsqr', tol=1e-8)
        clf.fit(X, Y)
        Y_hat = clf.predict(X_test)

    trS = p - np.trace(lam*np.linalg.pinv(X.T.dot(X)+lam*np.identity(p)))
    
    return trS, Y_hat
    

def comp_empirical_gcv(
    X, Y, X_test, Y_test, phi_s, lam, 
    M=2, replace=True, return_allM=False, full=True):
    n, p = X.shape
    phi = p/n
    Y_test = Y_test.reshape((-1,1))    
    Y_hat = np.zeros((Y_test.shape[0], M))
    
    if replace:
        k = int(p/phi_s)
        ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]        
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
    ids_union_list = [reduce(np.union1d, ids_list[:j+1]) for j in range(M)]
    n_ids = np.array([len(ids) for ids in ids_union_list])
    if full:
        ids_union_list = [np.arange(n) for _ in range(M)]
    else:        
        ids_union_list = [np.nonzero(np.in1d(ids_union_list[-1], ids))[0] for ids in ids_union_list]
    X_val, Y_val = X[ids_union_list[-1],:], Y[ids_union_list[-1],:]
    Y_val = Y_val.reshape((-1,1))
    X_eval = np.r_[X_val, X_test]
    

    if len(ids_list)>=2:
        with Parallel(n_jobs=8, temp_folder='~/tmp/', max_nbytes=None, verbose=0) as parallel:
            res = parallel(
                delayed(fit_predict_gcv)(
                    X[ids,:]/np.sqrt(len(ids)),
                    Y[ids,:]/np.sqrt(len(ids)), X_eval, lam)
                for ids in ids_list
            )
        trS, Y_hat = zip(*res)
    else:
        ids = ids_list[0]
        trS, Y_hat = fit_predict_gcv(
            X[ids,:]/np.sqrt(len(ids)), Y[ids,:]/np.sqrt(len(ids)), X_eval, lam)
        trS, Y_hat = [trS], [Y_hat]
    
    trS = np.array(trS)
    Y_hat = np.concatenate(Y_hat, axis=-1)

    if return_allM:
        Y_hat = np.cumsum(Y_hat, axis=1) / np.arange(1,M+1)
        trS = np.cumsum(trS) / np.arange(1,M+1)
        idM = np.arange(M)
    else:
        Y_hat = np.mean(Y_hat, axis=1, keepdims=True)
        trS = np.mean(trS, keepdims=True)
        idM = -1
    
    mse_train = (Y_hat[:-Y_test.shape[0],:]-Y_val)**2
    
    if return_allM:
        risk_val = np.array([np.mean(mse_train[ids_union_list[j],j]) for j in idM])
    else:
        risk_val = np.mean(mse_train[ids_union_list[-1],-1], axis=0)
    risk_test = np.mean((Y_hat[-Y_test.shape[0]:,:]-Y_test)**2, axis=0)[idM]
    
    if full:
        deno = (1 - trS[idM]/n)**2
    else:
        deno = (1 - trS[idM]/n_ids[idM])**2
    
    gcv = risk_val / deno
    
    return gcv, risk_test
    
    
def comp_theoretic_gcv_inf(Sigma, beta0, sigma2, lam, phi, phi_s, M=np.inf,
                          v=None, tc=None, tv=None, tv_s=None):

    if phi_s == np.inf:
        rho2 = beta0.T @ Sigma @ beta0
        return rho2 + sigma2, None, None, None, None
    else:
        if v is None:
            v = v_general(phi_s, lam, Sigma)
        if tc is None:
            tc = tc_general(phi_s, lam, Sigma, beta0, v)
        if tv is None:
            tv = tv_general(phi, phi_s, lam, Sigma, v)
        if tv_s is None:
            tv_s = tv_general(phi_s, phi_s, lam, Sigma, v)
        
        if lam==0. and phi_s<=1:
            lamv = 1 - phi_s
        else:
            lamv = lam * v
        D1 = lamv**2
        D2 = (1 - phi_s/(2*phi_s-phi)*( 1 - lamv))**2
        D = (1 - phi/phi_s*( 1 - lamv))**2
        
        Rte1 = (1 + tv_s) * (tc + sigma2)
        Rte2 = ((1 + tv_s) + (1 + tv) ) / 2 * (tc + sigma2)
        Rteinf = (1 + tv) * (tc + sigma2)
        Rtr1 = D1 * Rte1
        c1 = 2*(phi_s-phi)/(2*phi_s-phi)
        Rtr2 = 1/4 * c1 * Rte1 + \
            1/2 * D1 * phi_s/(2*phi_s-phi) * Rte1 + \
            1/2 * (c1 * lamv + (1-c1)*D1)* Rteinf
        
        if M==np.inf:
            gcv = (2*phi*(2*phi_s-phi)/phi_s**2 * Rtr2 + 2*(phi_s-phi)**2/phi_s**2 * Rte2 - phi/phi_s * Rtr1 - (phi_s-phi)/phi_s * Rte1) / D
        elif M==1:
            gcv = Rtr1/D1
        elif M==2:
            gcv = Rtr2/D2
        else:
            raise ValueError('No implementation for 2<M<inf.')
        return gcv, v, tc, tv, tv_s