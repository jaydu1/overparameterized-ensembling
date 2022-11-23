import numpy as np
from sklearn.linear_model import Ridge
from fixed_point_sol import *
import scipy as sp
from joblib import Parallel, delayed


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
        return rho**2, 0., rho**2 + sigma2
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



def comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, phi_s, M, replace=True):
    if phi_s == np.inf:
        rho2 = beta0.T @ Sigma @ beta0 #(1-rho_ar1**2)/(1-rho_ar1)**2/5
        return rho2, 0, rho2 + sigma2
    else:
        v = v_general(phi_s, lam, Sigma)
        tc = tc_general(phi_s, lam, Sigma, beta0, v)
        
        if np.any(M!=1):
            tv = tv_general(phi, phi_s, lam, Sigma, v) if replace else 0
        else:
            tv = 0
        if np.any(M!=np.inf):
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



def fit_predict(X, Y, X_test, lam):
    if lam==0:
        beta = sp.linalg.lstsq(X, Y, check_finite=False, lapack_driver='gelsy')[0]
        Y_hat = X_test @ beta
        assert len(Y_hat.shape)==2
    else:
        clf = Ridge(alpha=lam, fit_intercept=False, solver='lsqr')
        clf.fit(X, Y)
        Y_hat = clf.predict(X_test)
    return Y_hat


def comp_empirical_risk(X, Y, X_test, Y_test, 
                        phi_s, lam, M=2, data_val=None, replace=True, return_allM=False):
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
            delayed(fit_predict)(
                X[ids,:]/np.sqrt(len(ids)), 
                Y[ids,:]/np.sqrt(len(ids)), X_eval, lam)
            for ids in ids_list
        )
    Y_hat = np.concatenate(res, axis=-1)
#     for j in range(M):
#         ids = ids_list[j]
#         Y_hat[:,j:j+1] = fit_predict(X[ids,:]/np.sqrt(len(ids)), Y[ids,:]/np.sqrt(len(ids)), X_eval, lam)
        
    if return_allM:
        Y_hat = np.cumsum(Y_hat, axis=1) / np.arange(1,M+1)
        idM = np.arange(M)
    else:
        Y_hat = np.mean(Y_hat, axis=1, keepdims=True)
        idM = 0
        
    risk_test = np.mean((Y_hat[-Y_test.shape[0]:,:]-Y_test)**2, axis=0)[idM]
    if data_val is not None:
        risk_val = np.mean((Y_hat[:-Y_test.shape[0],:]-Y_val)**2, axis=0)[idM]
        return risk_val, risk_test
    else:
        return risk_test
    
    
    
def cross_validation(X, Y, X_test, Y_test, lam, M, nu=0.6, replace=True):
    assert 0.5 < nu < 1
    n, p = X.shape
    n_val = int(2 * np.sqrt(n))
    n_train = n - n_val
    n_base = int(n_train**nu)
    
    ids_val = np.sort(np.random.choice(n,n_val,replace=False))
    ids_train = np.setdiff1d(np.arange(n),ids_val)
    X_val, Y_val = X[ids_val,:], Y[ids_val,:]
    X_train, Y_train = X[ids_train,:], Y[ids_train,:]
    
    if replace:
        k_list = np.arange(n_base, n_train+1, n_base)
        if n_train!=k_list[-1]:
            k_list = np.append(k_list, n_train)
    else:
        k_list = n_train / np.arange(1,M+1)
        k_list = k_list[k_list>=n_base]
    
    res_val = np.full((len(k_list)+1,M), np.inf)
    res_test = np.full((len(k_list)+1,M), np.inf)
    
    for j,k in enumerate(k_list):
        if replace:
            res_val[j,:], res_test[j,:] = comp_empirical_risk(
                X_train, Y_train, X_test, Y_test, 
                p/k, lam, M, data_val=(X_val, Y_val), replace=replace, return_allM=True
            )
        else:
            m = j + 1
            res_val[j,:m], res_test[j,:m] = comp_empirical_risk(
                X_train, Y_train, X_test, Y_test, 
                p/k, lam, m, data_val=(X_val, Y_val), replace=replace, return_allM=True
            )
            res_val[j,m:] = res_val[j,m-1]
            res_test[j,m:] = res_test[j,m-1]
            
    # null predictor
    res_val[-1,:], res_test[-1,:] = np.mean(Y_val**2), np.mean(Y_test**2)
    
    j_cv = np.argmin(res_val, axis=0)
#     k_cv = k_list[j_cv]
    risk_cv = res_test[j_cv, np.arange(M)]

    return risk_cv