import numpy as np
from sklearn.linear_model import Ridge
from fixed_point_sol import *
import scipy as sp
from joblib import Parallel, delayed

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
    if phi_s == np.inf:
        return rho**2, sigma**2, rho**2 + sigma**2
    if M==1:
        return comp_theoretic_risk_M1(rho, sigma, lam, phi_s)
    elif replace and M==2:

        phi_0 = phi_s**2 /phi

        if lam==0 and phi_s<=1:
            if phi_s<1:
                B0 = 0.
                V0 = 0.5 * sigma**2 * (
                    phi_s / (1-phi_s) + 
                    phi / (1 - phi)
                )
            elif phi_s==1:
                B0 = 0.
                V0 = 0. if sigma==0 else np.inf

        else:
            v = v_phi_lam(phi_s,lam)
            vb = vb_lam_phis_phi(lam,phi_s,phi, v=v)
            B0 = 0.5 * rho**2 * (
                (1 + vb_phi_lam(phi_s,lam)) / (1 + v)**2 +
                 vb / (1 + v)**2
            )

            V0 = 0.5* sigma**2 * (
                phi_s * vv_phi_lam(phi_s,lam, v=v) / (1 + v)**2 +                
                (vb - 1)
            )
        return B0, V0, sigma**2+B0+V0
    elif replace and M>2:
        B1, V1, r1 = comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M=1)
        B2, V2, r2 = comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M=2)
        
        def decomp(x, y):
            b = 2 * (x - y)
            a = x - b
            return a, b
        
        Ba, Bb = decomp(B1, B2)
        Va, Vb = decomp(V1, V2)
        a, b = decomp(r1, r2)
        
        if M==np.inf:
            return Ba, Va, a
        else:
            return Ba + Bb / M, Va + Vb / M, a + b / M
    else:
        B1, V1, _ = comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M=1)
        
        # M is finite
        if lam==0 and phi_s<=1:
            BM = 0
        else:
            C = rho**2 / (1 + v_phi_lam(phi_s,lam))**2
            BM = B1 / M + (1 - 1/M) * C
        VM = V1 / M
        return BM, VM, sigma**2 + BM + VM


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
#     p = int(p * n_train / n)
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
    
    res_val = np.full((len(k_list),M), np.inf)
    res_test = np.full((len(k_list),M), np.inf)
    
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
    j_cv = np.argmin(res_val, axis=0)
    k_cv = k_list[j_cv]
    risk_cv = res_test[j_cv, np.arange(M)]

    return risk_cv