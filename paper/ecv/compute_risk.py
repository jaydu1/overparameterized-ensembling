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

n_jobs = 16
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
        return rho2, 0, rho2 + sigma2
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
                        M=2, data_val=None, replace=True, bootstrap=False,
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
        ids_list = [np.sort(np.random.choice(n,k,replace=bootstrap)) for j in range(M)]
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
        
    with Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=0) as parallel:
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
    
    
def cross_validation(
    X, Y, X_test, Y_test, method, param, M, nu=0.5, 
    replace=True, bootstrap=False, 
    val_size=None, Kfold=False, k_list=None, return_full=False):
    assert 0 < nu < 1
    n, p = X.shape
    
    if Kfold is False:
        if val_size is None:
            n_val = int(2 * np.sqrt(n))
        else:
            n_val = int(n * val_size)

        ids_val_list = [np.sort(np.random.choice(n,n_val,replace=False))]
        ids_train_list = [np.setdiff1d(np.arange(n),ids_val_list[0])]
    else:
        K = Kfold        
        kf = KFold(n_splits=Kfold)
        ids_train_list, ids_val_list = list(zip(*kf.split(np.arange(n))))
        n_val = len(ids_train_list[0])
    
    n_train = n - n_val
    n_base = int(n_train**nu)
    if k_list is not None:
        k_list = np.array(k_list)
        k_list = k_list[k_list<=n_train]
    else:
        if replace:
            k_list = np.arange(n_base, n_train+1, n_base)
            if n_train!=k_list[-1]:
                k_list = np.append(k_list, n_train)
        else:
            k_list = n_train / np.arange(1,M+1)
            k_list = k_list[k_list>=n_base]
        
    res_val = []
    res_test = []
    for ids_train, ids_val in zip(ids_train_list, ids_val_list):
        X_train, Y_train = X[ids_train,:], Y[ids_train,:]
        X_val, Y_val = X[ids_val,:], Y[ids_val,:]        

        _res_val = np.full((len(k_list)+1,M), np.inf)
        _res_test = np.full((len(k_list)+1,X_test.shape[0],M), np.inf)

        for j,k in enumerate(k_list):
            if replace:
                _res_val[j,:], _res_test[j,:,:] = comp_empirical_risk(
                    X_train, Y_train, X_test, Y_test, 
                    p/k, method, param, M, data_val=(X_val, Y_val), 
                    replace=replace, bootstrap=bootstrap, 
                    return_allM=True, return_pred_diff=True
                )
            else:
                m = j + 1
                _res_val[j,:m], _res_test[j,:,:m] = comp_empirical_risk(
                    X_train, Y_train, X_test, Y_test, 
                    p/k, method, param, m, data_val=(X_val, Y_val), 
                    replace=replace, bootstrap=bootstrap, 
                    return_allM=True, return_pred_diff=True
                )
                _res_val[j,:,m:] = _res_val[j,:,m-1]
                _res_test[j,:,m:] = _res_test[j,:,m-1]

        # null predictor
        _res_val[-1,:], _res_test[-1,:,:] = np.mean(Y_val**2), Y_test
        
        res_val.append(_res_val)
        res_test.append(_res_test)
        
    res_val = np.mean(np.array(res_val), axis=0)
    res_test = np.mean(np.mean(np.array(res_test), axis=0)**2, axis=1)
    
    
    if return_full:
        return np.append(k_list,0), res_val, res_test
    else:
        j_cv = np.argmin(res_val, axis=0)
        risk_cv = res_test[j_cv, np.arange(M)]
        return k_list[j_cv], risk_cv




############################################################################
#
# Out-of-bag cross-validation
#
############################################################################

def risk_estimate(sq_err, method, eta):
    if method=='AVG':
        risk = np.mean(sq_err)        
    else:
        n = sq_err.shape[0]
        B = int(np.maximum(
            np.minimum(np.ceil(8 * np.log(1/eta)), n), 1))
        ids_list = np.array_split(np.random.permutation(np.arange(n)), B)
        risk = np.median([np.mean(sq_err[ids]) for ids in ids_list])
    return risk
    


def comp_empirical_oobcv(
    X, Y, X_test, Y_test, phi_s, method, param, 
    M=2, M0=5, M_test=None, re_method='AVG', eta=None,
    oobcv=True, bootstrap=False):
    '''
    Parameters
    ----------
    X,Y,X_test,Y_test : numpy.array
        traning and testing samples.
    phi_s : float
        the subsample aspect ratio: phi_s=p/k.
    method : str
        The base predictor: 'ridge', 'lasso', 'tree'.
    param : float
        The regularization parameter for ridge or lasso predictors.
    M : int or numpy.array
        The number of bags to evaluate based on the oob estimates.
    M0 : int
        The number of bags to compute the oob estimates.
    M_test : int
        The number of bags to compute the test error.
    
    Return:
    ----------
        
    '''
    M_test = M0 if M_test is None else np.maximum(M_test, M0)

    n,p = X.shape
    n_test = int(n/np.log(n)) if oobcv else 0
    # n_test = np.minimum(int(n/np.log(n)), int(n*0.05)) if oobcv else 0
    Y_test = Y_test.reshape((-1,1))
    Y_hat = np.zeros((Y_test.shape[0]+Y.shape[0], M0))
    X_eval = np.r_[X, X_test]

    k = np.minimum(int(p/phi_s), n-n_test)
    
    if re_method=='MOM' and eta is None:
        eta = 1/n
    
    if method =='logistic':
        ids_list = [
            np.sort(
                np.r_[
                np.random.choice(np.where(Y[:n-n_test,0]==0)[0],
                                 int(np.mean(Y[:n-n_test,0]==0)*k),replace=bootstrap),
                np.random.choice(np.where(Y[:n-n_test,0]==1)[0],
                                 int(np.mean(Y[:n-n_test,0]==1)*k),replace=bootstrap)
                ]
            ) for j in range(M_test)]
    else:
        ids_list = [np.sort(np.random.choice(n-n_test,k,replace=bootstrap)) for j in range(M_test)]
    
    with Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=0) as parallel:
        res = parallel(
            delayed(fit_predict)(X[ids,:], Y[ids,:], X_eval, method, param)
            for ids in ids_list
        )
    Y_hat = np.concatenate(res, axis=-1)
    
    if np.isscalar(M):
        M = np.arange(1,M+1)
    else:
        M = np.array(M)
        
    if oobcv:
        dev_eval = Y_hat[:-Y_test.shape[0],:M0] - Y
        err_eval = dev_eval**2

        with Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=0) as parallel:
            res_1 = parallel(
                delayed(lambda j:risk_estimate(
                    np.delete(err_eval[:,j], ids_list[j]), re_method, eta)
                       )(j)
                for j in np.arange(M0)
            )
            res_2 = parallel(
                delayed(lambda i,j:risk_estimate(
                    np.mean(
                        np.delete(dev_eval[:,[i,j]], 
                                  np.union1d(ids_list[i], ids_list[j]), axis=0), axis=1)**2,
                    re_method, eta
                ))(i,j)
                for i,j in itertools.combinations(np.arange(M0), 2)
            )
        risk_oob_1 = np.mean(res_1)
        risk_oob_2 = np.mean(res_2)
        risk_oob = - (1-2/M) * risk_oob_1 + 2*(1-1/M) * risk_oob_2
    else:
        risk_oob = None
        
    risk_test = np.mean((
        np.cumsum(Y_hat[-Y_test.shape[0]:,:], axis=1) / np.arange(1,M_test+1) - Y_test)**2, axis=0)

    return risk_oob, risk_test


def cross_validation_oob(X, Y, X_test, Y_test, method, param, M, M0, M_test=None, M_max=np.inf,
                         nu=0.5, return_full=False, k_list=None, bootstrap=False,
                         delta=0., re_method='AVG', eta=None):
    n, p = X.shape
    n_base = int(n**nu)
    n_train = int(n*(1-1/np.log(n)))
    if k_list is not None:
        k_list = np.array(k_list)
        k_list = k_list[k_list<=n_train]
    else:
        k_list = np.arange(n_base, n_train+1, n_base)
        if n_train!=k_list[-1]:
            k_list = np.append(k_list, n_train)
    
    M_test = M0 if M_test is None else np.maximum(M_test, M0)
    
    res_val = np.full((len(k_list)+1,M), np.inf)
    res_test = np.full((len(k_list)+1,M_test), np.inf)    

    for j,k in enumerate(k_list):
        res_val[j,:], res_test[j,:] = comp_empirical_oobcv(
            X, Y, X_test, Y_test, 
            p/k, method, param, M, M0=M0, M_test=M_test, 
            re_method=re_method, eta=eta, bootstrap=bootstrap)
            
    # null predictor
    if method=='logistic':
        res_val[-1,:], res_test[-1,:] = np.mean((Y-0.5)**2), np.mean((Y_test-0.5)**2)
    else:
        res_val[-1,:], res_test[-1,:] = np.mean(Y**2), np.mean(Y_test**2)
        
    k_list = np.append(k_list,0)
        
    if return_full:
        return k_list, res_val, res_test
    else:
        j = np.nanargmin(2 * res_val[:,1] - res_val[:,0])
        k_hat = k_list[j]
        if delta==0.:
            M_hat = np.inf
        else:
            M_hat = int(np.ceil(2 / delta * 
                            (res_val[j,0] - res_val[j,1])))
        M_hat = np.minimum(M_hat, M_max)
        if M_hat>M_test and k_hat>0:
            M_hat = int(np.minimum(M_hat, M_test))
            _, risk_cv = comp_empirical_oobcv(X, Y, X_test, Y_test, 
                    p/k_hat, method, param, M_hat, M0=M_hat, M_test=M_hat, re_method=re_method, eta=eta, 
                                              oobcv=False, bootstrap=bootstrap)
        else:
            risk_cv = res_test[j, np.arange(M_test)]

        return k_hat, M_hat, risk_cv


def compute_prediction_risk(X, Y, X_test, Y_test, method, param, 
                            M, k_list=None, nu=0.5, bootstrap=False):
    n, p = X.shape
    n_base = int(n**nu)

    if k_list is not None:
        k_list = np.array(k_list)
    else:
        k_list = np.arange(n_base, n+1, n_base)
        if n!=k_list[-1]:
            k_list = np.append(k_list, n)
    
    res_val = np.full((len(k_list)+1,M), np.inf)
    res_test = np.full((len(k_list)+1,M), np.inf)
    
    for j,k in enumerate(k_list):
        res_val[j,:], res_test[j,:] = comp_empirical_oobcv(X, Y, X_test, Y_test, 
            p/k, method, param, M, M0=1, M_test=M, bootstrap=bootstrap)
            
    # null predictor
    res_val[-1,:], res_test[-1,:] = np.mean(Y**2), np.mean(Y_test**2)

    return np.append(k_list,0), res_val, res_test