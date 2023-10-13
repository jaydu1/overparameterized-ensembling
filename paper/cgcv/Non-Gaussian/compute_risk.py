import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, ElasticNet
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



def comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, psi, M, 
                                replace=True, v=None, tc=None, tv=None, tv_s=None):
    if psi == np.inf:
        rho2 = beta0.T @ Sigma @ beta0 #(1-rho_ar1**2)/(1-rho_ar1)**2/5
        return np.full_like(M, rho2, dtype=np.float32), np.zeros_like(M), np.full_like(M, rho2 + sigma2, dtype=np.float32)
    else:
        if v is None:
            v = v_general(psi, lam, Sigma)
        if tc is None:
            tc = tc_general(psi, lam, Sigma, beta0, v)
        
        if np.any(M!=1):
            if tv is None:
                tv = tv_general(phi, psi, lam, Sigma, v) if replace else 0
        else:
            tv = 0
        if np.any(M!=np.inf):
            if tv_s is None:
                tv_s = tv_general(psi, psi, lam, Sigma, v)
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
class Ridgeless(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        self.beta = sp.linalg.lstsq(X, Y, check_finite=False, lapack_driver='gelsy')[0]
        
    def predict(self, X_test):
        return X_test @ self.beta
    
    
class NegativeRidge(object):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fit(self, X, Y):
        n, p = X.shape
        if n<=p:
            L = np.linalg.cholesky(X.dot(X.T) + self.alpha * np.eye(n))
            self.beta = X.T @ np.linalg.solve(L.T, np.linalg.solve(L, y))
        else:
            L = np.linalg.cholesky(X.T.dot(X) + self.alpha * np.eye(p))
            self.beta = np.linalg.solve(L.T, np.linalg.solve(L, X.T.dot(y)))
        
    def predict(self, X_test):
        return X_test @ self.beta
    

class Ridge_Ridgeless(Ridge, Ridgeless, NegativeRidge):
    def __init__(self, alpha):
        if alpha==0:
            super(Ridgeless, self).__init__()
            self.cls = Ridgeless
        elif alpha<0:
            super(NegativeRidge, self).__init__(alpha=alpha)
            self.cls = NegativeRidge
        else:
            super(Ridge, self).__init__(alpha=alpha, fit_intercept=False, solver='lsqr')
            self.cls = Ridge
            
    def fit(self, X, Y):
        sqrt_n = X.shape[0]
        return super(self.cls, self).fit(X/sqrt_n, Y/sqrt_n)

    
def wrap_class(clf, p, **kwargs):
    class ClassWrapper(clf):
        def __init__(self, p=p, **kwargs):
            super(clf, self).__init__(**kwargs)
            self.p = p            

        def fit(self, X, Y):
            if self.p>0:
                return super(clf, self).fit(X[:,:self.p], Y)
            else:
                self.mean = np.mean(Y)
                return self

        def predict(self, X_test):
            if self.p>0:
                return super(clf, self).predict(X_test[:,:self.p])
            else:
                return np.full(X_test.shape[0], self.mean)
    return ClassWrapper(p, **kwargs)


PartialRidge = lambda lam, p : wrap_class(Ridge_Ridgeless, p, alpha=lam)
PartialLasso = lambda lam, p : wrap_class(Lasso, p, alpha=np.maximum(lam,1e-15), fit_intercept=False)


def fit_predict(X, Y, X_test, method, param):
    if method in ['tree', 'random_forest', 'NN', 'kNN']:
        if method=='tree':
            regressor = DecisionTreeRegressor(max_features=1./3, min_samples_split=5)#, splitter='random')
        elif method=='random_forest':
            regressor = RandomForestRegressor(n_estimators=param[0], max_leaf_nodes=param[1], max_depth=param[2],
                                              max_features='sqrt', bootstrap=False)
        elif method=='NN':
            regressor = MLPRegressor(random_state=0, max_iter=500)
        elif method=='kNN':
            regressor = KNeighborsRegressor(n_neighbors=param).fit(X, Y)
        regressor = regressor.fit(X, Y)

        Y_hat = regressor.predict(X_test)
    elif method=='logistic':
        clf = LogisticRegression(
            random_state=0, fit_intercept=False, C=1/np.maximum(param,1e-6)
        ).fit(X, Y.astype(int))
        Y_hat = clf.predict_proba(X_test)[:,1].astype(float)    
    else:
        if method.startswith('partial'):
            lam, p = param
        else:
            lam = param
            p = X.shape[1]
        method = method.replace('partial_','')
        if method.startswith('ridge'):
            regressor = PartialRidge(lam, p)
        elif method.startswith('lasso'):
            regressor = PartialLasso(lam, p)
        elif method=='elastic_net':
            lam_1, lam_2 = param
            alpha = lam_1 + lam_2
            l1_ratio = lam_1 / (lam_1 + lam_2)
            regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
        else:
            raise ValueError('No implementation for {}.'.format(method))
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
def fit_predict_gcv(X, Y, X_test, method, param):
    k, p = X.shape
    k_sqrt = np.sqrt(k)
    
    if method in ['ridge','lasso']:
        lam = np.maximum(param, 1e-8)
        
        if method=='ridge':
            regressor = PartialRidge(lam, p)
            X = X/k_sqrt
            Y = Y/k_sqrt
        else:
            regressor = PartialLasso(lam, p)
        regressor.fit(X, Y)
        
        if method=='ridge':            
            svds = np.linalg.svd(X, compute_uv=False)
            evds = svds[:k]**2
            dof = np.sum(evds/(evds + lam))
        elif method=='lasso':
            dof = np.sum(regressor.coef_!=0)
            
    elif method=='elastic_net':
        lam_1, lam_2 = param
        alpha = lam_1 + lam_2
        l1_ratio = lam_1 / (lam_1 + lam_2)
        regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
        regressor.fit(X, Y)
        if np.any(regressor.coef_!=0):
            svds = np.linalg.svd(X[:,regressor.coef_!=0], compute_uv=False)
        else:
            svds = np.array([0.])
        evds = svds[:k]**2
        dof = np.sum(evds/(evds + k * lam_2))
        
    
    Y_hat = regressor.predict(X_test)
    
    if len(Y_hat.shape)<2:
        Y_hat = Y_hat[:,None]
        
    return dof, Y_hat
    

def comp_empirical_gcv(
    X, Y, X_test, Y_test, psi, method, param, 
    M=2, return_allM=False, 
    estimator='cgcv'):
    
    n, p = X.shape
    phi = p/n
    Y_test = Y_test.reshape((-1,1))    
    Y_hat = np.zeros((Y_test.shape[0], M))
    
    k = int(p/psi)
    ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
    ids_union_list = [reduce(np.union1d, ids_list[:j+1]) for j in range(M)]
    n_ids = np.array([len(ids) for ids in ids_union_list])

#     if estimator=='cgcv' or estimator=='full':
        
#     elif estimator=='sub':
#         X_val, Y_val = X[ids_union_list[-1],:], Y[ids_union_list[-1],:]
        
    X_val, Y_val = X, Y
    ids_union_full_list = [np.arange(n) for _ in range(M)]
#     ids_union_sub_list = [np.nonzero(np.in1d(ids_union_list[-1], ids))[0] for ids in ids_union_list]
    Y_val = Y_val.reshape((-1,1))
    X_eval = np.r_[X_val, X_test]
        
    if len(ids_list)>=2:
        with Parallel(n_jobs=8, temp_folder='~/tmp/', max_nbytes=None, verbose=0) as parallel:
            res = parallel(
                delayed(fit_predict_gcv)(
                    X[ids,:], Y[ids,:], X_eval, method, param)
                for ids in ids_list
            )
        dof, Y_hat = zip(*res)
    else:
        ids = ids_list[0]
        dof, Y_hat = fit_predict_gcv(
            X[ids,:], Y[ids,:], X_eval, method, param)
        dof, Y_hat = [dof], [Y_hat]
    
    dof = np.array(dof)
    Y_hat_per = np.concatenate(Y_hat, axis=-1)
    
    if estimator=='cgcv':
        pass
    elif estimator=='full':
        pass
    elif estimator=='sub':
        pass
    
    if return_allM:
        Y_hat = np.cumsum(Y_hat_per, axis=1) / np.arange(1,M+1)
        dof = np.cumsum(dof) / np.arange(1,M+1)
        idM = np.arange(M)
    else:
        Y_hat = np.mean(Y_hat_per, axis=1, keepdims=True)
        dof = np.mean(dof, keepdims=True)
        idM = -1
    
    mse_train = (Y_hat[:-Y_test.shape[0],:]-Y_val)**2
    
    if return_allM:
        risk_val_full = np.array([np.mean(mse_train[ids_union_full_list[j],j]) for j in idM])
        risk_val_sub = np.array([np.mean(mse_train[ids_union_list[j],j]) for j in idM])
    else:
        risk_val_full = np.mean(mse_train[ids_union_full_list[-1],-1], axis=0)
        risk_val_sub = np.mean(mse_train[ids_union_list[-1],-1], axis=0)

    risk_test = np.mean((Y_hat[-Y_test.shape[0]:,:]-Y_test)**2, axis=0)[idM]
    
    deno_1 = (1 - dof[idM]/n)**2
    deno_2 = deno_1 + (psi-phi)/phi*(dof[idM]/n)**2
    C = (1/deno_1 - 1/deno_2) / np.arange(1,M+1)[idM]**2 * np.cumsum(
        np.mean((Y_hat_per[:-Y_test.shape[0],:]-Y_val)**2, axis=0))[idM]
    cgcv = risk_val_full / deno_1 - C
    
    deno = (1 - dof[idM]/n_ids[idM])**2
#     gcv = risk_val_sub / deno  
    fgcv = risk_val_full / deno_1
    
    return fgcv, cgcv, risk_test



    
def comp_theoretic_gcv(Sigma, beta0, sigma2, lam, phi, psi, M,
                          v=None, tc=None, tv=None, tv_s=None):

    if psi == np.inf:
        rho2 = beta0.T @ Sigma @ beta0
        return rho2 + sigma2, None, None, None, None
    else:
        if v is None:
            v = v_general(psi, lam, Sigma)
        if tc is None:
            tc = tc_general(psi, lam, Sigma, beta0, v)
        if tv is None:
            tv = tv_general(phi, psi, lam, Sigma, v)
        if tv_s is None:
            tv_s = tv_general(psi, psi, lam, Sigma, v)
        
        if lam==0. and psi<=1:
            lamv = 1 - psi
        else:
            lamv = lam * v
        D1 = lamv**2
        D2 = (1 - psi/(2*psi-phi)*( 1 - lamv))**2
        D = (1 - phi/psi*( 1 - lamv))**2
        
        Rte1 = (1 + tv_s) * (tc + sigma2)
        Rte2 = ((1 + tv_s) + (1 + tv) ) / 2 * (tc + sigma2)
        Rteinf = (1 + tv) * (tc + sigma2)
        Rtr1 = D1 * Rte1
        c1 = 2*(psi-phi)/(2*psi-phi)
        Rtr2 = 1/4 * c1 * Rte1 + \
            1/2 * D1 * psi/(2*psi-phi) * Rte1 + \
            1/2 * (c1 * lamv + (1-c1)*D1)* Rteinf
        
        if M==np.inf:
            gcv = (2*phi*(2*psi-phi)/psi**2 * Rtr2 + 2*(psi-phi)**2/psi**2 * Rte2 - phi/psi * Rtr1 - (psi-phi)/psi * Rte1) / D
        elif M==1:
            gcv = Rtr1/D1
        elif M==2:
            gcv = Rtr2/D2
        else:
            raise ValueError('No implementation for 2<M<inf.')
        return gcv, v, tc, tv, tv_s