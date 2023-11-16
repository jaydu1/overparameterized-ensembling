import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import scipy as sp
from joblib import Parallel, delayed
from functools import reduce
import itertools

import warnings
warnings.filterwarnings('ignore') 

n_jobs = 16


############################################################################
#
#   Predictors
#
############################################################################


class Ridgeless(object):
    def __init__(self, **kwargs):
        pass
    
    def fit(self, X, Y):
        self.beta = sp.linalg.lstsq(X, Y, check_finite=False, lapack_driver='gelsy')[0]
        
    def predict(self, X_test):
        return X_test @ self.beta
    
    
class NegativeRidge(object):
    def __init__(self, alpha, **kwargs):
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
    def __init__(self, alpha, **kwargs):
        if alpha==0:
            super(Ridgeless, self).__init__(**kwargs)
            self.cls = Ridgeless
        elif alpha<0:
            super(NegativeRidge, self).__init__(alpha=alpha, **kwargs)
            self.cls = NegativeRidge
        else:
            super(Ridge, self).__init__(alpha=alpha, fit_intercept=False, solver='lsqr', **kwargs)
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


PartialRidge = lambda lam, p, **kwargs : wrap_class(Ridge_Ridgeless, p, alpha=lam, **kwargs)
PartialLasso = lambda lam, p, **kwargs : wrap_class(Lasso, p, alpha=np.maximum(lam,1e-15), fit_intercept=False, **kwargs)


def fit_predict(X, Y, X_test, method, param, **kwargs):
    if method in ['tree', 'random_forest', 'NN', 'kNN']:
        if method=='tree':
            if 'max_features' not in kwargs:
                kwargs['max_features'] = 1./3
            if 'min_samples_split' not in kwargs:
                kwargs['min_samples_split'] = 5
            regressor = DecisionTreeRegressor(**kwargs)
        elif method=='NN':
            regressor = MLPRegressor(activation='identity', solver='sgd', 
                                     hidden_layer_sizes=[param[0]], random_state=param[1], 
                                     early_stopping=True, max_iter=5000, **kwargs)
        elif method=='kNN':
            if 'n_neighbors' not in kwargs:
                kwargs['n_neighbors'] = 5
            regressor = KNeighborsRegressor(**kwargs).fit(X, Y)
        regressor = regressor.fit(X, Y)
        Y_hat = regressor.predict(X_test)
    elif method=='logistic':
        if not np.any(Y==0):
            Y_hat = np.ones(X_test.shape[0])
        elif not np.any(Y==1):
            Y_hat = np.zeros(X_test.shape[0])
        else:
            clf = LogisticRegression(
                random_state=0, fit_intercept=False, C=1/np.maximum(param,1e-6), **kwargs
            ).fit(X, Y.astype(int))
            Y_hat = clf.predict_proba(X_test)[:,1].astype(float)    
    else:
        if method.startswith('partial'):
            lam, p = param
        else:
            lam = param
            p = X.shape[1]
        method = method.replace('partial','')
        if method.startswith('ridge'):
            regressor = PartialRidge(lam, p, **kwargs)
        else:
            regressor = PartialLasso(lam, p, **kwargs)
        regressor.fit(X, Y)
        Y_hat = regressor.predict(X_test)
    if len(Y_hat.shape)<2:
        Y_hat = Y_hat[:,None]
    return Y_hat



############################################################################
#
#   Split CV and K-fold CV
#
############################################################################
def comp_empirical_risk(X, Y, X_test, Y_test, phi_s, method, param, 
                        M, data_val=None, replace=True, bootstrap=False,
                        return_allM=False, return_pred_diff=False, **kwargs):
    '''Compute the empirical risk estimates.

    Parameters
    ----------
    X,Y,X_test,Y_test : numpy.array
        The training and testing samples.
    phi_s : float
        The subsample aspect ratio: phi_s=p/k.
    method : str
        The base predictor: 'ridge', 'lasso', 'tree', 'kNN', and 'logistic'.
    param : float
        The regularization parameter for ridge or lasso predictors.
    M : int
        The number of bags to evaluate.
    data_val : tuple of numpy.array
        The validation data.
    replace : bool
        Whether to use the bootstrap sampling.
    bootstrap : bool
        Whether to use the bootstrap sampling.
    return_allM : bool
        Whether to return the risk estimates for all M.
    return_pred_diff : bool
        Whether to return the prediction difference.
    **kwargs : dict
        The additional arguments for the base predictor.
    
    Returns
    -------
    risk_val : numpy.array
        The validation risk estimates.
    risk_test : numpy.array
        The testing risk estimates.
    '''
    n,p = X.shape
    
    if len(Y_test.shape)<2:
        Y_test = Y_test[:,None]

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
            delayed(fit_predict)(X[ids,:], Y[ids,:], X_eval, method, param, **kwargs)
            for ids in ids_list
        )
    Y_hat = np.concatenate(res, axis=-1)

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
    val_size=None, Kfold=False, k_list=None, return_full=False, **kwargs):
    '''Split-CV and K-fold CV.

    Pa
    
    '''
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
    if 0 not in k_list:
        k_list = np.insert(k_list,0,0)
        
    res_val = []
    res_test = []
    for ids_train, ids_val in zip(ids_train_list, ids_val_list):
        X_train, Y_train = X[ids_train,:], Y[ids_train,:]
        X_val, Y_val = X[ids_val,:], Y[ids_val,:]        

        _res_val = np.full((len(k_list),M), np.inf)
        _res_test = np.full((len(k_list),X_test.shape[0],M), np.inf)

        for j,k in enumerate(k_list):
            # null predictor
            if k==0:
                mu = 0.5 if method=='logistic' else 0.
                _res_val[j,:] = np.mean((Y-mu)**2)
                _res_test[j,:,:] = (mu - Y_test)
                continue
                
            if replace:
                _res_val[j,:], _res_test[j,:,:] = comp_empirical_risk(
                    X_train, Y_train, X_test, Y_test, 
                    p/k, method, param, M, data_val=(X_val, Y_val), 
                    replace=replace, bootstrap=bootstrap, 
                    return_allM=True, return_pred_diff=True, **kwargs
                )
            else:
                m = j + 1
                _res_val[j,:m], _res_test[j,:,:m] = comp_empirical_risk(
                    X_train, Y_train, X_test, Y_test, 
                    p/k, method, param, m, data_val=(X_val, Y_val), 
                    replace=replace, bootstrap=bootstrap, 
                    return_allM=True, return_pred_diff=True, **kwargs
                )
                _res_val[j,m:] = _res_val[j,m-1]
                _res_test[j,:,m:] = _res_test[j,:,m-1]
        
        res_val.append(_res_val)
        res_test.append(_res_test)
        
    res_val = np.mean(np.array(res_val), axis=0)
    res_test = np.mean(np.mean(np.array(res_test), axis=0)**2, axis=1)
    
    
    if return_full:
        return k_list, res_val, res_test
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
    '''Estimate the risk based on the square errors.
    
    Parameters
    ----------
    sq_err : numpy.array
        The square errors.
    method : str
        The estimator: 'AVG' or 'MOM'.
    eta : float
        The MOM estimator: eta=1/n.
    
    Returns
    -------
    risk : float
        The risk estimate.    
    '''
    if len(sq_err)<1:
        return np.nan
    
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
    M=2, M0=5, M_test=None, k_max=None, re_method='AVG', eta=None,
    oobcv=True, bootstrap=False, **kwargs):
    '''Compute the empirical ECV risk estimates.

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
    risk_oob : numpy.array
        The ECV risk estimates.
    risk_test : numpy.array
        The test risk estimates.
    '''
    M_test = M0 if M_test is None else np.maximum(M_test, M0)
    
    # null predictor
    if np.isinf(phi_s):        
        mu = 0.5 if method=='logistic' else 0.
        risk_oob = np.full(M if np.isscalar(M) else len(M), np.mean((Y-mu)**2))
        risk_test = np.full(M_test, np.mean((Y_test-mu)**2))
        return risk_oob, risk_test
    
    if np.isscalar(M):
        M = np.arange(1,M+1)
    else:
        M = np.array(M)
    
    n,p = X.shape
    
    if k_max is None:
        n_test = int(n/np.log(n)) if oobcv else 0
    else:
        n_test = int(n-k_max)
    # n_test = np.minimum(int(n/np.log(n)), int(n*0.05)) if oobcv else 0
    if len(Y_test.shape)<2:
        Y_test = Y_test[:,None]
    Y_hat = np.zeros((Y_test.shape[0]+Y.shape[0], M0))
    X_eval = np.r_[X, X_test]

    k = np.minimum(int(p/phi_s), n-n_test)
    
    if re_method=='MOM' and eta is None:
        eta = 1/n
    
    if method =='logistic':
        ids_list = [
            np.sort(
                np.r_[
                np.random.choice(np.where(Y[:n,0]==0)[0],
                                 int(np.mean(Y[:n,0]==0)*k),replace=bootstrap),
                np.random.choice(np.where(Y[:n,0]==1)[0],
                                 int(np.mean(Y[:n,0]==1)*k),replace=bootstrap)
                ]
            ) for j in range(M_test)]
    else:
        ids_list = [np.sort(np.random.choice(n,k,replace=bootstrap)) for j in range(M_test)]
    
    with Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=0) as parallel:
        res = parallel(
            delayed(fit_predict)(X[ids,:], Y[ids,:], X_eval, method, param, **kwargs)
            for ids in ids_list
        )
    Y_hat = np.concatenate(res, axis=-1)
        
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
                    
        risk_oob_1 = np.nanmean(res_1)
        risk_oob_2 = np.nanmean(res_2)
        risk_oob = - (1-2/M) * risk_oob_1 + 2*(1-1/M) * risk_oob_2
    else:
        risk_oob = None
        
    risk_test = np.mean((
        np.cumsum(Y_hat[-Y_test.shape[0]:,:], axis=1) / np.arange(1,M_test+1) - Y_test)**2, axis=0)

    return risk_oob, risk_test


def cross_validation_oob(X, Y, X_test, Y_test, method, param, M, M0, M_test=None, M_max=np.inf,
                         nu=0.5, return_full=False, k_list=None, k_max=None, bootstrap=False,
                         delta=0., re_method='AVG', eta=None, **kwargs):
    '''ECV.

    Parameters
    ----------
    X,Y,X_test,Y_test : numpy.array
        The training and testing samples.
    method : str
        The base predictor: 'ridge', 'lasso', 'tree', 'kNN', and 'logistic'.
    param : float
        The regularization parameter for ridge or lasso predictors.
    M : int
        The number of bags to evaluate.
    M0 : int
        The number of bags to compute the oob estimates.
    M_test : int
        The number of bags to compute the test error.
    M_max : int
        The maximum number of bags to extrapolate.
    nu : float
        The subsample aspect ratio: phi_s=p/k.
    return_full : bool
        Whether to return the risk estimates for all M.
    k_list : numpy.array
        The list of subsample sizes.
    k_max : int
        The maximum subsample size.
    bootstrap : bool
        Whether to use the bootstrap sampling.
    delta : float
        The extrapolation parameter: delta=0 corresponds to the ECV.
    re_method : str
        The risk estimator: 'AVG' or 'MOM'.
    eta : float
        The MOM estimator: eta=1/n.
    **kwargs : dict
        The additional arguments for the base predictor.

    Returns
    -------
    k_hat : int
        The optimal subsample size.
    M_hat : int
        The optimal number of bags.
    risk_cv : numpy.array
        The ECV risk estimates.
    '''
    n, p = X.shape
    n_base = int(n**nu)
    n_train = int(n*(1-1/np.log(n))) if k_max is None else k_max
    if k_list is not None:
        k_list = np.array(k_list)
    else:
        k_list = np.arange(n_base, n_train+1, n_base)
    k_list = k_list[k_list <= n_train]
    if 0 not in k_list:
        k_list = np.insert(k_list,0,0)
        
    M_test = M0 if M_test is None else np.maximum(M_test, M0)
    
    res_val = np.full((len(k_list),M), np.inf)
    res_test = np.full((len(k_list),M_test), np.inf)    

    for j,k in enumerate(k_list):
        res_val[j,:], res_test[j,:] = comp_empirical_oobcv(
            X, Y, X_test, Y_test, 
            p/k, method, param, M, M0=M0, M_test=M_test, 
            re_method=re_method, eta=eta, bootstrap=bootstrap, **kwargs)
        
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
                                              oobcv=False, bootstrap=bootstrap, **kwargs)
        else:
            risk_cv = res_test[j, np.arange(M_test)]

        return k_hat, M_hat, risk_cv


def compute_prediction_risk(X, Y, X_test, Y_test, method, param, 
                            M, k_list=None, nu=0.5, bootstrap=False, **kwargs):
    '''Compute the empirical prediction risk estimates.

    Parameters
    ----------
    X,Y,X_test,Y_test : numpy.array
        The training and testing samples.
    method : str
        The base predictor: 'ridge', 'lasso', 'tree', 'kNN', and 'logistic'.
    param : float
        The regularization parameter for ridge or lasso predictors.
    M : int
        The number of bags to evaluate.
    k_list : numpy.array
        The list of subsample sizes.
    nu : float
        The subsample aspect ratio: phi_s=p/k.
    bootstrap : bool
        Whether to use the bootstrap sampling.
    **kwargs : dict
        The additional arguments for the base predictor.

    Returns
    -------
    k_list : numpy.array
        The list of subsample sizes.
    res_val : numpy.array
        The validation risk estimates.
    res_test : numpy.array
        The testing risk estimates.
    '''
    n, p = X.shape
    n_base = int(n**nu)

    if k_list is not None:
        k_list = np.array(k_list)
    else:
        k_list = np.arange(n_base, n+1, n_base)
        if n!=k_list[-1]:
            k_list = np.append(k_list, n)
    if 0 not in k_list:
        k_list = np.insert(k_list,0,0)
    
    res_val = np.full((len(k_list),M), np.inf)
    res_test = np.full((len(k_list),M), np.inf)
    
    for j,k in enumerate(k_list):
        res_val[j,:], res_test[j,:] = comp_empirical_oobcv(X, Y, X_test, Y_test, 
            p/k, method, param, M, M0=1, M_test=M, bootstrap=bootstrap, **kwargs)

    return k_list, res_val, res_test