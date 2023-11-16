import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from compute_risk import *
from generate_data import *
from joblib import Parallel, delayed



func = 'quad'
bootstrap = True
bagging = 'bagging'
predictor = 'tree'
path_result = 'result/ex4/{}/{}/'.format(bagging,predictor)
os.makedirs(path_result, exist_ok=True)


def oobcv_tree(X, Y, X_test, Y_test, predictor, M_max, M0, 
               ratio_holdout=1./6, bootstrap=bootstrap):
    n, p = X.shape
    
    # ECV using OOB observations
    M_list = np.append(np.arange(1,M_max+1), np.inf)
    res_oob, res_test = comp_empirical_oobcv(X, Y, X_test, Y_test, 
        p/n, predictor, None, M_list, M0=M0, M_test=M_max, bootstrap=bootstrap)
    
    # ECV using holdout observations
    n_val = int(np.ceil(n*ratio_holdout))
    id_val = np.random.choice(n,n_val,replace=False)
    _, res_val = comp_empirical_oobcv(
        np.delete(X, id_val, axis=0), np.delete(Y, id_val, axis=0), X[id_val,:], Y[id_val], 
            p/n, predictor, None, M_list, M0=2, M_test=2, oobcv=False, bootstrap=bootstrap)

    risk_holdout = - (1-2/M_list) * res_val[0] + 2*(1-1/M_list) * res_val[1]
    
    return res_oob, np.append(res_test, np.full(len(M_list)-M_max, np.nan)), risk_holdout

                              
def run_one_simulation(n, phi, rho_ar1, sigma, M, M0, i):
    np.random.seed(i)
    p = int(n*phi)
    
    # Generate data
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data(
        n, p, coef='eig-5', func=func,
        rho_ar1=rho_ar1, sigma=sigma, df=np.inf, n_test=2000, sigma_quad=1.)
    
    # Compute the empirical risk estimates
    oobcv_emp, risk_emp, hold_emp = oobcv_tree(X, Y, X_test, Y_test,
                                     predictor, M, M0)

    res = np.c_[
        np.full((3,1), phi), np.full((3,1), i), np.array(['oobcv_emp', 'risk_emp', 'hold_emp'])[:,None],
        np.full((3,1), rho2), np.full((3,1), sigma2), 
        np.stack([oobcv_emp, risk_emp, hold_emp
    ])]
    return res



n = 500
n_simu = 50
sigma = .5
rho_ar1 = 0.5
M0 = 20 # number of bags for computing the OOB estimates
M_max = 500
M_list = np.append(np.arange(1,M_max+1), np.inf)
M_str = ['{:d}'.format(int(M)) for M in M_list[:-1]] + ['$\infty$']                            

with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:        
    file_saved = path_result+'res_cv_rho_{:.02f}.csv'.format(rho_ar1)
    df_res = pd.DataFrame()
    for phi in tqdm(np.logspace(-1.,1.,21), desc = 'phi'):
        res = parallel(
            delayed(run_one_simulation)(n, phi, rho_ar1, sigma, M_max, M0, i) for i in tqdm(range(n_simu))
        )
        res = pd.DataFrame(np.concatenate(res, axis=0), columns=np.append(['phi', 'seed', 'type', 'rho2', 'sigma2'], M_str))
        df_res = pd.concat([df_res, res],axis=0)
        df_res.to_csv(file_saved, index=False)
        


# The results can be summarized by the following code for plotting:
# import pandas as pd
# func = 'quad'
# bootstrap = True
# bagging = 'bagging'
# predictor = 'tree'
# rho_ar1= 0.5
# path_result = 'result/ex4/{}/{}/'.format(bagging,predictor)
# file_saved = path_result+'res_cv_rho_{:.02f}.csv'.format(rho_ar1)
# _df = pd.read_csv(file_saved)
# _df.to_pickle('result/ex4/res_path.pkl', compression='gzip')