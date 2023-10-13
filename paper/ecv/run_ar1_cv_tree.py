import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
sns.set_theme()
from compute_risk import *
from generate_data import *

bootstrap = True
bagging = 'bagging' if bootstrap else 'subagging'
method = 'tree'
path_result = 'result/ex3/{}/{}/'.format(bagging,method)
os.makedirs(path_result, exist_ok=True)


def oobcv_tree(X, Y, X_test, Y_test, method, param, M_max, M0, 
               ratio_holdout=1./6, bootstrap=bootstrap):
    n, p = X.shape
    
    M_list = np.append(np.arange(1,M_max+1), np.inf)
    res_oob, res_test = comp_empirical_oobcv(X, Y, X_test, Y_test, 
        p/n, method, param, M_list, M0=M0, M_test=M_max)
    
    n_val = int(np.ceil(n*ratio_holdout))
    id_val = np.random.choice(n,n_val,replace=False)
    _, res_val = comp_empirical_oobcv(
        np.delete(X, id_val, axis=0), np.delete(Y, id_val, axis=0), X[id_val,:], Y[id_val], 
            p/n, method, param, M_list, M0=2, M_test=2, oobcv=False, bootstrap=bootstrap)

    risk_holdout = - (1-2/M_list) * res_val[0] + 2*(1-1/M_list) * res_val[1]
    
    return res_oob, np.append(res_test, np.full(len(M_list)-M_max, np.nan)), risk_holdout

                              
def run_one_simulation(n, phi, rho_ar1, sigma, lam, M, M0, i):
    np.random.seed(i)
    p = int(n*phi)
    
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma, n_test=5000)
    
    oobcv_emp, risk_emp, hold_emp = oobcv_tree(X, Y, X_test, Y_test,
                                     method, lam, M, M0)
    res = np.c_[
        np.full((3,1), phi), np.full((3,1), i), np.array(['oobcv_emp', 'risk_emp', 'hold_emp'])[:,None],
        np.full((3,1), rho2), np.full((3,1), sigma2), 
        np.stack([
        oobcv_emp, risk_emp, hold_emp
    ])]
    
    return res


lam_list = np.linspace(0.,1.,11)


n = 1000
n_simu = 100
sigma = 1.
rho_ar1_list = [0.25, 0.5, 0.75]
rho_ar1 = rho_ar1_list[int(sys.argv[1])]
if method in ['tree', 'kNN']:
    lam = None
else:
    lam_list = np.linspace(0.,1.,11)
    lam = lam_list[int(sys.argv[2])]
M0 = 20 # number of bags for computing the OOB estimates
M_max = 1000
M_list = np.append(np.arange(1,M_max+1), np.inf)
M_str = ['{:d}'.format(int(M)) for M in M_list[:-1]] + ['$\infty$']
                              
                              
from joblib import Parallel, delayed

with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:        
    if lam is None:
        file_saved = path_result+'res_ar1_cv_ar1rho_{:.02f}.csv'.format(rho_ar1)
    else:
        file_saved = path_result+'res_ar1_cv_lam_{:.01f}_ar1rho_{:.02f}.csv'.format(lam, rho_ar1)

    df_res = pd.DataFrame()
    print(method,rho_ar1,os.path.exists(file_saved))

    df_res = pd.DataFrame()
    for phi in tqdm(np.logspace(-1.,1.,11), desc = 'phi'):
        res = parallel(
            delayed(run_one_simulation)(n, phi, rho_ar1, sigma, lam, M_max, M0, i) for i in tqdm(range(n_simu))
        )
        res = pd.DataFrame(np.concatenate(res, axis=0), columns=np.append(['phi', 'seed', 'type', 'rho2', 'sigma2'], M_str))
        df_res = pd.concat([df_res, res],axis=0)
        df_res.to_csv(file_saved)
        