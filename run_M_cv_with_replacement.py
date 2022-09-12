import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
sns.set_theme()
from compute_risk import *

path_result = 'result/ex3/'
os.makedirs(path_result, exist_ok=True)

def generate_data(n, phi, rho=1., sigma=1.):
    p = int(n*phi)

    if rho>0.:
        beta0 = np.random.randn(p)
        beta0 = beta0 / np.sqrt(p) * rho
    else:
        beta0 = np.zeros(p)
        
    X = np.random.randn(n,p)
    Y = X@beta0[:,None]

    X_test = np.random.randn(n,p)
    Y_test = X_test@beta0[:,None]
    
    if sigma>0.:
        Y += np.random.randn(n,1)*sigma
        Y_test += np.random.randn(n,1)*sigma

    return beta0, X, Y, X_test, Y_test


def run_one_simulation(n, phi, rho, sigma, lam, M, i):
    np.random.seed(i)
    beta0, X, Y, X_test, Y_test = generate_data(n, phi, rho, sigma)

    risk_emp_cv_M = cross_validation(X, Y, X_test, Y_test, lam, M, nu=0.6, replace=True)
    res = np.append([phi, i], risk_emp_cv_M)
    return res


lam_list = np.linspace(0.,1.,11)#np.append(0, np.logspace(-9, 0, 10))

M = 20
n = 1000
n_simu = 100
rho = 1.
from joblib import Parallel, delayed

with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
    for lam in [lam_list[int(sys.argv[1])]]:
        for SNR in [1.,2.,3.,4.]:
#             if os.path.exists(
#                 path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}.csv'.format(phi, lam, SNR)):
#                 continue
            
            df_res = pd.DataFrame()
            for phi in tqdm(np.logspace(-1, 1, 50), desc = 'phi'):
                sigma = rho / np.sqrt(SNR)
                res = parallel(
                    delayed(run_one_simulation)(n, phi, rho, sigma, lam, M, i) for i in tqdm(range(n_simu))
                )
                res = pd.DataFrame(np.r_[res], columns=np.append(['phi', 'seed'], np.arange(1,M+1)))
                df_res = pd.concat([df_res, res],axis=0)
                df_res.to_csv(path_result+'res_cv_lam_{:.01f}_SNR_{:.01f}.csv'.format(lam, SNR))
