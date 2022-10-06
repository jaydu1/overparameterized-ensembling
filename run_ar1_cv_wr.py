import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
sns.set_theme()
from compute_risk import *
from generate_data import generate_data_ar1

path_result = 'result/ex5/'
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(p, phi, rho_ar1, sigma, lam, M, i):
    np.random.seed(i)
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma)

    risk_emp_cv_M = cross_validation(X, Y, X_test, Y_test, lam, M, nu=0.6, replace=True)
    res = np.append([phi, i, rho2, sigma2], risk_emp_cv_M)
    return res


lam_list = np.linspace(0.,1.,11)#np.append(0, np.logspace(-9, 0, 10))
rho_ar1_list = [0.25, 0.5, 0.75]

M = 50
p = 500
n_simu = 100
sigma = 1.
from joblib import Parallel, delayed

with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
    for lam in [lam_list[int(sys.argv[1])]]:
        for rho_ar1 in rho_ar1_list:
#             if os.path.exists(
#                 path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}.csv'.format(phi, lam, SNR)):
#                 continue
            
            df_res = pd.DataFrame()
            for phi in tqdm(np.logspace(-1, 1, 50), desc = 'phi'):
                res = parallel(
                    delayed(run_one_simulation)(p, phi, rho_ar1, sigma, lam, M, i) for i in tqdm(range(n_simu))
                )
                res = pd.DataFrame(np.r_[res], columns=np.append(['phi', 'seed', 'rho2', 'sigma2'], np.arange(1,M+1)))
                df_res = pd.concat([df_res, res],axis=0)
                df_res.to_csv(path_result+'res_cv_rhoar1_{:.01f}_lam_{:.01f}_with_replacement.csv'.format(rho_ar1, lam))
