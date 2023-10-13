import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from fixed_point_sol import *
from generate_data import *
from compute_risk import *
from tqdm import tqdm


def run(method, phi, rho_ar1, sigma, i):
    M = 10
    M_str = ['{:d}'.format(int(_M)) for _M in np.arange(1,M+1)]
    res = []
    res_key = []
    
    np.random.seed(i)
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data(n, p, coef='random', func='linear',
        rho_ar1=rho_ar1, sigma=sigma, n_test=2000)

    for psi in tqdm(psi_list):
        try:
            gcv_emp, cgcv_emp, risk_emp = comp_empirical_gcv(
                X, Y, X_test, Y_test, psi, 
                method, param, M=M, return_allM=True)
        except:
            print('fail:', psi, i)
            gcv_emp, cgcv_emp, risk_emp = np.full(M, np.nan), np.full(M, np.nan), np.full(M, np.nan)
        _df = pd.DataFrame(
            np.stack([
                gcv_emp, cgcv_emp, risk_emp
            ]), index = ['gcv_emp', 'cgcv_emp', 'risk_emp'], columns=M_str
        )

        res.append(_df)
        res_key.append((psi,i))
            
    res = pd.concat(res, axis=0, keys=res_key, names=['psi','seed'])
    res['phi'] = phi
    res['lam'] = lam
    res['rho2'] = rho2
    res['sigma2'] = sigma2
    
    res.to_csv(path_result+'res_{}_phi_{:.01f}_sigma_{:.01f}_seed_{}.csv'.format(
        method, phi, sigma, i))
    return res



rho_ar1 = 0.
sigma = 1.
p = 1200
n_simu = 50

path_result = 'result/ex1/'
os.makedirs(path_result, exist_ok=True)

df_res_the = pd.DataFrame()

phi = 0.2
n = int(p/phi)
psi_list = np.logspace(np.log10(phi), np.log10(5), 101)
lam = 1e-2

# i_simu = int(sys.argv[1])
method_list = ['ridge', 'lasso', 'elastic_net']
method = method_list[int(sys.argv[1])]

param = lam if method!='elastic_net' else [lam,lam]

# for i in tqdm(range(25*i_simu,25*(i_simu+1))):
for i in tqdm(range(n_simu)):
    run(method, phi, rho_ar1, sigma, i)
