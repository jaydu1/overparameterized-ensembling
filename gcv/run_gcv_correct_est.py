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

path_result = 'result/ex5/'
os.makedirs(path_result, exist_ok=True)


def run(phi, lam, rho_ar1, sigma):
    p = 500
    n_simu = 50

    M = 500
    M_str = ['{:d}'.format(int(_M)) for _M in np.arange(1,M+1)]
    
    res = []
    res_key = []
    phi_s_list = np.logspace(-1,1, 10)
    phi_s_list[np.where(phi_s_list>=phi)[0][0]] = phi
    for phi_s in tqdm(phi_s_list):
        if phi_s<phi or (lam==0. and phi_s==1.):
            continue
            
        for i in tqdm(range(n_simu)):
            np.random.seed(i)
            Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma)
            try:
                gcv_emp, risk_emp = comp_empirical_gcv(X, Y, X_test, Y_test, phi_s, lam, 
                                                         M=M, return_allM=True, full=False)
            except:
                print('fail:', phi_s, i)
                gcv_emp, risk_emp = np.full(M, np.nan), np.full(M, np.nan)
            _df = pd.DataFrame(
                np.stack([
                    gcv_emp, risk_emp
                ]), index = ['gcv_emp', 'risk_emp'], columns=M_str
            )
    
            res.append(_df)
            res_key.append((phi_s,i))
            
    res = pd.concat(res, axis=0, keys=res_key, names=['phi_s','seed'])    
    res['rho2'] = rho2
    res['sigma2'] = sigma2
    res.to_csv(path_result+'res_phi_{:.01f}_lam_{:.01f}_ar1rho_{:.02f}.csv'.format(
        phi, lam, rho_ar1))
    

lam_list = [1e-8, 1e-1, 1.]
lam = lam_list[int(sys.argv[1])]
rho_ar1_list = [0.25, 0.5, 0.75]
rho_ar1 = rho_ar1_list[int(sys.argv[2])]
sigma = 1.
for phi in [0.1]:
    print(phi, lam, rho_ar1, sigma)
    run(phi, lam, rho_ar1, sigma)

