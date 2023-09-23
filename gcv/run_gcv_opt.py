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


def run(phi, phi_s, rho_ar1, sigma):
    p = 500
    n_simu = 10

    M = 500
    
    res = []

    for i in tqdm(range(n_simu)):
        i += 40
        np.random.seed(i)
        Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma)
        
        v = v_general(phi_s, 0., Sigma)
        lam = (phi_s-phi) * np.trace(np.linalg.solve(np.identity(p) + v * Sigma, v * Sigma)) / p
        
        for _lam, _phis, _M, name in zip([0, lam], [phi_s, phi], [M, 1], ['sample', 'ridge']):
            try:
                gcv_emp, risk_emp = comp_empirical_gcv(
                    X, Y, X_test, Y_test, _phis, 
                    _lam if _lam>0. else 1e-8, M=_M, return_allM=False)
            except:
                print('fail:', phi_s, i)
                gcv_emp, risk_emp = np.nan, np.nan
            
        
            res.append([_lam, phi, _phis, name, i, gcv_emp, risk_emp])
            
    res = pd.DataFrame(res, columns=['lam', 'phi', 'phi_s', 'type', 'seed', 'gcv_emp', 'risk_emp'])
    res['rho2'] = rho2
    res['sigma2'] = sigma2
    
    return res

sigma = 1.
p = 500


rho_ar1_list = [0.25,0.5]
rho_ar1 = rho_ar1_list[int(sys.argv[1])]


df_the = pd.DataFrame()
lam = 0.
df_the = pd.read_csv('result/ex2/res_the_lam_{:.01f}_ar1rho_{:.02f}.csv'.format(lam, rho_ar1))
df_the.columns = ['phi', 'phi_s', 'lam', 'risk_gcv', 'risk_the']
df_the = pd.wide_to_long(df_the, stubnames='risk', i=['phi', 'phi_s', 'lam'], j='Theoretical', sep='_', suffix='\w+').reset_index()
df_the = df_the.reset_index(drop=True)

phi_list = df_the['phi'].unique()
phi_list = phi_list[phi_list<=10.]
phi_list = phi_list[np.linspace(0,int(len(phi_list))-1, 11, dtype=int)]

path_result = 'result/ex3/'
os.makedirs(path_result, exist_ok=True)
print(rho_ar1)
df_emp = pd.DataFrame()
for phi in tqdm(phi_list):
    _df = df_the[(df_the['phi']==phi)&(df_the['Theoretical']=='the')].reset_index(drop=True)
    phi_s = _df.iloc[_df['risk'].argmin()]['phi_s']
    if phi_s<=1:
        print(phi,phi_s)
        continue
    _df = run(phi, phi_s, rho_ar1, sigma)
    df_emp = pd.concat([df_emp, _df])
    df_emp.to_csv(path_result+'res_optimal_rhoar1_{:.02f}.csv'.format(rho_ar1))