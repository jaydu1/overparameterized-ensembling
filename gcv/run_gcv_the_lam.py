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
#             gcv_emp, risk_emp = comp_empirical_gcv(
#                     X, Y, X_test, Y_test, _phis, 
#                     _lam if _lam>0. else 1e-8, M=M, return_allM=False)
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
df_the = pd.read_csv('result/gcv/ex2/res_the_lam_{:.01f}_ar1rho_{:.02f}.csv'.format(lam, rho_ar1))
df_the.columns = ['phi', 'phi_s', 'lam', 'risk_gcv', 'risk_the']
df_the = pd.wide_to_long(df_the, stubnames='risk', i=['phi', 'phi_s', 'lam'], j='Theoretical', sep='_', suffix='\w+').reset_index()
df_the = df_the.reset_index(drop=True)

phi_list = df_the['phi'].unique()
phi_list = phi_list[phi_list<=10.]
phi_list = phi_list[np.linspace(0,int(len(phi_list))-1, 11, dtype=int)]

path_result = 'result/ex3/'
os.makedirs(path_result, exist_ok=True)
print(rho_ar1)

df_res_the = pd.DataFrame()

lam_list = np.logspace(-2, 1.5, 1000)
for phi in tqdm(phi_list):
    _df = df_the[(df_the['phi']==phi)&(df_the['Theoretical']=='the')].reset_index(drop=True)
    
    res_the = []
    lam = 0.
    idx = (df_the['phi']==phi)&(df_the['phi_s']==phi)&(df_the['Theoretical']=='the')
    risk_the = df_the[idx].iloc[0]['risk']
    idx = (df_the['phi']==phi)&(df_the['phi_s']==phi)&(df_the['Theoretical']=='gcv')
    gcv_the = df_the[idx].iloc[0]['risk']
    res_the.append([phi, lam, gcv_the, risk_the])
    
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma)
    
    for lam in lam_list:    
        gcv_the, v, tc, tv, tv_s = comp_theoretic_gcv_inf(Sigma, beta0, sigma2, lam, phi, phi)
        _, _, risk_the = comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, phi, 1,
                                                     v=v, tc=tc, tv=tv, tv_s=tv_s)
        res_the.append([phi, lam, gcv_the, risk_the])
    res_the = pd.DataFrame(
            res_the, columns=['phi', 'lam', 'gcv_the', 'risk_the']
        )

    
    df_res_the = pd.concat([df_res_the, res_the], axis=0)
    df_res_the.to_csv('{}res_the_lam_ar1rho_{:.02f}.csv'.format(
        path_result, rho_ar1), index=False)
    