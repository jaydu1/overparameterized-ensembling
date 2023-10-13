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
method_list = ['ridge', 'lasso', 'logistic', 'kNN']
method = method_list[int(sys.argv[1])]
path_result = 'result/ex1/{}/{}/'.format(bagging, method)
os.makedirs(path_result, exist_ok=True)



def run(phi, method, lam, SNR, bootstrap=bootstrap):
    n = 1000

    p = int(n*phi)
    n_simu = 100

    rho = 1.
    sigma = rho / np.sqrt(SNR)
    M0 = 10 # number of bags for computing the OOB estimates
    M_max = 50
    M_list = np.append(np.arange(1,M_max+1), np.inf)
    M_str = ['{:d}'.format(int(M)) for M in M_list[:-1]] + ['$\infty$']
    
    res = []
    res_key = []
    phi_s_list = np.logspace(-1,1, 50)
    phi_s_list[np.where(phi_s_list>=phi)[0][0]] = phi
    for phi_s in tqdm(phi_s_list):
        if phi_s<phi or (lam==0. and phi_s==1.):
            continue
            
        for i in range(n_simu):
            np.random.seed(i)
            beta0, X, Y, X_test, Y_test = generate_data(n, phi, rho, sigma)
            if method=='logistic':
                Y_test = np.where(Y_test>=np.median(Y), 1, 0)
                Y = np.where(Y>=np.median(Y), 1, 0)

            oobcv_emp, risk_emp = comp_empirical_oobcv(
                X, Y, X_test, Y_test, phi_s, 
                method, lam, M0=M0, M=M_list, M_test=M_max, bootstrap=bootstrap)
#             B0, V0, risk_the = comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M_list)
            
            _df = pd.DataFrame(
                np.stack([
                    oobcv_emp, 
                    np.append(risk_emp, np.full(len(M_list)-M_max, np.nan)),
                ]), index = ['oobcv_emp', 'risk_emp'], columns=M_str
            )
    
            res.append(_df)
            res_key.append((phi_s,i))
            
    res = pd.concat(res, axis=0, keys=res_key, names=['phi_s','seed'])
    if lam is None:
        res.to_csv(path_result+'res_phi_{:.01f}_SNR_{:.02f}.csv'.format(
            phi, SNR))
    else:
        res.to_csv(path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.02f}.csv'.format(
            phi, lam, SNR))
    
    
if method in ['tree', 'kNN']:
    lam = None
else:
    lam_list = np.linspace(0.,1.,11)

    lam = lam_list[int(sys.argv[2])]
    
for phi in [0.1, 1.1]:
    for SNR in [0.01, 0.25, 1.]:
        print(phi, method, lam, SNR)
        run(phi, method, lam, SNR)
