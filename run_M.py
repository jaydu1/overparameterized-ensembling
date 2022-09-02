import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
from compute_risk import *


path_result = 'result/ex1/'
os.makedirs(path_result, exist_ok=True)

def generate_data(n, phi, rho=1., sigma=1.):
    p = int(n*phi)

    if rho>0.:
        beta0 = np.random.randn(p)
#         beta0 = beta0 / np.sqrt(np.sum(beta0**2)) * rho
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

def run(phi, lam, SNR):
    n = 1000

    p = int(n*phi)
    n_simu = 100

    rho = 1.
    sigma = rho / np.sqrt(SNR)

    res = []
    phi_s_list = np.logspace(-1,1, 50)
    phi_s_list[np.where(phi_s_list>=phi)[0][0]] = phi
    for phi_s in phi_s_list:
        if phi_s<phi or (lam==0. and phi_s==1.):
            continue
            
        phi_0 = phi_s ** 2 / phi # p/k * k/i0
        
        for M in np.append(np.arange(1,11),[15, 20, 25, 50, 100]):
            _, _, risk_the = comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M=M)
            for i in range(n_simu):
                np.random.seed(i)
                beta0, X, Y, X_test, Y_test = generate_data(n, phi, rho, sigma)

                risk_emp = comp_empirical_risk(X, Y, X_test, Y_test, phi_s, lam, M=M)
                res.append([phi_s, i, M,
                            risk_emp - sigma**2,
                            risk_the - sigma**2,
                           ])

    res = pd.DataFrame(res, columns=['phi_s', 'seed', 'M', 'risk_emp', 'risk_the'])
#     res = pd.read_csv(path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}.csv'.format(phi, lam, SNR),index_col=0)
#     res = pd.concat([res, _res]).reset_index(drop=True)
    res.to_csv(path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}.csv'.format(phi, lam, SNR))

lam_list = np.linspace(0.,1.,11)#np.append(0, np.logspace(-9, 0, 10))

for lam in [lam_list[int(sys.argv[1])]]:
    for phi in [0.1, 0.5, 1., 1.5]:
        for SNR in [1., 2., 3, 4.]:
#             if os.path.exists(
#                 path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}.csv'.format(phi, lam, SNR)):
#                 continue
            print(phi, lam, SNR)
            run(phi, lam, SNR)
