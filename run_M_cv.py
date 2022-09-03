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

def run(lam, SNR):
    n = 1000
    n_simu = 100
    rho = 1.
    M = 100
    sigma = rho / np.sqrt(SNR)
    res = []
    
    phi_list = np.logspace(-1,1, 50)
    for phi in tqdm(phi_list):
        for i in range(n_simu):
            np.random.seed(i)
            beta0, X, Y, X_test, Y_test = generate_data(n, phi, rho, sigma)

            risk_emp_cv_M = cross_validation(X, Y, X_test, Y_test, lam, M, nu=0.6, replace=True)
            res.append(np.append([phi, i], risk_emp_cv_M))

    res = pd.DataFrame(res, columns=np.append(['phi_s', 'seed'], np.arange(1,M+1)))
    res.to_csv(path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}.csv'.format(phi, lam, SNR))

lam_list = np.linspace(0.,1.,11)#np.append(0, np.logspace(-9, 0, 10))

for lam in [lam_list[int(sys.argv[1])]]:
    for SNR in [1., 2., 3, 4.]:
#             if os.path.exists(
#                 path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}.csv'.format(phi, lam, SNR)):
#                 continue
        print(lam, SNR)
        run(lam, SNR)
