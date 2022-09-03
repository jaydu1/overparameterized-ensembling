import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tqdm.contrib import tzip
import matplotlib.pyplot as plt
sns.set_theme()
from compute_risk import *


path_result = 'result/ex1_new/'
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
    M = 100
    
    res = []

    
    M_list = np.arange(1,100+1)
    phis_list = (n * phi) / (n / M_list)
    for M, phi_s in tzip(M_list, phis_list):
        if lam==0. and phi_s==1.:
            continue
        for i in range(n_simu):
            np.random.seed(i)
            beta0, X, Y, X_test, Y_test = generate_data(n, phi, rho, sigma)

            risk_emp = comp_empirical_risk(X, Y, X_test, Y_test, phi_s, lam, 
                                           M=M, replace=False, return_allM=False)
            res.append([phi_s, M, i, risk_emp])

    res = pd.DataFrame(res, columns=['phi_s', 'M', 'seed', 'risk_emp'])
    res.to_csv(path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}_without_replacement.csv'.format(
        phi, lam, SNR))

lam_list = np.linspace(0.,1.,11)#np.append(0, np.logspace(-9, 0, 10))

for lam in [lam_list[int(sys.argv[1])]]:
    for phi in [0.1, 1.5]:
        for SNR in [1., 2., 3, 4.]:
    #             if os.path.exists(
    #                 path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}.csv'.format(phi, lam, SNR)):
    #                 continue
            print(phi, lam, SNR)
            run(phi, lam, SNR)
