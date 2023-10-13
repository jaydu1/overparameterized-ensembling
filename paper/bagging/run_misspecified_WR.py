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


def run(phi, lam, rho_ar1, sigma=1):
    p = 500
    n_simu = 100
    M = 100
    
    res = []
    phi_s_list = np.logspace(-1,1, 50)
    phi_s_list[np.where(phi_s_list>=phi)[0][0]] = phi
    for phi_s in tqdm(phi_s_list):
        if phi_s<phi or (lam==0. and phi_s==1.):
            continue
            
        for i in range(n_simu):
            np.random.seed(i)
            Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma, misspecified=True)
            risk_emp = comp_empirical_risk(X, Y, X_test, Y_test, phi_s, lam, 
                                           M=M, replace=True, return_allM=True)
            res.append(np.append([phi_s, i, rho2, sigma2], risk_emp))

    res = pd.DataFrame(res, columns=np.append(['phi_s', 'seed', 'rho2', 'sigma2'], np.arange(1,M+1)))
    res.to_csv(path_result+'res_rhoar1_{:.02f}_phi_{:.01f}_lam_{:.01f}_with_replacement_misspecified.csv'.format(
        rho_ar1, phi, lam))



lam_list = np.linspace(0.,1.,11)
lam = lam_list[int(sys.argv[1])]

rho_ar1_list = [0.25, 0.5, 0.75]
rho_ar1 = rho_ar1_list[int(sys.argv[2])]
sigma = 1.
for phi in [0.1, 1.1]:
#             if os.path.exists(
#                 path_result+'res_phi_{:.01f}_lam_{:.01f}_SNR_{:.01f}_with_replacement.csv'.format(phi, lam, SNR)):
#                 continue
    print(phi, lam, rho_ar1, sigma)
    run(phi, lam, rho_ar1, sigma)
