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


sigma = 1.
p = 500

rho_ar1_list = [0.25,0.5]
rho_ar1 = rho_ar1_list[int(sys.argv[1])]



path_result = 'result/ex1/'
os.makedirs(path_result, exist_ok=True)
print(rho_ar1)
phi = 0.1
df_res_the = pd.DataFrame()


Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma)

phi_s_list = np.logspace(-1,1, 500)
phi_s_list = np.append([0.1, 1., 10., np.inf], phi_s_list)
phi_s_list = np.unique(phi_s_list)

lam_list = np.append([0, 0.1, 1., 10.], np.logspace(-2.,1.,500))
lam_list = np.unique(lam_list)
for lam in tqdm(lam_list, desc='lam'):
    res_the = []
    for phi_s in tqdm(phi_s_list, desc='phi_s'):
        if phi_s<phi or (lam==0 and phi_s==1.):
            continue

        gcv_the, v, tc, tv, tv_s = comp_theoretic_gcv_inf(Sigma, beta0, sigma2, lam, phi, phi_s)
        _, _, risk_the = comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, phi_s, np.inf,
                                                     v=v, tc=tc, tv=tv, tv_s=tv_s)

        res_the.append([phi, phi_s, lam, gcv_the, risk_the])


    res_the = pd.DataFrame(
            res_the, columns=['phi', 'phi_s', 'lam', 'gcv_the', 'risk_the']
        )

    df_res_the = pd.concat([df_res_the, res_the], axis=0)

    df_res_the.to_csv('{}res_the_ar1rho_{:.02f}.csv'.format(
        path_result, rho_ar1), index=False)