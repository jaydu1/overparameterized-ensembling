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
lam_list = [0., 1e-1, 1.]

rho_ar1_list = [0.25,0.5]
rho_ar1 = rho_ar1_list[int(sys.argv[1])]



path_result = 'result/ex5/'
os.makedirs(path_result, exist_ok=True)
print(rho_ar1)
phi = 0.1
for lam in tqdm(lam_list, desc='lam'):
    df_res_the = pd.DataFrame()
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma)

    phi_s_list = np.logspace(-2,1.5, 1000) if phi<1e-1 else np.logspace(-1,2, 1000)
    phi_s_list[np.where(phi_s_list>=phi)[0][0]] = phi
    phi_s_list = np.append(phi_s_list, np.inf)

    res_the = []
    for phi_s in tqdm(phi_s_list, desc='phi_s'):
        if phi_s<phi or phi_s==1.:
            continue
        
        M_list = np.append(np.arange(1,101), np.inf)
        _, _, risk_the = comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, phi_s, M_list)

        res_the.append(
            np.c_[
                np.full([len(risk_the), 1], phi),
                np.full([len(risk_the), 1], phi_s),
                np.full([len(risk_the), 1], lam),
                M_list[:,None],        
                risk_the[:,None]
            ]
        )
    res_the = np.concatenate(res_the, axis=0)
    res_the = pd.DataFrame(
            res_the, columns=['phi', 'phi_s', 'lam', 'M', 'risk_the']
        )

    df_res_the = pd.concat([df_res_the, res_the], axis=0)

    df_res_the.to_csv('{}res_the_lam_{:.1f}_ar1rho_{:.02f}.csv'.format(
        path_result, lam, rho_ar1), index=False)