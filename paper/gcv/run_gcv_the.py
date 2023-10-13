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
lam = lam_list[int(sys.argv[1])]

rho_ar1_list = [0.25,0.5]
rho_ar1 = rho_ar1_list[int(sys.argv[2])]



path_result = 'result/ex2/'
os.makedirs(path_result, exist_ok=True)
print(lam, rho_ar1)
df_res_the = pd.DataFrame()
for phi in tqdm(
    np.concatenate([
        np.logspace(-2, -1, 50, endpoint=False), 
        np.logspace(-1, 1, 200, endpoint=False), [1.1],
        np.logspace(1, 1.5, 25)]
    )
):

    if (phi < 0.1) and (lam > 0.):
        continue

    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma)

    phi_s_list = np.logspace(-2,1.5, 1000) if phi<1e-1 else np.logspace(-1,2, 1000)
    phi_s_list[np.where(phi_s_list>=phi)[0][0]] = phi
    phi_s_list = np.append(phi_s_list, np.inf)

    res_the = []
    for phi_s in phi_s_list:
        if phi_s<phi or phi_s==1.:
            continue

        gcv_the, v, tc, tv, tv_s = comp_theoretic_gcv_inf(Sigma, beta0, sigma2, lam, phi, phi_s)
        _, _, risk_the = comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, phi_s, np.inf,
                                                     v=v, tc=tc, tv=tv, tv_s=tv_s)

        res_the.append([phi, phi_s, lam, gcv_the, risk_the])


    res_the = pd.DataFrame(
            res_the, columns=['phi', 'phi_s', 'lam', 'gcv_the', 'risk_the']
        )

    df_res_the = pd.concat([df_res_the, res_the], axis=0)

    df_res_the.to_csv('{}res_the_lam_{:.1f}_ar1rho_{:.02f}.csv'.format(
        path_result, lam, rho_ar1), index=False)