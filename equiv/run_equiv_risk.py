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

n_simu = 50
sigma = 0.5
p = 500
M = 100
rho_ar1_list = [0.25,0.5]
rho_ar1 = rho_ar1_list[int(sys.argv[1])]



path_result = 'result/ex2/'
os.makedirs(path_result, exist_ok=True)
print(rho_ar1)

phi = 0.1


def run_one_simulation(X, Y, phi, phi_s, rho, sigma, lam, beta0, Sigma, Sigma_out, M, i):
    np.random.seed(i)

    stat = comp_empirical_generalized_risk(X, Y, phi_s, 'ridge', lam, beta0, Sigma, Sigma_out, M=M)
    res = np.c_[np.full(M, i), np.full(M, phi), np.full(M, phi_s), 
              np.full(M, lam), np.arange(1,M+1), stat]
    return res


phi_s_list = np.logspace(-1,1, 100)
phi_s_list = np.append([0.1, 1., 10., np.inf], phi_s_list)
phi_s_list = np.unique(phi_s_list)

lam_list = np.append([0, 0.1, 1., 10.], np.logspace(-2.,1.,100))
lam_list = np.unique(lam_list)


np.random.seed(0)
Sigma, beta0, X, Y, _, _, rho2, sigma2 = generate_data(
    p, phi, rho_ar1, sigma, func='quad', n_test=1)
a_gau = np.random.normal(0, 1/np.sqrt(p), size=p)
df = 5
a_nongau = np.random.standard_t(df=df, size=p) / np.sqrt(df / (df - 2)) /np.sqrt(p)

Sigma_out, _, _, _, _, _, _, _ = generate_data(
    p, phi, 0.25, sigma, func='quad', n_test=1)




j = int(sys.argv[2])
lam = lam_list[j]
with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
#         for phi_s in tqdm(phi_s_list, desc='phi_s'):
#             if phi_s<phi or (lam==0 and phi_s==1.):
#                 continue
    for i in range(n_simu):
        df_res = pd.DataFrame()
        res = parallel(
            delayed(run_one_simulation)(X, Y, phi, phi_s, rho_ar1, sigma, lam, 
                                        beta0, Sigma, Sigma_out, M, i) 
            for phi_s in tqdm(phi_s_list, desc='phi_s')# for lam in tqdm(lam_list, desc='lam')
            if (phi_s>=phi)
        )

        res = pd.DataFrame(np.concatenate(res,axis=0), columns=
            ['seed', 'phi', 'phi_s', 'lam', 'M', 'est_err', 'train_err', 'pred_in', 'pred_out']
        )
        df_res = pd.concat([df_res, res],axis=0)

        df_res.to_csv('{}res_ar1rho_{:.02f}_{}_{}.csv'.format(
            path_result, rho_ar1, i, j), index=False)
