import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from generate_data import *
from compute_risk import *
from tqdm import tqdm

n_simu = 20
sigma = .5
p = 1000
M = 100
rho_ar1_list = [0., 0.25,0.5]
rho_ar1 = rho_ar1_list[int(sys.argv[1])]
phi = 0.1

if int(sys.argv[2])==0:
    replace = False
    weight = None
elif int(sys.argv[2])==1:
    replace = True
    weight = None
elif int(sys.argv[2])==2:
    replace = False
    weight = (9/10)**np.arange(int(p/phi))

path_result = 'result/ex1/{}/'.format(int(sys.argv[2]))
os.makedirs(path_result, exist_ok=True)
print(rho_ar1)




def run_one_simulation(X, Y, phi, psi, rho, sigma, lam, a_gau, a_nongau, M, i):
    np.random.seed(i)
    k = int(p/psi)
    
    stat = comp_empirical_beta_stat(X, Y, k, 'ridge', lam, a_gau, a_nongau, M=M, replace=replace, weight=weight)
    res = np.c_[np.full(M, i), np.full(M, phi), np.full(M, psi), 
              np.full(M, lam), np.arange(1,M+1), stat]
    return res


psi_list = np.logspace(-1,1, 100)
psi_list = np.append([0.1, 1., 10., np.inf], psi_list)
psi_list = np.unique(psi_list)

lam_list = np.append([0, 0.1, 1., 10.], np.logspace(-2.,1.,100))
lam_list = np.unique(lam_list)


np.random.seed(0)
Sigma, beta0, X, Y, _, _, rho2, sigma2 = generate_data(
    p, phi, rho_ar1, sigma, func='quad', n_test=1)
a_gau = np.random.normal(0, 1/np.sqrt(p), size=p)
df = 5
a_nongau = np.random.standard_t(df=df, size=p) / np.sqrt(df / (df - 2)) /np.sqrt(p)


j = int(sys.argv[3])
lam = lam_list[j]
with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:

    for i in range(n_simu):
    # for i in range(j, j+1):
        if os.path.exists('{}res_ar1rho_{:.02f}_{}_{}.csv'.format(
            path_result, rho_ar1, i, j)):
            continue

        df_res = pd.DataFrame()
        res = parallel(
            delayed(run_one_simulation)(X, Y, phi, psi, rho_ar1, sigma, lam, a_gau, a_nongau, M, i) 
            for psi in tqdm(psi_list, desc='psi') #for lam in tqdm(lam_list, desc='lam')
            if (psi>=phi)
        )

        res = pd.DataFrame(np.concatenate(res,axis=0), columns=
            ['seed', 'phi', 'psi', 'lam', 'M', 'min', 'max',
             'mean', 'median', 'std', 'random_gau', 'random_t', 'dof']
        )
        df_res = pd.concat([df_res, res],axis=0)

        df_res.to_csv('{}res_ar1rho_{:.02f}_{}_{}.csv'.format(
            path_result, rho_ar1, i, j), index=False)


## merge results
# df = pd.DataFrame()
# for i in range(20):
#     for j in range(101):
#         try:
#             _df = pd.read_csv('{}res_ar1rho_{:.02f}_{}_{}.csv'.format(
#                         path_result, rho_ar1, i, j))
#             _df = _df[_df['M']==100]
#             df = pd.concat([df, _df], axis=0)
#         except:
#             print('{}res_ar1rho_{:.02f}_{}_{}.csv'.format(
#                         path_result, rho_ar1, i, j))
#             pass       
# df.to_pickle('{}res_ar1rho_{:.02f}.pkl'.format(path_result, rho_ar1))