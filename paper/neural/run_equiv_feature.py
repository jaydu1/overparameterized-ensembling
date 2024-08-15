import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from generate_data import *
from compute_risk import *
from tqdm import tqdm
from sklearn.kernel_approximation import Nystroem


n_simu = 20
sigma = .5
p = 500
M = 100
rho_ar1 = 0.25
phi = 0.1
n = int(p/phi)

feature_list = ['RMT', 'rf', 'kernel']
feature_name = feature_list[int(sys.argv[1])]


if feature_name == 'RMT':
    method = 'ridge'
    kwargs = {}
    d = p
    func = lambda x: x
    F = np.identity(p)
elif feature_name == 'rf':
    method = 'ridge'
    kwargs = {}
    d = 2 * p
    func = lambda x: x * (x > 0)
    F = np.random.normal(0, 1/np.sqrt(d), size=(d, p))
elif feature_name == 'kernel':
    method = 'kernelridge'
    d = p
    kernel = 'poly'
    # feature_map_nystroem = Nystroem(kernel='laplacian', random_state=1, n_components=p)
    F = np.identity(p)
    func = lambda x: pairwise_kernels(x, degree=3, coef0=0, metric=kernel)#np.sqrt(p) * feature_map_nystroem.fit_transform(x)

replace = False if int(sys.argv[2])==0 else True



path_result = 'result/ex2/{}/{}/'.format(feature_name,int(sys.argv[2]))
os.makedirs(path_result, exist_ok=True)
print(rho_ar1)




def run_one_simulation(X, Y, phi, psi, rho, sigma, lam, a_gau, a_nongau, M, i):
    np.random.seed(i)
    k = int(p/psi)
    stat = comp_empirical_beta_stat(X, Y, k, method, lam, a_gau, a_nongau, M=M, replace=replace)
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
    d, phi, rho_ar1, sigma, func='quad', n_test=1)
X = func(X @ F)

df = 5
a_gau = np.random.normal(0, 1/np.sqrt(p), size=p)
a_nongau = np.random.standard_t(df=df, size=p) / np.sqrt(df / (df - 2)) /np.sqrt(p)



j = int(sys.argv[3])
lam = lam_list[j]
with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:

    for i in range(n_simu):
    # for i in range(j, j+1):
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



# # merge results
# for k in range(2):
#     path_result = 'result/ex2/{}/{}/'.format(feature_name,k)
#     df = pd.DataFrame()
#     for i in range(20):
#         for j in range(101):
#             try:
#                 _df = pd.read_csv('{}res_ar1rho_{:.02f}_{}_{}.csv'.format(
#                             path_result, rho_ar1, i, j))
#                 _df = _df[_df['M']==100]
#                 df = pd.concat([df, _df], axis=0)
#             except:
#                 print(i,j)
#                 pass
#     df.to_pickle('{}res_ar1rho_{:.02f}.pkl'.format(path_result, rho_ar1))
