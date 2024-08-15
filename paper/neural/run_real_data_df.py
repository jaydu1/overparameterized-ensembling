import os
import sys


import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from generate_data import *
from compute_risk import *
from tqdm import tqdm

from sklearn_ensemble_cv import ECV
from sklearn_ensemble_cv.cross_validation import process_grid, comp_empirical_ecv, fit_ensemble
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler




data = "FashionMNIST"; model = 'resnet-18_init'
# data = 'CIFAR-10'; model = 'resnet-18_pretrained'
# data = 'Flowers-102'; model = 'resnet-50_pretrained'

path_result = 'result/ex4/{}_{}/'.format(data, model)
os.makedirs(path_result, exist_ok=True)

with open('data_{}_{}.npz'.format(data, model), 'rb') as f:
    data = np.load(f)
    X = data['X_train']
    X_test = data['X_test']
    Ys = data['Y_train']
    Ys_test = data['Y_test']

n, p = X.shape

X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(p)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True) * np.sqrt(p)

scaler = StandardScaler(with_std=False).fit(Ys)
Ys = scaler.transform(Ys)
Ys_test = scaler.transform(Ys_test)

Y, Y_test = Ys[:,:1], Ys_test[:,:1]



def run_one_simulation(X, Y, phi, psi, lam, a_gau, a_nongau, M, i):
    np.random.seed(i)
    k = int(p/psi)
    stat = comp_empirical_beta_stat(X, Y, k, 'ridge', lam, a_gau, a_nongau, M=M, replace=False)
    res = np.c_[np.full(M, i), np.full(M, phi), np.full(M, psi), 
              np.full(M, lam), np.arange(1,M+1), stat]
    return res

phi = p/n
psi_list = np.logspace(-2, 1, 100)
psi_list = np.append([phi, 1., np.inf], psi_list)
psi_list = np.unique(psi_list)

lam_list = np.append([0, 0.1, 1., 10.], np.logspace(-3.,1.,100))
lam_list = np.unique(lam_list)


a_gau = np.random.normal(0, 1/np.sqrt(p), size=p)
df = 5
a_nongau = np.random.standard_t(df=df, size=p) / np.sqrt(df / (df - 2)) /np.sqrt(p)
M = 100
n_simu = 20


j = int(sys.argv[1])
lam = lam_list[j]
with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:

    for i in range(n_simu):
    # for i in range(j, j+1):
        df_res = pd.DataFrame()
        res = parallel(
            delayed(run_one_simulation)(X, Y, phi, psi, lam, a_gau, a_nongau, M, i) 
            for psi in tqdm(psi_list, desc='psi') #for lam in tqdm(lam_list, desc='lam')
            if (psi>=phi)
        )

        res = pd.DataFrame(np.concatenate(res,axis=0), columns=
            ['seed', 'phi', 'psi', 'lam', 'M', 'min', 'max',
             'mean', 'median', 'std', 'random_gau', 'random_t', 'dof']
        )
        df_res = pd.concat([df_res, res],axis=0)

        df_res.to_csv('{}res_df_{}_{}.csv'.format(
            path_result, i, j), index=False)



# path_result = 'result/ex4/{}_{}/'.format(data, model)
# df = pd.DataFrame()
# for i in range(20):
#     for j in range(103):
#         try:
#             _df = pd.read_csv('{}res_df_{}_{}.csv'.format(
#                         path_result, i, j))
#             _df = _df[_df['M']==100]
#             df = pd.concat([df, _df], axis=0)
#         except:
#             print(i,j)
#             pass
# df.to_pickle('{}res_df.pkl'.format(path_result))            