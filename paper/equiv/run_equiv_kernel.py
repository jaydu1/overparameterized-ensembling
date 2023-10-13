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
from scipy.special import expit
n_simu = 50
sigma = 0.5
p = 500
phi = 0.05
M = 100
rho_ar1 = 0.

kernel_list = ['poly-3' ,'rbf', 'laplacian']
kernel = kernel_list[int(sys.argv[1])]
    
path_result = 'result/ex6/'
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(X, Y, X_test, Y_test, phi, phi_s, param, M, i):
    np.random.seed(i)

    stat = comp_empirical_risk(
        X, Y, X_test, Y_test, phi_s, 'kernelridge', param, 
        M=M)
    res = np.array([[i, phi, phi_s, param['lam'], stat]])
    return res

np.random.seed(0)
Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data(
    p, phi, rho_ar1, sigma, func='quad', df=np.inf)


phi_s_list = np.logspace(-1,1, 100)
phi_s_list = np.append([0.1, 1., 10., np.inf], phi_s_list)
phi_s_list = np.unique(phi_s_list)

lam_list = np.append([0, 0.1, 1., 10.], np.logspace(-2.,1.,100))
lam_list = np.unique(lam_list)

i = 0
j = int(sys.argv[2])
lam = lam_list[j]
with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
    l = int(sys.argv[2])
    for j in np.arange(101):
        lam = lam_list[j]
        print(j,lam)
        df_res = pd.DataFrame()
        res = parallel(
            delayed(run_one_simulation)(X, Y, X_test, Y_test, phi, phi_s, 
                                        {'lam':lam,
                                         'kernel':kernel.split('-')[0], 
                                         'degree':3 if '-' not in kernel else int(kernel.split('-')[1])
                                        }, M, i) 
            for phi_s in tqdm(phi_s_list, desc='phi_s')# for lam in tqdm(lam_list, desc='lam')
            if (phi_s>=phi)
        )

        res = pd.DataFrame(np.concatenate(res,axis=0), columns=
            ['seed', 'phi', 'phi_s', 'lam', 'risk']
        )
        df_res = pd.concat([df_res, res],axis=0)

        df_res.to_csv('{}res_{}_{}_{}.csv'.format(
            path_result, kernel, i, j), index=False)
