"""
This script runs simulations to estimate the risk of a predictor on a dataset with varying data and subsample aspect ratios.
The script takes 3 command line arguments:
    - i_func: integer index of the function to use for generating the data (0-2)
    - i_bs: integer index indicating whether to use bootstrap (1) or subagging (0)
    - i_predictor: integer index indicating which predictor to use (0-5)
The script generates data, computes the empirical ecv estimates, and saves the results in CSV files.
The results can be summarized by running the code block at the end of the script.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from compute_risk import *
from generate_data import *
from joblib import Parallel, delayed



i_func = int(sys.argv[1])
i_bs = int(sys.argv[2])
i_predictor = int(sys.argv[3])

func_list = ['quad', 'tanh', 'linear']
func = func_list[i_func]

bootstrap = i_bs==1
bagging = 'bagging' if bootstrap else 'subagging'

predictor_list = ['ridgeless', 'lassoless', 'ridge', 'lasso', 'logistic', 'kNN']
predictor = predictor_list[i_predictor]

path_result = 'result/ex2/{}/{}/{}/'.format(func, bagging, predictor)
os.makedirs(path_result, exist_ok=True)

if predictor in ['tree', 'kNN']:
    lam = None
else:
    lam = 0.1 if predictor in ['ridge', 'lasso'] else 0.



def run_one_simulation(n, phi, rho_ar1, sigma, predictor, lam, M, i, bootstrap=bootstrap):
    np.random.seed(i)
    p = int(n*phi)

    # Generate data
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data(
        n, p, coef='eig-5', func=func,
        rho_ar1=rho_ar1, sigma=sigma, df=np.inf, n_test=2000, sigma_quad=1.)
    
    if predictor=='logistic':
        Y_test = np.where(Y_test>=np.median(Y), 1, 0)
        Y = np.where(Y>=np.median(Y), 1, 0)
    
    # Cross-validation
    try:
        k_list, risk_val, risk_test = cross_validation_oob(
            X, Y, X_test, Y_test, predictor, lam, M, M0=M0, M_test=M, 
            bootstrap=bootstrap, return_full=True)
        res = np.concatenate([
            np.full((len(k_list),1), phi), np.full((len(k_list),1), i), 
            k_list[:,None], risk_val, risk_test], axis=-1)
    except:
        res = np.zeros((0,103))
    return res


M = 50
M0 = 10
n = 1000
n_simu = 50
sigma = 0.5
rho_ar1 = 0.5
phi_list = np.logspace(-1, 1, 25)
with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
    file_saved = path_result+'res_cv_rho_{:.02f}.csv'.format(rho_ar1)
    df_res = pd.DataFrame()
    for i_phi in tqdm(np.arange(len(phi_list)), desc = 'phi'):
        phi = phi_list[i_phi]

        res = parallel(
            delayed(run_one_simulation)(n, phi, rho_ar1, sigma, predictor, lam, M, i) for i in tqdm(range(n_simu))
        )

        res = pd.DataFrame(np.concatenate(res,axis=0), columns=np.concatenate(
            [['phi', 'seed', 'k_list'],
            np.char.add('val-', np.arange(1,M+1).astype(str)),
            np.char.add('test-', np.arange(1,M+1).astype(str))]
        ))
        df_res = pd.concat([df_res, res],axis=0)
        df_res.to_csv(file_saved, index=False)
        


# The results can be summarized by the following code for plotting:
# import pandas as pd
# predictor_list = ['ridgeless', 'lassoless', 'ridge', 'lasso', 'logistic', 'kNN']   
# func_list = ['quad', 'tanh','linear']
# ar1rho = 0.5
# for func in func_list:
#     for bagging in ['bagging', 'subagging']:
#         for predictor in predictor_list:
#             path_result = 'result/ex2/{}/{}/{}/'.format(func, bagging,predictor)
#             file_saved = path_result+'res_cv_rho_{:.02f}.csv'.format(ar1rho)
#             _df = pd.read_csv(file_saved)
#             _df['ar1rho'] = ar1rho
#             _df.to_pickle('result/ex2/res_cv_{}_{}_{}.pkl'.format(
#                 func, bagging, predictor), compression='gzip')
        