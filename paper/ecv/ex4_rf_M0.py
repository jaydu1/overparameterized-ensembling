'''
This script generates data and computes ECV estimates for random forests on data with different hyperparameters M0.
The script takes 1 command line argument:
    - i_M0: integer index indicating which hyperparameter M0 to use (0-4)
The results are saved to a CSV file and can be summarized by running the code block at the end of the script.
'''
import os
import sys
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from compute_risk import *
from generate_data import *               
from joblib import Parallel, delayed



func = 'quad'
bootstrap = True
bagging = 'bagging'
predictor = 'tree'
path_result = 'result/ex4/{}/{}/'.format(bagging,predictor)
os.makedirs(path_result, exist_ok=True)


def oobcv_tree(X, Y, X_test, Y_test, predictor, M_max, M0, bootstrap=bootstrap):
    n, p = X.shape
    
    # Compute the time of cross-validation
    t0 = time.time()
    _, _ = comp_empirical_oobcv(
        X, Y, X_test[:1,:], Y_test[:1],
        p/n, predictor, None, M0=M0, M=M_list, M_test=M0, bootstrap=bootstrap)
    t1 = time.time() - t0
    
    # Compute the ecv estimate and the test risk
    res_oob, res_test = comp_empirical_oobcv(X, Y, X_test, Y_test, 
        p/n, predictor, None, M0=M0, M=M_list, M_test=M_max, bootstrap=bootstrap)
    
    return res_oob, np.append(res_test, np.full(len(M_list)-M_max, np.nan)), t1

                              
def run_one_simulation(n, phi, rho_ar1, sigma, M, M0, i):
    np.random.seed(i)
    p = int(n*phi)
    
    # Generate data
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data(
        n, p, coef='eig-5', func=func,
        rho_ar1=rho_ar1, sigma=sigma, df=np.inf, n_test=2000, sigma_quad=1.)
    
    # Compute the empirical risk estimates
    oobcv_emp, risk_emp, t1 = oobcv_tree(X, Y, X_test, Y_test, predictor, M, M0)
    res = np.c_[
        np.full((2,1), phi), np.full((2,1), i), np.array(['oobcv_emp', 'risk_emp'])[:,None],
        np.full((2,1), rho2), np.full((2,1), sigma2), 
        np.stack([oobcv_emp, risk_emp]),
        np.full((2,1), t1)
    ]
    return res



n = 500
n_simu = 50
sigma = .5
rho_ar1 = 0.5
M_max = 500
M_list = np.append(np.arange(1,M_max+1), np.inf)
M_str = ['{:d}'.format(int(M)) for M in M_list[:-1]] + ['$\infty$']
M0_list = [5,10,15,20,25]
i_M0 = int(sys.argv[1])
M0 = M0_list[i_M0]


with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:        
    file_saved = path_result+'res_M0_{:d}.csv'.format(M0)
    df_res = pd.DataFrame()
    for phi in tqdm([0.1,10], desc = 'phi'):
        res = parallel(
            delayed(run_one_simulation)(n, phi, rho_ar1, sigma, M_max, M0, i) for i in tqdm(range(n_simu))
        )
        res = pd.DataFrame(np.concatenate(res, axis=0), columns=
                           np.append(np.append(['phi', 'seed', 'type', 'rho2', 'sigma2'], M_str), ['time']))
        res['M0'] = M0
        df_res = pd.concat([df_res, res],axis=0)
        df_res.to_csv(file_saved, index=False)
        
        
# The results can be summarized by the following code for plotting:
# import pandas as pd
# func = 'quad'
# bootstrap = True
# bagging = 'bagging'
# predictor = 'tree'
# M0_list = [5,10,15,20,25]
# path_result = 'result/ex4/{}/{}/'.format(bagging,predictor)
# df = pd.DataFrame()
# for M0 in M0_list:
#     file_saved = path_result+'res_M0_{:d}.csv'.format(M0)
#     _df = pd.read_csv(file_saved)
#     _df['M0'] = M0
#     df = pd.concat([df,_df], axis=0)
# df.to_pickle('result/ex4/res_M0.pkl', compression='gzip')