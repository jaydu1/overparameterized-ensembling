"""
This script runs simulations to evaluate the performance of a classification algorithm on generated data.
The script takes 1 command line argument:
    - i_prop: integer index of the proportion of positive class labels to use in the logistic regression algorithm (0-4)
The results of the simulations are saved to a CSV file and can be summarized by running the code block at the end of the script.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from compute_risk import *
from generate_data import *
from joblib import Parallel, delayed



func = 'quad'
bootstrap = True
bagging = 'bagging'
predictor = 'logistic'
prop_list = [0.5,0.6,0.7,0.8,0.9]
prop = prop_list[int(sys.argv[1])]
K_list = [3,5,10]
lam = 0.

M = 50
M0 = 10
n = 1000
n_simu = 50
sigma = 0.5
rho_ar1 = 0.5
path_result = 'result/ex3/{}/{}/{}/'.format(func, bagging, predictor)
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(n, phi, rho_ar1, sigma, predictor, lam, M, i, bootstrap=bootstrap):
    np.random.seed(i)
    p = int(n*phi)

    # Generate data
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data(
        n, p, coef='eig-5', func=func,
        rho_ar1=rho_ar1, sigma=sigma, df=np.inf, n_test=2000, sigma_quad=1.,        
    )
    
    if predictor=='logistic':
        binarize_thres = np.quantile(Y, prop)
        Y_test = np.where(Y_test>=binarize_thres, 1, 0)
        Y = np.where(Y>=binarize_thres, 1, 0)

    # ECV        
    k_list, risk_val, risk_test = cross_validation_oob(
        X, Y, X_test, Y_test, predictor, lam, M, M0=M0, M_test=M, 
        bootstrap=bootstrap, return_full=True)
    
    # K-fold CV
    _k_lists = []
    for K in K_list:
        _k_list, _risk_val, _risk_test = cross_validation(
            X, Y, X_test, Y_test, 
            predictor, lam, M, Kfold=K, k_list=k_list,
            return_full=True, bootstrap=bootstrap)
        _k_lists.append(_k_list)
        risk_val = np.r_[risk_val, _risk_val]
        risk_test = np.r_[risk_test, _risk_test]

    _k_lists_r = np.concatenate(_k_lists)
    res = np.concatenate([
        np.full((len(k_list)+len(_k_lists_r),1), phi), 
        np.full((len(k_list)+len(_k_lists_r),1), i),
        np.r_[k_list, _k_lists_r][:,None], 
        risk_val, risk_test,
        np.r_[np.full(len(k_list), np.nan), np.repeat(K_list, [len(l) for l in _k_lists])][:,None]
    ], axis=-1)

    return res



with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
    file_saved = path_result+'res_prop_{:.01f}.csv'.format(prop)
    df_res = pd.DataFrame()
    for phi in tqdm([0.1,10.], desc = 'phi'):

        res = parallel(
            delayed(run_one_simulation)(n, phi, rho_ar1, sigma, predictor, lam, M, i) for i in tqdm(range(n_simu))
        )

        res = pd.DataFrame(np.concatenate(res,axis=0), columns=np.concatenate(
            [['phi', 'seed', 'k_list'],
            np.char.add('val-', np.arange(1,M+1).astype(str)),
            np.char.add('test-', np.arange(1,M+1).astype(str)),
            ['K']]
        ))
        df_res = pd.concat([df_res, res],axis=0)
        df_res.to_csv(file_saved)

        
        
# The results can be summarized by the following code for plotting:
# import pandas as pd
# func = 'quad'
# bootstrap = True
# bagging = 'bagging'
# predictor = 'logistic'
# prop_list = [0.5,0.6,0.7,0.8,0.9]
# path_result = 'result/ex3/{}/{}/{}/'.format(func, bagging,predictor)
# df = pd.DataFrame()
# for prop in prop_list:
#     file_saved = path_result+'res_prop_{:.01f}.csv'.format(prop)
#     _df = pd.read_csv(file_saved)
#     _df['prop'] = prop
#     df = pd.concat([df,_df], axis=0)
# df.to_pickle('result/ex3/res_classification_{}_{}.pkl'.format(func,bagging), compression='gzip')