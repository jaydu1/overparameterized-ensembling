"""
This script runs simulations to estimate the risk of a predictor on a dataset with varying subsample aspect ratios.
The script takes 3 command line arguments:
    - i_func: integer index of the function to use for generating the data (0-2)
    - i_bs: integer index indicating whether to use bootstrap (1) or subagging (0)
    - i_predictor: integer index indicating which predictor to use (0-5)
The script generates data, computes the empirical ecv estimates and risks, and saves the results in CSV files.
The results can be summarized by running the code block at the end of the script.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from compute_risk import *
from generate_data import *



i_func = int(sys.argv[1])
i_bs = int(sys.argv[2])
i_predictor = int(sys.argv[3])

func_list = ['quad', 'tanh', 'linear']
func = func_list[i_func]

bootstrap = i_bs==1
bagging = 'bagging' if bootstrap else 'subagging'

predictor_list = ['ridgeless', 'lassoless', 'ridge', 'lasso', 'logistic', 'kNN']
predictor = predictor_list[i_predictor]

path_result = 'result/ex1/{}/{}/{}/'.format(func,bagging, predictor)
os.makedirs(path_result, exist_ok=True)

if predictor in ['tree', 'kNN']:
    lam = None
else:
    lam = 0.1 if predictor in ['ridge', 'lasso'] else 0.



def run(phi, predictor, rho_ar1, seed, bootstrap=bootstrap):
    p = 500 if phi==0.1 else 5000
    n = int(p/phi) # 5000 or 500
    
    # Generate data
    np.random.seed(seed)
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data(
        n, p, coef='eig-5', func=func,
        rho_ar1=rho_ar1, sigma=sigma, df=np.inf, n_test=2000, sigma_quad=1.)

    if predictor=='logistic':
        Y_test = np.where(Y_test>=np.median(Y), 1, 0)
        Y = np.where(Y>=np.median(Y), 1, 0)
    
    # Generate the grid of subsample aspect ratios
    if phi<1:
        phi_s_list = np.logspace(-1, 1, 25)
    else:
        phi_s_list = np.logspace(1, 2, 25)

    phi_s_list = np.append(phi_s_list, np.inf)
    phi_s_list[np.where(phi_s_list>=phi)[0][0]] = phi
    n_test = int(n/np.log(n))
    phi_s_list = phi_s_list[np.where(p/phi_s_list >= (n-n_test))[0][-1]:]
    
    # Compute the empirical ecv estimates and risks
    res = []
    res_key = []
    for phi_s in phi_s_list:
        if phi_s<phi or (lam==0. and phi_s==1.):
            continue

        oobcv_emp, risk_emp = comp_empirical_oobcv(
            X, Y, X_test, Y_test, phi_s, 
            predictor, lam, M0=M0, M=M_list, M_test=M_max, bootstrap=bootstrap)

        _df = pd.DataFrame(
            np.stack([
                oobcv_emp, 
                np.append(risk_emp, np.full(len(M_list)-M_max, np.nan)),
            ]), index = ['oobcv_emp', 'risk_emp'], columns=M_str
        )

        res.append(_df)
        res_key.append((phi_s,seed))

    res = pd.concat(res, axis=0, keys=res_key, names=['phi_s','seed'])
    res.to_csv(path_result+'res_phi_{:.01f}_rho_{:.02f}_seed_{}.csv'.format(
            phi, rho_ar1, seed))
    

n_simu = 50
sigma = 0.5
rho_ar1 = 0.5
M0 = 10 # number of bags for computing the OOB estimates
M_max = 50
M_list = np.append(np.arange(1,M_max+1), np.inf)
M_str = ['{:d}'.format(int(M)) for M in M_list[:-1]] + ['$\infty$']
for phi in [0.1, 10.]:
    for seed in tqdm(range(n_simu)):
        print(phi, seed)
        run(phi, predictor, rho_ar1, seed)





# The results can be summarized by the following code for plotting:
# predictor_list = ['ridgeless', 'lassoless', 'ridge', 'lasso', 'logistic', 'kNN']   
# func_list = ['quad', 'tanh', 'linear']
# for func in func_list:
#     for bagging in ['bagging', 'subagging']:
#         for predictor in predictor_list:
#             path_result = 'result/ex1/{}/{}/{}/'.format(func, bagging,predictor)
#             df = pd.DataFrame()
#             for phi in [0.1, 10]:               
#                 for seed in range(50):
#                     _df = pd.read_csv(path_result+'res_phi_{:.01f}_rho_{:.02f}_seed_{}.csv'.format(
#                         phi, ar1rho, seed))
#                     _df['ar1rho'] = ar1rho
#                     _df['phi'] = phi
#                     df = pd.concat([df,_df], axis=0)
#             df.to_pickle('result/ex1/res_{}_{}_{}.pkl'.format(
#                 func, bagging, predictor), compression='gzip')
