'''
This script generates data and applies ECV to tune both observation and feature subsampling ratios for random forests.
The results are saved to a CSV file and can be summarized by running the code block at the end of the script.
'''
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from compute_risk import *
from generate_data import *
import time
from joblib import Parallel, delayed


func = 'quad'
bootstrap = True
bagging = 'bagging'
predictor = 'tree'
path_result = 'result/ex4/{}/{}/'.format(bagging,predictor)
os.makedirs(path_result, exist_ok=True)



def tune_max_features(X, Y, X_test, Y_test, predictor, M_max, M0, **kwargs):
    n, p = X.shape

    np.random.seed(0)
    t0 = time.time()    
    res_oob_1, _ = comp_empirical_oobcv(X, Y, X_test[:1,:], Y_test[:1],
        p/n, predictor, None, M=M_list, M0=M0, M_test=M0, k_max=k_max, **kwargs)
    t1 = time.time() - t0

    np.random.seed(0)
    t0 = time.time()
    res_oob_2, _ = comp_empirical_oobcv(X, Y, X_test[:1,:], Y_test[:1],
        p/n, predictor, None, M=M_list, M0=M_max, M_test=M_max, k_max=k_max, **kwargs)
    t2 = time.time() - t0
    
    np.random.seed(0)
    _, res_test = comp_empirical_oobcv(X, Y, X_test, Y_test,
        p/n, predictor, None, M=M_list, M0=M0, M_test=M_max, k_max=k_max, **kwargs)

    res = np.stack([
        np.r_[np.array([kwargs['max_features'],'oobcv_emp', t1],dtype=object), res_oob_1],
        np.r_[np.array([kwargs['max_features'],'oobcv_full_emp', t2],dtype=object), res_oob_2],
        np.r_[np.array([kwargs['max_features'],'risk_emp', np.nan],dtype=object), res_test, np.full(len(M_list)-M_max, np.nan)]
    ])

    return res

                              
def run_one_simulation(n, phi, rho_ar1, sigma, M_max, M0, i, **kwargs):    
    p = int(n*phi)
    np.random.seed(i)

    # Generate data
    Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data(
        n, p, coef='eig-5', func=func,
        rho_ar1=rho_ar1, sigma=sigma, df=np.inf, n_test=2000, sigma_quad=1.)


    # Tune mtry/max_features for fixed subsample size
    with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
        res = parallel(
            delayed(tune_max_features)(
                X, Y, X_test, Y_test, predictor, M_max, M0, 
                max_features=max_features, **kwargs) 
            for max_features in tqdm(max_features_list)
        )
        res = np.concatenate(res, axis=0)
        
    res = np.c_[np.full((res.shape[0],1), phi), np.full((res.shape[0],1), i), res]
    df = pd.DataFrame(res, columns=['phi','seed','max_features','type','time']+M_str)
    file_saved = path_result+'res_max_features_rho_{:.02f}_sigma_{:.01f}_seed_{}.csv'.format(rho_ar1,sigma,i)
    df.to_csv(file_saved)
    
    # Tune the subsample size for fixed tuned mtry
    max_features = max_features_list[df[df['type']=='oobcv_emp'][M_str[-2]].values.argmin()]
    k_list, risk_val, risk_test = cross_validation_oob(
        X, Y, X_test, Y_test, predictor, None, M_max, M0=M0, M_test=M_max, 
        return_full=True, max_features=max_features, k_max=k_max, **kwargs)
    res = np.concatenate([
        np.full((len(k_list),1), phi), np.full((len(k_list),1), i), 
        k_list[:,None], risk_val, risk_test], axis=-1)
    df = pd.DataFrame(res, columns=np.concatenate(
            [['phi', 'seed', 'k_list'],
            np.char.add('val-', np.arange(1,M_max+1).astype(str)),
            np.char.add('test-', np.arange(1,M_max+1).astype(str))]
        ))
    file_saved = path_result+'res_mtry_rho_{:.02f}_sigma_{:.01f}_seed_{}.csv'.format(rho_ar1,sigma,i)
    df.to_csv(file_saved, index=False)



n = 100
k_max = 0.9*n
n_simu = 50
sigma = 5.
rho_ar1 = 0.5
M0 = 25 # number of bags for computing the OOB estimates
M_max = 100
M_list = np.append(np.arange(1,M_max+1), np.inf)
M_str = ['{:d}'.format(int(M)) for M in M_list[:-1]] + ['$\infty$']
phi = 10
p = int(n*phi)
max_features_list = (p * np.linspace(0, 1, 11)[1:]).astype(int)

for i in tqdm(range(n_simu)):
    run_one_simulation(n, phi, rho_ar1, sigma, M_max, M0, i, bootstrap=bootstrap)
    


# The results can be summarized by the following code for plotting:
# import pandas as pd
# func = 'quad'
# bootstrap = True
# bagging = 'bagging'
# predictor = 'tree'
# rho_ar1_list = [-0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75]
# path_result = 'result/ex4/{}/{}/'.format(bagging,predictor)
# df = pd.DataFrame()
# for rho_ar1 in rho_ar1_list:
#     file_saved = path_result+'res_rho_{:.02f}.csv'.format(rho_ar1)
#     _df = pd.read_csv(file_saved)
#     _df['rho_ar1'] = rho_ar1
#     df = pd.concat([df,_df], axis=0)
# df.to_pickle('result/ex4/res_rho.pkl', compression='gzip')