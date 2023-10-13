import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
sns.set_theme()
from compute_risk import *
from generate_data import *


bootstrap = True
bagging = 'bagging' if bootstrap else 'subagging'
method_list = ['ridge', 'lasso', 'logistic', 'kNN']
method = method_list[int(sys.argv[1])]
path_result = 'result/ex2/{}/{}/'.format(bagging, method)
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(n, phi, rho, sigma, method, lam, M, i, bootstrap=bootstrap):
    np.random.seed(i)
    beta0, X, Y, X_test, Y_test = generate_data(n, phi, rho, sigma)
    if method=='logistic':
        Y_test = np.where(Y_test>=np.median(Y), 1, 0)
        Y = np.where(Y>=np.median(Y), 1, 0)
        
    k_list, risk_val, risk_test = cross_validation_oob(
        X, Y, X_test, Y_test, method, lam, M, M0=M0, M_test=M, 
        bootstrap=bootstrap, return_full=True)
    res = np.concatenate([
        np.full((len(k_list),1), phi), np.full((len(k_list),1), i), 
        k_list[:,None], risk_val, risk_test], axis=-1) 
    return res


lam_list = np.linspace(0.,1.,11)

M = 50
M0 = 10
n = 1000
n_simu = 100
sigma = 1.
from joblib import Parallel, delayed

if method in ['tree', 'kNN']:
    lam = None
else:
    lam_list = np.linspace(0.,1.,11)

    lam = lam_list[int(sys.argv[2])]
    
with Parallel(n_jobs=8, verbose=0, timeout=99999) as parallel:
    for SNR in [1.]:
        if lam is None:
            file_saved = path_result+'res_iso_cv_SNR_{:.02f}.csv'.format(SNR)
        else:
            file_saved = path_result+'res_iso_cv_lam_{:.01f}_SNR_{:.02f}.csv'.format(lam, SNR)
        if os.path.exists(file_saved):        
            df_res = pd.read_csv(file_saved, index_col=0)
        else:
            df_res = pd.DataFrame()
        print(method,SNR,os.path.exists(file_saved))
        for phi in tqdm(np.logspace(-1, 1, 25), desc = 'phi'):
            if df_res.shape[0]>0 and phi<=df_res['phi'].max():
                continue

            rho = sigma * np.sqrt(SNR)
            res = parallel(
                delayed(run_one_simulation)(n, phi, rho, sigma, method, lam, M, i) for i in tqdm(range(n_simu))
            )

            res = pd.DataFrame(np.concatenate(res,axis=0), columns=np.concatenate(
                [['phi', 'seed', 'k_list'],
                np.char.add('val-', np.arange(1,M+1).astype(str)),
                np.char.add('test-', np.arange(1,M+1).astype(str))]
            ))
            df_res = pd.concat([df_res, res],axis=0)
            df_res.to_csv(file_saved)

