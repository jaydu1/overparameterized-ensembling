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
from sklearn.linear_model import Ridge



n_simu = 20
sigma = 0.5
p = 500
phi = 0.1
n = int(p/phi)
M = 100
rho_ar1 = 0.25

path_result = 'result/ex3/'
os.makedirs(path_result, exist_ok=True)
print(rho_ar1)

d = 2 * p
func = lambda x: x * (x > 0)
F = np.random.normal(0, 1/np.sqrt(d), size=(d, p))

np.random.seed(0)
Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data(
    d, phi, rho_ar1, sigma, func='quad', n_test=n)
X = func(X @ F)
X_test = func(X_test @ F)

nu = 1.
mod = Ridge(alpha = nu * np.sqrt(n)).fit(X, Y)
mse = np.mean((mod.predict(X_test) - Y_test)**2)
mse # 1.9370091515536012

# estimate equivalence pairs on the path
lam_list = [0., nu/100, nu/10]
k_list = [est_k_theory(X=X, nu=nu, lam=lam) for lam in lam_list]


for i in range(len(k_list)):
    lam, k = lam_list[i], k_list[i]

    # Hyperparameters for the base regressor
    grid_regr = {    
        'alpha':np.array([lam*k]), 
        'fit_intercept':False,
        }
    # Hyperparameters for the ensemble
    grid_ensemble = {
        'random_state':np.arange(20),
        'max_samples':np.array([k/n]),
        'bootstrap':False,
        'n_jobs':-1
    }

    # Build 50 trees and get estimates until 100 trees
    res_ecv, info_ecv = ECV(
        X, Y, Ridge, grid_regr, grid_ensemble, 
        M0=20, M=100, M_max=100, return_df=True,
        X_test=X_test, Y_test=Y_test,
    )
    res_ecv.to_pickle(path_result+'res_{}.pkl'.format(i+1))