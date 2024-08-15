import os
import sys


import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from generate_data import *
from compute_risk import *
from tqdm import tqdm

from sklearn_ensemble_cv import GCV
from sklearn_ensemble_cv.cross_validation import process_grid, comp_empirical_ecv, fit_ensemble
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler




# data = "FashionMNIST"; model = 'resnet-18_init'
# data = 'CIFAR-10'; model = 'resnet-18_pretrained'
data = 'Flowers-102'; model = 'resnet-50_pretrained'
# data = 'Food-101'; model = 'resnet-101_pretrained'

path_result = 'result/ex6/{}_{}/'.format(data, model)
os.makedirs(path_result, exist_ok=True)

with open('data_{}_{}.npz'.format(data, model), 'rb') as f:
    dat = np.load(f)
    if data=='Flowers-102':
        X = np.r_[dat['X_train'], dat['X_val']]
        Ys = np.r_[dat['Y_train'], dat['Y_val']]
    else:
        X = dat['X_train']
        Ys = dat['Y_train']
    X_test = dat['X_test']    
    Ys_test = dat['Y_test']

n, p = X.shape

X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(p)
X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True) * np.sqrt(p)

scaler = StandardScaler(with_std=False).fit(Ys)
Ys = scaler.transform(Ys)
Ys_test = scaler.transform(Ys_test)

nu_list = np.append([0,], np.logspace(-4.,-2, 100))
nu_list = np.unique(nu_list)
lam_0 = 1e-4

# estimate equivalence pairs on the path
if data=='Flowers-102':
    # X, X_test = X_test, X
    # Ys, Ys_test = Ys_test, Ys
    # n, p = X.shape
    k_list = (np.logspace(-2.,0, 50) * n).astype(int)
    M0 = 25
else:
    k_list = (np.logspace(-3.,0, 50) * n).astype(int)
    M0 =25

i = int(sys.argv[1])

random_state = 0

for j in range(10):
    Y, Y_test = Ys[:,j], Ys_test[:,j]

    # fit subsample ridge ensemble
    lam, k = lam_0, k_list[i]
    print('fit subsample ridge ensemble', lam, k)

    # Hyperparameters for the base regressor
    grid_regr = {    
        'alpha':np.array([lam*k]), 
        'fit_intercept':False,
        }
    # Hyperparameters for the ensemble
    grid_ensemble = {
        'random_state':np.array([random_state]),
        'max_samples':np.array([k/n]),
        'bootstrap':False,
        'n_jobs':-1
    }

    # Build 50 trees and get estimates until 100 trees
    res_ecv, info_ecv = GCV(
        X, Y, Ridge, grid_regr, grid_ensemble, 
        M0=M0, M=100, M_max=np.inf, return_df=True,
        X_test=X_test, Y_test=Y_test
    )
    res_ecv.to_pickle(path_result+'res_sub_{}_{}_{}.pkl'.format(j, i+1, random_state))


# # merge results
# df = pd.DataFrame()
# for i in range(50):
#     for j in range(10):
#         try:
#             _df = pd.read_pickle(path_result+'res_sub_{}_{}_{}.pkl'.format(j, i+1, random_state))
#             _df['j'] = j
#             df = pd.concat([df, _df], axis=0)
#         except:
#             continue

# df.to_pickle('{}res_sub.pkl'.format(path_result))