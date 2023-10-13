import os
import sys
os.environ["OMP_NUM_THREADS"] = "11"
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "11" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "11" # export NUMEXPR_NUM_THREADS=6
os.environ["NUMBA_CACHE_DIR"]='/tmp/numba_cache'
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse
import h5py

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from tqdm import tqdm
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from compute_risk import *


with h5py.File('pbmc_count.h5', 'r') as f:
    print(f.keys())
    ADT_names = np.array(f['ADT_names'], dtype='S32').astype(str)
    gene_names = np.array(f['gene_names'], dtype='S32').astype(str)
    X = sp.sparse.csc_matrix(
        (np.array(f['RNA.data'], dtype=np.float32), 
         np.array(f['RNA.indices'], dtype=np.int32),
         np.array(f['RNA.indptr'], dtype=np.int32)
        ), 
        shape = np.array(f['RNA.shape'], dtype=np.int32)).tocsc().astype(np.float32).T.toarray()
    Y = np.array(f['ADT'], dtype=np.float32)
    cell_types = np.array(f['celltype'], dtype='S32').astype(str)
    cell_ids = np.array(f['cell_ids'], dtype='S32').astype(str)
    
X = np.log(X/np.sum(X, axis=1, keepdims=True)*1e4+1.)
Y = np.log(Y/np.sum(Y, axis=1, keepdims=True)*1e4+1.)



df_split = pd.read_csv('df_split.csv', index_col=[0])


cell_types_list = ['B', 'CD4 T', 'CD8 T', 'DC', 'Mono', 'NK']
celltype = cell_types_list[int(sys.argv[1])]



id_train = (df_split['split'].values=='train')&(cell_types==celltype)
id_test = (df_split['split'].values=='test')&(cell_types==celltype)
X_train, Y_train = X[id_train], Y[id_train]
X_test, Y_test = X[id_test], Y[id_test]

scaler = StandardScaler()
scaler.fit(Y_train)
Y_train = scaler.transform(Y_train)
Y_test = scaler.transform(Y_test)


bootstrap = int(sys.argv[3])==1
bagging = 'bagging' if bootstrap else 'subagging'
method = 'tree'
path_result = 'result/ex7/{}/{}/{}/'.format(celltype,bagging,method)
os.makedirs(path_result, exist_ok=True)

M = 50
M0 = 20
M_list = [50, 100, 250]
file_name_time = path_result+'res_time_{}.csv'.format(method)
res_time = pd.DataFrame(columns=['ADT_name','lam','splitcv','kfoldcv-3','kfoldcv-5']
        +['oobcv-{}'.format(_M) for _M in M_list])

lam = None
for j,ADT_name in tqdm(enumerate(ADT_names)):

    _res_time = [ADT_name, lam]

    # compute the grid of subsample sizes that are valid for all three cv methods.
    n, p = X_train.shape
    nu = 0.5
    n_base = int(n**nu)
    n_train = int(n*(1-1/np.log(n)))
    k_list = np.arange(n_base, n_train+1, n_base)
    if n_train!=k_list[-1]:
        k_list = np.append(k_list, n_train)
    _k_list = k_list[k_list <= (n//5 + 1)]


    # test for all M and k
    for i in range(5):
        print(i)
        file_name = path_result+'res_ADT_{}_test_{}.csv'.format(j,i)
        k_list, risk_val, risk_test = compute_prediction_risk(
            X_train, Y_train[:,j:j+1], X_test, Y_test[:,j:j+1], 
            method, lam, M, k_list=_k_list, bootstrap=bootstrap)
        res = np.concatenate([k_list[:,None], risk_val, risk_test], axis=-1)
        res = pd.DataFrame(res, columns=np.concatenate([
            ['k_list'],
            np.char.add('val-', np.arange(1,M+1).astype(str)),
            np.char.add('test-', np.arange(1,M+1).astype(str))
        ]))
        res['ADT_name'] = ADT_name
        res.to_csv(file_name)


    # split validation
    file_name = path_result+'res_ADT_{}_splitcv.csv'.format(j)
    t0 = time.time()
    k_list, risk_val, risk_test = cross_validation(
        X_train, Y_train[:,j:j+1], X_test, Y_test[:,j:j+1], 
        method, lam, M, val_size=1/6., Kfold=False, k_list=_k_list, 
        return_full=True, bootstrap=bootstrap)        
    t1 = time.time() - t0
    res = np.concatenate([k_list[:,None], risk_val, risk_test], axis=-1)
    res = pd.DataFrame(res, columns=np.concatenate([
        ['k_list'],
        np.char.add('val-', np.arange(1,M+1).astype(str)),
        np.char.add('test-', np.arange(1,M+1).astype(str))
    ]))
    res['ADT_name'] = ADT_name
    print(t1) 
    _res_time.append(t1)

    
    # K-fold CV
    for K in [3,5]:
        file_name = path_result+'res_ADT_{}_kfoldcv_{}.csv'.format(j,K)
        t0 = time.time()
        k_list, risk_val, risk_test = cross_validation(
            X_train, Y_train[:,j:j+1], X_test, Y_test[:,j:j+1], 
            method, lam, M, Kfold=K, k_list=_k_list, 
            return_full=True, bootstrap=bootstrap)        
        t1 = time.time() - t0
        res = np.concatenate([k_list[:,None], risk_val, risk_test], axis=-1)
        res = pd.DataFrame(res, columns=np.concatenate([
            ['k_list'],
            np.char.add('val-', np.arange(1,M+1).astype(str)),
            np.char.add('test-', np.arange(1,M+1).astype(str))
        ]))
        res['ADT_name'] = ADT_name
        print(t1)
        _res_time.append(t1)

        
    # OOBCV M bags
    for _M in M_list:    
        t0 = time.time()
        k_list, risk_val, risk_test = cross_validation_oob(
            X_train, Y_train[:,j:j+1], X_test[:1,:], Y_test[:1,j:j+1], 
            method, lam, _M, M0=M0, M_test=_M, k_list=_k_list, 
            return_full=True, bootstrap=bootstrap)
        t1 = time.time() - t0
        print(t1)
        _res_time.append(t1)


        file_name = path_result+'res_ADT_{}_oobcv_{}.csv'.format(j, _M)
        t0 = time.time()
        k_list, risk_val, risk_test = cross_validation_oob(
            X_train, Y_train[:,j:j+1], X_test, Y_test[:,j:j+1], 
            method, lam, _M, M0=M0, M_test=_M, 
            return_full=True, bootstrap=bootstrap)
        t1 = time.time() - t0
        res = np.concatenate([k_list[:,None], risk_val, risk_test], axis=-1)
        res = pd.DataFrame(res, columns=np.concatenate(
            [['k_list'],
            np.char.add('val-', np.arange(1,_M+1).astype(str)),
            np.char.add('test-', np.arange(1,_M+1).astype(str))]
        ))
        res['ADT_name'] = ADT_name
        res.to_csv(file_name)
        print('oobcv-{}:'.format(_M),t1)
    

    res_time = pd.concat([
        res_time,
        pd.DataFrame([_res_time], columns=
        ['ADT_name','lam','splitcv','kfoldcv-3','kfoldcv-5'] + ['oobcv-{}'.format(_M) for _M in M_list])
    ])
    res_time.to_csv(file_name_time)
    
