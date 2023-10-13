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
celltype = cell_types_list[0]



id_train = (df_split['split'].values=='train')&(cell_types==celltype)
id_test = (df_split['split'].values=='test')&(cell_types==celltype)
X_train, Y_train = X[id_train], Y[id_train]
X_test, Y_test = X[id_test], Y[id_test]

scaler = StandardScaler()
scaler.fit(Y_train)
Y_train = scaler.transform(Y_train)
Y_test = scaler.transform(Y_test)

n_pcs = 500
pca = PCA(n_components=n_pcs)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

method = 'ridge'

path_result = 'result/ex4/{}/{}/'.format(celltype,method)
os.makedirs(path_result, exist_ok=True)

M = 500
# compute the grid of subsample sizes that are valid for all three cv methods.
n, p = X_train.shape
nu = 0.5
n_base = int(n**nu)
n_train = int(n)
k_list = np.logspace(np.log10(n_base), np.log10(n_train), 25).astype(int)
phi = p/n
for j in tqdm(np.arange(39,30,-1)):

    res = []
    for lam in tqdm(np.logspace(-2,2,100), desc='lam'):
        gcv_emp, risk_emp = comp_empirical_gcv(
                X_train, Y_train[:,j:j+1], X_test, Y_test[:,j:j+1], phi, 
                lam, M=1, return_allM=False)
        res.append([phi, phi, lam, 1, gcv_emp, risk_emp])


    for phi_s in tqdm(p/k_list, desc='phi_s'):
        gcv_emp, risk_emp = comp_empirical_gcv(
                    X_train, Y_train[:,j:j+1], X_test, Y_test[:,j:j+1], phi_s, 
                    1e-8, M=M, return_allM=False)    
        res.append([phi, phi_s, 0, M, gcv_emp, risk_emp])

    pd.DataFrame(res, columns=['phi', 'phi_s', 'lam', 'M', 'gcv_emp', 'risk_emp']).to_csv(path_result+'res_{}_{}.csv'.format(celltype, j))
