import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from fixed_point_sol import *
from generate_data import *
from compute_risk import *
from tqdm import tqdm
import torchvision
from sklearn import datasets


data_set_list = ['cifar10', 'mnist', 'usps', 'svhn', 'lfw']
dataset = data_set_list[int(sys.argv[1])]

if dataset=='cifar10':
    data_train = torchvision.datasets.CIFAR10('data/cifar10/train', train=True)
    data_test = torchvision.datasets.CIFAR10('data/cifar10/test', train=False)
    def preprocess(data):
        y = np.array(data.targets)
        X = data.data[(y==3) | (y==1)]
        y = y[(y==3) | (y==1)]
        X = X.reshape(len(y), -1)
        X = X/255.        
        y = np.where(y==3, 0, 1)
        return X, y[:,None]
elif dataset=='lfw':
    
    data_train = datasets.fetch_lfw_pairs(subset='train', data_home='data/fetch_lfw_pairs/')
    data_test = datasets.fetch_lfw_pairs(subset='test', data_home='data/fetch_lfw_pairs/')
    
    def preprocess(data): 
        X = data['data']
        Y = data['target']
        return X, Y[:,None]
else:    
    if dataset=='mnist':
        data_train = torchvision.datasets.MNIST('data/mnist/train', train=True, download=True)
        data_test = torchvision.datasets.MNIST('data/mnist/test', train=False, download=True)
    elif dataset=='usps':
        data_train = torchvision.datasets.USPS('data/usps/train', train=True, download=True)
        data_test = torchvision.datasets.USPS('data/usps/test', train=False, download=True)
    elif dataset=='svhn':
        data_train = torchvision.datasets.SVHN('data/svhn/train', split='train', download=True)
        data_test = torchvision.datasets.SVHN('data/svhn/test', split='test', download=True)
    def preprocess(data):
        if dataset=='svhn':
            y = np.array(data.labels)
        else:
            y = np.array(data.targets)        
        X = np.array(data.data)[(y==3) | (y==1)]
        y = y[(y==3) | (y==1)]
        X = X.reshape(len(y), -1)
        X = X/255.
        y = np.where(y==3, 0, 1)
        return X, y[:,None]

    
X, Y = preprocess(data_train)
X_test, Y_test = preprocess(data_test)

n, p = X.shape
phi = p/n
M = 500

print(dataset, n,p)

path_result = 'result/ex4/'.format()
os.makedirs(path_result, exist_ok=True)


def run_one_simulation(X, Y, X_test, Y_test, phi, phi_s, lam, a_gau, a_nongau, M,):
    np.random.seed(0)
    if phi==phi_s:
        M = 1
    stat = comp_empirical_beta_stat(X, Y, phi_s, 'ridge', lam, a_gau, a_nongau, M=M)
    stat_2 = comp_empirical_generalized_risk(X, Y, phi_s, 'ridge', lam, X_test=X_test, Y_test=Y_test, M=M)
    
    res = np.c_[np.full(M, phi), np.full(M, phi_s), 
              np.full(M, lam), stat, stat_2]
    return res[-1:,:]


np.random.seed(0)
a_gau = np.random.normal(0, 1/np.sqrt(p), size=p)
df = 5
a_nongau = np.random.standard_t(df=df, size=p) / np.sqrt(df / (df - 2)) /np.sqrt(p)


with Parallel(n_jobs=1, verbose=1, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
    

    
    for psi in [2,4,6,8,10]:
        lam = 1e-3
        df_res = pd.DataFrame()
        lam_bar = est_lam(X, psi, 'ridge', lam, M=100)
        print(lam_bar, phi, psi)

        lam_list = (lam - lam_bar) * np.linspace(0,1,2) + lam_bar
        phis_list = (psi - phi) * np.linspace(0,1,2) + phi

        res = parallel(
            delayed(run_one_simulation)(X, Y, X_test, Y_test, phi, phi_s, lam, a_gau, a_nongau, M) 
            for lam, phi_s in zip(lam_list, phis_list)
        )

        res = pd.DataFrame(np.concatenate(res,axis=0), columns=
            ['phi', 'phi_s', 'lam', 'min', 'max', 'mean', 
             'median', 'std', 'random_gau', 'random_t', 
             'train_est_err', 'train_pred_err', 'test_est_err', 'test_pred_err']
        )
        df_res = pd.concat([df_res, res],axis=0)

        df_res.to_csv('{}res_{}_{}.csv'.format(path_result,dataset,psi), index=False)
