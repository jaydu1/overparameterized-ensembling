# %%
import numpy as np
from scipy import linalg as la
from numpy.linalg import norm
from joblib import Parallel, delayed
import pandas as pd
from math import comb
import math
import pdb # pdb.set_trace()
from sklearn import linear_model

print('ElasticNet')

# fixed setting
n = 2000
p = 500
M_max = 10
sigma = 1
rep = 100    # repeat simulation 100 times

# %%
def find_min_k(n):
    # find the minimal subsample size such that Pr(intersection of two subsamples is not empty) > 0.99
    for k in range(1, n+1):
        p0 = comb(n-k,k)/comb(n,k) # prob of non intersection for each repeatition
        prob = 1 - (1- p0)**rep
        if prob < .01:
            break
    return k
k_min = find_min_k(n)
k_min = math.ceil(k_min/100)*100
k_list = np.unique(np.append([p-100, p, p+100], np.arange(start=k_min, stop=n-200, step=100))) # do not include n

# Covariance structure
eigenvalues = np.linspace(0.1, 10.0, num=p)
Sigma = np.diag(eigenvalues)
Sigma_sq_root = np.diag(eigenvalues**0.5)
Sigma_inverse = np.diag(eigenvalues**-1)

# def sqrt_mat_sym(M):
#     # square root for symmetric matrix
#     s, v = la.eigh(M)
#     result = v @ np.diag(s**0.5) @ v.T
#     # assert np.isclose(M, v @ np.diag(s) @ v.T).all()
#     # assert np.isclose(result @ result, M).all()
#     # assert result.shape == M.shape
#     return result

# Sigma = la.toeplitz(0.5 ** np.arange(p))
# Sigma_sq_root = sqrt_mat_sym(Sigma)

s = p-100
beta0 = np.zeros(p)
rng = np.random.default_rng(42)
beta0[:s] = rng.normal(size=(s,))

# generate coefficient that satisfies to signal-noise ratio
snr = 1
beta0 = beta0 * np.sqrt(snr * sigma**2 / (beta0 @ Sigma @ beta0) )
# print('snr:', (beta0 @ Sigma @ beta0) / sigma**2)

# compute the range of lambda for lasso
rng = np.random.default_rng(42)
G = rng.normal(size=(n, p)) # G = X Sigma^{-1/2} with iid N(0,1) entries
X = G @ Sigma_sq_root
y = X @ beta0 + sigma * rng.normal(size=(n, ))


lam_max = np.max(X.T @ y)/n/3
# lam_min = lam_max * 1e-2
lam_min = .002 # make sure that l1_ratio > .01 (otherwise, skelarn not stable)
lam_list = np.unique(np.append([.002, .004, .006, .008, .01], np.logspace(np.log10(lam_min), np.log10(lam_max), num=30)))
lam_list = np.unique(np.round(lam_list, 3))
# %%
def one_run(seed, k=n//5, M=5, lam1=.005, lam2=.1):
    """ 
    Compute errors
    
    Parameters
    ----------
    seed: random seed, for reproducible results
    k: size of subsample
    a: regularization parameter for ||beta||_1
    b: regularization parameter for ||beta||^2

    ElasticNet optimization: ||y-X beta||^2/(2n) + lam1 ||beta||_1  + lam2/2 ||beta||^2 
    Note that n is the full sample size (for subsample it should be k)

    # Equivalent to the sklearn optimzation below
    # ||y-Xbeta||^2/(2n) + alpha * l1_ratio ||beta||_1  + 0.5 * alpha * (1 - l1_ratio) ||beta||^2 
    # Relationship: alpha = lam1 + lam2 and l1_ratio = lam1 /(lam1 + lam2)
    
    Returns
    -------
    results: xxx
    
    """
    
    
    rng = np.random.default_rng(seed)
    subsamples = np.stack([rng.choice(n, size=k, replace=False)
                           for m in range(M)])
    I_cum = [np.unique(subsamples[:m]) for m in range(1, M+1)] # cumulative subsamples

    G = rng.normal(size=(n, p)) # G = X Sigma^{-1/2} with iid N(0,1) entries
    X = G @ Sigma_sq_root
    y = X @ beta0 + sigma * rng.normal(size=(n, ))
    # shape (M, M), compute cardinalities
    n_intersections = np.array([[np.intersect1d(a,b).shape[0] 
                                 for a in subsamples]
                                 for b in subsamples ])

    psi = np.zeros((n, M))       # full residual: m-th column is y - X \hat\beta_m
    psi_sub = np.zeros((n, M))   # sub residual: m-th column is y_{I_m} - X_{I_m} \hat\beta_m
    
    df = np.zeros(M)
    v = np.zeros(M)
    B = np.zeros((p, M))

    assert lam1/(lam1+lam2) > .01

    for m, sub in enumerate(subsamples):
        X_I = np.zeros_like(X)
        y_I = np.zeros_like(y)
        
        X_I[sub,] = X[sub, :]
        y_I[sub] = y[sub]

        clf = linear_model.ElasticNet(fit_intercept=False, alpha=lam1+lam2, l1_ratio=lam1/(lam1+lam2))
        clf.fit(X_I, y_I)
        B[:, m] = clf.coef_

        # calculate df for elastic net
        supp = np.nonzero(clf.coef_)[0] # support set
        X_I_S = X_I[:, supp]
        val = np.linalg.svd(X_I_S, compute_uv=False) # eigenvalues of X_S
        df[m]  = np.sum(val**2/(val**2 + k*lam2))

        # print(df[m], len(supp))
        # assert( np.abs(df[m]-len(supp))< .001 )

        v[m] = 1 - df[m]/k
        
        psi[:, m] = y - X @ B[:, m]         # full residual
        psi_sub[:, m] = y_I - X_I @ B[:, m] # sub residual

    # prediction error
    H = B - beta0[:, np.newaxis]
    R0 = sigma**2 + H.T @ Sigma @ H
    
    # estimated error 1: using subsamples (require large enough subsample size)
    R1 = (psi_sub.T @ psi_sub) / (np.outer(v, v) * n_intersections)
    # estimated error 2: using all the samples
    R2 = (psi.T @ psi) / (n - df[np.newaxis,:] - df[:,np.newaxis] + np.outer(df,df)*n_intersections/k**2)
    
    R3 = (psi.T @ psi)
    
    results = []
    c = k/n
    for m in range(1, M+1):
        dic = {'lam': lam1, 'k': k,'M': m}
        
        true = np.mean(R0[:m,:m])
        est1 = np.mean(R1[:m,:m])
        est2 = np.mean(R2[:m,:m])

        #----naive gcv---------------
        beta_m = np.mean(B[:,:m], axis=1)
        est2_5 = norm(y[I_cum[m-1]] - X[I_cum[m-1]] @ beta_m) ** 2 / (len(I_cum[m-1]) * (1 - df[:m].mean()/len(I_cum[m-1]))**2)

        # gcv error (full)
        temp = df[:m].mean()/n 
        est3 = np.mean(R3[:m,:m])/n/(1 - temp)**2
        

        # corrected gcv error_full
        correction = temp**2/(m**2 * (1-temp)**2) * (1/c-1) * np.trace(R2[:m,:m]) # correction using full est R2[m,m]
        est4 = est3 - correction

        # corrected gcv error_sub
        correction_sub = temp**2/(m**2 * (1-temp)**2) * (1/c-1) * np.trace(R1[:m,:m])
        est5 = est3 - correction_sub

                
        results.append({**dic, 'risk': true, 'Type': 'Risk'})
        results.append({**dic, 'risk': est1, 'Type': 'est_sub'})
        results.append({**dic, 'risk': est2, 'Type': 'est_full'})
        results.append({**dic, 'risk': est2_5, 'Type': 'GCV_naive'})
        results.append({**dic, 'risk': est3, 'Type': 'GCV'})
        results.append({**dic, 'risk': est4, 'Type': 'CGCV'})
        results.append({**dic, 'risk': np.abs(est5-true)/true, 'Type': 'CGCV_sub'})
        results.append({**dic, 'risk': np.abs(est4-true)/true, 'Type': 'CGCV_full'})
        results.append({**dic, 'risk': np.abs(est1-true)/true, 'Type': 'sub_error'})
        results.append({**dic, 'risk': np.abs(est2-true)/true, 'Type': 'full_error'})


        
        # # corrected gcv error 
        # correction = temp**2/(m**2 * (1-temp)**2) * (1/c-1) * np.trace(R2[:m,:m]) # correction using full est R2[m,m]
        # est4 = est3 - correction
        # # print(correction)
                
        # results.append({**dic, 'risk': true, 'Type': 'Risk'})
        # results.append({**dic, 'risk': est1, 'Type': 'est_sub'})
        # results.append({**dic, 'risk': est2, 'Type': 'est_full'})
        # results.append({**dic, 'risk': est3, 'Type': 'GCV'})
        # results.append({**dic, 'risk': est4, 'Type': 'CGCV'})
        # results.append({**dic, 'risk': np.abs(est1-true)/true, 'Type': 'sub_error'})
        # results.append({**dic, 'risk': np.abs(est2-true)/true, 'Type': 'full_error'})
        
    return results

# total number of experimentes: 
print(rep * len(k_list) * len(lam_list))

# k should not be too small, otherwise the intersection of subsamples would be NAN
data = Parallel(n_jobs=-1, verbose=2)(delayed(one_run)(seed, k, M, lam1, lam2)
                                      for seed in range(rep)
                                      for k in k_list
                                      for M in [M_max]
                                      for lam1 in lam_list
                                      for lam2 in [.01]
)
data_flat = [it for sublist in data for it in sublist]
df = pd.DataFrame(data_flat).round(4)
df.head()

# save results 
fname = f'df_ElasticNet_n{n}_p{p}.pkl'
df.to_pickle(fname, compression='gzip')           # save dataframe as pkl
# df = pd.read_pickle(fname, compression='gzip')  # load pkl back as dataframe
