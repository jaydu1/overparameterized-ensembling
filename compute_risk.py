import numpy as np
from sklearn.linear_model import Ridge
from fixed_point_sol import *



def comp_theoretic_risk_M1(rho, sigma, lam, phi_s):
    if lam==0.:
        if phi_s<1:
            B0 = 0.
            V0 = sigma**2 * phi_s / (1-phi_s)
        elif phi_s==1:
            B0 = 0.
            V0 = 0. if sigma==0 else np.inf
        else:
            v = v_phi_lam(phi_s, lam)
            B0 = rho**2 * (1 + vb_phi_lam(phi_s, lam, v=v)) / (1 + v)**2
            V0 = sigma**2 * phi_s * vv_phi_lam(phi_s, lam, v=v) / (1 + v)**2
        
    else:
        v = v_phi_lam(phi_s, lam)
        B0 = rho**2 * (1 + vb_phi_lam(phi_s,lam,v=v)) / (1 + v)**2
        V0 = sigma**2 * phi_s * vv_phi_lam(phi_s,lam,v=v) / (1+v)**2
    return B0, V0, sigma**2+B0+V0


def comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M, replace=True):
    if M==1:
        return comp_theoretic_risk_M1(rho, sigma, lam, phi_s)
    elif replace and M==2:

        phi_0 = phi_s**2 /phi

        if lam==0 and phi_s<=1:
            if phi_s<1:
                B0 = 0.
                V0 = 0.5 * sigma**2 * (
                    phi_s / (1-phi_s) + 
                    phi / (1 - phi)
                )
            elif phi_s==1:
                B0 = 0.
                V0 = 0. if sigma==0 else np.inf

        else:
            v = v_phi_lam(phi_s,lam)
#             c = (phi_s/phi-1) * (1 + v)
#             tv_c = v_phi_lam_extend(phi_0, lam, c=c)
#             tv_v = vv_phi_lam_extend(phi_0, lam, c=c, v=tv_c)
#             tv_b = vb_phi_lam_extend(phi_0, lam, c=c, v=tv_c)
#             print(1+tv_b, vb_lam_phis_phi(phi_s,phi,lam, v=v))
            B0 = 0.5 * rho**2 * (
                (1 + vb_phi_lam(phi_s,lam)) / (1 + v)**2 +
                vb_lam_phis_phi(lam,phi_s,phi, v=v) / (1 + v)**2
            )

            V0 = 0.5* sigma**2 * (
                phi_s * vv_phi_lam(phi_s,lam, v=v) / (1 + v)**2 +
                vv_lam_phis_phi(lam,phi_s,phi, v=v)/(1 + v)**2
            )
        return B0, V0, sigma**2+B0+V0
    elif replace and M>2:
        r1 = comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M=1)[-1]
        r2 = comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M=2)[-1]
        b = 2 * (r1 - r2)

        a = r1 - b
        if M==np.inf:
            return None, None, a
        else:
            return None, None, a + b / M
    else:
        B1, V1, _ = comp_theoretic_risk(rho, sigma, lam, phi, phi_s, M=1)
        
        # M is finite
        if lam==0 and phi_s<=1:
            BM = 0
        else:
            C = rho**2 / (1 + v_phi_lam(phi_s,lam))**2
            BM = B1 / M + (1 - 1/M) * C
        VM = V1 / M
        return BM, VM, sigma**2 + BM + VM


def compute_cov(Xs, lam):
    n,p = Xs.shape
    Sigma = Xs.T @ Xs / n
    if lam==0.:
        M = np.linalg.pinv(Sigma, hermitian=True)
        T1 = np.identity(p) - M @ Sigma
        T2 = M
    else:
        M = np.linalg.inv(Sigma + lam * np.identity(p))
        T1 = lam * M
        T2 = M @ Sigma @ M

    return Sigma, M/n, T1, T2/n

def comp_true_empirical_risk(X, Y, phi_s, lam, rho, sigma, beta0, M, replace=True):
    n,p = X.shape
    phi = p/n  
    
    assert replace is False or M==2
    
    if replace:
        k = int(p/phi_s)
        
        # randomly sample
        ids_list = [
            np.sort(np.random.choice(n,k,replace=False)) for _ in range(M)]
    else:
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)

    Sigma_list, M_list, T1_list, T2_list = zip(
        *[compute_cov(X[ids,:], lam) for ids in ids_list])
    T1_list = np.stack(T1_list, axis=0)
    T1_list = np.mean(T1_list, 0)
        
#         ids1 = np.sort(np.random.choice(n,k,replace=False))
#     ids2 = np.sort(np.random.choice(n,k,replace=False))
    
        
    
#     Sigma1, M1, T1_1, T2_1 = compute_cov(X[ids1,:], lam)
#     Sigma2, M2, T1_2, T2_2 = compute_cov(X[ids2,:], lam)

    
    if rho==0:
        B0 = 0
    else:
        B0 = beta0[:,None].T @ T1_list @ T1_list @ beta0[:,None]
        B0 = B0[0,0]
        
    if sigma==0:
        V0 = 0.
    else:
#         eps = Y - X @ beta0[:, None]
#         L1 = np.zeros(n)
#         L1[ids1] = 1
#         L1 = np.diag(L1)
#         L2 = np.zeros(n)
#         L2[ids2] = 1
#         L2 = np.diag(L2)
#         V0 = eps.T @ (M1 @ X.T @ L1 + M2 @ X.T @ L2).T @ (M1 @ X.T @ L1 + M2 @ X.T @ L2) @ eps / k**2 / 4
#         V0 = V0[0,0]
  
#         V0 = sigma**2/k/4 * (np.trace(T2_1) + np.trace(T2_2))
        
        V0 = sigma**2/M**2 * np.sum([np.trace(T2) for T2 in T2_list])

        if replace:
            for i in range(M):
                for j in range(i+1,M):
                    ids0 = np.intersect1d(ids_list[i],ids_list[j])
                    i0 = len(ids0)
                    if i0>0:
                        Sigma0 = X[ids0,:].T @ X[ids0,:] / i0
                        V0 += 2*sigma**2*i0/M**2 * np.trace(M_list[i] @ Sigma0 @ M_list[j])

    return B0 + V0 + sigma**2


def fit_predict(X, Y, X_test, lam):
    if lam==0:        
        beta = np.linalg.pinv(X.T @ X, hermitian=True) @ (X.T @ Y)
        Y_hat = X_test @ beta
        assert len(Y_hat.shape)==2
    else:
        clf = Ridge(alpha=lam, fit_intercept=False, solver='svd')
        clf.fit(X, Y)
        Y_hat = clf.predict(X_test)
    return Y_hat


def comp_empirical_risk(X, Y, X_test, Y_test, 
                        phi_s, lam, M=2, data_val=None, replace=True):
    n,p = X.shape
    
    if data_val is not None:
        X_val, Y_val = data_val
        Y_val_hat = np.r_[
            np.zeros_like(Y_val), np.zeros_like(Y_test)]
        X_eval = np.r_[X_val, X_test]
    else:
        Y_hat = np.zeros_like(Y_test)
        X_eval = X_test
        
    # rescale training data
#     X = X/np.sqrt(k)
#     Y = Y/np.sqrt(k)
#     i0 = int(k**2/n)
#     i0 = np.minimum(int(p/phi_0), k)
#     ids_com = np.sort(np.random.choice(n,i0,replace=False))
#     if i0<n:
#         ids_ind = np.random.choice(np.setdiff1d(np.arange(n), ids_com),2*(k-i0),replace=False)
#     else:
#         ids_ind = []
    
    if replace:
        k = int(p/phi_s)
        ids_list = [np.sort(np.random.choice(n,k,replace=False)) for j in range(M)]
    else:
        k = np.floor(n/M)
        assert 1 <= k <= n
        ids_list = np.array_split(np.random.permutation(np.arange(n)), M)
    for j in range(M):
        ids = ids_list[j]
#         ids = np.r_[ids_com, ids_ind[j*(k-i0):(j+1)*(k-i0)]].astype(int)
        Y_hat += fit_predict(X[ids,:]/np.sqrt(len(ids)), Y[ids,:]/np.sqrt(len(ids)), X_eval, lam)
    Y_hat /= M
    risk_test = np.mean((Y_hat[-Y_test.shape[0]:]-Y_test)**2)
    if data_val is not None:
        risk_val = np.mean((Y_hat[:-Y_test.shape[0]]-Y_val)**2)
        return risk_val, risk_test
    else:
        return risk_test
    