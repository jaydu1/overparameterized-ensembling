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


sigma = 1.

rho_ar1 = 0.25
p = 500
M_list = [1, 2, 5, 10, 50, 100]

lam_list = np.linspace(0.,1.,11)

# df_res_the = pd.DataFrame()
# for lam in lam_list:
#     for phi in [0.1, 1.1]:
#         Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma)

#         phi_s_list = np.logspace(-1,1, 2000)
#         phi_s_list[np.where(phi_s_list>=phi)[0][0]] = phi

#         tmp = []
#         for phi_s in tqdm(phi_s_list):
#             if phi_s<phi or phi_s==1.:
#                 continue
#             for M in np.append(M_list, [np.inf]):
#                 B, V, risk_the = comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, phi_s, M, replace=True)
#                 tmp.append([phi_s,
#                             '{:d}'.format(int(M)) if M<np.inf else '$\infty$',
#                             B,V,risk_the])        
#         res_the = pd.DataFrame(tmp, columns=['phi_s', 'M', 'B_M', 'V_M', 'risk_the'])

#         res_the['phi'] = phi
#         res_the['lam'] = lam
#         df_res_the = pd.concat([df_res_the, res_the])

#     df_res_the.to_csv('result/ex5/res_the_rho_{:.02f}_WR.csv'.format(rho_ar1), index=False)
    
    
M_list = [1, 2, 3, 4, 5]
df_res_the = pd.DataFrame()
for lam in lam_list:
    for phi in [0.1, 1.1]:
        Sigma, beta0, X, Y, X_test, Y_test, rho2, sigma2 = generate_data_ar1(p, phi, rho_ar1, sigma)

        phi_s_list = np.logspace(-1,1, 5000)
        phi_s_list[np.where(phi_s_list>=phi)[0][0]] = phi

        tmp = []
        for phi_s in tqdm(phi_s_list):
            if phi_s<phi or phi_s==1.:
                continue
            for M in np.append(M_list, [phi_s/phi]):
                if M > phi_s/phi:
                    continue
                B, V, risk_the = comp_theoretic_risk_general(Sigma, beta0, sigma2, lam, phi, phi_s, M, replace=False)
                tmp.append([phi_s,
                            '{:d}'.format(int(M)) if M<phi_s/phi else '$\phi_s/\phi$',
                            B,V,risk_the])        
        res_the = pd.DataFrame(tmp, columns=['phi_s', 'M', 'B_M', 'V_M', 'risk_the'])

        res_the['phi'] = phi
        res_the['lam'] = lam
        df_res_the = pd.concat([df_res_the, res_the])

        df_res_the.to_csv('result/ex5/res_the_rho_{:.02f}_WOR.csv'.format(rho_ar1), index=False)    
