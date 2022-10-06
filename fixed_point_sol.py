import warnings
import numpy as np

# isotopic features

def v_phi_lam(phi, lam, a=1):
    '''
    The unique solution v for fixed-point equation
        1 / v(-lam;phi) = lam + phi * int r / 1 + r * v(-lam;phi) dH(r)
    where H is the distribution of eigenvalues of Sigma.
    For isotopic features Sigma = a*I, the solution has a closed form, which reads that
        lam>0:
            v(-lam;phi) = (-(phi+lam/a-1)+np.sqrt((phi+lam/a-1)**2+4*lam/a))/(2*lam)
        lam=0, phi>1
            v(-lam;phi) = 1/(a*(phi-1))
    and undefined otherwise.
    '''
    assert a>0
    
    min_lam = -(1 - np.sqrt(phi))**2 * a
    if phi<=0. or lam<min_lam:
        raise ValueError("The input parameters should satisfy phi>0 and lam>=min_lam.")
        
    if lam!=0:
        return (-(phi+lam/a-1)+np.sqrt((phi+lam/a-1)**2+4*lam/a))/(2*lam)
    elif phi<=1.:
        raise ValueError("v is undefined for 0<phi<=1 and lam=0.")
    else:
        return 1/(a*(phi-1))
    

def vb_phi_lam(phi, lam, a=1, v=None):    
    if lam==0:
        if phi>1:
            return 1/(phi-1)
        else:
            return phi/(phi-1)
    else:
        if v is None:
            v = v_phi_lam(phi,lam,a)
        return phi/(1/a+v)**2/(
            1/v**2 - phi/(1/a+v)**2)
    
    
def vv_phi_lam(phi, lam, a=1, v=None):
    if lam==0:
        if phi>1:
            return phi/(a**2*(phi-1)**3)
        else:
            return np.inf
    else:
        if v is None:
            v = v_phi_lam(phi,lam,a)
        return 1./(
            1/v**2 - phi/(1/a+v)**2)
    
    
def tv_phi_lam(phi, phi_s, lam, v=None):
    if lam==0:
        if phi>1:
            return phi/(phi_s^2 - phi)
        else:
            return phi_s/(1 - phi_s)
    else:
        if v is None:
            v = v_phi_lam(phi_s,lam)
        tmp = phi/(1+v)**2
        return tmp/(1/v**2 - tmp)
    

def tc_phi_lam(phi_s, lam, v=None):
    if lam==0:
        if phi_s>1:
            return (phi_s - 1)**2/phi_s**2
        else:
            return 0
    else:
        if v is None:
            v = v_phi_lam(phi_s,lam)
        return 1/(1+v)**2
    
    

def vb_lam_phis_phi(lam, phi_s, phi, v=None):    
    if lam==0 and phi_s<=1:
        return 1+phi_s/(phi_s-1)
    else:
        if v is None:
            v = v_phi_lam(phi_s,lam)
        vsq_inv = 1/v**2
        return vsq_inv/(vsq_inv - phi/(1+v)**2)
    
    
def vv_lam_phis_phi(lam, phi_s, phi, v=None):
    if lam==0 and phi_s<=1:
        return np.inf
    else:
        if v is None:
            v = v_phi_lam(phi_s,lam)
        return phi/(
            1/v**2 - phi/(1+v)**2)


    
def v_general(phi, lam, Sigma=None, v0=None):
    if Sigma is None:
        return v_phi_lam(phi, lam)
    else:
        p = Sigma.shape[0]
        
        if phi==np.inf:
            return 0
        elif lam==0 and phi<=1:
            return np.inf        
        
        if v0 is None:
            v0 = v_phi_lam(phi, lam)
            
        v = v0
        eps = 1.
        n_iter = 0
        while eps>1e-3:
            if n_iter>1e4:
                if eps>1e-2: 
                    warnings.warn("Not converge within 1e4 steps.")
                break
            v = 1/(lam + phi * np.trace(np.linalg.solve(np.identity(p) + v * Sigma, Sigma)) / p)
            eps = np.abs(v-v0)/(np.abs(v0)+1e-3)
            v0 = v
            n_iter += 1
        return v


def tv_general(phi, phi_s, lam, Sigma=None, v=None):
    if lam==0 and phi_s<1:
        return phi/(1 - phi)
    if Sigma is None:
        return tv_phi_lam(phi, phi_s, lam, v)
    else:
        if v is None:
            v = v_general(phi_s, lam, Sigma)
        if v==np.inf:
            return phi/(1-phi)
        
        p = Sigma.shape[0]
        tmp = phi * np.trace(
                np.linalg.matrix_power(
                np.linalg.solve(np.identity(p) + v * Sigma, Sigma), 2)
        ) / p
        tv = tmp/(1/v**2 - tmp)
        return tv


def tc_general(phi_s, lam, Sigma=None, beta=None, v=None):
    if lam==0 and phi_s<1:
        return 0
    if Sigma is None:
        return tc_phi_lam(phi_s, lam, v)
    else:
        if v is None:
            v = v_general(phi_s, lam, Sigma)
        if v==np.inf:
            return 0.
        p = Sigma.shape[0]
        tmp = np.linalg.solve(np.identity(p) + v * Sigma, beta[:,None])
        tc = np.trace(tmp.T @ Sigma @ tmp)
        return tc
