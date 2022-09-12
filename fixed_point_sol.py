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
    
    if phi<=0. or lam<0.:
        raise ValueError("The input parameters should satisfy phi>0 and lam>=0.")

    if lam>0.:
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
            return np.inf
    else:
        if v is None:
            v = v_phi_lam(phi_s,lam)
        tmp = phi/(1+v)**2
        return tmp/(1/v**2 - tmp)
    
    

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

    
def v_phi_lam_extend(phi, lam, a=1, 
                     c=None, C=None):
    if c is None and C is None:
        warnings.warn('Setting C be to zero matrix.')
        return v_phi_lam(phi, lam, a)
    elif c is not None and c>=0.:
        return v_phi_lam(phi, lam, a/(1+c))
    elif C is not None:
        p = C.shape[0]    
        v0 = v_phi_lam(phi, lam, a)
        v = v0
        eps = 1.
        while eps>1e-3:
            v = 1/(lam + phi * a * np.trace(np.linalg.inv((a*v +1)*np.identity(p)+ C)) / p)
            eps = np.abs(v-v0)
            v0 = v
        return v
    else:
        raise ValueError("Invalid input.")
            
def vb_phi_lam_extend(phi, lam, a=1, c=None, C=None, v=None):
    if c is None and C is None:
        warnings.warn('Setting C be to zero matrix.')
        return vb_phi_lam(phi, lam, a, v=v)
    elif c is not None and c>=0.:
        return vb_phi_lam(phi, lam, a/(1+c), v=v)
    elif C is not None:
        v = v_phi_lam_extend(phi, lam, a, C=C)
        p = C.shape[0]
        tmp = phi * a**2 * np.trace(
                np.linalg.matrix_power(
                np.linalg.inv((a*v +1)*np.identity(p)+ C)) / p, 2
        )
        tvb = tmp/(1/v**2 - tmp)
        return v
    else:
        raise ValueError("Invalid input.")

def vv_phi_lam_extend(phi, lam, a=1, c=None, C=None, v=None):
    if c is None and C is None:
        warnings.warn('Setting C be to zero matrix.')
        return vv_phi_lam(phi, lam, a, v=v)
    elif c is not None and c>=0.:
        return vv_phi_lam(phi, lam, a/(1+c), v=v)
    elif C is not None:
        if v is None:
            v = v_phi_lam_extend(phi, lam, a, C=C)
        p = C.shape[0]
        tvv = 1/(1/v**2 - phi * a**2 * np.trace(
                np.linalg.matrix_power(
                np.linalg.inv((a*v +1)*np.identity(p)+ C), 2)) / p)
        return tvv
    else:
        raise ValueError("Invalid input.")