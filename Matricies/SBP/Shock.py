
import numpy as np 
#from SBP.mesh_1d import Element1D



def nodal_to_modal(u:np.ndarray, w:np.ndarray, V:np.ndarray):
    """
    Convert nodal values u at points x with weights w
    into modal Legendre coefficients a.
    
    u : array_like, shape (n+1,)   — nodal values
    w : array_like, shape (n+1,)   — corresponding LGL weights
    V : array, shape (n+1,n+1)     — Vandermonde Matrix evaluated at the LGL collocation points
    
    returns
    a : ndarray, shape (n+1,)      — modal Legendre coefficients
    """
    n = len(u) - 1

    
    # 2) Compute weighted inner‐products β_i = Σ_j w[j]*P_i(x[j])*u[j]
    beta = V.T.dot(w * u)         # shape = (n+1,)
    
    # 3) Scale by (2i+1)/2 to get a_i
    i = np.arange(n+1)
    a = beta * (2*i + 1) / 2      # shape = (n+1,)
    return a

def perrson_sensor(a:np.ndarray,kill_mode:int = -1,  eps:float = 1e-30) -> float: 
    """
    Calculates the Persson and Peraire from a vector of modal coefficients. 
    a : array, shape (n+1,)  — modal values
    kill_mode : float, scalar — determines the mode to be killed. 
    
    Note: The sensor essentially wants to compare the energy in the highest mode to the total energy. 
    """
    a2 = np.dot(a,a) + eps # Denomenator
    S = a[kill_mode]**2/ a2
    S = max(S, eps) # To avoid inf
    return np.log10(S)

def av(s:float, s0:float = np.log(1/3**4), kappa:float = 1, e0: float = 1e-1): 
    """
    u       : array_like, shape (n+1,)    — solution vector of nodal values
    s       : float,                      — shock indicator 
    s0      : float,                      — AV shock threshold
    kappa   : float,                      — AV smoothing range 
    """
    #return np.where(s<= (s0 - kappa), 0.0, 
    #                np.where((s >= s0 - kappa) & (s < s0 + kappa), 
    #                         e0*0.5*(1 + np.sin(np.pi*(s - s0)/(2 * kappa))), e0))
    return np.float64(np.where(
        s <= (s0 - kappa),          0.0,
        np.where(
            s >= (s0 + kappa),      e0,
            e0 * 0.5 * (1.0 + np.sin(np.pi * (s - s0) / (2.0 * kappa)))
        )
    ))

