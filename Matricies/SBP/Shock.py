
import numpy as np 
from numpy.polynomial.legendre import legvander
from SBP.mesh_1d import Element1D


def nodal_to_modal(u:np.ndarray, x:np.ndarray, w:np.ndarray, V:np.ndarray):
    """
    Convert nodal values u at points x with weights w
    into modal Legendre coefficients a.
    
    u : array_like, shape (n+1,)   — nodal values
    x : array_like, shape (n+1,)   — LGL nodes
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

#def lower_order_projection(u:np.ndarray): 

