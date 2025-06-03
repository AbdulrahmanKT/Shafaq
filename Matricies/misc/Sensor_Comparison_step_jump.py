###################################################################################################
###################################### Imports ####################################################

import SBP as sb
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
np.set_printoptions(linewidth=np.inf)
from numpy.polynomial.legendre import legvander

###################################### Imports ####################################################
###################################################################################################



###################################################################################################
###################################### Functions ##################################################

# This function is for converting from nodal to modal represntation of the solution vecotr
def nodal_to_modal(u, x, w):
    """
    Convert nodal values u at points x with weights w
    into modal Legendre coefficients a.
    
    u : array_like, shape (n+1,)   — nodal values
    x : array_like, shape (n+1,)   — LGL nodes
    w : array_like, shape (n+1,)   — corresponding LGL weights
    
    returns
    a : ndarray, shape (n+1,)      — modal Legendre coefficients
    """
    n = len(u) - 1
    
    # 1) Build Legendre‐Vandermonde: V[j,i] = P_i(x[j]) for i=0..n
    V = legvander(x, n)           # shape = (n+1, n+1)
    
    # 2) Compute weighted inner‐products β_i = Σ_j w[j]*P_i(x[j])*u[j]
    beta = V.T.dot(w * u)         # shape = (n+1,)
    
    # 3) Scale by (2i+1)/2 to get a_i
    i = np.arange(n+1)
    a = beta * (2*i + 1) / 2      # shape = (n+1,)
    return a

# This function takes in a lower order solution and kills the highest frequency 
def modal_frequency_reduction(u, x, w):
    """
    Convert nodal values u at points x with weights w
    into modal Legendre coefficients a.
    
    u : array_like, shape (n+1,)   — nodal values
    x : array_like, shape (n+1,)   — LGL nodes
    w : array_like, shape (n+1,)   — corresponding LGL weights
    
    returns
    a : ndarray, shape (n+1,)      — modal Legendre coefficients
    """
    n = len(u) - 1
    
    # 1) Build Legendre‐Vandermonde: V[j,i] = P_i(x[j]) for i=0..n
    V = legvander(x, n)           # shape = (n+1, n+1)
    
    # 2) Compute weighted inner‐products β_i = Σ_j w[j]*P_i(x[j])*u[j]
    beta = V.T.dot(w * u)         # shape = (n+1,)
    
    # 3) Scale by (2i+1)/2 to get a_i
    i = np.arange(n+1)
    a = beta * (2*i + 1) / 2      # shape = (n+1,)
    
    # 4) Killing the highest frequency term
    a[-1] = 0
    
    # 5) Return Modal to Nodal 
    a = V.dot(a)
    return a

# This function calculates the energy of the highest mode in a modal representation of a solution
def fractional_energy_highest_mode(a):
    """
    Compute the fraction of total L²‐energy contained in the highest Legendre mode.
    
    Parameters
    ----------
    a : array_like, shape (n+1,)
        Modal coefficients [a₀, a₁, …, aₙ] of a polynomial of degree n.
    
    Returns
    -------
    float
        Fractional energy in mode n:
          Eₙ / Σ_{i=0}ⁿ Eᵢ,
        where Eᵢ = aᵢ²·(2/(2i+1)).
    """
    a = np.asarray(a)
    n = a.size - 1
    i = np.arange(a.size)
    # energy in each mode: E[i] = a[i]**2 * (2/(2*i+1))
    E = a**2 * (2/(2*i + 1))
    return E[n] / E.sum()

###################################### Functions ##################################################
###################################################################################################


###################################################################################################
###################################### Step Jump ##################################################


E   = np.zeros(11) # Fractional Energy of the highest mode without any modification to the highest energy term
E_n = np.zeros(11) # Fractional Energy of the highest mode using the nodal interpolation
E_m = np.zeros(11) # Fractional Energy of the highest mode using the modal projection

for i in range(10): # Looping over polynomial degrees from 1 - 10
    n = i+2 
    k = i+1
    x7 , w7 = sb.lgl(n) # the 7 and 6 are old place holders. x7 means the absiccas of the nth polynomial and 6 means the same thing but for the kth order. 
    x6 , w6 = sb.lgl(k)

    u7 = np.where(x7 < 0, 0.0, 1.0) # Defining the step jump

# Interpolation onto a lower solution order 
    
    L6 = np.array([sb.lagrange(k,x7[i]) for i in range(n+1)]) # Operator that interpolates a nth polynomial into a kth polyniomial (k = n - 1 )
    P6 = np.linalg.inv(sb.sbp_p(k)) # Inverse of the mass matrix of order k. Again the 6 here just means the kth order. 
    P7 = sb.sbp_p(n) # Mass Matrix of order n. 7 just means of order n.       
    u_lower = P6@(L6.T.dot(P7@u7)) # This lower is the kth solution interpolated from the nth solution. 

# Interpolation back to original solution degree
    u7_n = L6.dot(u_lower) # This is essentially the lower order solution interpolated back into nth order. 

# Projection to the modal space and killing the highest frequency
    u7_m = nodal_to_modal(u7,x7,w7) # Projecting the nodal representation to the modal representation. 
    u7_m[-1] = 0 # Setting the coefficient of the highest degree mode to 0, effectivly killing the highest frequency. 
    

## Calcating the energy of the Solution 

# Projecting the Nodal solution onto the modal space 
    u7_nm = nodal_to_modal(u7_n,x7,w7)
# Calculating the fractional energy of the nodal approach
    E_n[i]   = fractional_energy_highest_mode(u7_nm)

# Calculating the fractional energy of the modal approach 
    E_m[i]   = fractional_energy_highest_mode(u7_m)

# Calculating the fractional energy of the un-modified jump
    u7_um    = nodal_to_modal(u7,x7,w7)
    E[i]     = fractional_energy_highest_mode(u7_um)
    
    
    # Comparing the Scaling of the Sensor - From Persson 
    
    diff_n = u7_um-u7_nm
    diff_m = u7_um-u7_m
    S_n = (n**2)*diff_n.T.dot(P7@diff_n)/(u7_um.T.dot(P7@u7_um)) # the coefficient n^2 is here to proove that the scalling of the sensor when applied to a jump discontinuity is indeed 1/n^2
    S_m = (n**2)*diff_m.T.dot(P7@diff_m)/(u7_um.T.dot(P7@u7_um))
    
    
    print("============================================================")
    print(f"Finished Calculating the Fractional Energy of {i+1}th Mode")
    print(f"Persson Sensor using the Nodal Interpolation Approach S_n = {S_n}")
    print(f"Persson Sensor using the Modal Projection Approach    S_m = {S_m}")
    print(f"Nodal Energy Approach = {E_n[i]}")
    print(f"Modal Energy Approach = {E_m[i]}")
    print(f"Energy of the unmodified jump = {E[i]}")
    print("============================================================")
