import numpy as np 
import SBP.Equations as eq1
from SBP.mesh_1d import Element1D

class Advection(eq1): 
    """
    1D linear advection with optional diffusion. 
    Implements:
      - interior_flux(elem):    –a * ∂_x u
      - diffusion_term(elem):    ∂_x( ν ∂_x u )
      - sat_penalty(elem, ...): upwind SAT for advection + SIPG viscous penalties
    """
    def __init__(self, a: float, nu: float = 0.0, v_off: float = 1.0):
        """
        Parameters:
          a     : advection speed
          nu    : constant viscosity (if nu>0, adds diffusion ∂_x(ν ∂_x u))
          v_off : flag (0 or 1) to turn viscous SAT on/off; multiply viscous SAT by v_off
        """
        self.a     = a
        self.nu    = nu
        self.v_off = v_off
    def interior_advective_flux(self, elem: Element1D): 
        """
        Computes the interior advective flux (a * u_x)
        which is a @ D @ u
        """
        return - elem.a * (elem.D_phys @ elem.u) # Linear Advection 
    
    def interior_diffusive_flux(self, elem: Element1D): 
        """
        Computes the interior diffusive fluxes (nu * u_xx)
        which is D @ (nu * D @ u), where nu is the viscocity
        """
        return elem.D_phys.dot(elem.nu * elem.D_phys.dot(elem.u))
    
    def SAT(self, elem: Element1D, gl: float, gr: float, dgl: float, dgr: float): 
        """
        Compute the SAT (boundary‐penalty) terms (length‐(n+1) array) combining:
          1) Upwind‐SAT for linear advection and assuming a > 0
          2) SIPG viscous SAT if nu>0 (scaled by v_off)
        """
        # Definition of Variables
        ul, ur       = elem.left_boundary(), elem.right_boundary() # Extracting the element boundaries
        grad_local = elem.D_phys.dot(elem.u) # Extracting Local gradients
        duL, duR   = grad_local[0], grad_local[-1] # Extracting gradients at the element faces
        jump_duL =    duL - dgl   # Jump in the gradient in the left face 
        jump_duR =   -duR + dgr   # Jump in the gradient in the right face 
        jump_uL  =    ul  - gl    # Jump in the state variables on the left face
        jump_uR  =   -ur  + gr    # Jump in the state variables on the right face
        nu       =    self.nu
        # Inviscid SAT Arithmatic
        tau_left     = -self.a / 2 # Linear Penalty proposional to the advection strength
        tau_right    = self.a / 2 # Linear Penalty proposional to the advection strength
        sat_inv      =  tau_left*(ul - gl)*elem.el + tau_right*(ur - gr)*elem.er # Weak enforcement of the inviscid fluxes

        # Viscous SAT Arithmatic 
        sat_visc_L = (0.5*nu*jump_duL + 0.5*nu*jump_uL) * elem.el
        sat_visc_R = (0.5*nu*jump_duR + 0.5*nu*jump_uR) * elem.er
