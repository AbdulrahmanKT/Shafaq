import numpy as np 
from SBP.Equations import Equation1D
from SBP.mesh_1d import Element1D
import SBP as sb

class Burger(Equation1D): 
    """
    1D linear advection with optional diffusion. 
    Implements:
      - interior_flux(elem):    –a * ∂_x u
      - diffusion_term(elem):    ∂_x( ν ∂_x u )
      - sat_penalty(elem, ...): upwind SAT for advection + SIPG viscous penalties
    """
    def __init__(self, c_off: float, nu: float = 0.0, v_off: float = 1.0):
        """
        Parameters:
          c_off : flag (0 or 1) to turn inviscid SAT on/off; multiply inviscid SAT by v_off
          nu    : constant viscosity (if nu>0, adds diffusion ∂_x(ν ∂_x u))
          v_off : flag (0 or 1) to turn viscous SAT on/off; multiply viscous SAT by v_off
        """
        self.c_off = c_off # This parameter turns off convection
        self.nu    = nu
        self.v_off = v_off
    def interior_flux(self, elem: Element1D): 
        """
        Computes the interior convective inviscid flux (-2 * Hadmard(D, F_num)). 
        F_num comes from the 2-point entropy flux function, which is defined in the legendre.py module.
        """
        return sb.two_point_flux_function(elem.n, elem.D_phys, elem.u) * self.c_off
    
    def interior_diffusive_flux(self, elem: Element1D): 
        """
        Computes the interior diffusive fluxes (nu * u_xx)
        which is D @ (nu * D @ u), where nu is the viscocity
        """
        return elem.D_phys.dot(self.nu * elem.D_phys.dot(elem.u))
    
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
        tau_left     = -np.abs((ul + gl))*self.m / 2.0 # Linear Penalty proposional to the convection strength
        tau_right    =  np.abs((ur + gr))*self.m / 2.0 # Linear Penalty proposional to the convection strength
        sat_inv      =  tau_left*(ul - gl)*elem.el + tau_right*(ur - gr)*elem.er # Weak enforcement of the inviscid fluxes

        # Viscous SAT Arithmatic 
        sat_visc_L = (0.5*nu*jump_duL + 0.5*nu*jump_uL) * elem.el # Calculation of the left Viscous SAT
        sat_visc_R = (0.5*nu*jump_duR + 0.5*nu*jump_uR) * elem.er # Calculation of the right Viscous SAT
        
        # Forming the Total SAT for the Advection Equation
        sat_total = self.c_off*sat_inv + self.v_off*(sat_visc_L + sat_visc_R) 
        sat_rhs = elem.P_inv.dot(sat_total)
        return sat_rhs