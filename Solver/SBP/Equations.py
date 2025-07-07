from abc import ABC, abstractmethod
import numpy as np 
from SBP.mesh_1d import Element1D
import SBP as sb
from .fluxes import *





class Equation1D(ABC): 
    """
    Abstract base class defining the interface for a 1D PDE.
    Any concrete subclass must implement:
      - interior_flux(self, elem):    inviscid/advective contribution
      - interior_diffusive_flux(self, elem):   diffusive (or AV) contribution
      - SAT(self, elem, left_ghost_u, right_ghost_u, left_ghost_du, right_ghost_du):
                                        SAT (boundary‐penalty) terms
    """

    @abstractmethod
    def inviscid_flux(self, element): 
        """
        Compute the inviscid (or advective) flux‐derivative RHS for thias element.

        Parameters:
          elem: an Element1D instance, whose .u and .D_phys (and other data) are populated.

        Returns:
          A NumPy array of length (n+1) giving –∂_x(f(u)) (or −a ∂_x u, etc.) evaluated
          at each node inside the element (excluding SAT penalties).
        """
        pass


    @abstractmethod
    def viscous_aux(self, elem: Element1D, gl:float, gr:float) -> np.ndarray:
        """
        Compute the diffusive‐term RHS (e.g. ∂_x(ν ∂_x u) or AV term) for this element.

        Parameters:
          elem: an Element1D instance, whose .u and .D_phys are populated.

        Returns:
          A NumPy array of length (n+1) giving ∂_x( ε(x) ∂_x u ) (or zero if purely advective),
          evaluated at each node inside the element (excluding SAT penalties).
        """
        pass

    @abstractmethod
    def SAT(
        self,
        elem,
        left_ghost_u: float,
        right_ghost_u: float,
        left_ghost_du: float,
        right_ghost_du: float
    ) -> np.ndarray:
        """
        Compute the SAT (boundary‐penalty) contribution for this element’s RHS.

        Parameters:
          elem            : an Element1D instance
          left_ghost_u    : value of u just to the left of this element
          right_ghost_u   : value of u just to the right of this element
          left_ghost_du   : ∂_x u just to the left of this element
          right_ghost_du  : ∂_x u just to the right of this element

        Returns:
          A NumPy array of length (n+1) containing the SAT terms (inviscid + viscous)
          that must be added to the interior RHS so that boundary/interface coupling
          is enforced.
        """
        pass


# =================================================================================================================================== #

class Advection(Equation1D): 
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
    def inviscid_flux(self, elem: Element1D): 
        """
        Computes the interior advective flux (a * u_x)
        which is a @ D @ u
        """
        return - self.a * (elem.D_phys @ elem.u) # Linear Advection 
    
    def viscous_aux(self, elem: Element1D): 
        """
        Computes the interior diffusive fluxes (nu * u_xx)
        which is D @ (nu * D @ u), where nu is the viscocity
        """
        nu = self.nu
        eps = elem.av_eps
        nu_sum = nu + eps
        return elem.D_phys.dot(nu_sum * elem.D_phys.dot(elem.u))
    
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
        nu       =    self.nu + elem.av_eps
        # Inviscid SAT Arithmatic
        tau_left     =  self.a / 2 # Linear Penalty proposional to the advection strength
        tau_right    =  self.a / 2 # Linear Penalty proposional to the advection strength
        sat_inv      =  tau_left*(ul - gl)*elem.el + tau_right*(ur - gr)*elem.er # Weak enforcement of the inviscid fluxes
        





        # Viscous SAT Arithmatic 
        sat_visc_L = (0.5*nu*jump_duL + 0.5*nu*jump_uL) * elem.el # Calculation of the left Viscous SAT
        sat_visc_R = (0.5*nu*jump_duR + 0.5*nu*jump_uR) * elem.er # Calculation of the right Viscous SAT
        
        # Forming the Total SAT for the Advection Equation
        sat_total = sat_inv + self.v_off*(sat_visc_L + sat_visc_R) 
        sat_rhs = elem.P_inv.dot(sat_total)
        return sat_rhs
      
      
# =================================================================================================================================== #
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

    def inviscid_flux(self, elem: Element1D): 
        """
        Computes the interior convective inviscid flux (-2 * Hadmard(D, F_num)). 
        F_num comes from the 2-point entropy flux function, which is defined in the legendre.py module.
        """
        
        return f_burger_ec(elem.n, elem.Q_phys, elem.u) * self.c_off
    
    def viscous_aux(self, elem: Element1D, gl:float, gr:float): 
        """
        LDG gradient reconstruction with penalty terms.
        This implementation uses entropy variables (since w = u for burgers equation)
        Parameters
        ----------
        elem : Element1D
            Element holding :pyattr:`u`, :pyattr:`D_phys`, :pyattr:`P_inv`,
            and the boundary pick-off vectors :pyattr:`el`, :pyattr:`er`.
        u_star_L, u_star_R : float
            Numerical traces (ghost states) supplied by the Riemann solver
            on the *left* and *right* faces of the element.

        Returns
        -------
        ndarray
            Length ``n+1`` vector `theta` ≈ ∂ₓu on the element.
        """
        # Defining the relevant variables 
        nu = self.nu
        eps = elem.av_eps
        nu_sum = nu + eps
        kl       = -1    # Normal to the left face of element k 
        kr       =  1    # Normal to the right face of element k
        el       = elem.el 
        er       = elem.er
        alpha    = elem.alpha
        u        = elem.u
        grad_rot = 1     # This is the conversion from primitve to entropy variables

        # Forming the gradient theta 
        dw = elem.D_phys@u
        theta = elem.P_inv@(dw +(-kr/2)*(1-kr*alpha)*(grad_rot*u[-1] - grad_rot*gr)*er + (-kl/2)*(1-kl*alpha)*(grad_rot*u[0] - grad_rot*gl)*el) 
        return theta
    
    def SAT(self, elem: Element1D, gl: float, gr: float, dgl: float, dgr: float, av_l:float, av_r:float): 
        """
        Full SAT = SAT_inv + SAT_visc. 
        SAT_inv => Entropy Stable Lax-Friedrichs with Merriam \lambda = u_l. 
        SAT_visc => uses standard jump in the gradient to evaluate the viscous SAT + IP-term that uses the same logic as Dalcin implemented in SSDC.

        gl => Ghost cell on the left
        gr => Ghost cell on the right
        dgl => Gradient of the gl at the left interface
        dgr => Gradient of the gr at the right interface
        av_l => av value at the left ghost cell
        av_r => av value at the right ghost cell
        """
        # Definition of Variables
        ul, ur   = elem.left_boundary(), elem.right_boundary() # Extracting the element boundaries
        nu       =    self.nu + elem.av_eps
        kl       = -1    # Normal to the left face of element k 
        kr       =  1    # Normal to the right face of element k
        el       = elem.el 
        er       = elem.er
        du       = elem.du # The gradients are stored element wise - this method assumes that they have been calculated by viscous_aux

        # Forming the rhs
        sat_inv  = (kr*(f_burger(ur)-f_ssr_meriam(ur,gr,ur,gr,f_burger_ec))*er                # Right Interface Flux -> SAT_inv_r
                   + kl*(f_burger(ul)-f_ssr_meriam(ul,gl,ul,gl,f_burger_ec))*el)             # Left Interface Flux -> SAT_inv_l
        sat_visc = (kr*(1/2)*(nu*du[-1]))*er + (kl*(1/2)*(nu*du[0]))*el
        
        return sat_visc + self.c_off*sat_inv 
      
      
      
      
      