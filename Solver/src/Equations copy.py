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
    def volume_flux(self, element): 
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

    def volume_flux(self, elem: Element1D): 
        """
        Computes the interior convective inviscid flux (-2 * Hadmard(D, F_num)). 
        F_num comes from the 2-point entropy flux function, which is defined in the legendre.py module.
        """
        return f_adv_ec_vol(self.a, elem.Q_phys, elem.u) + elem.D_phys@((self.nu + elem.av_eps)*elem.du)
    
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
        kl       = -1    # Normal to the left face of element k 
        kr       =  1    # Normal to the right face of element k
        el       = elem.el 
        er       = elem.er
        alpha    = 0
        u        = elem.u
        grad_rot = 1     # This is the conversion from primitve to entropy variables

        # Forming the gradient theta 
        dw = elem.D_phys@u
        theta = elem.P_inv@(dw +(-kr/2)*(1-kr*alpha)*(grad_rot*u[-1] - grad_rot*gr)*er + (-kl/2)*(1-kl*alpha)*(grad_rot*u[0] - grad_rot*gl)*el) 
        return theta
    
    def SAT(self, elem: Element1D, gl: float, gr: float, dgl: float, dgr: float, av_l:float, av_r:float): 
        """
        Full SAT = SAT_inv + SAT_visc. 
        SAT_inv => Entropy Stable Lax-Friedrichs with Merriam lambda = u_l. 
        SAT_visc => uses standard jump in the gradient to evaluate the viscous SAT + IP-term that uses the same logic as Dalcin implemented in SSDC.

        gl => Ghost cell on the left
        gr => Ghost cell on the right
        dgl => Gradient of the gl at the left interface
        dgr => Gradient of the gr at the right interface
        av_l => av value at the left ghost cell
        av_r => av value at the right ghost cell

        For the coupling of the IP term, the following the dalcin recipe is followed: 
          IP = P^-1 ((1/2)*lambda_v (w_l - w_gl)el + (1/2)*lambda_v (w_r - w_gr)er)

          1 - Rotate from primitive variables to the entropy variables
          2 - Take the difference in the normal direction of the face (This is important in higher dimensions 2D or 3D). 
          3 - Rotate back to the primitive variables. 
          4 - Construct the Lambda_V = L (from Parsani Discontious Interfaces paper) => L(ui, gi) = -B(c_ui + c_gi)/det(J) -> B = 1, c_ui and c_gi is the element viscocity at that interface.    
          5 - Form IP term for every interface.      
          """
        # Definition of Variables
        ul, ur   = elem.left_boundary(), elem.right_boundary() # Extracting the element boundaries
        nu       =    self.nu + elem.av_eps
        nu_l     =    self.nu + av_l 
        nu_r     =    self.nu + av_r       
        kl       = -1    # Normal to the left face of element k 
        kr       =  1    # Normal to the right face of element k
        el       = elem.el 
        er       = elem.er
        du       = elem.du # The gradients are stored element wise - this method assumes that they have been calculated by viscous_aux
        J        = elem.J  # Determinante of the jacobian of the element
        # Forming the rhs
        sat_inv  = (kr*(f_adv(self.a, ur)-f_ssr_meriam(ur,gr,ur,gr,f_adv_ec))*er                # Right Interface Flux -> SAT_inv_r
                   + (kl)*(f_adv(self.a, ul)-f_ssr_meriam(ul,gl,ul,gl,f_adv_ec))*el)             # Left Interface Flux -> SAT_inv_l
        sat_visc = ((-kr*(1/2)*(nu*du[-1] - nu_r*dgr) + IP_term_burger(nu_i=nu, nu_gi=nu_r, det_J=J)*(ur - gr))*er 
                    + (-kl*(1/2)*(nu*du[0] - nu_l*dgl) + IP_term_burger(nu_i=nu, nu_gi=nu_l, det_J=J)*(ul - gl))*el)
        
        return elem.P_inv@ (sat_visc + self.c_off*sat_inv) 
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

    def volume_flux(self, elem: Element1D): 
        """
        Computes the interior convective inviscid flux (-2 * Hadmard(D, F_num)). 
        F_num comes from the 2-point entropy flux function, which is defined in the legendre.py module.
        """
        
        return f_burger_ec_vol(elem.Q_phys, elem.u) * self.c_off + elem.D_phys@((self.nu + elem.av_eps)*elem.du)
    
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
        kl       = -1    # Normal to the left face of element k 
        kr       =  1    # Normal to the right face of element k
        el       = elem.el 
        er       = elem.er
        alpha    = 0
        u        = elem.u
        grad_rot = 1     # This is the conversion from primitve to entropy variables

        # Forming the gradient theta 
        dw = elem.D_phys@u
        theta = elem.P_inv@(dw +(-kr/2)*(1-kr*alpha)*(grad_rot*u[-1] - grad_rot*gr)*er + (-kl/2)*(1-kl*alpha)*(grad_rot*u[0] - grad_rot*gl)*el) 
        return theta
    
    def SAT(self, elem: Element1D, gl: float, gr: float, dgl: float, dgr: float, av_l:float, av_r:float): 
        """
        Full SAT = SAT_inv + SAT_visc. 
        SAT_inv => Entropy Stable Lax-Friedrichs with Merriam lambda = u_l. 
        SAT_visc => uses standard jump in the gradient to evaluate the viscous SAT + IP-term that uses the same logic as Dalcin implemented in SSDC.

        gl => Ghost cell on the left
        gr => Ghost cell on the right
        dgl => Gradient of the gl at the left interface
        dgr => Gradient of the gr at the right interface
        av_l => av value at the left ghost cell
        av_r => av value at the right ghost cell

        For the coupling of the IP term, the following the dalcin recipe is followed: 
          IP = P^-1 ((1/2)*lambda_v (w_l - w_gl)el + (1/2)*lambda_v (w_r - w_gr)er)

          1 - Rotate from primitive variables to the entropy variables
          2 - Take the difference in the normal direction of the face (This is important in higher dimensions 2D or 3D). 
          3 - Rotate back to the primitive variables. 
          4 - Construct the Lambda_V = L (from Parsani Discontious Interfaces paper) => L(ui, gi) = -B(c_ui + c_gi)/det(J) -> B = 1, c_ui and c_gi is the element viscocity at that interface.    
          5 - Form IP term for every interface.      
          """
        # Definition of Variables
        ul, ur   = elem.left_boundary(), elem.right_boundary() # Extracting the element boundaries
        nu       =    self.nu + elem.av_eps
        nu_l     =    self.nu + av_l 
        nu_r     =    self.nu + av_r       
        kl       = -1    # Normal to the left face of element k 
        kr       =  1    # Normal to the right face of element k
        el       = elem.el 
        er       = elem.er
        du       = elem.du # The gradients are stored element wise - this method assumes that they have been calculated by viscous_aux
        J        = elem.J  # Determinante of the jacobian of the element
        # Forming the rhs
        sat_inv  = (kr*(f_burger(ur)-f_ssr_meriam(ur,gr,ur,gr,f_burger_ec))*er                # Right Interface Flux -> SAT_inv_r
                   + (kl)*(f_burger(ul)-f_ssr_meriam(ul,gl,ul,gl,f_burger_ec))*el)             # Left Interface Flux -> SAT_inv_l
        sat_visc = ((-kr*(1/2)*(nu*du[-1] - nu_r*dgr) + IP_term_burger(nu_i=nu, nu_gi=nu_r, det_J=J)*(ur - gr))*er 
                    + (-kl*(1/2)*(nu*du[0] - nu_l*dgl) + IP_term_burger(nu_i=nu, nu_gi=nu_l, det_J=J)*(ul - gl))*el)
        
        return elem.P_inv@ (sat_visc + self.c_off*sat_inv) 
      
      
      
      
      