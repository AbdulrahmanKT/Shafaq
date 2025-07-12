from abc import ABC, abstractmethod
import numpy as np 
from Shafaq.mesh_1d import Element1D
import Shafaq as sb
from .fluxes import *             ## Contains the Flux class. 





class Equation1D(ABC): 
    """
    Abstract Class defining the Semi-discrete system. Every Flux class must implement the following: 

          - Flux function. 
          - Entropy Conservative Flux. 
          - Entropy Conservative Volume Flux. 
          - Transformation from Conservative Variables to Entropy variables and vice-versa. 
    """
    def __init__(self,
                 flux:Flux,
                 nu : float,
                 c_off: float):
        
        self.flux = flux # Inherits flux type (e.g. Burgers, Advecion, Euler)
        self.nu          # Viscocity of the equation - Later to be defined by another class called viscous fluxes for (NS and Euler)
        self.c_off       # Coefficient for turning off convection - Debugging




    @abstractmethod
    def volume_flux(self, elem: Element1D): 
        """
        Computes the interior convective inviscid flux (-2 * Hadmard(D, F_num)). 
        F_num comes from the 2-point entropy flux function, which is defined in the legendre.py module.
        """
        
        return self.flux.ec_volume(elem.Q_phys, elem.u) * self.c_off + elem.D_phys@((self.nu + elem.av_eps)*elem.du)


    @abstractmethod
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
      
      
      
      
      