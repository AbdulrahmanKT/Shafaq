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
                 nu : float = 0.0,
                 c_off: float = 1.0):
        
        self.flux = flux # Inherits flux type (e.g. Burgers, Advecion, Euler)
        self.nu = nu         # Viscocity of the equation - Later to be defined by another class called viscous fluxes for (NS and Euler)
        self.c_off  = c_off     # Coefficient for turning off convection - Debugging




    


    
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
        theta = dw + elem.P_inv@( +(-kr)*(u[-1] - gr)*er + (-kl)*(u[0] - gl)*el) # The jump sign convention is " -(normal)(Inner - Outer) " 
        return theta
    

    def volume_flux(self, elem: Element1D): 
        """
        Computes the interior convective inviscid flux (-2 * Hadmard(D, F_num)). 
        F_num comes from the 2-point entropy flux function, which is defined in the legendre.py module.
        """
        
        return (self.flux.flux_ec_vol(elem.Q_phys, elem.u)*self.c_off  + elem.D_phys@((self.nu + elem.av_eps)*elem.du))
    
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
        ul = elem.left_boundary()
        ur = elem.right_boundary()
        nu       =    float(self.nu + elem.av_eps)
        nu_l     =    float(self.nu + av_l) 
        nu_r     =    float(self.nu + av_r)       
        kl       = -1    # Normal to the left face of element k 
        kr       =  1    # Normal to the right face of element k
        el       = elem.el 
        er       = elem.er
        du       = elem.du # The gradients are stored element wise - this method assumes that they have been calculated by viscous_aux
        J        = elem.J  # Determinante of the jacobian of the element
        
        # Forming the inviscid SAT
        sat_inv_r  =    kr*(self.flux.flux(ur)-self.flux.f_ssr_meriam(ur,gr,ur,gr,self.flux.flux_ec))*er # Right Interface Flux -> SAT_inv_r
        sat_inv_l  =    kl*(self.flux.flux(ul)-self.flux.f_ssr_meriam(ul,gl,ul,gl,self.flux.flux_ec))*el             # Left Interface Flux -> SAT_inv_l
        sat_inv = sat_inv_r + sat_inv_l
        

        # Forming the viscous SAT
        sat_visc_r = (-kr*(nu*du[-1] - nu_r*dgr) - self.flux.ip_term(nu_i=nu, nu_gi=nu_r, det_J=J)*(ur - gr))*er # The gradient jump should be opposite of normal, and the IP term should penalize opposite of that
        sat_visc_l = (-kl*(nu*du[ 0] - nu_l*dgl) + self.flux.ip_term(nu_i=nu, nu_gi=nu_l, det_J=J)*(ul - gl))*el # The gradient jump should be opposite of normal, and the IP term should penalize opposite of that
        sat_visc = sat_visc_r + sat_visc_l
    
        #print("IP = ", self.flux.ip_term(nu_i=nu, nu_gi=nu_r, det_J=J)*(ur - gr)*er + self.flux.ip_term(nu_i=nu, nu_gi=nu_l, det_J=J)*(ul - gl)*el)
        #print("Diffusive SAT = ", kr*(nu*du[-1] - nu_r*dgr)*er + kl*(nu*du[ 0] - nu_l*dgl)*el)
        #print("Diffusive SAT face = ", sat_visc_r*er + sat_visc_l*el ) #*kl*(nu_l*dgl - nu*du[ 0] )
        return elem.P_inv@ (sat_visc + sat_inv*self.c_off) 

      
      
      