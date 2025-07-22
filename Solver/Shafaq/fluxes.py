"""
fluxes.py
~~~~~~~~~
Elementary numerical‐flux functions used in finite-volume / DG
schemes for 1-D scalar conservation laws.

Defining fluxes here, will allow for the Equations class to "grab" a flux type and use it in the solver.
"""

# Imports       -----------------------------------------------------
from __future__ import annotations  # for Python <3.11
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class Flux(ABC):
    """Interface all concrete fluxes must satisfy."""
    @abstractmethod
    def flux(self, u): ...
    @abstractmethod
    def flux_grad(self, u): ...
    @abstractmethod
    def flux_ec(self, u1: NDArray, u2: NDArray): ...
    @abstractmethod
    def flux_ec_vol(self, Q: NDArray, u :NDArray): ...
    @abstractmethod
    def ip_term(self, nu_i:float, nu_gi:float , det_J:float, B:float = 1): ...
    @abstractmethod
    def f_ssr_meriam(self, u_int:float | NDArray[np.float64],
             g_int:float | NDArray[np.float64], 
             w_uint:float | NDArray[np.float64], 
             w_gint:float | NDArray[np.float64]): ...
    
# ---------- Burgers --------------------------------------------
class BurgerFlux(Flux):
    
    def flux(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Element-wise nonlinear **Burger’s** flux *f(u) = ½ u²*.

        Parameters
        ----------
        u : array_like
            State values.

        Returns
        -------
        ndarray
            `0.5 * u**2`, same shape as *u*.
        """
        return 0.5 * u ** 2
    
    def flux_grad(self, u):
        return u
    
    def flux_ec(self, u1: NDArray[np.float64],
                u2: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Entropy-conservative semi-discrete volume term for Burger’s
        equation using the symmetric form

        .. math:: F_{ij} = \\frac16 \\bigl(u_i^2 + u_i u_j + u_j^2\bigr),

        then contracting with the SBP operator *Q*.

        Parameters
        ----------
        u1 : (n+1,) ndarray
            Nodal state vector on that element boundary (float 1D or array 2D-3D).
        u2 : (n+1,) ndarray
            Nodal state vector to calculate the two-point flux function (float 1D or array 2D-3D).

        Returns
        -------
        ndarray
            Length-`n+1` RHS contribution ``F( u1 , u2 )`` (scalar).
        """
        # Broadcast u_i and u_j onto a 2-D grid without explicit loops   
        return (u1**2 + u1 * u2 + u2**2) / 6.0    # shape (n+1, n+1)

    def flux_ec_vol(self, Q: NDArray[np.float64],
                u: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Entropy-conservative semi-discrete volume term for Burger’s
        equation using the symmetric form

        .. math:: F_{ij} = \\frac16 \\bigl(u_i^2 + u_i u_j + u_j^2\\bigr),

        then contracting with the SBP operator *Q*.

        Parameters
        ----------
        Q : (n+1, n+1) ndarray
            First-derivative SBP matrix for one element.
        F_ec : (n+1, n+1) ndarray
            Array of Entropy conservative flux values.

        Returns
        -------
        ndarray
            Length-`n+1` RHS contribution ``-2 * Q @ F @ 1`` (vector form).
        """
        F_ec = self.flux_ec(u[:, None], u[None, :]) #self.flux_ec(u,u)
        return -2.0 * (Q * F_ec).sum(axis=1)
    

    def ip_term(self, nu_i:float, nu_gi:float , det_J:float, B:float = 1): 
        """ This function calculates the lamda_V or the L term (from Fisher and Carpenter or Parsani Discontinous Interfaces respectivly)). 

        Parameters
        ----------

        nu_i : float
            Viscous term at the ith interface of the kth element. 
        nu_gi : float
            Viscous term at the ith interface of the neighbor to kth element. 

        det_J : float
            Determinent of the interface. 
        Returns
        -------
        L : float
            The IP term. 
        """
        return -B*(nu_i + nu_gi)/det_J
    
    def f_ssr_meriam(self, u_int:float | NDArray[np.float64],
             g_int:float | NDArray[np.float64], 
             w_uint:float | NDArray[np.float64], 
             w_gint:float | NDArray[np.float64]):
        """
        Entropy Stable interface flux. From the Carpenter and Fisher paper 2014 eq (4.13). 
        Using the merriam convention. 
        """
        f_ssr = self.flux_ec(u_int,g_int) + 0.5*np.abs(self.flux_grad(u_int))*(w_uint - w_gint) 
        return f_ssr

# ---------- Linear advection -----------------------------------
class AdvectiveFlux(Flux):

    def __init__(self, a: float): self.a = float(a)


    def flux(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Element-wise **linear advection** flux *f(u) = a u*.

        Parameters
        ----------
        a : float
            Constant advection speed.
        u : array_like
            State values at solution points (any NumPy-broadcastable shape).

        Returns
        -------
        ndarray
            `a * u`, with the same shape as *u*.
        """
        return self.a * u
    
    
    
    def flux_grad(self, u):
        return self.a
    
    
    def flux_ec(self,
                 u_L: NDArray[np.float64],
                 u_R: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        **Entropy-conservative** two-point flux for linear advection
        (Tadmor 1987, *J. Comp. Phys.*).

        .. math:: \\hat f^{ec}(u_L,u_R) = a \\frac{u_L + u_R}{2}.

        Parameters
        ----------
        a : float
            Advection speed (may be ±).
        u_L, u_R : array_like
            Left and right states.  Shapes must be broadcast-compatible.

        Returns
        -------
        ndarray
            Central flux of the same broadcasted shape as `np.broadcast(u_L, u_R)`.
        """
        return 0.5 * self.a * (u_L + u_R)


    def flux_ec_vol(self, Q : NDArray , u: NDArray[np.float64]): 
        
        f_ec = 0.5 * self.a * (u[:,None] + u[None,:])
        return -2.0 * (Q * f_ec).sum(axis=1)


    def ip_term(self, nu_i:float, nu_gi:float , det_J:float, B:float = 1): 
        """ This function calculates the lamda_V or the L term (from Fisher and Carpenter or Parsani Discontinous Interfaces respectivly)). 

        Parameters
        ----------

        nu_i : float
            Viscous term at the ith interface of the kth element. 
        nu_gi : float
            Viscous term at the ith interface of the neighbor to kth element. 

        det_J : float
            Determinent of the interface. 
        Returns
        -------
        L : float
            The IP term. 
        """
        return -B*(nu_i + nu_gi)


    def f_ssr_meriam(self, u_int:float | NDArray[np.float64],
                 g_int:float | NDArray[np.float64], 
                 w_uint:float | NDArray[np.float64], 
                 w_gint:float | NDArray[np.float64]):
        """
        Entropy Stable interface flux. From the Carpenter and Fisher paper 2014 eq (4.13). 
        Using the merriam convention. 
        """
        f_ssr = self.flux_ec(u_int,g_int) + 0.5*np.abs(self.flux_grad(u_int))*(w_uint - w_gint) 
        return f_ssr



# ----------------------------------------------------------------------
# f_ssr_meriam 
# ----------------------------------------------------------------------

