from abc import ABC, abstractmethod
import numpy as np 


class Equation(ABC): 
    """
    Abstract base class defining the interface for a 1D PDE.
    Any concrete subclass must implement:
      - interior_flux(self, elem):    inviscid/advective contribution
      - diffusion_term(self, elem):   diffusive (or AV) contribution
      - sat_penalty(self, elem, left_ghost_u, right_ghost_u, left_ghost_du, right_ghost_du):
                                        SAT (boundary‐penalty) terms
    """

    @abstractmethod
    def interior_flux(self, element): 
        """
        Compute the inviscid (or advective) flux‐derivative RHS for this element.

        Parameters:
          elem: an Element1D instance, whose .u and .D_phys (and other data) are populated.

        Returns:
          A NumPy array of length (n+1) giving –∂_x(f(u)) (or −a ∂_x u, etc.) evaluated
          at each node inside the element (excluding SAT penalties).
        """
        pass


    @abstractmethod
    def diffusion_term(self, elem) -> np.ndarray:
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
    def sat_penalty(
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
