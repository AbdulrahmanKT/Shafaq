import numpy as np 
import matplotlib.pyplot as plt 



#######################################
############ Elements
#######################################

class Element1D:
    """
    A 1D element that only stores:
      - left/right physical coords
      - reference nodes xi
      - reference SBP matrices D_ref, P_ref
    and computes its Jacobian & physical‐space operators once, at __init__.
    """
    def __init__(self,
                 index: int,
                 left: float,
                 right: float,
                 xi: np.ndarray,      # reference nodes
                 D_ref: np.ndarray,   # reference D
                 P_ref: np.ndarray,
                 Q_ref: np.ndarray):  # reference P
        self.index   = index
        self.left    = left
        self.right   = right
        self.xi      = xi        # shape (n+1,)
        self.n       = xi.size-1

        # physical nodes: x(ξ) = h*ξ + c
        h         = (right - left)/2
        c         = (right + left)/2
        self.x    = h*xi + c     # shape (n+1,)

        # Jacobian & scaled SBP
        self.J        = h
        self.D_ref    = D_ref
        self.P_ref    = P_ref
        self.Q_ref    = Q_ref
        # physical‐space SBP operators:
        self.D_phys = D_ref / h         # ∂/∂x = (1/J) ∂/∂ξ
        self.P_phys = P_ref * h         # ∫_x = ∫_ξ J dξ
        self.Q_phys = self.P_phys.dot(self.D_phys)

        # placeholder for solution in reference space
        self.solution = np.zeros(self.n+1)

    def set_solution_reference(self, u_ref: np.ndarray):
        """Set the nodal values on reference nodes."""
        assert u_ref.shape == (self.n+1,)
        self.solution = u_ref.copy()

    def solution_physical(self) -> np.ndarray:
        """
        Since values are stored at the same nodes,
        U_phys(x_i) = U_ref(xi_i).
        """
        return self.solution
    
    def set_initial_condition(self, f):
        """ In: f - function that is callable. 
    """
        u_0 = np.zeros(self.n+1)
        u_0 = f(self.x)
        self.solution = u_0

    def map_to_reference(self, x_phys: float) -> float:
        return (2*x_phys - (self.left + self.right))/(self.right - self.left)

    def map_to_physical(self, xi: float) -> float:
        return ( (self.right - self.left)/2 )* xi + (self.right + self.left)/2
    
    def plot(self, ax=None):
        """
        Plot this element’s nodal solution using its own style.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,3))
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.grid(True)
            ax.set_title("Solution per element")

        ax.plot(self.x, self.solution)
        ax.grid(True)
        return ax


#######################################
############ Elements
#######################################



