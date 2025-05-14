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

        # ----------------------------------> 
        ## RHS and Selectors
        self.el      = np.eye(self.n + 1)[0]
        self.er      = np.eye(self.n + 1)[-1]
        self.irhs    = np.zeros_like(u) # This is the interior RHS without element coupling
        # ---------------------------------->
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
        self.u = np.zeros(self.n+1)

    def left_boundary(self): 
        return self.u[0] # This should select the left boundary node
    
    def right_boundary(self): 
        return self.u[-1] # This should select the left boundary node
    


    def set_solution_reference(self, u_ref: np.ndarray):
        """Set the nodal values on reference nodes."""
        assert u_ref.shape == (self.n+1,)
        self.u = u_ref.copy()

    def solution_physical(self) -> np.ndarray:
        """
        Since values are stored at the same nodes,
        U_phys(x_i) = U_ref(xi_i).
        """
        return self.u
    
    def set_initial_condition(self, f):
        """ In: f - function that is callable. 
    """
        u_0 = np.zeros(self.n+1)
        u_0 = f(self.x)
        self.u = u_0

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

        ax.plot(self.x, self.u)
        ax.grid(True)
        return ax
#######################################
############ Elements
#######################################



#######################################
############ Meshing
#######################################



class Mesh1D:
    """
    1D mesh composed of equally spaced Element1D objects,
    with all SBP operators provided externally.
    """
    def __init__(self,
                 x_min: float,
                 x_max: float,
                 nex: int,
                 n: int,
                 xi: np.ndarray,
                 w: np.ndarray,
                 D_ref: np.ndarray,
                 P_ref: np.ndarray,
                 Q_ref: np.ndarray):
        """
        Initialize the mesh and its elements.

        Parameters
        ----------
        x_min, x_max : float
            Physical domain endpoints.
        nex : int
            Number of elements.
        n : int
            Polynomial degree (n+1 LGL nodes per element).
        xi : np.ndarray
            Reference nodes (length n+1).
        w : np.ndarray
            Reference weights (length n+1).
        D_ref : np.ndarray
            Differentiation matrix on reference.
        P_ref : np.ndarray
            Norm (quadrature) matrix on reference.
        Q_ref : np.ndarray
            Skew-symmetric SBP matrix on reference.
        """
        # store mesh parameters
        self.x_min, self.x_max = x_min, x_max
        self.nex, self.n       = nex, n

        # store reference SBP data
        self.xi    = xi.copy()
        self.w     = w.copy()
        self.D_ref = D_ref.copy()
        self.P_ref = P_ref.copy()
        self.Q_ref = Q_ref.copy()

        # build physical elements
        self.elements = []
        dx = (self.x_max - self.x_min) / self.nex
        for i in range(self.nex):
            L = self.x_min + i * dx
            R = L + dx
            elem = Element1D(
                index=i,
                left=L,
                right=R,
                xi=self.xi,
                D_ref=self.D_ref,
                P_ref=self.P_ref,
                Q_ref=self.Q_ref
            )
            self.elements.append(elem)

    def set_initial_condition(self, f):
        """
        Apply an initial-condition function f(x_phys) to every element.

        f may be vectorized (accept a NumPy array) or scalar-only.
        If f(x_phys) raises an exception, fallback to scalar loop.

        After calling this, each element's reference solution is set
        to f evaluated at its physical nodes.
        """
        for elem in self.elements:
            x_phys = elem.x
            try:
                u0 = f(x_phys)
            except Exception:
                u0 = np.array([f(x) for x in x_phys])
            elem.set_solution_reference(u0)

    def set_solution(self, U: np.ndarray):
        """
        Assign a full solution array to mesh elements.

        U must have shape (nex, n+1).
        """
        assert U.shape == (self.nex, self.n+1), \
            f"Expected U shape ({self.nex},{self.n+1}), got {U.shape}"
        for elem, u_row in zip(self.elements, U):
            elem.set_solution_reference(u_row)

    def global_coordinates(self) -> np.ndarray:
        """
        Return sorted unique global node coordinates across elements.
        """
        coords = []
        for elem in self.elements:
            coords.extend(elem.x.tolist())
        return np.unique(coords)
    
    
    def x(self) -> np.ndarray:
        """
        Return a 2D NumPy array of physical node coordinates for each element.

        Output shape is (nex, n+1), where each row gives `elem.x`.
        """
        # Stack each element's x array into a 2D array
        return np.vstack([elem.x for elem in self.elements])

    def plot(self,
             ax=None,
             edge_opts=None,
             node_opts=None,
             boundary_opts=None,
             figsize=None):
        """
        Draw the mesh edges, nodes, and element boundaries on the given Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. If None, creates new figure if figsize provided or uses plt.gca().
        edge_opts : dict, optional
            Style for element edges.
        node_opts : dict, optional
            Style for element nodes.
        boundary_opts : dict, optional
            Style for element boundary lines.
        figsize : tuple, optional
            Figure size (width, height) in inches, used only if ax is None to create a new figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Axes with the mesh drawn.
        """
        # create new figure/axes if none provided
        if ax is None:
            if figsize is not None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                ax = plt.gca()

        # default styles
        edge_defaults = {"linestyle": "-", "linewidth": 1, "color": "k"}
        node_defaults = {"marker": "o", "linestyle": "", "color": "C0"}
        boundary_defaults = {"linestyle": "--", "linewidth": 1, "color": "gray"}

        eopts = {**edge_defaults, **(edge_opts or {})} # Edge defaults for global plotting 
        nopts = {**node_defaults, **(node_opts or {})} # Node defaults for global plotting
        bopts = {**boundary_defaults, **(boundary_opts or {})} # Boundary defaults for global plotting

        # plot element edges and nodes
        for elem in self.elements:
            x = elem.x
            y = np.zeros_like(x)
            ax.plot(x, y, **eopts)
            ax.plot(x, y, **nopts)
            ax.set_title("Mesh Nodes")
            ax.set_xlabel("x - coordinate")
            ax.set_ylabel("u")
        # plot element boundaries
        ymin, ymax = ax.get_ylim()
        for elem in self.elements:
            #ax.plot([elem.left, elem.left], [ymin, ymax], **bopts)
            ax.axvline(elem.left, **bopts) # This makes sure that the boundary lines span to the maximum edges of the figure
            ax.axvline(elem.right, **bopts) # This makes sure that the boundary lines span to the maximum edges of the figure
        ax.plot([self.x_max, self.x_max], [ymin, ymax], **bopts)

        return ax















#######################################
############ Meshing
#######################################
