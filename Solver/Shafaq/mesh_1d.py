import numpy as np 
import matplotlib.pyplot as plt 
import Shafaq as sb
from . import Shock



############ For VSCODE to import the autocomplete ###################
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import Shock
######################################################################



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
                 Q_ref: np.ndarray, 
                 equation):  # viscocity
        self.index   = index
        self.left    = left
        self.right   = right
        self.xi      = xi        # shape (n+1,)
        self.n       = xi.size-1
        self.equation = equation # Stores Equation to be solved
        

        
        
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
        self.D_phys  = D_ref / ( self.J)         # ∂/∂x = (1/J) ∂/∂ξ
        self.P_phys  = P_ref * ( self.J)         # ∫_x = ∫_ξ J dξ
        self.Q_phys  = Q_ref # Notice the use of the physical operators 
        self.P_inv   = np.linalg.inv(self.P_phys)
        self.el      = np.eye(self.n + 1)[0] # Using the E operator that is consistant with SBP theory 
        self.er      = np.eye(self.n + 1)[-1] # Using the E operator that is consistant with SBP theory
        self.alpha   = 0 # LDG constant
        # shock capturing 
        self.av_eps = 0        # This is the initialization of artificial viscocity to be added
        self.S      = 0        # This is the initialization of the Sensor
        # Initialization for Solution vector and for RHS vector
        self.u          = np.zeros(self.n+1)
        self.du         = np.zeros(self.n+1)
        self.irhs       = np.zeros_like(self.u)
        self.sat_rhs    = np.zeros_like(self.u)
        self.rhs        = np.zeros_like(self.u)
        # Debug Variables




        # Runge-Kutta 4 Stage Solution Vectors
        self.K1 = np.zeros_like(self.u)
        self.K2 = np.zeros_like(self.u)
        self.K3 = np.zeros_like(self.u)
        self.K4 = np.zeros_like(self.u)
# -----------------------------------------------------------------------------
    def left_boundary(self): 
        return self.u[0] # This should select the left boundary node
# -----------------------------------------------------------------------------   
    def right_boundary(self): 
        return self.u[-1] # This should select the left boundary node
# -----------------------------------------------------------------------------
    def set_solution_reference(self, u_ref: np.ndarray):
        """Set the nodal values on reference nodes."""
        assert u_ref.shape == (self.n+1,)
        self.u = u_ref.copy()
# -----------------------------------------------------------------------------
    def solution_physical(self) -> np.ndarray:
        """
        Since values are stored at the same nodes,
        U_phys(x_i) = U_ref(xi_i).
        """
        return self.u
# -----------------------------------------------------------------------------    
    def set_initial_condition(self, f):
        """ In: f - a vector of data that must match u size. 
        """
        self.u = f
# -----------------------------------------------------------------------------
    def map_to_reference(self, x_phys: float) -> float:
        return (2*x_phys - (self.left + self.right))/(self.right - self.left)
# -----------------------------------------------------------------------------
    def map_to_physical(self, xi: float) -> float:
        return ( (self.right - self.left)/2 )* xi + (self.right + self.left)/2
# -----------------------------------------------------------------------------    
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

        ax.plot(self.x, self.u, label="u")
        ax.grid(True)
        return ax
# -----------------------------------------------------------------------------
    def SAT_rhs(self, gl, gr, dgl, dgr, avl, avr): 
        """This method calculates the RHS contribution to this element from the neighboring elements via gl and gr.
        Note: Each call of this method will calculate and store the output in its corresponding vector. 
        """
        
        return self.equation.SAT(self, gl=gl, gr=gr, dgl=dgl, dgr=dgr, av_l=avl, av_r=avr)  
# -----------------------------------------------------------------------------    
    def volume_flux(self):
        """This method calculates the RHS contribution from the interior operator (all of the volumetric term INV and VISC).
        Note: Each call of this method will calculate and store the output in its corresponding vector. 
        """

        return self.equation.volume_flux(self)
# -----------------------------------------------------------------------------
    def viscous_flux(self, gl, gr): 
        """This method calculates the RHS contribution to this element from the neighboring elements via gl and gr.
        Note: Each call of this method will calculate and store the output in its corresponding vector. 
        """
        
        return self.equation.viscous_aux(self, gl, gr)  
# -----------------------------------------------------------------------------
#    def element_rhs(self, gl, gr, duL_n, duR_n): 
#        """This method calculates the total RHS contribution from the IRHS and SAT_RHS. 
#        Note: Each call of this method will calculate and store the output in its corresponding vector. 
#        """
#        inv     = self.inviscid_flux()
#        sat_rhs  = self.SAT_rhs(gl, gr, duL_n, duR_n)
#        self.rhs = irhs + sat_rhs
#        return self.rhs
# -----------------------------------------------------------------------------
    def __dir__(self):
        """
        Return the usual dir() list *plus* all runtime attributes so that
        IPython / Jupyter Tab-completion shows x, D_ref, etc.
        """
        # standard names from the superclass
        std = super().__dir__()
        # everything created in __init__
        dynamic = list(self.__dict__.keys())
        # merge and deduplicate
        return sorted(set(std + dynamic))

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
                 Q_ref: np.ndarray,
                 equation, 
                 shock_capture:bool = False):
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
        equation: Equation1D
            The equation to be solved, providing the necessary methods
            for computing fluxes and SAT terms.
        shock_capture : bool
            This option turns on or off (True or False) the shock
            capturing scheme.
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
        self.el    = np.eye(self.n + 1)[0]
        self.er    = np.eye(self.n + 1)[-1]
        self.E0    = 6 # This Quantity is to normalize the total energy
        self.equation = equation  # Store the equation to be solved
        

        # Shock Capturing 
        self.use_shock_capture = shock_capture # This option controls if the shock capturing is initialzed
        if self.use_shock_capture == True: 
            self.V = sb.legendre.Vmonde_orthonormal(self.xi, self.w, self.n) # Bulding the Vandermonde Matrix for Converting to modal representation of the solution
        self.s0 = -1 #np.log(1/self.n**4)
        self.kappa = 1 
        self.eps_max = 0.01   

        
         
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
                Q_ref=self.Q_ref,
                equation=equation
            )
            self.elements.append(elem)
# -----------------------------------------------------------------------------
    def global_coordinates(self) -> np.ndarray:
        """
        Return sorted unique global node coordinates across elements.
        """
        coords = []
        for elem in self.elements:
            coords.extend(elem.x.tolist())
        return np.unique(coords)
# -----------------------------------------------------------------------------   
    def x(self) -> np.ndarray:
        """
        Return a 2D NumPy array of physical node coordinates for each element.

        Output shape is (nex, n+1), where each row gives `elem.x`.
        """
        # Stack each element's x array into a 2D array
        return np.vstack([elem.x for elem in self.elements])
# -----------------------------------------------------------------------------
    def set_initial_condition(self, f):
        """ F is a function that takes in the coordinates. 
        """
        for idx, elem in enumerate(self.elements):
            x_phys = elem.x
            y_num = f(x_phys)
            elem.set_initial_condition(y_num)
        self.E0 = self.total_energy()
# -----------------------------------------------------------------------------
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

        eopts = {**edge_defaults, **(edge_opts or {})}
        nopts = {**node_defaults, **(node_opts or {})}
        bopts = {**boundary_defaults, **(boundary_opts or {})}

        # plot element edges and n
        for elem in self.elements:
            x = elem.x
            y = np.zeros_like(x)
            ax.plot(x, y, **eopts)
            ax.plot(x, y, **nopts)
            #ax.set_title("Mesh Nodes")
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
# -----------------------------------------------------------------------------    
    def rhs(self):
        """
        Compute du/dt for every element under periodic BCs,
        storing each element’s residual in elem.rhs.
        """
        NE = self.nex
        # Gradient Sweap - LDG step
        for e_id, elem in enumerate(self.elements):
            # periodic neighbor indices
            left_id  = (e_id - 1) % NE
            right_id = (e_id + 1) % NE # To properly find the next element

            # neighbor boundary values
            gl = self.elements[left_id].right_boundary()
            gr = self.elements[right_id].left_boundary()

            # LDG step to calculate the gradients
            elem.du = elem.viscous_flux(gl, gr)
            
        

        # Full RHS sweap
        for e_id, elem in enumerate(self.elements):
            # periodic neighbor indices
            left_id  = (e_id - 1) % NE
            right_id = (e_id + 1) % NE # To properly find the next element

            # neighbor boundary values
            gl = self.elements[left_id].right_boundary()
            gr = self.elements[right_id].left_boundary()
            
            # neighbor ∂u/∂x at the faces from neighboring elements
            dgl = self.elements[left_id].du[-1] # Gradients at the neighboring elements at the left element at the left boundary
            dgr = self.elements[right_id].du[0]  # Gradients at the neighboring elements at the right element at the right boundary
            
            # Neighboring AV 
            avl = self.elements[left_id].av_eps # Gradients at the neighboring elements at the left element at the left boundary
            avr = self.elements[right_id].av_eps  # Gradients at the neighboring elements at the right element at the right boundary
            
            # Forming the full rhs
            sat_visc = elem.SAT_rhs( gl, gr, dgl, dgr, avl, avr)
            elem.rhs = elem.volume_flux() + sat_visc
            elem.debug_sat_visc = sat_visc
        
# ----------------------------------------------------------------------------- 
    def export_global_rhs(self) -> np.ndarray:
        """
        Compute and return the complete mesh‐wide RHS as a 1D vector
        (periodic BCs assumed). Does NOT store it on self.
        """
        NE = self.nex
        # First, update each element’s local rhs
        for e_id, elem in enumerate(self.elements):
            left_id  = (e_id - 1) % NE
            right_id = (e_id + 1) % NE
            gL = self.elements[left_id].right_boundary()
            gR = self.elements[right_id].left_boundary()
            elem.element_rhs(gL, gR)

        # Then concatenate all element.rhs into one vector
        return np.concatenate([elem.rhs for elem in self.elements])
# -----------------------------------------------------------------------------       
    def step_rk4(self, dt):
        """
        One classical RK4 step of size dt based on mesh.rhs().
        """
        # --------  snapshot of the solution ------------------------------------
        U0 = [elem.u.copy() for elem in self.elements]

        # -------------------------------------------------------------------- K1
        self.shock_capture()        # update artificial-viscosity eps_i
        self.rhs()                  # fills elem.rhs for all elements
        for elem in self.elements:
            elem.K1 = elem.rhs.copy()

        # -------------------------------------------------------------------- K2
        for e, elem in enumerate(self.elements):
            elem.u = U0[e] + 0.5*dt*elem.K1
        self.shock_capture()
        self.rhs()
        for elem in self.elements:
            elem.K2 = elem.rhs.copy()

        # -------------------------------------------------------------------- K3
        for e, elem in enumerate(self.elements):
            elem.u = U0[e] + 0.5*dt*elem.K2
        self.shock_capture()
        self.rhs()
        for elem in self.elements:
            elem.K3 = elem.rhs.copy()

        # -------------------------------------------------------------------- K4
        for e, elem in enumerate(self.elements):
            elem.u = U0[e] + dt*elem.K3
        self.shock_capture()
        self.rhs()
        for elem in self.elements:
            elem.K4 = elem.rhs.copy()

        # ---------------------- final RK4 combine ------------------------------
        for e, elem in enumerate(self.elements):
            elem.u = (
                U0[e] + (dt/6.0)*(elem.K1 + 2*elem.K2 + 2*elem.K3 + elem.K4)
            )
        self.rhs()
# -----------------------------------------------------------------------------       
    def total_energy_normalized(self) -> float:
        """
        Compute  dS/dt.
        """
        E = 0.0
        for elem in self.elements:
            # u^T P_phys u  = sum_i (P_phys_ii * u_i^2)
            E += elem.u.T @ (elem.P_phys @ elem.rhs) # This formulation allows for the user to observe the dE/dt
        return E / self.E0
# -----------------------------------------------------------------------------
    def total_energy(self) -> float:
        """
        Compute  E = 1/2 * sum_e (u_e^T P_phys u_e).
        """
        E = 0.0
        for elem in self.elements:
            # u^T P_phys u  = sum_i (P_phys_ii * u_i^2)
            E += elem.u @ (elem.P_phys @ elem.u)
        return 0.5 * E       
# ----------------------------------------------------------------------------- 
    def shock_capture(self): 
        """Computes the Persson Sensor and the Artificial Viscocity. 
        Stores the AV in elem.av_eps."""
        if not self.use_shock_capture:
            return                           # This option exits early when this option is not true
        
        V = self.V 
        w = self.w
        s0 = self.s0
        kappa = self.kappa
        eps_max = self.eps_max

        for elem in self.elements: 
            a = Shock.nodal_to_modal(u=elem.u,w=w,V=V)
            elem.S = Shock.perrson_sensor(a=a)
            elem.av_eps = Shock.av(elem.S, s0=s0, kappa=kappa, e0=eps_max)
# -----------------------------------------------------------------------------
    def print_av(self) -> float:
       """Prints a field of AV values accross all elements.
       """
       return np.vstack([elem.av_eps for elem in self.elements])
# -----------------------------------------------------------------------------
    def print_S(self) -> float:
       """Prints a field of AV values accross all elements.
       """
       return np.vstack([elem.S for elem in self.elements])
           
#######################################
############ Meshing
#######################################
