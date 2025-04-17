

import numpy as np 
import matplotlib.pyplot as plt 
import sympy as sp


def p_n(x,n): 
    """
     This is a function that returns the nth order legendre polynomial evaluated at the point x, with order n.

     x: is the point of evaluation 
     n: is the order of the Legendre polynomial

    """
    if n == 0:
        return 1
    
    if n == 1:
        return x
    
    p = np.zeros(n+1)
    p[0] = 1
    p[1] = x
    for i in range(2,n+1):
        p[i] = ((2*i-1)*x*p[i-1]-(i-1)*p[i-2])/i
        
    return p[-1]



def legplot(f,n,*args):
    
    """
    This is a function that uses the matplotlib library. Where f is a function 
    handle which should a Legndre Polynomial or a derivative of a polynomial, n is 
    the order of that polynomial. 
    The function will plot all polynomials up to n. 

    f: Function handle
    n: order of such polynomial
    args: the left and right bounds of the interval of evaluation

    ** Each function uses 5000 evaluations 
    """
    plt.figure
    print(args)
    if len(args)==0:
        points = np.linspace(-1,1,5000)
    else: 
        [left,right] = [*args[0]]
        points = np.linspace(left,right,5000)
    plt.plot(points,np.zeros(5000),".")
    for i in range(n): 
        poly = [f(c,i) for c in points]
        plt.plot(points,poly,label = "n = " + str(i))
    
    
    if len(args) > 1 and isinstance(args[1], str):
        title = args[1]
        plt.title("Plot of the " +title + " up to n")
    else: 
        plt.title("Plot of the polynomials up to n")

    plt.show()
    plt.legend()
    plt.grid()


    



def dp_n(x,n): 
    
    """ This is a function that returns the first derivative of the nth order 
    legendre polynomial evaluated at the point x. 

    x: point of evaluation 
    n: is the order of the polynomial 
    
    """
    if n == 0:
        return 0
    
    if n == 1:
        return 1
    else:
        p = np.zeros(n+1)
        p[0] = 0
        p[1] = 1
        for i in range(2,n+1):
            p[i] = i*(x*p_n(x,i)-p_n(x,i-1))/(x**2 -1) # From the Bonnet Formula 
    return p[-1]



def p_n_c(n): 
    """
    Generates the coefficients of the Legendre polynomial of degree n.
    Returns the coefficients as a list, from the highest degree to the lowest.
    """

    x = sp.symbols("x")  # Symbols are used to eventually extract the coefficients


    if n == 0: 
        return [1]
    
    if n == 1: 
        return [0, 1]
    if n < 0: 
        raise ValueError("n cannot be negative, it is a natural number!!!")
    else:
        size = n + 1
        p = sp.zeros(size,1)
        p[0] = 1
        p[1] = 1*x 

        for i in range(2,n+1):
            p[i] = sp.simplify((((2*i - 1)* x * p[i-1]) - (i-1)*p[i-2])/i)
        out = sp.Poly(p[-1], x)
        out = out.all_coeffs()


    return out


def dp_n_c(n):
    """
    Generates the coefficients of the derivative of the Legendre polynomial of degree n.
    """
    x = sp.symbols("x")

    if n == 0:
        return 0
    
    if n == 1:
        return [1, 0]
    else:
        size = n + 1
        p = sp.zeros(size,1)
        p_prime = sp.zeros(size,1)

        ## From Bonnet's formula, one requires the legendre polynomials in order to 
        ## calculate the derivative. 
        p[0] = 1
        p[1] = 1*x 
        for i in range(2,n+1):
            p[i] = (((2*i - 1)* x * p[i-1]) - (i-1)*p[i-2])/i




        p_prime[0] = 0
        p_prime[1] = 1
        for i in range(2,n+1):
            p_prime[i] = sp.simplify(i*(x*p[i]-p[i-1])/(x**2 -1)) # From the Bonnet Formula

        out = sp.Poly(p_prime[-1], x)
        out = out.all_coeffs()
        
    return out


def lgl(n): 
    """
    This is a function that takes n as an argument. n is the polynomial order.
    It calculates the weights of the nth order Legendre-Gauss-Lobatto
    points (LGL) and the roots. Output = [roots_dp, weights] 
    """
    dp = dp_n_c(n) # generates the coefficients of the dp/dx polynomial. 
    roots_dp = np.array(np.roots(dp)) # solves for the roots of the dp/dx polynomial (the Absiccas)
    w = np.zeros(n+1)
    w[0] = 2/(n*(n+1))
    w[-1] = w[0]
    roots_dp = sorted(roots_dp)
    if n > 2: 
        for i in range(1,n):
            w[i] = w[0]*(1/(p_n(roots_dp[i-1],n))**2)
    out = np.zeros(n+1)
    out[0] = -1
    out[-1] = 1
    out[1:-1] = roots_dp
    ### Note: I did not include the jacobians in here!!!!
    return np.matrix([out,w])

def lagrange(n,x): 
    """
    This function creates a vector of lagrange polynomials that interpolate a unitary polynomial through the 
    Legendre-Gauss-Lobatto points. 
    n - order of the polynomial, which means n + 1 points. 
    The output will be a vector evaluated at the n LGL points. 
    """
    points = lgl(n) # uses the previous function to generate the roots and the weights
    roots = np.zeros(n+1)
    roots[:] = points[0,:]

    w = np.zeros(n+1)
    w[:] = points[1,:]

    l = np.ones(n+1) # initalization of the vector
    for i in range(n+1): 
        for j in range(n+1): 
            if j != i:
                l[i] = l[i]*(x-roots[j])/((roots[i]-roots[j]))
    return l


def dlagrange(n, *args): 
    """
    This function takes in the degree of the lagrange polynomial - n along with a collocation point - x. 
    The output is a matrix that satifies the SBP property for differentiation. 
    """
    points = lgl(n) # uses the previous function to generate the roots and the weights
    roots = np.zeros(n+1)
    roots[:] = points[0,:]


    if len(args) == 1: # This is to add a custom range of values (domain different than [-1,1]). Note: The negative number is because lgl(n) is defined from small to big. 
        jacob = args[0]
    else: 
        jacob = -1
    w = np.zeros(n+1)
    w[:] = points[1,:]
    dl = np.zeros((n+1,n+1))
    prod = 1 # Just to save the intermediate steps of multiplication 
    for i in range(n+1): # Note, since there are 2 exclusions (from the definition of the derivative of the lagrange polynomials)
        for k in range(n+1):
            if k != i: 
                prod = 1
                for j in range(n+1): 
                    if j!= i and j!= k: 
                        prod= prod*(roots[i]-roots[j])/((roots[k]-roots[j]))
                dl[i,k] = prod*(1/(roots[i] - roots[k]))
 
        
    for i in range(n+1):
        dl[i, i] = jacob*np.sum(dl[i, :])  # since off-diagonals sum to -D_ii
    return -1*dl ###The minus sign is for the 1D jacobian



def Vmonde(n, *args):
    """
    This function generates a vandermonde matrix of order nxn. This matrix uses the monomials as its basis, 
    and they are evaluated at the LGL points.

    Note: NOT YET IMPLIMENTED!!
    There are different basis implimented: 
        - *args = none, this function will use the default monomial basis. 
        - *args = p, will use the Legendre Polynomials
        - *args = l, will use the Lagrange Interpolating Polynomials
    """
    
    x = np.zeros(n)
    out = lgl(n-1)
    x[:] = out[0,:]
    V = np.zeros((n,n))


    #if args == "l":
    #    raise ValueError("The Vandermonde with Lagrange Basis has not been implimented yet")
    #if args == "p":
    #    raise ValueError("The Vandermonde with Legendre Polynomial Basis has not been implimented yet")
    #else:
    for j in range(n):
        V[j,:] = [x[j]**i for i in range(n)]
    return V

def sbp_d(n): 

    """
    Generates the differentiation matrix D of size n + 1 by n + 1. Where n is the degree of the solution
    """
    D = dlagrange(n)
    return D 


def sbp_p(n):
    out = lgl(n)
    roots = np.zeros(n+1)
    w = np.zeros(n+1)
    w[:] = out[1,:]
    roots[:] = out[0,:]
    result = np.zeros((n+1,n+1))
    l1 = np.zeros(n+1)
    for i in range(n+1):
        result = result + (np.outer(lagrange(n,roots[i]),lagrange(n,roots[i])))*w[i]
    P = -1*result # The negative one stems from the formulation of the code. The points were sorted from right to left, hence the jacobian is negative.
    return P 

###################################
#---------Element-----------------#
###################################
class Element: 
    def __init__(self, index, time , solution, left_bound, right_bound, top_bound, bottom_bound, lgl):
        """
        This class defines a single element. 
        - index: ID of the element 
        - time: Time step
        - solution: solution values 
        - bounds: the physical boundaries of the element
        - lgl: is the set of the lgl points that were used to define the SBP operators
        """
        
        
        self.index = index
        self.time = time
        self.solution = solution
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.top_bound = top_bound
        self.bottom_bound = bottom_bound
        self.lgl = lgl 
        
        
        
    def print_computational_points(self): 
        X, Y = np.meshgrid(self.lgl,self.lgl)
        return X, Y
     
    def print_box(self):
        t = np.array([[self.left_bound, self.right_bound],[self.top_bound,self.top_bound]])
        b = np.array([[self.left_bound, self.right_bound],[self.bottom_bound,self.bottom_bound]])
        l = np.array([ [self.left_bound, self.left_bound], [self.bottom_bound,self.top_bound]])
        r = np.array([[self.right_bound, self.right_bound], [self.bottom_bound,self.top_bound]])
        return plt.plot(t[0],t[1]),plt.plot(b[0],b[1]),  plt.plot(l[0],l[1]), plt.plot(r[0],r[1])
    
    def print_box_with_nodes(self):
        t = np.array([[self.left_bound, self.right_bound],[self.top_bound,self.top_bound]])
        b = np.array([[self.left_bound, self.right_bound],[self.bottom_bound,self.bottom_bound]])
        l = np.array([ [self.left_bound, self.left_bound], [self.bottom_bound,self.top_bound]])
        r = np.array([[self.right_bound, self.right_bound], [self.bottom_bound,self.top_bound]])
        x     = ((self.right_bound-self.left_bound)/2)*self.lgl + ((self.right_bound+self.left_bound)/2)*np.ones_like(self.lgl) 
        y     = ((self.top_bound-self.bottom_bound)/2)*self.lgl + ((self.top_bound+self.bottom_bound)/2)*np.ones_like(self.lgl) 
        X, Y  = np.meshgrid(x, y)
        return plt.plot(t[0],t[1]),plt.plot(b[0],b[1]),  plt.plot(l[0],l[1]), plt.plot(r[0],r[1]), plt.plot(X, Y, "+")
        
        
            
    def print_nodes_physical(self):
        x     = ((self.right_bound-self.left_bound)/2)*self.lgl + ((self.right_bound+self.left_bound)/2)*np.ones_like(self.lgl) 
        y     = ((self.top_bound-self.bottom_bound)/2)*self.lgl + ((self.top_bound+self.bottom_bound)/2)*np.ones_like(self.lgl) 
        X, Y  = np.meshgrid(x, y)
        return X, Y
    
    def Jacob(self): 
        A_physical = (self.right_bound - self.left_bound)*(self.top_bound - self.bottom_bound)
        S_x = (self.right_bound - self.left_bound)/2
        S_y = (self.top_bound - self.bottom_bound)/2  
        inv_jacobian = (S_x*S_y)**(-1)
        return inv_jacobian, A_physical
    

###################################
#---------Element-----------------#
###################################

###################################
#---------Mesh--------------------#
###################################
### Creating the class for the mesh element

class Mesh:
    def __init__(self, x_min, x_max,  y_max, y_min , nex, ney, n):
        lgl = np.zeros(n+1)
        lgl[:]= sb.lgl(n)[0,:]
        self.lgl     = lgl
        self.x_min   = x_min
        self.x_max   = x_max
        self.y_min   = y_min
        self.y_max   = y_max
        self.nex     = nex
        self.ney     = ney
        self.elements = [[None for _ in range(ney)] for _ in range(nex)]  # 2D list of elements
        #self.generate_mesh()
        self.t = 0
        
        
    def generate_mesh(self):
        """
        Creates elements and assigns them to the mesh.
        """
        self.solution = None
        dx = (self.x_max - self.x_min) / self.nex
        dy = (self.y_max - self.y_min) / self.ney
        index = 0 # For intializing element IDs 
        
        for i in range(self.nex):
            for j in range(self.ney):
                left_bound = self.x_min + i * dx
                right_bound = self.x_min + (i+1) * dx
                bottom_bound = self.y_min + j * dy
                top_bound = self.y_min + (j+1) * dy

                element = Element( index, self.t , self.solution, left_bound, right_bound, top_bound, bottom_bound, lgl)
                self.elements[i][j] = element
                index += 1

    def get_element(self, i, j):
        """
        Retrieves an element by its (i, j) position in the grid.
        """
        return self.elements[i][j]
    def print_all_elements(self): 
        k = []
        plt.figure(figsize=(6,6))  # Create a figure
        for i in range(self.nex):
            for j in range(self.ney): 
                k = self.elements[i][j]
                k.print_box_with_nodes()
                
                
    #def set_solution_at_time(self,solution):
        
    