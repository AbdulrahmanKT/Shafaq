

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
    """
    Return the n-th order Legendre–Gauss–Lobatto nodes (xi) and weights (w).
    xi: array shape (n+1,)
    w : array shape (n+1,)
    """
    # 1) find interior roots of P_n'
    dp       = dp_n_c(n)               # sympy coeffs of P_n'
    roots_dp = np.sort(np.roots(dp))   # length n-1

    # 2) assemble full xi
    xi = np.empty(n+1)
    xi[0]   = -1.0
    xi[-1]  =  1.0
    if n > 1:
        xi[1:-1] = roots_dp

    # 3) evaluate P_n at all xi (scalar p_n → list comprehension)
    Pn_xi = np.array([p_n(xi_i, n) for xi_i in xi])

    # 4) compute weights in one vectorized expression
    w = 2.0 / (n*(n+1) * (Pn_xi**2))
    out = xi
    return np.array([out,w])

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


def dlagrange(n, tol=1e-12): 
    """
    This function takes in the degree of the lagrange polynomial - n along with a collocation point - x. 
    The output is a matrix that satifies the SBP property for differentiation. 
    """
    #####Note: This function is coded to match the logarithmic derivative form of the lagrange polynomial. A more efficient implementation is the 
    ##### Baycentric form. 
    
    points = lgl(n) # uses the previous function to generate the roots and the weights
    roots = np.zeros(n+1)
    roots[:] = points[0,:]


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
                dl[i,k] = prod*(1/(-roots[i] + roots[k]))
    
    # a boolean mask that’s True for all off-diagonal positions
    mask_offdiag = ~np.eye(n+1, dtype=bool)

    # a boolean mask of entries whose magnitude is below the tolerance
    small = np.abs(dl) < tol

    # combine them: spots that are both off-diagonal AND tiny
    dl[mask_offdiag & small] = 0.0

        
    for i in range(n+1):
        dl[i, i] = -1*np.sum(dl[i, :])  # since off-diagonals sum to -D_ii

    return dl ###The minus sign is for the definition of the points



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


#def sbp_p(n):
    #out = lgl(n)
    #roots = np.zeros(n+1)
    #w = np.zeros(n+1)
    #w[:] = out[1,:]
    #roots[:] = out[0,:]
    #result = np.zeros((n+1,n+1))
    #l1 = np.zeros(n+1)
    #for i in range(n+1):
    #    result = result + (np.outer(lagrange(n,roots[i]),lagrange(n,roots[i])))*w[i]
    #P = result # The negative one stems from the formulation of the code. The points were sorted from right to left, hence the jacobian is negative.
    #return P 
    
def sbp_p(n):
    out = lgl(n)
    roots = np.zeros(n+1)
    w = np.zeros(n+1)
    w[:] = out[1,:]
    roots[:] = out[0,:]
    result = np.zeros((n+1,n+1))
    P = np.diag(w)
    return P 


def sbp_q(n): 
    out = lgl(n)
    roots = np.zeros(n+1)
    w = np.zeros(n+1)
    w[:] = out[1,:]
    roots[:] = out[0,:]
    dq = sbp_d(n)
    result1 = np.zeros((n+1,n+1))
    result1 = w[:,None]*dlagrange(n)   # Q = PD - using the numpy broadcast
    return result1 

def two_point_flux_function(n ,D, u): 
    ## Using numpy broadcasting is much faster than loops 
    #n = len(u) - 1
    ui = u[:,None]
    uj = u[None,:]
    F = np.zeros((n+1,n+1)) 
    F = (1/6)*(ui**2 + ui*uj + uj**2)
    
    return -2*(D * F).sum(axis=1)





