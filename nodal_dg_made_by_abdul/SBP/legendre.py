

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

