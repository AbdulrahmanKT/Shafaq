

import numpy as np 
import matplotlib.pyplot as plt 


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
    
    plt.show
    plt.legend()
    plt.grid()
    plt.title("Plot of the polynomials up to n")



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