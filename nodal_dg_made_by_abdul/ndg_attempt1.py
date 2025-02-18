import SBP as sb 
import matplotlib.pyplot as plt 
import numpy as np 
import sys
np.set_printoptions(threshold=sys.maxsize)
### Up to 9 significant digits!!!!!!



### Plotting the Legendre Polynomials
#sb.legplot(sb.p_n, 5, [-0.999, 0.999], "legendre polynomial")
#sb.legplot(sb.dp_n, 5, [-0.999, 0.999], "Derivative of Legendre polynomial")

#test = sb.p_n_c(4)
#x = np.linspace(-1, 1, 500)
#
#
#plt.plot(x, sb.p_n(x,n))
a = sb.p_n_c(4)
#print(a)
print(np.roots(a))
dp = sb.dp_n_c(4)
roots_dp = np.array(np.roots(dp))
print(roots_dp)


## In order to generate the weights, the following 


#def weights(n): 
#    dp = sb.dp_n_c(n)
#    roots_dp = np.array(np.roots(dp))
#    w = np.zeros(n+1)
#    w[0] = 2/(n*(n+1))
#    w[-1] = w[0]
#
#    if n > 2: 
#        for i in range(1,-2):
#            w[i] = w[0]*(1/(p_n())**2)