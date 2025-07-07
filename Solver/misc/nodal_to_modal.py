import numpy as np 
import matplotlib.pyplot as plt 
import SBP as sb

n = 5 # Solution Order 
a = 1 # Reduction in the polynomial order
o = n - a # The lower order of the solution
V = sb.Vmonde(n) # Vandermonde Matrix 
Vinv = np.linalg.inv(V) 
P = np.zeros((n,n))
P[0:o,0:o] = np.diag(np.ones(o))



u_n = np.ones(n) # Place holder for solution vector 
u_o = V @ P @ (Vinv @ u_n)
print(u_o)