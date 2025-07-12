import SBP as sb 
import matplotlib.pyplot as plt 
import matplotlib
#matplotlib.use('Qt5Agg')
import numpy as np 
import sys
np.set_printoptions(threshold=sys.maxsize)
### Up to 9 significant digits!!!!!!

#Matrix P
n = 4
out = sb.lgl(n)
roots = np.zeros(n+1)
w = np.zeros(n+1)
w[:] = out[1,:]
roots[:] = out[0,:]
result = np.zeros((n+1,n+1))
l1 = np.zeros(n+1)
for i in range(n+1):
    result = result + (np.outer(sb.lagrange(n,roots[i]),sb.lagrange(n,roots[i])))*w[i]
P = -1*result # The negative one stems from the formulation of the code. The points were sorted from right to left, hence the jacobian is negative. 
print("The matrix P \n", result)


#Matrix Q from P
dq = sb.dlagrange(n)
result1 = np.zeros_like(result)
for i in range(n+1):
    result1[:,i] = sb.lagrange(n,roots[i]) @ dq* w[i]
result1 = result1  
Q = result1
print("The matrix Q from P \n", Q) 

# Matrix D 
D = sb.dlagrange(n)
print("The matrix D \n", D) 

