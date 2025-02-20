import SBP as sb 
import matplotlib.pyplot as plt 
import numpy as np 
import sys
np.set_printoptions(threshold=sys.maxsize)
### Up to 9 significant digits!!!!!!

#Matrix P
n = 5
out = sb.lgl(5)
roots = np.zeros(n+1)
w = np.zeros(n+1)
w[:] = out[1,:]
roots[:] = out[0,:]
result = np.zeros((n+1,n+1))
l1 = np.zeros(n+1)
for i in range(n+1):
    result = result + np.outer(sb.lagrange(n,roots[i]),sb.lagrange(n,roots[i]))

result = result * w
#print(result)


#Matrix Q from P
dq = sb.dlagrange(n)
result1 = np.zeros_like(result)
for i in range(n+1):
    result1[:,i] = sb.lagrange(n,roots[i]) @ np.transpose(sb.dlagrange(n))
result1 = result1 * w
print(result1) 


dpq = np.linalg.inv(result)@ result1



#Q Another way
q = result @ sb.dlagrange(n)
print(q)

#Matrix D
#D = sb.dlagrange(n) 
#print("\n", D)

