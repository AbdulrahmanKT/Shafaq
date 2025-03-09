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
    result1[:,i] = sb.lagrange(n,roots[i]) @ np.transpose(dq)
result1 = result1 @ w
print("The matrix Q from P \n", result1) 

# Matrix D 
D = sb.dlagrange(n)


#### Testing Differentiation and integration 
m = -1 # Slope of test 
y = np.array([m*roots[i] for i in range(n+1)])
dy = D@y 
y_d = m*np.ones_like(y)
yp = P@(dy)
fig , ax = plt.subplots(figsize=(10,6))
ax.plot(roots, y, label="y")
ax.plot(roots, dy, label="dy")
#ax.plot(roots, y_d, label='y_d')
ax.plot(roots, yp, label="yp")
# Add grid, legend and set axis limits
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=10)
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-10, 10])

# Improve overall appearance
#plt.tight_layout()
plt.show()

dpq = np.linalg.inv(P)@result1
error = np.transpose(D - dpq)@(D - dpq)
plt.figure
plt.matshow(error)

plt.show()
