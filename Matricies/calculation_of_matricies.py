import SBP as sb 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np 
import sys
np.set_printoptions(threshold=sys.maxsize)
### Up to 9 significant digits!!!!!!

#Matrix P
n = 5
out = sb.lgl(n)
roots = np.zeros(n+1)
w = np.zeros(n+1)
w[:] = out[1,:]
roots[:] = out[0,:]
result = np.zeros((n+1,n+1))
l1 = np.zeros(n+1)
for i in range(n+1):
    result = result + w*(np.outer(sb.lagrange(n,roots[i]),sb.lagrange(n,roots[i])))


P = result
print("The matrix P \n", result)


#Matrix Q from P
dq = sb.dlagrange(n)
result1 = np.zeros_like(result)
for i in range(n+1):
    result1[:,i] = sb.lagrange(n,roots[i]) @ np.transpose(sb.dlagrange(n))
result1 = result1 * w
print("The matrix Q from P \n", result1) 

# Matrix D 
D = sb.dlagrange(n)


#### Testing Differentiation and integration 
m = 2 # Slope of test 
y = np.array([m*roots[i] for i in range(n+1)])
dy = D@y 
yp = P@y 

fig , ax = plt.subplots(figsize=(10,6))
ax.plot(roots, y, label="y")
ax.plot(roots, dy, label="dy")
ax.plot(roots, yp, label="yp")
# Add grid, legend and set axis limits
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=10)
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-10, 10])

# Improve overall appearance
#plt.tight_layout()
plt.show()


#error = D - dpq
#print(error)
print(w)