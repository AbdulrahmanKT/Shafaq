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
print("The matrix P \n", result)


#Matrix Q from P
dq = sb.dlagrange(n)
result1 = np.zeros_like(result)
for i in range(n+1):
    result1[:,i] = sb.lagrange(n,roots[i]) @ np.transpose(sb.dlagrange(n))
result1 = result1 * w
print("The matrix Q from P \n", result1) 


dpq = np.linalg.inv(result)@ result1
print("D from P and Q \n ", dpq)


#Q Another way
q = result @ sb.dlagrange(n)
print("Q from D \n", q)

#Matrix D
D = sb.dlagrange(n) 
print("D first \n", D)


##########################
##########################
#######Validation#########
x = np.zeros(n+1) # Initialization
x[:] = sb.lgl(n)[0,:]
y_1 = 5*np.ones_like(x)
y_1_d = D@y_1
y_1_dpq = dpq@y_1
print(y_1_d, y_1_dpq)
# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))  # Create figure with specified size
ax.plot(x, y_1, 'b-', linewidth=2, label="$y_1$ = 1")
ax.plot(x, y_1_d, 'r.',linewidth=2, label="y1_d")
ax.plot(x, y_1_dpq, 'g*', linewidth=2, label="y1dpq")


# Set plot labels and title
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.set_title('Plot of y = 1', fontsize=14)

# Add grid, legend and set axis limits
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=10)
ax.set_xlim([-2, 2])
ax.set_ylim([-10, 10])
print(D)
# Improve overall appearance
#plt.tight_layout()
plt.show()
## difference 
error = D - dpq
print(error)