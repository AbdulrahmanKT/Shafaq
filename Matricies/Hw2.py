import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
#matplotlib.use('Qt5Agg')             #Use this on Linux 
import SBP as sb 

###########################################
"""
In this script, the SBP operators will be used to numerically differentiate multiple analytic functions. 
"""



### Setting up the colocation points 
n = 4 # poly nomial order
x = np.zeros(n+1) # Initialization
x[:] = sb.lgl(n)[0,:]

### Setting up analytic functions 
y_1 = np.ones_like(x)
print(np.shape(y_1))
# Defining a linear functions
m = 1.5
y_2 = [m*x[i] for i in range(n+1)]
y_3 = [x[i]**2 for i in range(n+1)]



### Numerical Differentiation 
D = sb.sbp_d(n)
print(D)
#print([np.sum(D[i,:]) for i in range(n+1)])
print(y_2 , "/n", D@y_2)
# 
# 
# P 
y_1_d = D@y_1
y_2_d = D@y_2
# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))  # Create figure with specified size

# Plot the data
ax.plot(x, y_1, 'b-', linewidth=2, label="$y_1$ = 1")
ax.plot(x, y_2, 'r-', linewidth=2, label="$y_2$ =  $mx$")
ax.plot(x, y_3, 'g-', linewidth=2, label="$y_3= x^2$")
ax.plot(x, y_1_d, 'b*', linewidth=2, label="$Dy_1$")
ax.plot(x, y_2_d, 'r*', linewidth=2, label="$Dy_2$")
ax.plot(x, D@y_3, 'g*', linewidth=2, label="$Dy_3$")
# Set plot labels and title
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.set_title('Plot of analytical function and numerical derivatives', fontsize=14)

# Add grid, legend and set axis limits
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=10)
ax.set_xlim([-1.01, 1.01])
ax.set_ylim([-2.2, 2.2])
plt.show()