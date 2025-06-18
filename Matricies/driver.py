#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from SBP.legendre import * 
from SBP.mesh_1d import * 
from SBP.Equations import *
import SBP.Shock # For using the shock capturing


# --- 1) Problem parameters ---
Lx      = 5           # domain length
nex     = 11           # number of elements
poly_p  = 9              # polynomial degree (n)
t_final = 1          # final time
dt      = 1e-4
plot_every = 100
# --- 2) Build SBP operators on reference ---
n     = poly_p
xi, w = lgl(n)
D_ref = sbp_d(n)
P_ref = sbp_p(n)
Q_ref = sbp_q(n)


# Option A: Viscous Burgers
#  base_nu = 1e-3     # constant viscosity
#  eq = BurgersEquation(base_nu=base_nu, sensor_fn=None)

# Option B: Linear advection + constant viscosity (u_t + a u_x = ν u_xx)
a    = 1
nu   = 0.0     # viscosity
v_off = 1      # turn viscous SAT on/off (1→on, 0→off)
#eq   = Advection(a=a, nu=nu, v_off=v_off)
eq   = Burger(c_off=1, nu=nu, v_off=v_off)




# --- 3) Create mesh and set initial condition u(x,0) = sin(2πx/Lx) + 1 ---
mesh = Mesh1D(x_min=0.0,
              x_max=Lx,
              nex=nex,
              n=n,
              xi=xi,
              w=w,
              D_ref=D_ref,
              P_ref=P_ref,
              Q_ref=Q_ref, equation=eq, shock_capture=False)

######---------------#######
# Initial Condition 
x = mesh.x()
x_all  = mesh.x().flatten()
xmin, xmax = x_all.min(), x_all.max()

# Precompute your target interval [x0, x1] in global space
x0 = xmin + 0.1*(xmax - xmin)
x1 = xmin + 0.40*(xmax - xmin)

# Define ivbc to mask on those fixed values
def ivbc(x):
    """
    Zero everywhere except on the global [x0,x1], 
    where u = sin(x).
    """
    u = 0.1*np.ones_like(x)
    mask = (x >= x0) & (x <= x1)
    u[mask] = np.sin(x[mask])
    return u

def constant(x): 
    return np.ones_like(x)

def gaussian(x, mu=3, sigma=0.1):
    return 0.8*np.exp(-((x - mu)**2) / (2 * sigma**2))


mesh.set_initial_condition(gaussian)
mesh.rhs()
fig, ax = plt.subplots(figsize=(20,5))
mesh.plot()
for i, elem in enumerate(mesh.elements):
    ax.plot(elem.x, elem.u, label="Solution")
    #ax.plot(elem.x, elem.irhs, "+-", label="Interior RHS")
    #ax.plot(elem.x, elem.sat_rhs, "*", label="SAT RHS")
    #ax.plot(elem.x, elem.rhs, "o", label="Total RHS")
    ax.grid(True)
ax.set_title(" Initial Condition")
ax.set_ylim(-0.01, 0.5)
ax.grid(True)
plt.tight_layout()


# --- 4) Determine stable dt via max wave-speed ---
dx_elem = Lx / nex
#u_max   = np.max([np.max(np.abs(elem.u)) for elem in mesh.elements])

# --- 5) Time‐stepping loop using RK4 ---
t = 0.0
it = 0
#snapshots = [mesh]  # store initial snapshot if needed
fig, ax = plt.subplots(figsize=(20,5))
mesh.plot(ax=ax)
times = []
energies = []

while t < t_final:
    if t + dt > t_final:  # final short step
        dt = t_final - t

    mesh.step_rk4(dt)
    t += dt
    it += 1
    ax.clear()
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_ylim(-0.5, 1)
    
    if it % plot_every == 0:
        for i, elem in enumerate(mesh.elements):
            ax.plot(elem.x, elem.u, label="Solution")
            #ax.plot(elem.x, 1000*elem.av_eps*np.ones_like(elem.x),"*", label="AV")
            #ax.plot(elem.x, elem.S*np.ones_like(elem.x), label="Sensor")
            #ax.plot(elem.x, elem.irhs, "+-", label="Interior RHS")
            #ax.plot(elem.x, elem.sat_rhs, "*", label="SAT RHS")
            #ax.plot(elem.x, elem.rhs, ".", label="Total RHS")
            ax.set_title(f"Solution to the Viscous Burgers Equation using {nex*n} DOF at Time {t:.3f}")

        plt.pause(1/10000)
        plt.tight_layout()

        #plt.savefig("Final_Step.png")
    #ax.plot( mesh.total_energy(), label="Total Energy")
    
    
    times.append(t)
    E = mesh.total_energy_normalized()

    energies.append(E)
    print("===================================================")
    print(f"t = {t:.4f},  E = {E:.6e}")
    print(f"Max AV = {np.max(mesh.print_av())}")
    print("===================================================")
   
#ax.legend()
plt.tight_layout()
plt.show()

print(f"Driven to t={t:.3f} in {it} steps, final dt={dt:.3e}")

# --- 6) Plot final solution ---
plt.plot(times,energies)
plt.title("Energy of the System vs Time")
plt.xlabel("Time")
plt.ylabel("Total Energy of the System")
plt.grid(True)
plt.show()

