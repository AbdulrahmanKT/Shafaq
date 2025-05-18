#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from SBP.legendre import * 
from SBP.mesh_1d import * 


# --- 1) Problem parameters ---
Lx      = 8           # domain length
nex     = 100              # number of elements
poly_p  = 6               # polynomial degree (n)
t_final = 1.2            # final time
dt      = 1e-2
# --- 2) Build SBP operators on reference ---
n     = poly_p
xi, w = lgl(n)
D_ref = sbp_d(n)
P_ref = sbp_p(n)
Q_ref = sbp_q(n)

# --- 3) Create mesh and set initial condition u(x,0) = sin(2πx/Lx) + 1 ---
mesh = Mesh1D(x_min=0.0,
              x_max=Lx,
              nex=nex,
              n=n,
              xi=xi,
              w=w,
              D_ref=D_ref,
              P_ref=P_ref,
              Q_ref=Q_ref, nu = 1e-3)

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
    return np.zeros_like(x)
ic = constant(mesh.x())
mesh.set_initial_condition(np.sin)
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
ax.set_ylim(-1, 2.0)
ax.grid(True)
plt.tight_layout()


# --- 4) Determine stable dt via max wave-speed ---
dx_elem = Lx / nex
u_max   = np.max([np.max(np.abs(elem.u)) for elem in mesh.elements])

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
    for i, elem in enumerate(mesh.elements):
        ax.plot(elem.x, elem.u, label="Solution")
        #ax.plot(elem.x, elem.irhs, "+-", label="Interior RHS")
        #ax.plot(elem.x, elem.sat_rhs, "*", label="SAT RHS")
        #ax.plot(elem.x, elem.rhs, ".", label="Total RHS")
    #ax.plot( mesh.total_energy(), label="Total Energy")
    ax.set_title(f"Solution to the Viscous Burgers Equation using {nex*n} DOF")
    ax.set_ylim(-2, 2.0)
    ax.grid(True)
    times.append(t)
    E = mesh.total_energy_normalized()
    energies.append(E)
    print("===================================================")
    print(f"t = {t:.4f},  E = {E:.6e}")
    print("===================================================")
    plt.pause(1/100)
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
