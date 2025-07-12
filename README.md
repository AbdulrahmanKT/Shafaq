# Shafaq
*A Python CFD solver inspired by the pink-hued dusk (“الشفق”) over KAUST.*

---

## About

**Shafaq** is a research-grade, high-order Computational Fluid Dynamics (CFD) solver written in Python.  
Its design centers on **entropy-stable Discontinuous Galerkin (DG)** methods for conservation laws, targeting both academic exploration.

> *“Just as dusk blends day and night, Shafaq blends numerical rigor with ease of experimentation.”*

---

## Key Features

| Capability | Details |
|------------|---------|
| **High-order DG** | Nodal DG with modal/nodal basis switch, arbitrary polynomial order. |
| **Entropy stability** | SBP–SAT framework, Roe–Ismail & Chandrashekar fluxes, entropy-stable viscous terms. |
| **Shock capturing** | Persson/Tuten sensor + artificial viscosity. |
| **Modular physics** | Compressible Euler ↔ Navier–Stokes toggle, Sutherland viscosity, γ-law or tabulated EOS. |
| **HPC ready** | MPI via `mpi4py`, GPU kernels via CuPy/Numba (optional). |
| **Pythonic UX** | NumPy first, no external build step; Jupyter-friendly. |

---

## Installation

```bash
# Clone
git clone https://github.com/AbdulrahmanKT/shafaq.git
cd shafaq

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt      # numpy, scipy, matplotlib …

