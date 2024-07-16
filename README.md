# Catalan_MC

A Python-based project focused on generating Worm-Like Chain (WLC) model trajectories for the study of polymer physics, specifically in the context of the lattice Monte Carlo simulations.
![Catalan_MC](https://socialify.git.ci/ch-tung/Catalan_MC/image?description=1&font=Inter&language=1&name=1&owner=1&pattern=Solid&theme=Light)

## Features

- Generation of semiflexible WLC chain trajectories using lattice MC or the continuous Kratky-Porod model.
- Calculation of radially averaged S(Q), 2D spectrum, and RSHE of S(Q).

<img src="https://github.com/ch-tung/Catalan_MC/blob/main/chain.png?raw=true" width="640">
<img src="https://github.com/ch-tung/Catalan_MC/blob/main/sq.png?raw=true" width="640">
<img src="https://github.com/ch-tung/Catalan_MC/blob/main/sq2D.png?raw=true" width="320">

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.x
- NumPy
- Matplotlib
- SciPy

## Usage
### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/ch-tung/Catalan_MC.git
```

### Generate polymer chain trajectory

```python
import numpy as np
from WLM import WLChain

# Chain parameters
# Coordinate of C atoms in each unit
unit_C = np.zeros((3,1))

# Degree of polymerization
N_backbone = int(input_params[2])

# Chain stiffness (placeholder)
a_backbone = 1

# Unit persistence
lambda_backbone = 1

# Affine deformation matrix
F = np.array([[1,0,0],[0,1,0],[0,0,1]])

# Call WLChain class
chain = WLChain(N_backbone,a_backbone,lambda_backbone,unit_C)
chain.apply_SA = 1
chain.d_exc = 0.1
chain.f = 0.0

# Q points
n_q = 101
qq = (np.logspace(-1,3,n_q))/N_backbone

# Chain stiffness
chain.kappa = kappa

# Stretching strength
chain.epsilon = epsilon

# Specify grid for lattice MC, available grids in ["SC","RB","RT"]
chain.grid = grid

# Run MC simulation and get trajectory
chain.chain_grid()
chain.affine()
```
Now the beads trajectories can be accessed via `chain.Cc`, formatted in an n by 3 numpy array.

### Calculate scattering function

After doing `chain.chain_grid()` or `chain.chain()`
```python
# Q points
n_q = 101
qq = (np.logspace(-1,3,n_q))/N_backbone
qq_2D = np.concatenate((-np.flip(qq), np.array([0.0]), qq))

# radially averaged S(Q)
chain.scatter_direct(qq,n_merge=n_merge)
S_q = chain.S_q 

# 2D spectrum along velocity gradient–vorticity(y–z), flow–vorticity (x–z), and flow–velocity gradient (x–y) planes
chain.scatter_direct_aniso(qq,n_merge=n_merge)
S_q_2D = chain.S_q_2D

# RSHE of S(Q)
lm=[(0,0),(2,0),(4,0),(6,0)]
chain.scatter_direct_SHE(qq, rr, lm, n_merge=n_merge, calculate_g_r=calculate_g_r, real=real)
S_q_lm = chain.S_q_lm 
g_r_lm = chain.g_r_lm
```
`n_merge` can be used to merge the concatnated beads to accelerate the calculation.

