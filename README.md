# Heat PINN Validation

Building a complete physics-informed deep learning pipeline from scratch: benchmarking a neural network solver against a classical finite difference scheme to see where data-driven physics breaks down.

***

![comparison](plots/comparison.gif)  
*What the neural network learned versus the classical ground truth. The PINN reached 0.5% relative error, but struggled with boundary stiffness.*

***

## Summary

* **Goal:** validate a Physics-Informed Neural Network (PINN) against a trusted numerical solver  
* **Pipeline:** PyTorch MLP, automatic differentiation, explicit finite difference (FD) scheme  
* **Result:** 0.5% relative L2 error (0.005 absolute) vs. FD ground truth  
* **Main finding:** PINNs capture the global physics well but "cheat" at the boundaries, requiring careful weighting of loss terms  
* **Time spent:** 1 week (solver implementation, training loop, error analysis)

Built independently to understand the practical trade-offs between classical numerics and deep learning solvers.

***

## Motivation

I wanted to work through the full process of training a PINN, not just on a toy problem where I knew the analytical solution, but against a numerical solver I built myself. This project implements a 1D Heat Equation solver using both a classical Finite Difference (FD) scheme and a deep neural network, then pits them against each other.

The core question was simple: can a neural network learn calculus as well as a discrete grid?

I had done standard deep learning (classification/regression), but not physics-informed learning. I wanted to see what changes when you deal with:

* Loss functions that fight each other (PDE residual vs. Boundary Conditions)
* Spectral bias (networks learning low frequencies first)
* Evaluation metrics that go beyond "accuracy" to "physical consistency"

I was also curious whether the PINN would fail in interpretable ways.

***

## Results at a glance

* Relative L2 Error: **0.5%**
* Max Absolute Error: **0.015** (concentrated at $t=0$ and boundaries)
* Training Time: ~5000 epochs to convergence
* FD Convergence: Confirmed $O(\Delta x^2)$ second-order accuracy

Main takeaway: The PINN is incredibly sample-efficient but struggles with "stiff" gradients at the boundaries, whereas the FD solver is robust but grid-dependent.

***

## Methods

### 1. The Classical Solver (Finite Difference)

I implemented an explicit forward-time, centered-space (FTCS) difference scheme.

* **Grid:** 100 spatial points, 10,000 time steps
* **Stability:** Verified the CFL condition ($\alpha \Delta t / \Delta x^2 \leq 0.5$) to prevent numerical explosion.

### 2. The Neural Solver (PINN)

I built a fully connected network (MLP) with 3 hidden layers of 32 neurons.

* **Inputs:** $(x, t)$ coordinates
* **Outputs:** Temperature $u(x,t)$
* **Physics Loss:** Automatic differentiation calculates $\frac{\partial u}{\partial t}$ and $\frac{\partial^2 u}{\partial x^2}$ directly from the computation graph, enforcing the heat equation $u_t - \alpha u_{xx} = 0$ without a grid.

***

## Convergence Analysis

### Finite Difference Verification

I ran a convergence study to ensure my "ground truth" was actually true. By halving the grid spacing $\Delta x$ repeatedly, I confirmed the error dropped quadratically, proving the solver is second-order accurate.

![fd_convergence](plots/fd_convergence.png)  
*Spatial convergence of the FD solver. The slope confirms O(Δx²) behavior.*

### PINN Training Dynamics

The network minimizes a composite loss: `Loss = MSE_pde + MSE_bc + MSE_ic`.

![pinn_loss](plots/pinn_loss.png)  
*Loss landscape during training. The initial drop is rapid, but the "long tail" of convergence shows the network struggling to refine the solution at the boundaries.*

***

## Discussion

### Why the boundaries are hard

My initial expectation was that the PINN would fail in the center of the domain where dynamics are fastest. In practice, the error heatmap reveals the opposite: the largest errors (red/yellow zones) are pinned to the boundaries ($x=0, x=1$).

![error_heatmap](plots/error_heatmap.png)

This is a known pathology in PINNs; gradient pathologies makes it hard for the network to balance "satisfying the PDE" with "satisfying the boundary." The network effectively "cheats" by smoothing out the sharp gradients at the edges to lower the global residual.

### The trade-off

* **FD Solver:** Fast, exact (up to discretization), but requires a mesh.
* **PINN:** Mesh-free, differentiable, but harder to train and less accurate at the edges.

This project highlighted that while PINNs are powerful for inverse problems or irregular geometries, for a simple forward problem on a square domain, classical methods are still superior in speed and precision.

***

## Limitations and future work

Things I would improve first:

* Implement **Hard Constraint** enforcement (forcing the network architecture to satisfy BCs by design, rather than soft penalties).
* Add **Self-Adaptive Weights** to balance the PDE loss and Boundary loss dynamically.
* Test on a more complex, non-linear PDE (like Burgers' equation) where FD schemes often require expensive stabilization.

***

## Usage

```bash
pip install -r requirements.txt
python experiments/compare_methods.py
```

## Run individual experiments

```bash
python experiments/fd_convergence.py    # FD convergence plot
python experiments/pinn_training.py     # Train PINN, save model + loss curve
python compare_methods.py   # Side-by-side comparison
```

## Project structure

```
heat-pinn-validation/
├── README.md
├── requirements.txt
├── heat_fd.py              # Explicit FD solver
├── heat_pinn.py            # PyTorch PINN
├── validate.py             # Metrics + plotting utilities
├── experiments/
│   ├── fd_convergence.py   # dx convergence study
│   ├── pinn_training.py    # Train + save model
│   └── compare_methods.py  # Full comparison pipeline
├── notebooks/
│   └── 01_demo.ipynb       # Interactive walkthrough
├── plots/                  # Generated outputs
└── .gitignore
```
