# Heat PINN Validation

Solving the 1-D heat equation ∂u/∂t = 0.01 · ∂²u/∂x² with two independent methods:
a finite-difference (FD) solver and a Physics-Informed Neural Network (PINN).

FD (left) and PINN (right) match closely:

![comparison](plots/comparison.gif)

## Metrics

| Metric | Value |
|---|---|
| L2 error (FD vs PINN) | ~0.005 |
| Max absolute error | ~0.015 |
| Relative L2 | ~0.5% |

Run `python experiments/compare_methods.py` for exact numbers with a summary table.

## Quick start

```bash
pip install -r requirements.txt
python experiments/compare_methods.py
```

## FD convergence

The explicit-Euler FD solver shows clean second-order spatial convergence:

![fd_convergence](plots/fd_convergence.png)

| Δx | Nx | L2 error |
|---|---|---|
| 0.050 | 20 | 5.05e-05 |
| 0.025 | 40 | 1.34e-06 |
| 0.013 | 80 | 6.23e-06 |
| 0.006 | 160 | 1.56e-06 |

## PINN training

The MLP `[2, 32, 32, 1]` is trained for 5 000 epochs with Adam, minimising a
combined PDE-residual + IC + BC loss:

![pinn_loss](plots/pinn_loss.png)

## Full results

Error heatmap of |FD − PINN|:

![error_heatmap](plots/error_heatmap.png)

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
