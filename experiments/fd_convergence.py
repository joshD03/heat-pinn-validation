#!/usr/bin/env python
"""FD spatial-convergence study.

Runs the explicit FD solver at several grid spacings, computes L2 error
against the analytical solution, and saves a log-log convergence plot.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from heat_fd import HeatSolverFD


def main():
    dx_vals = np.array([0.05, 0.025, 0.0125, 0.00625])
    Nx_vals = np.array([20, 40, 80, 160])
    l2_errors = []

    for dx, Nx in zip(dx_vals, Nx_vals):
        # Nt=50 baseline; increase if CFL would be violated
        dx_actual = 1.0 / (Nx - 1)
        Nt_min = int(np.ceil(0.5 * 0.01 / (0.45 * dx_actual ** 2)))
        Nt = max(50, Nt_min)
        solver = HeatSolverFD(Nx=Nx, Nt=Nt, alpha=0.01)
        u_fd = solver.solve()
        u_exact = solver.analytical()
        err = np.linalg.norm(u_fd - u_exact) / np.linalg.norm(u_exact)
        l2_errors.append(err)
        print(f"dx={dx:.5f}  Nx={Nx:4d}  Nt={Nt:5d}  r={solver.r:.4f}  L2={err:.4e}")
    errors = l2_errors

    # --- log-log plot ---
    out = pathlib.Path(__file__).resolve().parent.parent / "plots"
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(dx_vals, errors, "o-", lw=2, markersize=7, label="FD L2 error")

    # Reference O(dx²) slope
    c = errors[0] / dx_vals[0] ** 2
    dx_ref = np.array(dx_vals)
    ax.loglog(dx_ref, c * dx_ref ** 2, "k--", alpha=0.5, label=r"$O(\Delta x^{2})$")

    ax.set_xlabel("Δx")
    ax.set_ylabel("L2 error")
    ax.set_title("FD Convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    fname = out / "fd_convergence.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"\nSaved {fname}")


if __name__ == "__main__":
    main()
