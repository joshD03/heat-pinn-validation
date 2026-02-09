#!/usr/bin/env python
"""Compare FD and PINN solutions.

If no saved model exists, trains the PINN first.
Produces: comparison.gif, error_heatmap.png, and prints L2/max errors.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from heat_fd import HeatSolverFD
from heat_pinn import HeatPINN, train_pinn, evaluate_grid
from validate import l2_error, max_error, summary_table, plot_error_heatmap, make_comparison_gif


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    plots = root / "plots"
    plots.mkdir(exist_ok=True)

    # --- FD solution ---
    fd = HeatSolverFD(Nx=100, Nt=5000, alpha=0.01)
    u_fd = fd.solve()

    # --- PINN solution ---
    model = HeatPINN()
    model_path = root / "model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded saved PINN model.")
    else:
        print("No saved model found — training PINN ...")
        train_pinn(model, epochs=5000, n_colloc=1000, verbose=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model → {model_path}")

    u_pinn = evaluate_grid(model, fd.x, fd.t)

    # --- Metrics ---
    l2 = l2_error(u_fd, u_pinn)
    mx = max_error(u_fd, u_pinn)
    print(f"\n{'='*40}")
    print(f"  L2 error (FD vs PINN): {l2:.6f}")
    print(f"  Max absolute error:    {mx:.6f}")
    print(f"{'='*40}")
    print(summary_table(u_fd, u_pinn))

    # --- Plots ---
    plot_error_heatmap(u_fd, u_pinn, fname=str(plots / "error_heatmap.png"))
    make_comparison_gif(u_fd, u_pinn, fd.t,
                        fname=str(plots / "comparison.gif"),
                        n_frames=10, frame_duration=0.4)


if __name__ == "__main__":
    main()
