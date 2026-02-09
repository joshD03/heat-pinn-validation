#!/usr/bin/env python
"""Train the PINN, save model weights, and plot the loss curve."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
from heat_pinn import HeatPINN, train_pinn


def main():
    out = pathlib.Path(__file__).resolve().parent.parent / "plots"
    out.mkdir(exist_ok=True)

    model = HeatPINN()
    print("Training PINN (5 000 epochs) ...")
    loss_history = train_pinn(model, epochs=5000, n_colloc=1000, verbose=True)

    # Save model
    model_path = pathlib.Path(__file__).resolve().parent.parent / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model → {model_path}")

    # Loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(loss_history, lw=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("PINN training loss")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    fname = out / "pinn_loss.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


if __name__ == "__main__":
    main()
