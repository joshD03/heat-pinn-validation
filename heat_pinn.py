"""Physics-Informed Neural Network (PINN) for the 1-D heat equation.

Equation:  ∂u/∂t = α · ∂²u/∂x²
Domain:    x ∈ [0, 1], t ∈ [0, 0.5]
IC:        u(x, 0) = sin(πx)
BC:        u(0, t) = u(1, t) = 0

Architecture: MLP [2, 32, 32, 1] with tanh activations.
Training uses Adam with PDE-residual + IC + BC losses via autograd.
"""

import numpy as np
import torch
import torch.nn as nn

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

ALPHA = 0.01  # thermal diffusivity


class HeatPINN(nn.Module):
    """Small MLP that maps (x, t) → u(x, t)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate network.

        Parameters
        ----------
        x, t : Tensor, shape (N, 1)

        Returns
        -------
        u : Tensor, shape (N, 1)
        """
        return self.net(torch.cat([x, t], dim=1))


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def _pde_residual(model: HeatPINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute PDE residual r = u_t − α·u_xx via autograd."""
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    return u_t - ALPHA * u_xx


def train_pinn(model: HeatPINN, epochs: int = 5000, n_colloc: int = 1000,
               lr: float = 1e-3, verbose: bool = True):
    """Train the PINN and return the loss history.

    Parameters
    ----------
    model : HeatPINN
    epochs : int
    n_colloc : int
        Number of interior collocation points resampled each epoch.
    lr : float
    verbose : bool

    Returns
    -------
    loss_history : list[float]
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: list[float] = []

    # Fixed IC / BC sample sizes
    n_ic = 200
    n_bc = 200

    for epoch in range(1, epochs + 1):
        optimiser.zero_grad()

        # --- Interior (PDE) collocation ---
        x_c = torch.rand(n_colloc, 1)
        t_c = torch.rand(n_colloc, 1) * 0.5
        res = _pde_residual(model, x_c, t_c)
        loss_pde = (res ** 2).mean()

        # --- Initial condition: u(x, 0) = sin(πx) ---
        x_ic = torch.rand(n_ic, 1)
        t_ic = torch.zeros(n_ic, 1)
        u_ic_pred = model(x_ic, t_ic)
        u_ic_true = torch.sin(np.pi * x_ic)
        loss_ic = ((u_ic_pred - u_ic_true) ** 2).mean()

        # --- Boundary conditions: u(0,t) = u(1,t) = 0 ---
        t_bc = torch.rand(n_bc, 1) * 0.5
        u_bc0 = model(torch.zeros(n_bc, 1), t_bc)
        u_bc1 = model(torch.ones(n_bc, 1), t_bc)
        loss_bc = ((u_bc0 ** 2).mean() + (u_bc1 ** 2).mean())

        loss = loss_pde + loss_ic + loss_bc
        loss.backward()
        optimiser.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch:5d}/{epochs}  loss={loss_val:.6f}  "
                  f"(pde={loss_pde.item():.4e}  ic={loss_ic.item():.4e}  "
                  f"bc={loss_bc.item():.4e})")

    return loss_history


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_grid(model: HeatPINN,
                  x_grid: np.ndarray,
                  t_grid: np.ndarray) -> np.ndarray:
    """Evaluate trained PINN on a meshgrid.

    Parameters
    ----------
    model : HeatPINN
    x_grid : 1-D array, shape (Nx,)
    t_grid : 1-D array, shape (Nt,)

    Returns
    -------
    u : np.ndarray, shape (Nt, Nx)
    """
    model.eval()
    X, T_ = np.meshgrid(x_grid, t_grid)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    t_flat = torch.tensor(T_.flatten(), dtype=torch.float32).unsqueeze(1)
    u_flat = model(x_flat, t_flat).numpy().flatten()
    return u_flat.reshape(X.shape)
