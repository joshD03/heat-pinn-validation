"""Finite-difference solver for the 1-D heat equation.

Equation:  ∂u/∂t = α · ∂²u/∂x²
Domain:    x ∈ [0, L], t ∈ [0, T]
IC:        u(x, 0) = sin(πx)
BC:        u(0, t) = u(1, t) = 0
Method:    explicit Euler in time, centred differences in space.
"""

import warnings
import numpy as np


class HeatSolverFD:
    """Explicit finite-difference solver for the 1-D heat equation."""

    def __init__(self, L: float = 1.0, T: float = 0.5,
                 Nx: int = 100, Nt: int = 50, alpha: float = 0.01):
        """Initialise grid and check CFL stability.

        Parameters
        ----------
        L : float
            Spatial domain length.
        T : float
            Total simulation time.
        Nx : int
            Number of spatial grid points (including boundaries).
        Nt : int
            Number of time steps.
        alpha : float
            Thermal diffusivity.
        """
        self.L = L
        self.T = T
        self.Nx = Nx
        self.Nt = Nt
        self.alpha = alpha

        self.dx = L / (Nx - 1)
        self.dt = T / Nt
        self.x = np.linspace(0.0, L, Nx)
        self.t = np.linspace(0.0, T, Nt + 1)

        # CFL stability condition: r = α·dt/dx² ≤ 0.5
        self.r = alpha * self.dt / self.dx ** 2
        if self.r > 0.5:
            warnings.warn(
                f"CFL condition violated: r = {self.r:.4f} > 0.5. "
                "Solution may be unstable. Increase Nt or decrease Nx."
            )

    # ------------------------------------------------------------------
    def solve(self) -> np.ndarray:
        """Run the explicit Euler time-stepping scheme.

        Returns
        -------
        u : np.ndarray, shape (Nt+1, Nx)
            Solution snapshots at every saved time level.
        """
        u = np.zeros((self.Nt + 1, self.Nx))
        # Initial condition
        u[0, :] = np.sin(np.pi * self.x)

        r = self.r
        for n in range(self.Nt):
            u[n + 1, 1:-1] = (u[n, 1:-1]
                               + r * (u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2]))
            # Dirichlet BCs
            u[n + 1, 0] = 0.0
            u[n + 1, -1] = 0.0

        return u

    # ------------------------------------------------------------------
    def analytical(self) -> np.ndarray:
        """Exact (Fourier) solution for the first-mode IC sin(πx).

        Returns
        -------
        u_exact : np.ndarray, shape (Nt+1, Nx)
        """
        X, T_ = np.meshgrid(self.x, self.t)
        return np.sin(np.pi * X) * np.exp(-self.alpha * np.pi ** 2 * T_)


git config --global user.email "j.das191003@gmail.com"
git config --global user.name "joshD03"