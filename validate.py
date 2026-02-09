"""Validation and visualisation utilities.

Provides error metrics, an error heatmap, and a side-by-side comparison GIF
for the FD and PINN solutions to the 1-D heat equation.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def l2_error(u1: np.ndarray, u2: np.ndarray) -> float:
    """Relative L2 error between two solution arrays.

    Parameters
    ----------
    u1, u2 : np.ndarray of same shape

    Returns
    -------
    float
        ||u1 - u2||_2 / ||u1||_2
    """
    diff_norm = np.linalg.norm(u1 - u2)
    ref_norm = np.linalg.norm(u1)
    if ref_norm == 0.0:
        return float(diff_norm)
    return float(diff_norm / ref_norm)


def max_error(u1: np.ndarray, u2: np.ndarray) -> float:
    """Maximum absolute pointwise error.

    Parameters
    ----------
    u1, u2 : np.ndarray of same shape

    Returns
    -------
    float
    """
    return float(np.max(np.abs(u1 - u2)))


def summary_table(u_fd: np.ndarray, u_pinn: np.ndarray) -> str:
    """Return a formatted Markdown summary table of error metrics.

    Parameters
    ----------
    u_fd, u_pinn : np.ndarray of same shape

    Returns
    -------
    str
        Markdown-formatted table.
    """
    l2 = l2_error(u_fd, u_pinn)
    max_e = max_error(u_fd, u_pinn)
    rel_l2 = l2  # l2_error already returns relative L2
    return (
        "\n"
        "  | Metric       | Value      |\n"
        "  |--------------|------------|\n"
        f"  | L2 error     | {l2:.6f}   |\n"
        f"  | Max error    | {max_e:.6f}   |\n"
        f"  | Relative L2  | {rel_l2:.1%}  |\n"
    )


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def plot_error_heatmap(u_fd: np.ndarray, u_pinn: np.ndarray,
                       fname: str = "plots/error_heatmap.png") -> None:
    """Save a heatmap of |u_fd − u_pinn| as a PNG.

    Parameters
    ----------
    u_fd, u_pinn : np.ndarray, shape (Nt, Nx)
    fname : str
    """
    pathlib.Path(fname).parent.mkdir(parents=True, exist_ok=True)
    err = np.abs(u_fd - u_pinn)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(err, aspect="auto", origin="lower",
                   extent=[0, 1, 0, 0.5], cmap="hot")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Absolute error")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Error heatmap  |FD − PINN|")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def make_comparison_gif(u_fd: np.ndarray, u_pinn: np.ndarray,
                        t_grid: np.ndarray,
                        fname: str = "plots/comparison.gif",
                        n_frames: int = 10,
                        frame_duration: float = 0.4) -> None:
    """Create a side-by-side GIF of FD (left) and PINN (right) evolution.

    Parameters
    ----------
    u_fd, u_pinn : np.ndarray, shape (Nt, Nx)
    t_grid : 1-D array of time values
    fname : str
    n_frames : int
    frame_duration : float  (seconds per frame)
    """
    pathlib.Path(fname).parent.mkdir(parents=True, exist_ok=True)
    Nt = u_fd.shape[0]
    indices = np.linspace(0, Nt - 1, n_frames, dtype=int)
    x = np.linspace(0, 1, u_fd.shape[1])
    ymax = max(np.max(np.abs(u_fd)), np.max(np.abs(u_pinn))) * 1.1

    frames: list[np.ndarray] = []
    for idx in indices:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        ax1.plot(x, u_fd[idx], "b-", lw=2)
        ax1.set_title(f"FD   t = {t_grid[idx]:.3f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x, t)")
        ax1.set_ylim(-0.05, ymax)
        ax1.grid(True, alpha=0.3)

        ax2.plot(x, u_pinn[idx], "r-", lw=2)
        ax2.set_title(f"PINN   t = {t_grid[idx]:.3f}")
        ax2.set_xlabel("x")
        ax2.set_ylim(-0.05, ymax)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(fname, frames, duration=frame_duration, loop=0)
    print(f"Saved {fname}  ({n_frames} frames, {frame_duration}s each)")
