#!/usr/bin/env python
"""Visualize the bijective multi-scale decomposition on Ising configurations.

Generates spin configs via Metropolis MCMC at three temperatures
(ordered / critical / disordered), encodes them into multi-scale tokens,
and plots the decomposition alongside the reconstruction.

Usage:
    python scripts/visualize_multiscale.py [--L 16]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax.numpy as jnp

from dsflow_ising.multiscale import encode, decode, token_sizes


# ── Metropolis sampler (numpy, for generating test data) ──────────────

def metropolis_ising(L, beta, n_sweeps=5000, seed=42):
    """Generate an Ising config via Metropolis at inverse temperature beta."""
    rng = np.random.RandomState(seed)
    sigma = rng.choice([-1, 1], size=(L, L)).astype(np.float32)
    for _ in range(n_sweeps):
        for _ in range(L * L):
            i, j = rng.randint(0, L), rng.randint(0, L)
            nn_sum = (
                sigma[(i + 1) % L, j] + sigma[(i - 1) % L, j]
                + sigma[i, (j + 1) % L] + sigma[i, (j - 1) % L]
            )
            dE = 2.0 * sigma[i, j] * nn_sum
            if dE <= 0 or rng.random() < np.exp(-beta * dE):
                sigma[i, j] *= -1
    return sigma


# ── Plotting helpers ──────────────────────────────────────────────────

SPIN_CMAP = plt.cm.colors.ListedColormap(["#2563eb", "#f59e0b"])
SPIN_NORM = plt.cm.colors.BoundaryNorm([-1, 0, 1], SPIN_CMAP.N)


def plot_spins(ax, grid, title="", show_grid=True):
    """Plot a 2D ±1 spin grid."""
    ax.imshow(grid, cmap=SPIN_CMAP, norm=SPIN_NORM,
              interpolation="nearest", origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_grid and grid.shape[0] <= 32:
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.3)


def plot_details(ax, details, channel, title=""):
    """Plot one channel (h/v/d) of a detail tensor."""
    grid = np.array(details[..., channel])
    plot_spins(ax, grid, title=title)


# ── Main figure ───────────────────────────────────────────────────────

def make_figure(L):
    beta_c = np.log(1 + np.sqrt(2)) / 2  # ≈ 0.4407
    temps = [
        ("T=1.5 (ordered)", 1.0 / 1.5),
        (f"T=Tc≈{1/beta_c:.2f}", beta_c),
        ("T=4.0 (disordered)", 1.0 / 4.0),
    ]
    n_levels = int(np.log2(L))  # number of detail levels + 1 for s0
    detail_labels = ["horiz (a·b)", "vert (a·c)", "diag (a·d)"]

    n_cols = len(temps)
    # rows: original | s0 | d1 h,v,d | d2 h,v,d | ... | dK h,v,d | recon
    n_detail_rows = n_levels * 3  # 3 channels per level
    n_rows = 1 + 1 + n_detail_rows + 1  # orig + s0 + details + recon

    fig = plt.figure(figsize=(3.5 * n_cols, 1.4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.45, wspace=0.25)

    for col, (label, beta) in enumerate(temps):
        sigma = metropolis_ising(L, beta, n_sweeps=8000, seed=col + 10)
        sigma_jnp = jnp.array(sigma)

        # Encode
        tokens = encode(sigma_jnp)
        # Decode and verify
        recon = decode(tokens)
        err = float(jnp.max(jnp.abs(recon - sigma_jnp)))

        # Row 0: original
        ax = fig.add_subplot(gs[0, col])
        plot_spins(ax, sigma, title=f"{label}")
        if col == 0:
            ax.set_ylabel("original", fontsize=9)

        # Row 1: s0 (1×1 global spin)
        ax = fig.add_subplot(gs[1, col])
        s0_val = float(tokens[0][0, 0])
        ax.text(0.5, 0.5, f"s₀ = {s0_val:+.0f}",
                ha="center", va="center", fontsize=14,
                color="#f59e0b" if s0_val > 0 else "#2563eb",
                transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#1e1e2e")
        if col == 0:
            ax.set_ylabel("s₀ (1×1)", fontsize=9)

        # Detail levels
        row = 2
        for lvl_idx in range(1, len(tokens)):
            details = np.array(tokens[lvl_idx])
            m = details.shape[0]
            for ch in range(3):
                ax = fig.add_subplot(gs[row, col])
                plot_details(ax, details, ch,
                             title=f"d{lvl_idx} {m}×{m}" if ch == 0 else "")
                if col == 0:
                    ax.set_ylabel(detail_labels[ch], fontsize=8)
                row += 1

        # Last row: reconstruction
        ax = fig.add_subplot(gs[n_rows - 1, col])
        plot_spins(ax, np.array(recon), title=f"recon (err={err:.0f})")
        if col == 0:
            ax.set_ylabel("decoded", fontsize=9)

    # Token count summary
    sizes = token_sizes(L)
    parts = [f"{name}: {n}" for name, n in sizes]
    fig.suptitle(
        f"Bijective multi-scale decomposition  L={L}   "
        f"tokens: {' + '.join(parts)} = {sum(n for _, n in sizes)}",
        fontsize=10, y=0.995,
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=16)
    args = parser.parse_args()

    fig = make_figure(args.L)
    out = f"multiscale_decomposition_L{args.L}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()
