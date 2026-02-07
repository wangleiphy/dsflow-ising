#!/usr/bin/env python
"""Plot training metrics from CSV log file."""

import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot training log")
    parser.add_argument("log_file", help="CSV log file from run_training.py")
    parser.add_argument("-o", "--output", default="train_plot.png", help="Output image file")
    parser.add_argument("--L", type=int, required=True, help="Lattice side length (for per-site normalization)")
    args = parser.parse_args()

    N = args.L ** 2

    steps, f_var, energy, entropy = [], [], [], []
    with open(args.log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            f_var.append(float(row['f_var']))
            energy.append(float(row['energy']))
            entropy.append(float(row['entropy']))

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    import numpy as np
    f_var_n = np.array(f_var) / N
    energy_n = np.array(energy) / N
    entropy_n = np.array(entropy) / N

    # Running average for smoothing
    window = max(1, len(steps) // 50)
    kernel = np.ones(window) / window if window > 1 else None

    axes[0].plot(steps, f_var_n, 'b-', alpha=0.3)
    if kernel is not None:
        axes[0].plot(steps[window-1:], np.convolve(f_var_n, kernel, mode='valid'), 'b-', lw=2)
    axes[0].set_ylabel('F/N')
    axes[0].set_title('Variational Free Energy per Site')

    axes[1].plot(steps, energy_n, 'r-', alpha=0.3)
    if kernel is not None:
        axes[1].plot(steps[window-1:], np.convolve(energy_n, kernel, mode='valid'), 'r-', lw=2)
    axes[1].set_ylabel('E/N')
    axes[1].set_title('Energy per Site ⟨E(σ)⟩/N')

    axes[2].plot(steps, entropy_n, 'g-', alpha=0.3)
    if kernel is not None:
        axes[2].plot(steps[window-1:], np.convolve(entropy_n, kernel, mode='valid'), 'g-', lw=2)
    axes[2].set_ylabel('S/N')
    axes[2].set_title('Entropy per Site H[p_θ]/N')
    axes[2].set_xlabel('Training Step')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
