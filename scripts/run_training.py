#!/usr/bin/env python
"""Entry point for training discrete normalizing flow on 2D Ising model."""

import argparse
import jax.numpy as jnp

from dsflow_ising.config import ModelConfig, TrainConfig
from dsflow_ising.train import train
from dsflow_ising.diagnostics import variational_free_energy, base_entropy
import jax


def main():
    parser = argparse.ArgumentParser(description="Train discrete flow for 2D Ising model")
    parser.add_argument("--L", type=int, default=8, help="Lattice side length")
    parser.add_argument("--n-flow-layers", type=int, default=4, help="Number of coupling layers")
    parser.add_argument("--T", type=float, default=2.269, help="Temperature")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr-theta", type=float, default=1e-3, help="Learning rate for MADE")
    parser.add_argument("--lr-phi", type=float, default=1e-3, help="Learning rate for flow")
    parser.add_argument("--num-steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--made-hidden-dims", type=int, nargs='+', default=None,
                        help="MADE hidden layer sizes, e.g. --made-hidden-dims 512 512")
    parser.add_argument("--z2", action="store_true", help="Enable Z2 spin-flip symmetry")
    parser.add_argument("--beta-anneal", type=float, default=0.0,
                        help="Beta annealing rate (0=disabled, e.g. 0.998)")
    parser.add_argument("--log-every", type=int, default=100, help="Log interval")
    parser.add_argument("--log-file", type=str, default=None, help="CSV log file path")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="dsflow-ising", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    args = parser.parse_args()

    model_cfg = ModelConfig(
        L=args.L,
        n_flow_layers=args.n_flow_layers,
        mask_features=(16, 16),
        made_hidden_dims=tuple(args.made_hidden_dims) if args.made_hidden_dims else (),
        z2=args.z2,
    )
    train_cfg = TrainConfig(
        T=args.T,
        J=args.J,
        batch_size=args.batch_size,
        lr_theta=args.lr_theta,
        lr_phi=args.lr_phi,
        num_steps=args.num_steps,
        seed=args.seed,
        beta_anneal=args.beta_anneal,
    )

    print(f"Training: L={model_cfg.L}, layers={model_cfg.n_flow_layers}, "
          f"T={train_cfg.T}, batch={train_cfg.batch_size}, z2={model_cfg.z2}")
    print(f"Total sites: {model_cfg.L**2}")

    state, history, made_model, flow_model, pairs = train(
        model_cfg, train_cfg, log_every=args.log_every, log_file=args.log_file,
        use_wandb=args.wandb, wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )

    # Final diagnostics
    key = jax.random.PRNGKey(train_cfg.seed + 999)
    k1, k2 = jax.random.split(key)
    F_final = variational_free_energy(
        made_model, state.made_params, flow_model, state.flow_params,
        pairs, train_cfg.J, train_cfg.T, k1, num_samples=1000,
    )
    H_final = base_entropy(made_model, state.made_params, k2, num_samples=1000)

    N = model_cfg.L ** 2
    print(f"\nFinal F_var = {F_final:.4f}")
    print(f"Final F_var/N = {F_final / N:.4f}")
    print(f"Base entropy H = {H_final:.4f}")
    print(f"Max entropy (uniform) = {N * jnp.log(2.0):.4f}")


if __name__ == "__main__":
    main()

