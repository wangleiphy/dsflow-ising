"""Diagnostics: free energy, entropy, and layer-by-layer metrics."""

import jax
import jax.numpy as jnp

from dsflow_ising.ising import energy
from dsflow_ising.made import MADE, log_prob, sample
from dsflow_ising.flow import DiscreteFlow
from dsflow_ising.coupling import checkerboard_indices
from dsflow_ising.flow import get_partition


def variational_free_energy(made_model, made_params, flow_model, flow_params,
                            pairs, J, T, key, num_samples=1000):
    """Estimate F_var = ⟨E(σ) + T ln p_θ(z)⟩ by sampling.

    Returns:
        F_var: scalar estimate
    """
    z_samples, z_log_probs = sample(made_model, made_params, key, num_samples=num_samples)
    sigma = flow_model.apply(flow_params, z_samples, use_ste=False)
    energies = jax.vmap(lambda s: energy(s, pairs, J))(sigma)
    return jnp.mean(energies + T * z_log_probs)


def base_entropy(made_model, made_params, key, num_samples=1000):
    """Estimate base distribution entropy H[p_θ] = -E[ln p_θ(z)].

    Returns:
        H: scalar estimate (non-negative)
    """
    _, z_log_probs = sample(made_model, made_params, key, num_samples=num_samples)
    return -jnp.mean(z_log_probs)


def conditional_entropy_profile(made_model, made_params, key, num_samples=1000):
    """Compute H(z_k | z_{<k}) for each site k.

    Returns:
        Array of shape (N,) with per-site conditional entropies.
    """
    N = made_model.n_sites
    z_samples, _ = sample(made_model, made_params, key, num_samples=num_samples)

    # Get logits for all conditionals
    logits = jax.vmap(lambda z: made_model.apply(made_params, z))(z_samples)  # (batch, N)

    # H(z_k | z_{<k}) = -E[p_k log p_k + (1-p_k) log(1-p_k)]
    # where p_k = sigmoid(logit_k)
    p = jax.nn.sigmoid(logits)  # (batch, N)
    # Binary entropy: -p log p - (1-p) log(1-p)
    eps = 1e-7
    h = -p * jnp.log(p + eps) - (1 - p) * jnp.log(1 - p + eps)  # (batch, N)
    return jnp.mean(h, axis=0)  # (N,)


def layer_free_energy_reduction(made_model, made_params, flow_model, flow_params,
                                pairs, J, T, key, num_samples=1000):
    """Compute the free energy reduction contributed by each flow layer.

    We compute F_var with 0, 1, 2, ... layers and take differences.
    ΔF_l = F_var(l layers) - F_var(l+1 layers)

    Positive ΔF means layer l+1 reduces the free energy.

    Returns:
        Array of shape (n_layers,) with per-layer ΔF.
    """
    z_samples, z_log_probs = sample(made_model, made_params, key, num_samples=num_samples)

    n_layers = flow_model.n_layers
    L = flow_model.L
    N = L * L

    # F_var with 0 layers (identity flow): σ = z
    energies_0 = jax.vmap(lambda s: energy(s, pairs, J))(z_samples)
    f_var_0 = jnp.mean(energies_0 + T * z_log_probs)

    # Apply flow with increasing numbers of layers via partial models
    f_vars = [f_var_0]
    for n in range(1, n_layers + 1):
        partial_flow = DiscreteFlow(L=L, n_layers=n, mask_features=flow_model.mask_features)
        # Extract params for first n layers
        partial_params = _extract_partial_params(flow_params, n)
        sigma = partial_flow.apply(partial_params, z_samples, use_ste=False)
        energies = jax.vmap(lambda s: energy(s, pairs, J))(sigma)
        f_vars.append(jnp.mean(energies + T * z_log_probs))

    # ΔF_l = F(l layers) - F(l+1 layers)
    deltas = jnp.array([f_vars[i] - f_vars[i + 1] for i in range(n_layers)])
    print(f"Per-layer free energy contributions: {deltas}")
    return deltas


def _extract_partial_params(flow_params, n_layers):
    """Extract parameters for the first n_layers from a full flow params dict."""
    # Flax linen stores params as params['params']['layer_0'], params['params']['layer_1'], ...
    full_inner = flow_params['params']
    partial_inner = {k: v for k, v in full_inner.items()
                     if k.startswith('layer_') and int(k.split('_')[1]) < n_layers}
    return {'params': partial_inner}
