"""MADE (Masked Autoencoder for Distribution Estimation) for binary spins.

Implements an autoregressive model over ±1 spins using raster-scan ordering.
Each conditional p(z_k | z_{<k}) is a Bernoulli parameterized by masked dense layers.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class MaskedDense(nn.Module):
    """Dense layer with a fixed binary mask applied to the weight matrix."""
    features: int
    mask: jnp.ndarray  # binary mask, shape (in_features, features)

    @nn.compact
    def __call__(self, x):
        # Mask-aware initialization: correct variance for zeroed-out weights
        mask_correction = jnp.sqrt(self.mask.size / jnp.maximum(jnp.sum(self.mask), 1.0))
        def corrected_init(key, shape, dtype=jnp.float32):
            return nn.initializers.lecun_normal()(key, shape, dtype) * mask_correction

        kernel = self.param(
            'kernel',
            corrected_init,
            (x.shape[-1], self.features),
        )
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        return x @ (kernel * self.mask) + bias


class PReLU(nn.Module):
    """Parametric ReLU with per-feature learnable slope, initialized to 0.5."""
    features: int

    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', lambda key, shape: jnp.full(shape, 0.5), (self.features,))
        return jnp.where(x >= 0, x, alpha * x)


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation over ±1 spins.

    Uses raster-scan ordering: z_0, z_1, ..., z_{N-1}.
    Output logit_k depends only on z_0, ..., z_{k-1}.
    logit_k > 0 means p(z_k = +1 | z_{<k}) > 0.5.

    Masking follows VAN's block-channel convention:
      - Each hidden unit h is assigned site h % N
      - Input→hidden (exclusive): input site i < hidden site h % N
      - Hidden→hidden/output (non-exclusive): prev site <= next site
    """
    n_sites: int
    hidden_dims: Sequence[int] = ()

    def setup(self):
        N = self.n_sites
        hdims = self.hidden_dims if self.hidden_dims else (4 * N,)

        input_site = jnp.arange(N)

        masks = []
        prev_site = input_site
        exclusive = True

        for h_dim in hdims:
            hidden_site = jnp.arange(h_dim) % N
            if exclusive:
                mask = (prev_site[:, None] < hidden_site[None, :]).astype(jnp.float32)
                exclusive = False
            else:
                mask = (prev_site[:, None] <= hidden_site[None, :]).astype(jnp.float32)
            masks.append(mask)
            prev_site = hidden_site

        output_site = jnp.arange(N)
        output_mask = (prev_site[:, None] <= output_site[None, :]).astype(jnp.float32)
        masks.append(output_mask)

        self.layers = [MaskedDense(features=m.shape[1], mask=m) for m in masks]
        self.activations = [PReLU(features=m.shape[1]) for m in masks[:-1]]

    def __call__(self, z):
        """Compute logits for each conditional p(z_k | z_{<k}).

        Args:
            z: spin configuration in {-1, +1}, shape (..., N).

        Returns:
            Logits, shape (..., N).
        """
        x = (z + 1) / 2  # {0, 1}

        for layer, act in zip(self.layers[:-1], self.activations):
            x = layer(x)
            x = act(x)
        logits = self.layers[-1](x)
        return logits


def _raw_log_prob(model, params, z):
    """Compute raw (non-Z2) log p_MADE(z) = Σ_k log p(z_k | z_{<k})."""
    logits = model.apply(params, z)
    targets = (z + 1) / 2
    log_probs_per_site = (-jax.nn.softplus(-logits) * targets
                          + (-jax.nn.softplus(logits)) * (1 - targets))
    return jnp.sum(log_probs_per_site, axis=-1)


def log_prob(model, params, z, z2=False):
    """Compute log p(z), optionally with Z2 symmetry.

    Without Z2: log p(z) = Σ_k log p(z_k | z_{<k})
    With Z2:    log p(z) = logsumexp(log p_MADE(z), log p_MADE(-z)) - log(2)

    Args:
        model: MADE instance
        params: model parameters
        z: spin configuration in {-1, +1}, shape (..., N)
        z2: if True, enforce Z2 symmetry

    Returns:
        Log probability, shape (...)
    """
    lp = _raw_log_prob(model, params, z)
    if z2:
        lp_inv = _raw_log_prob(model, params, -z)
        lp = jnp.logaddexp(lp, lp_inv) - jnp.log(2.0)
    return lp


def sample(model, params, key, num_samples, z2=False):
    """Autoregressive sampling, optionally with Z2 symmetry.

    With Z2: after autoregressive sampling, randomly flip all spins
    with probability 0.5. The returned log_probs are Z2-symmetrized.

    Args:
        model: MADE instance
        params: model parameters
        key: JAX PRNGKey
        num_samples: number of samples to draw
        z2: if True, enforce Z2 symmetry

    Returns:
        (samples, log_probs): samples shape (num_samples, N), log_probs shape (num_samples,)
    """
    N = model.n_sites
    key, flip_key = jax.random.split(key)

    def scan_fn(carry, k):
        z, lp = carry
        key_k = jax.random.fold_in(key, k)
        logits = model.apply(params, z)
        logit_k = logits[:, k]
        prob_plus = jax.nn.sigmoid(logit_k)
        u = jax.random.uniform(key_k, shape=(num_samples,))
        bit = (u < prob_plus).astype(jnp.float32)
        spin = 2 * bit - 1
        z = z.at[:, k].set(spin)
        lp_k = (-jax.nn.softplus(-logit_k) * bit
                 + (-jax.nn.softplus(logit_k)) * (1 - bit))
        lp = lp + lp_k
        return (z, lp), None

    z_init = jnp.zeros((num_samples, N))
    lp_init = jnp.zeros(num_samples)
    (z_final, lp_raw), _ = jax.lax.scan(scan_fn, (z_init, lp_init), jnp.arange(N))

    if z2:
        # Random global flip: multiply all spins by ±1
        flip = jax.random.bernoulli(flip_key, 0.5, shape=(num_samples, 1))
        flip = 2.0 * flip - 1.0  # {-1, +1}
        z_final = z_final * flip
        # Z2-symmetrized log prob: logsumexp(lp(z), lp(-z)) - log(2)
        lp_inv = jax.vmap(lambda z: _raw_log_prob(model, params, z))(z_final * -1.0)
        lp_fwd = jax.vmap(lambda z: _raw_log_prob(model, params, z))(z_final)
        lp_z2 = jnp.logaddexp(lp_fwd, lp_inv) - jnp.log(2.0)
        return z_final, lp_z2
    else:
        return z_final, lp_raw

