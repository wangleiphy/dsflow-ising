"""Full discrete flow: stacks coupling layers with alternating checkerboard partitions.

The flow maps latent spins z to physical spins σ:
    σ = f_φ(z) = layer_{K-1} ∘ ... ∘ layer_1 ∘ layer_0 (z)

The inverse maps physical spins σ back to latent spins z:
    z = f_φ⁻¹(σ) = layer_0 ∘ layer_1 ∘ ... ∘ layer_{K-1} (σ)

Each coupling layer is self-inverse, so the inverse just reverses the application order.
No Jacobian correction is needed for discrete bijections.
"""

import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

from dsflow_ising.coupling import MaskNet, forward_layer


def get_partition(layer_idx: int) -> str:
    """Alternating checkerboard partition: even layers use 'even', odd layers use 'odd'."""
    return "even" if layer_idx % 2 == 0 else "odd"


class DiscreteFlow(nn.Module):
    """Stack of discrete coupling layers with alternating checkerboard partitions.

    Attributes:
        L: lattice side length
        n_layers: number of coupling layers
        mask_features: hidden channel sizes for MaskNet ConvNets
        z2: if True, symmetrize mask networks for Z2 equivariance
    """
    L: int
    n_layers: int = 4
    mask_features: Sequence[int] = (16, 16)
    z2: bool = False

    @nn.compact
    def __call__(self, z, use_ste=True, inverse=False):
        """Forward or inverse pass through all coupling layers.

        Args:
            z: input spins in {-1, +1}, shape (..., N) where N = L*L
            use_ste: use straight-through estimator for sign gradients
            inverse: if True, apply layers in reverse order (σ → z)

        Returns:
            Output spins, shape (..., N)
        """
        x = z
        layer_order = range(self.n_layers)
        if inverse:
            layer_order = reversed(layer_order)

        for i in layer_order:
            partition = get_partition(i)
            mask_net = MaskNet(L=self.L, features=self.mask_features, name=f"layer_{i}")
            x = _apply_coupling(mask_net, x, self.L, partition, use_ste, self.z2)
        return x


def _apply_coupling(mask_net, z, L, partition, use_ste, z2=False):
    """Apply a single coupling layer using a MaskNet defined in the parent scope.

    When z2=True, the mask network output is symmetrized:
        g(z_A) = h(z_A) + h(-z_A)
    This makes g an even function, ensuring Z2 equivariance of the coupling layer.
    """
    from dsflow_ising.coupling import checkerboard_indices, _ste_sign
    N = L * L
    a_idx, b_idx = checkerboard_indices(L, partition)

    batch_shape = z.shape[:-1]
    z_flat = z.reshape(-1, N) if batch_shape else z[None]
    B = z_flat.shape[0]

    # Build grid with only A-sublattice values
    z_grid = jnp.zeros((B, L, L))
    z_grid = z_grid.at[:, a_idx // L, a_idx % L].set(z_flat[:, a_idx])

    # Get logits from mask network
    logits = mask_net(z_grid)  # (B, L, L)

    if z2:
        # Symmetrize: g(x) = h(x) + h(-x), ensuring g(-x) = g(x)
        logits = logits + mask_net(-z_grid)

    logits_flat = logits.reshape(B, N)
    logits_b = logits_flat[:, b_idx]

    # Compute signs
    sign_fn = _ste_sign if use_ste else jnp.sign
    m = sign_fn(logits_b)

    # Apply: σ_B = z_B * m, σ_A = z_A
    sigma_flat = z_flat.at[:, b_idx].set(z_flat[:, b_idx] * m)

    if batch_shape:
        return sigma_flat.reshape(*batch_shape, N)
    else:
        return sigma_flat.squeeze(0)

