"""Multi-scale discrete flow with squeeze/unsqueeze operations.

Inspired by Glow/RealNVP multi-scale architecture, adapted for discrete ±1 spins.
Uses a U-Net-like structure:

  Scale 0 (L×L×1):       coupling layers
  squeeze
  Scale 1 (L/2×L/2×4):   coupling layers
  squeeze
  ...
  Scale K (bottleneck):   coupling layers
  unsqueeze
  ...
  Scale 0 (L×L×1):       coupling layers

Squeeze groups 2×2 spatial blocks into channels: (B, H, W, C) → (B, H/2, W/2, 4C).
Each coupling layer uses checkerboard partitions with a dilated ResNet as mask network.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


def squeeze(x):
    """Space-to-depth: (B, H, W, C) → (B, H/2, W/2, 4C).

    Groups each 2×2 spatial block into 4 channel values.
    Block at (2r, 2t) maps to channels in order: (0,0), (0,1), (1,0), (1,1).
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // 2, 2, W // 2, 2, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H/2, W/2, 2, 2, C)
    x = x.reshape(B, H // 2, W // 2, 4 * C)
    return x


def unsqueeze(x):
    """Depth-to-space: (B, H/2, W/2, 4C) → (B, H, W, C).

    Inverse of squeeze.
    """
    B, h, w, C4 = x.shape
    C = C4 // 4
    x = x.reshape(B, h, w, 2, 2, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, h, 2, w, 2, C)
    x = x.reshape(B, h * 2, w * 2, C)
    return x


def checkerboard_mask(H, W, parity):
    """Checkerboard mask: 1 at A-sites, 0 at B-sites.

    Returns: (H, W, 1), broadcastable over batch and channels.
    """
    rows = jnp.arange(H)[:, None]
    cols = jnp.arange(W)[None, :]
    mask = ((rows + cols) % 2 == parity).astype(jnp.float32)
    return mask[:, :, None]


class ScaleMaskNet(nn.Module):
    """Dilated ResNet ConvNet for multi-channel coupling.

    Maps (B, H, W, C_in) → (B, H, W, out_channels) with exponentially
    growing receptive field via dilated convolutions.
    """
    hidden_features: int = 32
    n_res_blocks: int = 2
    out_channels: int = 1

    @nn.compact
    def __call__(self, z_grid):
        x = nn.Conv(self.hidden_features, (3, 3), padding='SAME')(z_grid)
        for i in range(self.n_res_blocks):
            dilation = 2 ** i
            residual = x
            x = nn.relu(x)
            x = nn.Conv(
                self.hidden_features, (3, 3), padding='SAME',
                kernel_dilation=(dilation, dilation),
            )(x)
            x = nn.relu(x)
            x = nn.Conv(self.hidden_features, (3, 3), padding='SAME')(x)
            x += residual
        x = nn.relu(x)
        x = nn.Conv(self.out_channels, (1, 1))(x)
        return x


def _binary_sign(x):
    """Binary sign: x >= 0 → +1, x < 0 → -1."""
    return jnp.where(x >= 0, 1.0, -1.0)


def _ste_sign(x):
    """Sign with straight-through estimator."""
    return x + jax.lax.stop_gradient(_binary_sign(x) - x)


def _grid_coupling(mask_net, x, parity, use_ste, z2=False):
    """Apply checkerboard coupling on a (B, H, W, C) grid.

    A-sites ((i+j)%2 == parity): unchanged.
    B-sites: multiplied by sign(mask_net(A-sites)).
    """
    a_mask = checkerboard_mask(x.shape[1], x.shape[2], parity)
    b_mask = 1.0 - a_mask
    z_a = x * a_mask
    logits = mask_net(z_a)
    if z2:
        logits = logits + mask_net(-z_a)
    sign_fn = _ste_sign if use_ste else _binary_sign
    m = sign_fn(logits)
    return x * a_mask + (x * m) * b_mask


class MultiScaleFlow(nn.Module):
    """Multi-scale discrete flow with squeeze/unsqueeze.

    U-Net-like architecture operating on ±1 spins.
    Same interface as DiscreteFlow: accepts flat (..., N) inputs.

    Total coupling layers = (2 * n_scales - 1) * layers_per_scale.

    Attributes:
        L: lattice side length (must be divisible by 2^(n_scales-1))
        n_scales: number of spatial scales (1 = flat, no squeeze)
        layers_per_scale: coupling layers per scale per pass
        hidden_features: channels in MaskNet hidden layers
        n_res_blocks: residual blocks per MaskNet
        z2: Z2 spin-flip symmetry
    """
    L: int
    n_scales: int = 2
    layers_per_scale: int = 4
    hidden_features: int = 32
    n_res_blocks: int = 2
    z2: bool = False

    @nn.compact
    def __call__(self, z, use_ste=True, inverse=False):
        N = self.L * self.L
        batch_shape = z.shape[:-1]
        x = z.reshape(-1, N) if batch_shape else z[None]
        B = x.shape[0]
        x = x.reshape(B, self.L, self.L, 1)

        ops = self._build_ops()
        if inverse:
            ops = self._invert_ops(ops)
        x = self._execute_ops(x, ops, use_ste)

        x = x.reshape(B, N)
        if batch_shape:
            return x.reshape(*batch_shape, N)
        else:
            return x.squeeze(0)

    def _build_ops(self):
        """Build forward operation sequence.

        Descending: for each scale, apply coupling layers then squeeze.
        Ascending: unsqueeze then apply coupling layers, back to finest.
        """
        ops = []
        layer_idx = 0
        # Descending path
        for scale in range(self.n_scales):
            channels = 4 ** scale
            for _ in range(self.layers_per_scale):
                parity = layer_idx % 2
                ops.append(('couple', layer_idx, channels, parity))
                layer_idx += 1
            if scale < self.n_scales - 1:
                ops.append(('squeeze',))
        # Ascending path
        for scale in range(self.n_scales - 2, -1, -1):
            ops.append(('unsqueeze',))
            channels = 4 ** scale
            for _ in range(self.layers_per_scale):
                parity = layer_idx % 2
                ops.append(('couple', layer_idx, channels, parity))
                layer_idx += 1
        return ops

    def _invert_ops(self, ops):
        """Invert: reverse order, swap squeeze/unsqueeze."""
        inv = []
        for op in reversed(ops):
            if op[0] == 'squeeze':
                inv.append(('unsqueeze',))
            elif op[0] == 'unsqueeze':
                inv.append(('squeeze',))
            else:
                inv.append(op)
        return inv

    def _execute_ops(self, x, ops, use_ste):
        """Execute a sequence of operations on grid tensor x."""
        for op in ops:
            if op[0] == 'squeeze':
                x = squeeze(x)
            elif op[0] == 'unsqueeze':
                x = unsqueeze(x)
            else:
                _, layer_idx, channels, parity = op
                net = ScaleMaskNet(
                    hidden_features=self.hidden_features,
                    n_res_blocks=self.n_res_blocks,
                    out_channels=channels,
                    name=f"mask_{layer_idx}",
                )
                x = _grid_coupling(net, x, parity, use_ste, self.z2)
        return x
