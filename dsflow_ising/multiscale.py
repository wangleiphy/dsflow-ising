"""Bijective multi-scale decomposition for ±1 Ising spins.

Discrete Haar-like wavelet on {±1} spins. Each 2×2 block [a,b;c,d]
decomposes into:
    coarse   = a          (representative, recurses to next level)
    detail_h = a·b        (horizontal relative sign)
    detail_v = a·c        (vertical relative sign)
    detail_d = a·d        (diagonal relative sign)

Total tokens = 1 + 3 + 12 + ... + 3(L/2)² = L²  (no redundancy).
The transform is bijective: decode(encode(σ)) = σ for all σ ∈ {±1}^{L×L}.
"""

import jax.numpy as jnp


def encode(sigma):
    """Encode L×L spin config into multi-scale relative-sign tokens.

    Args:
        sigma: shape (L, L), values in {-1, +1}.  L must be power of 2.

    Returns:
        List [s0, d1, d2, ..., dK] where
          s0: shape (1, 1)           — global representative spin
          dk: shape (Mk, Mk, 3)     — detail (h, v, d) at level k
    """
    L = sigma.shape[0]
    assert sigma.shape == (L, L) and L >= 2 and (L & (L - 1)) == 0
    levels = []
    cur = sigma
    while cur.shape[0] > 1:
        a = cur[0::2, 0::2]
        b = cur[0::2, 1::2]
        c = cur[1::2, 0::2]
        d = cur[1::2, 1::2]
        details = jnp.stack([a * b, a * c, a * d], axis=-1)
        levels.append(details)
        cur = a
    return [cur] + levels[::-1]


def decode(tokens):
    """Decode multi-scale tokens back to L×L spin config.

    Args:
        tokens: list [s0, d1, ..., dK] as returned by encode().

    Returns:
        sigma: shape (L, L), values in {-1, +1}.
    """
    cur = tokens[0]  # (1,1)
    for details in tokens[1:]:
        a = cur
        b = a * details[..., 0]
        c = a * details[..., 1]
        d = a * details[..., 2]
        M = a.shape[0] * 2
        out = jnp.zeros((M, M), dtype=a.dtype)
        out = out.at[0::2, 0::2].set(a)
        out = out.at[0::2, 1::2].set(b)
        out = out.at[1::2, 0::2].set(c)
        out = out.at[1::2, 1::2].set(d)
        cur = out
    return cur


def encode_batch(sigma):
    """Batched encode: (B, L, L) → list of (B, ...) arrays."""
    L = sigma.shape[1]
    assert sigma.ndim == 3 and sigma.shape[2] == L
    levels = []
    cur = sigma
    while cur.shape[1] > 1:
        a = cur[:, 0::2, 0::2]
        b = cur[:, 0::2, 1::2]
        c = cur[:, 1::2, 0::2]
        d = cur[:, 1::2, 1::2]
        details = jnp.stack([a * b, a * c, a * d], axis=-1)
        levels.append(details)
        cur = a
    return [cur] + levels[::-1]


def decode_batch(tokens):
    """Batched decode: list of (B, ...) arrays → (B, L, L)."""
    cur = tokens[0]
    for details in tokens[1:]:
        a = cur
        b = a * details[..., 0]
        c = a * details[..., 1]
        d = a * details[..., 2]
        M = a.shape[1] * 2
        B = a.shape[0]
        out = jnp.zeros((B, M, M), dtype=a.dtype)
        out = out.at[:, 0::2, 0::2].set(a)
        out = out.at[:, 0::2, 1::2].set(b)
        out = out.at[:, 1::2, 0::2].set(c)
        out = out.at[:, 1::2, 1::2].set(d)
        cur = out
    return cur


def token_sizes(L):
    """Return list of (level_name, n_spins) for each token level."""
    sizes = [("s0", 1)]
    m = 1
    while m < L:
        sizes.append((f"d{len(sizes)}", 3 * m * m))
        m *= 2
    return sizes
