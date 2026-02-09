"""Tests for multi-scale discrete flow."""

import jax
import jax.numpy as jnp
import pytest

from dsflow_ising.multiscale_flow import (
    squeeze, unsqueeze, checkerboard_mask,
    ScaleMaskNet, _grid_coupling, _binary_sign,
    MultiScaleFlow,
)


# ---------------------------------------------------------------------------
# squeeze / unsqueeze
# ---------------------------------------------------------------------------

class TestSqueezeUnsqueeze:
    def test_squeeze_shape(self):
        x = jnp.ones((3, 8, 8, 2))
        assert squeeze(x).shape == (3, 4, 4, 8)

    def test_unsqueeze_shape(self):
        x = jnp.ones((3, 4, 4, 8))
        assert unsqueeze(x).shape == (3, 8, 8, 2)

    def test_squeeze_unsqueeze_roundtrip(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(2, 8, 8, 1))
        assert jnp.allclose(unsqueeze(squeeze(x)), x)

    def test_unsqueeze_squeeze_roundtrip(self):
        key = jax.random.PRNGKey(1)
        x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(2, 4, 4, 4))
        assert jnp.allclose(squeeze(unsqueeze(x)), x)

    def test_squeeze_multichannel_roundtrip(self):
        key = jax.random.PRNGKey(2)
        x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(2, 6, 6, 3))
        assert jnp.allclose(unsqueeze(squeeze(x)), x)

    def test_squeeze_values(self):
        """Verify squeeze correctly groups 2×2 blocks into channels."""
        x = jnp.arange(16).reshape(1, 4, 4, 1).astype(jnp.float32)
        # x[0,:,:,0] = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
        y = squeeze(x)
        assert y.shape == (1, 2, 2, 4)
        # Top-left block [[0,1],[4,5]] → [0, 1, 4, 5]
        assert jnp.allclose(y[0, 0, 0, :], jnp.array([0, 1, 4, 5]))
        # Top-right block [[2,3],[6,7]] → [2, 3, 6, 7]
        assert jnp.allclose(y[0, 0, 1, :], jnp.array([2, 3, 6, 7]))
        # Bottom-left block [[8,9],[12,13]] → [8, 9, 12, 13]
        assert jnp.allclose(y[0, 1, 0, :], jnp.array([8, 9, 12, 13]))
        # Bottom-right block [[10,11],[14,15]] → [10, 11, 14, 15]
        assert jnp.allclose(y[0, 1, 1, :], jnp.array([10, 11, 14, 15]))


# ---------------------------------------------------------------------------
# checkerboard_mask
# ---------------------------------------------------------------------------

class TestCheckerboardMask:
    def test_shape(self):
        assert checkerboard_mask(4, 4, 0).shape == (4, 4, 1)

    def test_pattern_parity0(self):
        m = checkerboard_mask(4, 4, 0)
        assert m[0, 0, 0] == 1.0   # (0+0)%2==0 → A
        assert m[0, 1, 0] == 0.0   # (0+1)%2==1 → B
        assert m[1, 0, 0] == 0.0   # (1+0)%2==1 → B
        assert m[1, 1, 0] == 1.0   # (1+1)%2==0 → A

    def test_complement(self):
        m0 = checkerboard_mask(4, 4, 0)
        m1 = checkerboard_mask(4, 4, 1)
        assert jnp.allclose(m0 + m1, jnp.ones_like(m0))

    def test_half_sites(self):
        m = checkerboard_mask(8, 8, 0)
        assert jnp.sum(m) == 32  # half of 64


# ---------------------------------------------------------------------------
# grid coupling (standalone, outside Flax scope)
# ---------------------------------------------------------------------------

class TestGridCoupling:
    def _make_coupling_fn(self, L, C, key):
        """Create a coupling function with fixed params."""
        net = ScaleMaskNet(hidden_features=8, n_res_blocks=1, out_channels=C)
        dummy = jnp.zeros((1, L, L, C))
        params = net.init(key, dummy)
        return lambda z: net.apply(params, z), params

    def test_self_inverse(self):
        L, C = 4, 1
        key = jax.random.PRNGKey(0)
        x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(2, L, L, C))
        fn, _ = self._make_coupling_fn(L, C, jax.random.PRNGKey(1))

        y = _grid_coupling(fn, x, parity=0, use_ste=False)
        z = _grid_coupling(fn, y, parity=0, use_ste=False)
        assert jnp.allclose(z, x)

    def test_output_binary(self):
        L, C = 4, 1
        key = jax.random.PRNGKey(0)
        x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(3, L, L, C))
        fn, _ = self._make_coupling_fn(L, C, jax.random.PRNGKey(1))

        y = _grid_coupling(fn, x, parity=0, use_ste=False)
        assert jnp.all(jnp.abs(y) == 1.0)

    def test_a_sites_unchanged(self):
        L, C = 4, 1
        key = jax.random.PRNGKey(0)
        x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(2, L, L, C))
        fn, _ = self._make_coupling_fn(L, C, jax.random.PRNGKey(1))

        y = _grid_coupling(fn, x, parity=0, use_ste=False)
        a_mask = checkerboard_mask(L, L, 0)
        assert jnp.allclose(y * a_mask, x * a_mask)

    def test_multichannel_self_inverse(self):
        L, C = 4, 4
        key = jax.random.PRNGKey(0)
        x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(2, L, L, C))
        fn, _ = self._make_coupling_fn(L, C, jax.random.PRNGKey(1))

        y = _grid_coupling(fn, x, parity=0, use_ste=False)
        z = _grid_coupling(fn, y, parity=0, use_ste=False)
        assert y.shape == x.shape
        assert jnp.all(jnp.abs(y) == 1.0)
        assert jnp.allclose(z, x)

    def test_z2_equivariance(self):
        """With z2=True, coupling(-x) = -coupling(x)."""
        L, C = 4, 1
        key = jax.random.PRNGKey(0)
        x = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(2, L, L, C))
        fn, _ = self._make_coupling_fn(L, C, jax.random.PRNGKey(1))

        y_pos = _grid_coupling(fn, x, parity=0, use_ste=False, z2=True)
        y_neg = _grid_coupling(fn, -x, parity=0, use_ste=False, z2=True)
        assert jnp.allclose(y_neg, -y_pos)


# ---------------------------------------------------------------------------
# MultiScaleFlow (full module tests)
# ---------------------------------------------------------------------------

class TestMultiScaleFlow:
    @pytest.mark.parametrize("L,n_scales", [(4, 1), (8, 2), (8, 1), (16, 3)])
    def test_output_shape(self, L, n_scales):
        N = L * L
        flow = MultiScaleFlow(L=L, n_scales=n_scales, layers_per_scale=2,
                               hidden_features=8, n_res_blocks=1)
        params = flow.init(jax.random.PRNGKey(0), jnp.ones(N))

        # Unbatched
        assert flow.apply(params, jnp.ones(N)).shape == (N,)
        # Batched
        assert flow.apply(params, jnp.ones((5, N))).shape == (5, N)

    @pytest.mark.parametrize("L,n_scales", [(4, 1), (8, 2), (16, 3), (4, 2)])
    def test_bijective(self, L, n_scales):
        """forward(inverse(σ)) = σ and inverse(forward(z)) = z."""
        N = L * L
        flow = MultiScaleFlow(L=L, n_scales=n_scales, layers_per_scale=2,
                               hidden_features=8, n_res_blocks=1)
        params = flow.init(jax.random.PRNGKey(0), jnp.ones(N))

        z = jax.random.choice(jax.random.PRNGKey(1),
                              jnp.array([-1.0, 1.0]), shape=(4, N))

        sigma = flow.apply(params, z, use_ste=False)
        z_rec = flow.apply(params, sigma, use_ste=False, inverse=True)
        assert jnp.allclose(z_rec, z), f"max err: {jnp.max(jnp.abs(z_rec - z))}"

        sigma2 = flow.apply(params, z_rec, use_ste=False)
        assert jnp.allclose(sigma2, sigma)

    def test_output_binary(self):
        L, N = 8, 64
        flow = MultiScaleFlow(L=L, n_scales=2, layers_per_scale=2,
                               hidden_features=8, n_res_blocks=1)
        params = flow.init(jax.random.PRNGKey(0), jnp.ones(N))

        z = jax.random.choice(jax.random.PRNGKey(1),
                              jnp.array([-1.0, 1.0]), shape=(10, N))
        sigma = flow.apply(params, z, use_ste=False)
        assert jnp.all(jnp.abs(sigma) == 1.0)

    def test_nontrivial(self):
        """Flow with random init is not the identity."""
        L, N = 8, 64
        flow = MultiScaleFlow(L=L, n_scales=2, layers_per_scale=2,
                               hidden_features=8, n_res_blocks=1)
        params = flow.init(jax.random.PRNGKey(0), jnp.ones(N))

        z = jax.random.choice(jax.random.PRNGKey(1),
                              jnp.array([-1.0, 1.0]), shape=(N,))
        sigma = flow.apply(params, z, use_ste=False)
        assert not jnp.allclose(sigma, z)

    def test_different_inputs_different_outputs(self):
        L, N = 8, 64
        flow = MultiScaleFlow(L=L, n_scales=2, layers_per_scale=2,
                               hidden_features=8, n_res_blocks=1)
        params = flow.init(jax.random.PRNGKey(0), jnp.ones(N))

        z1 = jnp.ones(N)
        z2 = -jnp.ones(N)
        s1 = flow.apply(params, z1, use_ste=False)
        s2 = flow.apply(params, z2, use_ste=False)
        assert not jnp.allclose(s1, s2)

    def test_z2_equivariance(self):
        """With z2=True, flow(-z) = -flow(z)."""
        L, N = 8, 64
        flow = MultiScaleFlow(L=L, n_scales=2, layers_per_scale=2,
                               hidden_features=8, n_res_blocks=1, z2=True)
        params = flow.init(jax.random.PRNGKey(0), jnp.ones(N))

        z = jax.random.choice(jax.random.PRNGKey(1),
                              jnp.array([-1.0, 1.0]), shape=(3, N))
        s_pos = flow.apply(params, z, use_ste=False)
        s_neg = flow.apply(params, -z, use_ste=False)
        assert jnp.allclose(s_neg, -s_pos)

    def test_ste_matches_hard(self):
        """STE forward values match hard sign."""
        L, N = 8, 64
        flow = MultiScaleFlow(L=L, n_scales=2, layers_per_scale=2,
                               hidden_features=8, n_res_blocks=1)
        params = flow.init(jax.random.PRNGKey(0), jnp.ones(N))

        z = jax.random.choice(jax.random.PRNGKey(1),
                              jnp.array([-1.0, 1.0]), shape=(5, N))
        assert jnp.allclose(
            flow.apply(params, z, use_ste=True),
            flow.apply(params, z, use_ste=False),
        )

    def test_ste_gradients_exist(self):
        """STE allows non-zero gradients through the flow."""
        L, N = 4, 16
        flow = MultiScaleFlow(L=L, n_scales=1, layers_per_scale=2,
                               hidden_features=8, n_res_blocks=1)
        params = flow.init(jax.random.PRNGKey(0), jnp.ones(N))

        z = jax.random.choice(jax.random.PRNGKey(1),
                              jnp.array([-1.0, 1.0]), shape=(3, N))

        def loss_fn(p):
            return jnp.mean(flow.apply(p, z, use_ste=True))

        grads = jax.grad(loss_fn)(params)
        has_nonzero = any(jnp.any(g != 0) for g in jax.tree.leaves(grads))
        assert has_nonzero

    def test_layer_count(self):
        """Total coupling layers = (2*n_scales - 1) * layers_per_scale."""
        for n_scales, lps in [(1, 4), (2, 4), (3, 2)]:
            flow = MultiScaleFlow(L=16, n_scales=n_scales, layers_per_scale=lps)
            ops = flow._build_ops()
            n_couple = sum(1 for op in ops if op[0] == 'couple')
            assert n_couple == (2 * n_scales - 1) * lps

    def test_single_scale_no_squeeze(self):
        """n_scales=1 produces no squeeze/unsqueeze ops."""
        flow = MultiScaleFlow(L=8, n_scales=1, layers_per_scale=4)
        ops = flow._build_ops()
        assert all(op[0] == 'couple' for op in ops)
        assert len(ops) == 4

    def test_compatible_interface(self):
        """Same calling convention as DiscreteFlow."""
        L, N = 8, 64
        flow = MultiScaleFlow(L=L, n_scales=2, layers_per_scale=2,
                               hidden_features=8, n_res_blocks=1)
        params = flow.init(jax.random.PRNGKey(0), jnp.ones(N))

        z = jax.random.choice(jax.random.PRNGKey(1),
                              jnp.array([-1.0, 1.0]), shape=(10, N))

        # Forward (no STE) — evaluation
        sigma = flow.apply(params, z, use_ste=False)
        assert sigma.shape == (10, N)
        # Forward (STE) — flow gradient
        sigma_ste = flow.apply(params, z, use_ste=True)
        assert sigma_ste.shape == (10, N)
        # Inverse — diagnostics
        z_rec = flow.apply(params, sigma, use_ste=False, inverse=True)
        assert z_rec.shape == (10, N)
        assert jnp.allclose(z_rec, z)
