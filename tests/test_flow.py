"""Tests for the full discrete flow (stacked coupling layers)."""

import jax
import jax.numpy as jnp
import pytest
from itertools import product

from dsflow_ising.flow import DiscreteFlow, get_partition


@pytest.fixture
def flow_4x4():
    """4x4 flow with 4 layers."""
    L = 4
    model = DiscreteFlow(L=L, n_layers=4, mask_features=(8, 8))
    key = jax.random.PRNGKey(42)
    z_dummy = jnp.ones(L * L)
    params = model.init(key, z_dummy)
    return model, params, L


@pytest.fixture
def flow_4x4_z2():
    """4x4 Z2-equivariant flow with 4 layers."""
    L = 4
    model = DiscreteFlow(L=L, n_layers=4, mask_features=(8, 8), z2=True)
    key = jax.random.PRNGKey(42)
    z_dummy = jnp.ones(L * L)
    params = model.init(key, z_dummy)
    return model, params, L


class TestBijectivity:
    def test_forward_inverse_roundtrip(self, flow_4x4):
        """flow.inverse(flow(z)) == z for random z."""
        model, params, L = flow_4x4
        key = jax.random.PRNGKey(0)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))

        sigma = model.apply(params, z, use_ste=False)
        z_rec = model.apply(params, sigma, use_ste=False, inverse=True)
        assert jnp.allclose(z_rec, z), f"Max diff: {jnp.max(jnp.abs(z_rec - z))}"

    def test_roundtrip_multiple_seeds(self, flow_4x4):
        """Bijectivity holds across multiple random parameter initializations."""
        L = 4
        for seed in range(5):
            key = jax.random.PRNGKey(seed * 100)
            model = DiscreteFlow(L=L, n_layers=4, mask_features=(8, 8))
            params = model.init(key, jnp.ones(L * L))
            z = jax.random.choice(jax.random.PRNGKey(seed), jnp.array([-1.0, 1.0]), shape=(L * L,))
            sigma = model.apply(params, z, use_ste=False)
            z_rec = model.apply(params, sigma, use_ste=False, inverse=True)
            assert jnp.allclose(z_rec, z), f"Failed for seed={seed}"

    def test_roundtrip_batched(self, flow_4x4):
        """Bijectivity works with batched inputs."""
        model, params, L = flow_4x4
        key = jax.random.PRNGKey(1)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(20, L * L))
        sigma = model.apply(params, z, use_ste=False)
        z_rec = model.apply(params, sigma, use_ste=False, inverse=True)
        assert jnp.allclose(z_rec, z)

    def test_exhaustive_permutation_2x2(self):
        """For 2x2 system (16 configs), flow output is a permutation of all configs."""
        L = 2
        N = L * L  # 4 sites, 16 configs
        model = DiscreteFlow(L=L, n_layers=4, mask_features=(4, 4))
        key = jax.random.PRNGKey(99)
        params = model.init(key, jnp.ones(N))

        # Enumerate all 2^4 configs
        all_configs = jnp.array(
            [list(c) for c in product([-1.0, 1.0], repeat=N)]
        )  # (16, 4)

        # Apply flow to each
        sigma_all = model.apply(params, all_configs, use_ste=False)

        # Verify all outputs are in {-1, +1}
        assert jnp.all((sigma_all == 1.0) | (sigma_all == -1.0))

        # Verify uniqueness (it's a permutation)
        # Convert to tuples for comparison
        sigma_set = set(tuple(s.tolist()) for s in sigma_all)
        assert len(sigma_set) == 2 ** N, \
            f"Expected {2**N} unique outputs, got {len(sigma_set)}"


class TestComposition:
    def test_alternating_partitions(self):
        """Verify layers alternate even/odd partitions."""
        assert get_partition(0) == "even"
        assert get_partition(1) == "odd"
        assert get_partition(2) == "even"
        assert get_partition(3) == "odd"

    def test_flow_changes_input(self, flow_4x4):
        """Flow should generally change the input (not identity)."""
        model, params, L = flow_4x4
        key = jax.random.PRNGKey(7)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))
        sigma = model.apply(params, z, use_ste=False)
        # It's extremely unlikely that a random flow is the identity
        assert not jnp.allclose(sigma, z), "Flow appears to be identity"

    def test_outputs_are_spins(self, flow_4x4):
        """All flow outputs should be in {-1, +1}."""
        model, params, L = flow_4x4
        key = jax.random.PRNGKey(8)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(50, L * L))
        sigma = model.apply(params, z, use_ste=False)
        assert jnp.all((sigma == 1.0) | (sigma == -1.0))


class TestZ2Equivariance:
    """Tests for Z2 equivariant flow: f(-z) = -f(z)."""

    def test_equivariance_single(self, flow_4x4_z2):
        """f(-z) == -f(z) for a single random configuration."""
        model, params, L = flow_4x4_z2
        key = jax.random.PRNGKey(0)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))

        fz = model.apply(params, z, use_ste=False)
        f_neg_z = model.apply(params, -z, use_ste=False)
        assert jnp.allclose(f_neg_z, -fz), \
            f"Z2 equivariance violated: max diff = {jnp.max(jnp.abs(f_neg_z + fz))}"

    def test_equivariance_batch(self, flow_4x4_z2):
        """f(-z) == -f(z) for a batch of random configurations."""
        model, params, L = flow_4x4_z2
        key = jax.random.PRNGKey(1)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(50, L * L))

        fz = model.apply(params, z, use_ste=False)
        f_neg_z = model.apply(params, -z, use_ste=False)
        assert jnp.allclose(f_neg_z, -fz), \
            f"Z2 equivariance violated in batch"

    def test_equivariance_multiple_seeds(self):
        """Z2 equivariance holds across different parameter initializations."""
        L = 4
        for seed in range(5):
            key = jax.random.PRNGKey(seed * 100)
            model = DiscreteFlow(L=L, n_layers=4, mask_features=(8, 8), z2=True)
            params = model.init(key, jnp.ones(L * L))
            z = jax.random.choice(jax.random.PRNGKey(seed), jnp.array([-1.0, 1.0]), shape=(L * L,))
            fz = model.apply(params, z, use_ste=False)
            f_neg_z = model.apply(params, -z, use_ste=False)
            assert jnp.allclose(f_neg_z, -fz), f"Failed for seed={seed}"

    def test_equivariance_exhaustive_2x2(self):
        """Z2 equivariance holds for all 16 configs of a 2x2 system."""
        L = 2
        N = L * L
        model = DiscreteFlow(L=L, n_layers=4, mask_features=(4, 4), z2=True)
        key = jax.random.PRNGKey(99)
        params = model.init(key, jnp.ones(N))

        all_configs = jnp.array(
            [list(c) for c in product([-1.0, 1.0], repeat=N)]
        )
        fz = model.apply(params, all_configs, use_ste=False)
        f_neg_z = model.apply(params, -all_configs, use_ste=False)
        assert jnp.allclose(f_neg_z, -fz), "Z2 equivariance violated exhaustively"

    def test_z2_flow_still_bijective(self, flow_4x4_z2):
        """Z2 equivariant flow is still a bijection."""
        model, params, L = flow_4x4_z2
        key = jax.random.PRNGKey(2)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))
        sigma = model.apply(params, z, use_ste=False)
        z_rec = model.apply(params, sigma, use_ste=False, inverse=True)
        assert jnp.allclose(z_rec, z)

    def test_z2_flow_outputs_are_spins(self, flow_4x4_z2):
        """Z2 flow outputs are in {-1, +1}."""
        model, params, L = flow_4x4_z2
        key = jax.random.PRNGKey(3)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(50, L * L))
        sigma = model.apply(params, z, use_ste=False)
        assert jnp.all((sigma == 1.0) | (sigma == -1.0))

    def test_non_z2_flow_breaks_equivariance(self, flow_4x4):
        """Standard (non-Z2) flow should generally NOT satisfy f(-z) = -f(z)."""
        model, params, L = flow_4x4
        key = jax.random.PRNGKey(0)
        z = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(L * L,))
        fz = model.apply(params, z, use_ste=False)
        f_neg_z = model.apply(params, -z, use_ste=False)
        # Very unlikely to be equivariant by accident
        assert not jnp.allclose(f_neg_z, -fz), \
            "Non-Z2 flow unexpectedly satisfies equivariance"

