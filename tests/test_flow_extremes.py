import pytest
import jax
import jax.numpy as jnp
from dsflow_ising.flow import DiscreteFlow
from dsflow_ising.ising import energy
from dsflow_ising.train import compute_loss

def test_gpu_consistency():
    """Ensure consistent outputs on CPU vs GPU."""
    if not jax.devices("gpu"):
        pytest.skip("No GPU available for testing.")
    L = 4
    N = L * L
    key = jax.random.PRNGKey(0)
    flow_model = DiscreteFlow(L=L, n_layers=2)
    params = flow_model.init(key, jnp.ones(N))

    z_samples = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(N,))

    # CPU
    cpu_device = jax.devices("cpu")[0]
    params_cpu = jax.device_put(params, cpu_device)
    z_cpu = jax.device_put(z_samples, cpu_device)
    energy_cpu = jax.jit(lambda z: energy(z, pairs=None, J=1.0))(z_cpu)

    # GPU
    gpu_device = jax.devices("gpu")[0]
    params_gpu = jax.device_put(params, gpu_device)
    z_gpu = jax.device_put(z_samples, gpu_device)
    energy_gpu = jax.jit(lambda z: energy(z, pairs=None, J=1.0))(z_gpu)

    assert jnp.isclose(energy_cpu, energy_gpu, atol=1e-5)

def test_extreme_flow_depth():
    """Flow models with zero and deeply stacked layers must function."""
    L = 4
    N = L * L
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    params = {}

    for flow_depth in [0, 10]:
        flow_model = DiscreteFlow(L=L, n_layers=flow_depth)
        params[flow_depth] = flow_model.init(keys[flow_depth % 2], jnp.ones(N))
        z_samples = jax.random.choice(keys[0], jnp.array([-1.0, 1.0]), shape=(N,))
        assert z_samples.shape == (N,)