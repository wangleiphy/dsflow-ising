"""Training loop: REINFORCE for base distribution (θ), STE for flow (φ).

The variational free energy is:
    F_var = ⟨E(σ)⟩_q + T ⟨ln p_θ(z)⟩_q

where q(σ) = p_θ(f_φ⁻¹(σ)), z ~ p_θ, σ = f_φ(z).

θ-gradient (REINFORCE):
    ∇_θ F_var = E_z[ (R(z) - b) ∇_θ ln p_θ(z) ]
    where R(z) = E(f_φ(z)) + T ln p_θ(z) and b is a running baseline.

φ-gradient (STE):
    ∇_φ F_var ≈ ∇_φ E(f_φ(z))  (using straight-through estimator through sign)
    The entropy term T ln p_θ(z) doesn't depend on φ, so only energy matters.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from dsflow_ising.ising import nearest_neighbor_pairs, energy
from dsflow_ising.made import MADE, log_prob, sample
from dsflow_ising.flow import DiscreteFlow
from dsflow_ising.config import ModelConfig, TrainConfig


class TrainState(NamedTuple):
    made_params: dict
    flow_params: dict
    made_opt_state: optax.OptState
    flow_opt_state: optax.OptState
    baseline: float
    step: int


def compute_loss(made_model, made_params, flow_model, flow_params,
                 z_samples, z_log_probs, pairs, J, T):
    """Compute variational free energy F_var = mean(E(σ) + T * ln p_θ(z)).

    Args:
        made_model, made_params: base distribution
        flow_model, flow_params: discrete flow
        z_samples: samples from p_θ, shape (batch, N)
        z_log_probs: ln p_θ(z) for each sample, shape (batch,)
        pairs: nearest-neighbor bond indices
        J: coupling constant
        T: temperature

    Returns:
        F_var: scalar, mean variational free energy per sample
        energies: shape (batch,), E(σ) per sample
    """
    sigma = flow_model.apply(flow_params, z_samples, use_ste=False)
    energies = jax.vmap(lambda s: energy(s, pairs, J))(sigma)
    rewards = energies + T * z_log_probs
    return jnp.mean(rewards), energies


def make_train_step(made_model, flow_model, pairs, J, T, made_optimizer, flow_optimizer):
    """Create a JIT-compiled training step function.

    Returns a function: (state, key) -> (state, metrics)
    """

    def train_step(state, key):
        made_params, flow_params, made_opt, flow_opt, baseline, step = state

        # Sample from base distribution
        z_samples, z_log_probs = sample(made_model, made_params, key, num_samples=z_log_probs_shape)

        # --- REINFORCE gradient for θ (made_params) ---
        # R(z) = E(f_φ(z)) + T * ln p_θ(z)
        sigma = flow_model.apply(flow_params, z_samples, use_ste=False)
        energies = jax.vmap(lambda s: energy(s, pairs, J))(sigma)
        rewards = energies + T * z_log_probs
        f_var = jnp.mean(rewards)

        # Advantage with baseline
        advantage = rewards - baseline

        # ∇_θ F_var ≈ mean( advantage * ∇_θ ln p_θ(z) )
        def made_loss_fn(mp):
            lps = jax.vmap(lambda z: log_prob(made_model, mp, z))(z_samples)
            return jnp.mean(jax.lax.stop_gradient(advantage) * lps) + T * jnp.mean(lps)

        # Actually: F_var = E[E(σ)] + T * E[ln p_θ(z)]
        # ∇_θ F_var = E[ (E(σ) + T ln p_θ(z) - b) ∇_θ ln p_θ(z) ] + T * E[∇_θ ln p_θ(z)]
        # But the second term is zero in expectation (score function identity).
        # So REINFORCE: ∇_θ F_var ≈ mean( advantage * ∇_θ ln p_θ(z) )
        # But we can also use the "reparameterization" through the log-prob itself:
        # The clean way is: loss_θ = mean( stop_grad(advantage) * ln p_θ(z) )

        def made_reinforce_loss(mp):
            lps = jax.vmap(lambda z: log_prob(made_model, mp, z))(z_samples)
            return jnp.mean(jax.lax.stop_gradient(advantage) * lps)

        made_grads = jax.grad(made_reinforce_loss)(made_params)
        made_updates, new_made_opt = made_optimizer.update(made_grads, made_opt, made_params)
        new_made_params = optax.apply_updates(made_params, made_updates)

        # --- STE gradient for φ (flow_params) ---
        # Only the energy term depends on φ: ∇_φ E(f_φ(z))
        # Using STE through the sign function
        def flow_loss_fn(fp):
            sigma_ste = flow_model.apply(fp, z_samples, use_ste=True)
            e = jax.vmap(lambda s: energy(s, pairs, J))(sigma_ste)
            return jnp.mean(e)

        flow_grads = jax.grad(flow_loss_fn)(flow_params)
        flow_updates, new_flow_opt = flow_optimizer.update(flow_grads, flow_opt, flow_params)
        new_flow_params = optax.apply_updates(flow_params, flow_updates)

        # Update baseline (exponential moving average)
        new_baseline = 0.99 * baseline + 0.01 * f_var

        new_state = TrainState(
            made_params=new_made_params,
            flow_params=new_flow_params,
            made_opt_state=new_made_opt,
            flow_opt_state=new_flow_opt,
            baseline=new_baseline,
            step=step + 1,
        )
        metrics = {
            'f_var': f_var,
            'energy': jnp.mean(energies),
            'entropy': -jnp.mean(z_log_probs),
            'baseline': baseline,
        }
        return new_state, metrics

    # This closure needs batch_size, so we'll use a different approach.
    # Return a factory function instead.
    return train_step


def init_train_state(model_cfg: ModelConfig, train_cfg: TrainConfig):
    """Initialize all models, parameters, and optimizers.

    Returns:
        (made_model, flow_model, state, pairs, made_opt, flow_opt)
    """
    key = jax.random.PRNGKey(train_cfg.seed)
    N = model_cfg.L ** 2
    hidden_dim = model_cfg.made_hidden_dim if model_cfg.made_hidden_dim > 0 else 4 * N

    # Initialize MADE
    made_model = MADE(n_sites=N, hidden_dims=(hidden_dim,))
    key, subkey = jax.random.split(key)
    made_params = made_model.init(subkey, jnp.ones(N))

    # Initialize flow
    flow_model = DiscreteFlow(
        L=model_cfg.L,
        n_layers=model_cfg.n_flow_layers,
        mask_features=model_cfg.mask_features,
    )
    key, subkey = jax.random.split(key)
    flow_params = flow_model.init(subkey, jnp.ones(N))

    # Optimizers
    made_opt = optax.adam(train_cfg.lr_theta)
    flow_opt = optax.adam(train_cfg.lr_phi)
    made_opt_state = made_opt.init(made_params)
    flow_opt_state = flow_opt.init(flow_params)

    pairs = nearest_neighbor_pairs(model_cfg.L)

    state = TrainState(
        made_params=made_params,
        flow_params=flow_params,
        made_opt_state=made_opt_state,
        flow_opt_state=flow_opt_state,
        baseline=0.0,
        step=0,
    )
    return made_model, flow_model, state, pairs, made_opt, flow_opt


def train_step(made_model, flow_model, pairs, J, T, batch_size,
               made_optimizer, flow_optimizer, state, key):
    """Single training step.

    Returns:
        new_state: updated TrainState
        metrics: dict with f_var, energy, entropy, baseline
    """
    made_params, flow_params, made_opt, flow_opt, baseline, step = state

    # Sample from base distribution
    z_samples, z_log_probs = sample(made_model, made_params, key, num_samples=batch_size)

    # Compute energies and rewards
    sigma = flow_model.apply(flow_params, z_samples, use_ste=False)
    energies = jax.vmap(lambda s: energy(s, pairs, J))(sigma)
    rewards = energies + T * z_log_probs
    f_var = jnp.mean(rewards)
    advantage = rewards - baseline

    # REINFORCE gradient for θ
    def made_reinforce_loss(mp):
        lps = jax.vmap(lambda z: log_prob(made_model, mp, z))(z_samples)
        return jnp.mean(jax.lax.stop_gradient(advantage) * lps)

    made_grads = jax.grad(made_reinforce_loss)(made_params)
    made_updates, new_made_opt = made_optimizer.update(made_grads, made_opt, made_params)
    new_made_params = optax.apply_updates(made_params, made_updates)

    # STE gradient for φ (only energy depends on flow params)
    def flow_loss_fn(fp):
        sigma_ste = flow_model.apply(fp, z_samples, use_ste=True)
        e = jax.vmap(lambda s: energy(s, pairs, J))(sigma_ste)
        return jnp.mean(e)

    flow_grads = jax.grad(flow_loss_fn)(flow_params)
    flow_updates, new_flow_opt = flow_optimizer.update(flow_grads, flow_opt, flow_params)
    new_flow_params = optax.apply_updates(flow_params, flow_updates)

    new_baseline = 0.99 * baseline + 0.01 * f_var

    new_state = TrainState(
        made_params=new_made_params,
        flow_params=new_flow_params,
        made_opt_state=new_made_opt,
        flow_opt_state=new_flow_opt,
        baseline=new_baseline,
        step=step + 1,
    )
    metrics = {
        'f_var': f_var,
        'energy': jnp.mean(energies),
        'entropy': -jnp.mean(z_log_probs),
        'baseline': baseline,
    }
    return new_state, metrics


def train(model_cfg: ModelConfig, train_cfg: TrainConfig,
          log_every: int = 100, log_file: str = None):
    """Full training loop.

    Args:
        model_cfg: model configuration
        train_cfg: training configuration
        log_every: print to stdout every N steps
        log_file: if provided, write CSV log (step, f_var, energy, entropy, baseline)

    Returns:
        state: final TrainState
        history: list of metrics dicts
    """
    made_model, flow_model, state, pairs, made_opt, flow_opt = init_train_state(
        model_cfg, train_cfg
    )

    key = jax.random.PRNGKey(train_cfg.seed + 1)
    history = []

    fh = None
    if log_file:
        fh = open(log_file, 'w')
        fh.write("step,f_var,energy,entropy,baseline\n")

    try:
        for i in range(train_cfg.num_steps):
            key, subkey = jax.random.split(key)
            state, metrics = train_step(
                made_model, flow_model, pairs, train_cfg.J, train_cfg.T,
                train_cfg.batch_size, made_opt, flow_opt, state, subkey,
            )
            history.append(metrics)

            if fh:
                fh.write(f"{i+1},{float(metrics['f_var']):.6f},"
                         f"{float(metrics['energy']):.6f},"
                         f"{float(metrics['entropy']):.6f},"
                         f"{float(metrics['baseline']):.6f}\n")
                fh.flush()

            if (i + 1) % log_every == 0:
                print(f"Step {i+1}: F_var={metrics['f_var']:.4f}, "
                      f"E={metrics['energy']:.4f}, S={metrics['entropy']:.4f}")
    finally:
        if fh:
            fh.close()

    return state, history, made_model, flow_model, pairs
