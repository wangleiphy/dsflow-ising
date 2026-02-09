"""Training loop: REINFORCE for base distribution (θ), STE for flow (φ).

The variational free energy is:
    F_var = ⟨E(σ)⟩_q + T ⟨ln p_θ(z)⟩_q

where q(σ) = p_θ(f_φ⁻¹(σ)), z ~ p_θ, σ = f_φ(z).

θ-gradient (REINFORCE):
    ∇_θ F_var = E_z[ (R(z) - b) ∇_θ ln p_θ(z) ]
    where R(z) = E(f_φ(z)) + T ln p_θ(z) and b is the batch mean (zero-bias baseline).

φ-gradient (STE):
    ∇_φ F_var ≈ ∇_φ E(f_φ(z))  (using straight-through estimator through sign)
    The entropy term T ln p_θ(z) doesn't depend on φ, so only energy matters.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from dsflow_ising.ising import nearest_neighbor_pairs, energy, magnetization
from dsflow_ising.made import MADE, log_prob, sample
from dsflow_ising.flow import DiscreteFlow
from dsflow_ising.multiscale_flow import MultiScaleFlow
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
    """Compute variational free energy F_var = mean(E(σ) + T * ln p_θ(z))."""
    sigma = flow_model.apply(flow_params, z_samples, use_ste=False)
    energies = jax.vmap(lambda s: energy(s, pairs, J))(sigma)
    rewards = energies + T * z_log_probs
    return jnp.mean(rewards), energies


def make_train_step(made_model, flow_model, pairs, J, T, batch_size,
                    made_optimizer, flow_optimizer, z2=False,
                    beta_anneal=0.0):
    """Create a JIT-compiled training step function.

    When beta_anneal > 0, uses annealed inverse temperature:
        beta_eff = (1/T) * (1 - beta_anneal^step)
    This starts training at high temperature (easy) and gradually cools
    to the target T, matching the VAN beta-annealing schedule.

    Returns a function: (state, key) -> (state, metrics)
    """
    beta_target = 1.0 / T

    def _step(state, key):
        made_params, flow_params, made_opt, flow_opt, baseline, step = state

        # Annealed effective temperature
        if beta_anneal > 0:
            beta_eff = beta_target * (1.0 - beta_anneal ** (step + 1))
            T_eff = 1.0 / jnp.maximum(beta_eff, 1e-10)
        else:
            T_eff = T

        z_samples, z_log_probs = sample(
            made_model, made_params, key, num_samples=batch_size, z2=z2)

        sigma = flow_model.apply(flow_params, z_samples, use_ste=False)
        energies = jax.vmap(lambda s: energy(s, pairs, J))(sigma)
        rewards = energies + T_eff * z_log_probs
        f_var = jnp.mean(rewards)

        advantage = rewards - jnp.mean(rewards)

        def made_reinforce_loss(mp):
            lps = jax.vmap(lambda z: log_prob(made_model, mp, z, z2=z2))(z_samples)
            return jnp.mean(jax.lax.stop_gradient(advantage) * lps)

        made_grads = jax.grad(made_reinforce_loss)(made_params)
        made_updates, new_made_opt = made_optimizer.update(made_grads, made_opt, made_params)
        new_made_params = optax.apply_updates(made_params, made_updates)

        def flow_loss_fn(fp):
            sigma_ste = flow_model.apply(fp, z_samples, use_ste=True)
            e = jax.vmap(lambda s: energy(s, pairs, J))(sigma_ste)
            return jnp.mean(e)

        flow_grads = jax.grad(flow_loss_fn)(flow_params)
        flow_updates, new_flow_opt = flow_optimizer.update(flow_grads, flow_opt, flow_params)
        new_flow_params = optax.apply_updates(flow_params, flow_updates)

        # Report F_var at the *target* temperature for fair comparison
        f_var_target = jnp.mean(energies + T * z_log_probs)

        new_state = TrainState(
            made_params=new_made_params,
            flow_params=new_flow_params,
            made_opt_state=new_made_opt,
            flow_opt_state=new_flow_opt,
            baseline=f_var_target,
            step=step + 1,
        )
        metrics = {
            'f_var': f_var_target,
            'energy': jnp.mean(energies),
            'entropy': -jnp.mean(z_log_probs),
            'baseline': f_var_target,
            'mag': jnp.mean(sigma),
            'T_eff': T_eff,
        }
        return new_state, metrics

    return jax.jit(_step)


def init_train_state(model_cfg: ModelConfig, train_cfg: TrainConfig):
    """Initialize all models, parameters, and optimizers."""
    key = jax.random.PRNGKey(train_cfg.seed)
    N = model_cfg.L ** 2
    hidden_dims = model_cfg.made_hidden_dims if model_cfg.made_hidden_dims else (4 * N,)

    made_model = MADE(n_sites=N, hidden_dims=hidden_dims)
    key, subkey = jax.random.split(key)
    made_params = made_model.init(subkey, jnp.ones(N))

    if model_cfg.flow_type == "multiscale":
        flow_model = MultiScaleFlow(
            L=model_cfg.L,
            n_scales=model_cfg.n_scales,
            layers_per_scale=model_cfg.layers_per_scale,
            hidden_features=model_cfg.ms_hidden_features,
            n_res_blocks=model_cfg.ms_n_res_blocks,
            z2=model_cfg.z2,
        )
    else:
        flow_model = DiscreteFlow(
            L=model_cfg.L,
            n_layers=model_cfg.n_flow_layers,
            mask_features=model_cfg.mask_features,
            z2=model_cfg.z2,
        )
    key, subkey = jax.random.split(key)
    flow_params = flow_model.init(subkey, jnp.ones(N))

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
               made_optimizer, flow_optimizer, state, key, z2=False,
               beta_anneal=0.0):
    """Single training step (convenience wrapper; builds and calls JIT step)."""
    step_fn = make_train_step(
        made_model, flow_model, pairs, J, T, batch_size,
        made_optimizer, flow_optimizer, z2=z2, beta_anneal=beta_anneal,
    )
    return step_fn(state, key)


def train(model_cfg: ModelConfig, train_cfg: TrainConfig,
          log_every: int = 100, log_file: str = None,
          use_wandb: bool = False, wandb_project: str = "dsflow-ising",
          wandb_name: str = None):
    """Full training loop."""
    if use_wandb:
        import wandb
        from dataclasses import asdict
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={**asdict(model_cfg), **asdict(train_cfg)},
        )

    made_model, flow_model, state, pairs, made_opt, flow_opt = init_train_state(
        model_cfg, train_cfg
    )

    z2 = model_cfg.z2
    key = jax.random.PRNGKey(train_cfg.seed + 1)
    history = []

    # Create JIT-compiled training step (built once, reused every iteration)
    step_fn = make_train_step(
        made_model, flow_model, pairs, train_cfg.J, train_cfg.T,
        train_cfg.batch_size, made_opt, flow_opt, z2=z2,
        beta_anneal=train_cfg.beta_anneal,
    )

    fh = None
    if log_file:
        fh = open(log_file, 'w')
        fh.write(f"# L={model_cfg.L} T={train_cfg.T} J={train_cfg.J} beta_anneal={train_cfg.beta_anneal}\n")
        fh.write("step,f_var,energy,entropy,baseline,mag,T_eff\n")

    try:
        for i in range(train_cfg.num_steps):
            key, subkey = jax.random.split(key)
            state, metrics = step_fn(state, subkey)
            history.append(metrics)

            if fh:
                fh.write(f"{i+1},{float(metrics['f_var']):.6f},"
                         f"{float(metrics['energy']):.6f},"
                         f"{float(metrics['entropy']):.6f},"
                         f"{float(metrics['baseline']):.6f},"
                         f"{float(metrics['mag']):.6f},"
                         f"{float(metrics['T_eff']):.6f}\n")
                fh.flush()

            if use_wandb:
                wandb.log({
                    'f_var': float(metrics['f_var']),
                    'energy': float(metrics['energy']),
                    'entropy': float(metrics['entropy']),
                    'baseline': float(metrics['baseline']),
                }, step=i + 1)

            if (i + 1) % log_every == 0:
                print(f"Step {i+1}: F_var={metrics['f_var']:.4f}, "
                      f"E={metrics['energy']:.4f}, S={metrics['entropy']:.4f}, "
                      f"T_eff={float(metrics['T_eff']):.4f}")

        if use_wandb:
            from dsflow_ising.diagnostics import (
                variational_free_energy, base_entropy,
                conditional_entropy_profile, layer_free_energy_reduction,
            )
            diag_key = jax.random.PRNGKey(train_cfg.seed + 999)
            k1, k2, k3, k4 = jax.random.split(diag_key, 4)

            f_final = variational_free_energy(
                made_model, state.made_params, flow_model, state.flow_params,
                pairs, train_cfg.J, train_cfg.T, k1, num_samples=1000,
            )
            h_final = base_entropy(made_model, state.made_params, k2, num_samples=1000)
            cond_ent = conditional_entropy_profile(
                made_model, state.made_params, k3, num_samples=1000,
            )
            layer_df = layer_free_energy_reduction(
                made_model, state.made_params, flow_model, state.flow_params,
                pairs, train_cfg.J, train_cfg.T, k4, num_samples=1000,
            )

            N = model_cfg.L ** 2
            wandb.summary['F_var_final'] = float(f_final)
            wandb.summary['F_var_per_site'] = float(f_final) / N
            wandb.summary['base_entropy'] = float(h_final)

            cond_data = [[k, float(cond_ent[k])] for k in range(len(cond_ent))]
            wandb.log({
                'conditional_entropy_profile': wandb.Table(
                    data=cond_data, columns=['site', 'H(z_k|z_<k)']),
            })
            layer_data = [[l, float(layer_df[l])] for l in range(len(layer_df))]
            wandb.log({
                'layer_free_energy_reduction': wandb.Table(
                    data=layer_data, columns=['layer', 'delta_F']),
            })
    finally:
        if fh:
            fh.close()
        if use_wandb:
            wandb.finish()

    return state, history, made_model, flow_model, pairs

