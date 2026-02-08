"""Configuration dataclasses."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    L: int = 8
    n_flow_layers: int = 4
    mask_features: tuple = (16, 16)
    made_hidden_dims: tuple = ()  # empty means (4*N,)
    z2: bool = False  # Z2 spin-flip symmetry


@dataclass
class TrainConfig:
    T: float = 2.269  # Near Tc for 2D Ising
    J: float = 1.0
    batch_size: int = 256
    lr_theta: float = 1e-3
    lr_phi: float = 1e-3
    num_steps: int = 10000
    seed: int = 42
    beta_anneal: float = 0.0  # Annealing rate: beta_eff = beta*(1 - beta_anneal**step), 0 = disabled

