from dataclasses import dataclass, field
from typing import Optional

STEPS = 252


@dataclass
class PPOConfig:
    learning_rate: float = 3e-5
    n_steps: int = STEPS
    batch_size: int = STEPS * 2
    n_epochs: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    stats_window_size: int = 100


@dataclass
class PolicyConfig:
    full_std: bool = True
    use_expln: bool = True
    squash_output: bool = False
    activation_fn: str = "relu"
    critic_dim: int = 64
    actor_dim: int = 64


@dataclass
class Config:
    run_name: str = "trader"
    data_dir: str = "data"

    env: str = "v1"

    initial_balance: float = 200000
    nb_stock: int = 20
    transaction_fee: float = 0.005
    slippage: float = 0.0
    timesteps_per_day: int = 4

    drawdown_penalty: float = 0.1
    seed: int = 42

    algorithm: str = "ppo"
    ppo: PPOConfig = field(default_factory=PPOConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    weight_decay: float = 0.00001
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95

    normalize_features: bool = True

    train_steps: int = 200000
    eval_episodes: int = 256
    n_env: int = 8

    device: str = "cpu"

    @property
    def out_dir(self):
        return f"models/{self.run_name}"
