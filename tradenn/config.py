from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PPOConfig:
    learning_rate: float = 2e-5
    n_steps: int = 252
    batch_size: int = 252 // 2
    n_epochs: int = 16
    gamma: float = 0.8
    gae_lambda: float = 0.9
    clip_range: float = 0.02
    clip_range_vf: Optional[float] = 1.7
    normalize_advantage: bool = True
    ent_coef: float = 0.001
    vf_coef: float = 0.934
    max_grad_norm: float = 0.22
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    stats_window_size: int = 100


@dataclass
class PolicyConfig:
    full_std: bool = True
    use_expln: bool = False
    squash_output: bool = False
    activation_fn: str = "silu"
    critic_dim: int = 128
    actor_dim: int = 224


@dataclass
class Config:
    run_name: str = "trader"
    data_dir: str = "data"

    turbulence_threshold: int = 140
    initial_balance: float = 1000000
    nb_stock: int = 20
    transaction_fee: float = 0.01
    slippage: float = 0.0
    time_steps_per_day: int = 8

    reward_scaling: float = 1.0
    drawdown_penalty: float = 0.1
    seed: int = 42

    algorithm: str = "ppo"
    ppo: PPOConfig = field(default_factory=PPOConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    weight_decay: float = 0.01
    adam_beta1: float = 0.84
    adam_beta2: float = 0.919

    normalize_features: bool = True

    train_steps: int = 100000
    eval_episodes: int = 128
    n_env: int = 8

    device: str = "cpu"

    @property
    def out_dir(self):
        return f"models/{self.run_name}"
