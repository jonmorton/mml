from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    n_steps: int = 252
    batch_size: int = 252 // 4
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 1.0
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = True
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    stats_window_size: int = 100


@dataclass
class Config:
    run_name: str = "trader"
    data_dir: str = "data"

    turbulence_threshold: int = 140
    hmax: int = 100
    initial_balance: float = 1000000
    nb_stock: int = 10
    transaction_fee: float = 0.01
    slippage: float = 0.0
    time_steps_per_day: int = 8

    reward_scaling: float = 1.0
    seed: int = 42

    algorithm: str = "ppo"
    ppo: PPOConfig = field(default_factory=PPOConfig)
    weight_decay: float = 0.01

    train_steps: int = 50000
    eval_episodes: int = 64
    n_env: int = 8

    device: str = "cpu"

    @property
    def out_dir(self):
        return f"models/{self.run_name}"
