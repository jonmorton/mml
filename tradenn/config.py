from dataclasses import dataclass, field
from typing import Optional, Union

DAYS = 252


@dataclass
class PPOConfig:
    learning_rate: float = 3e-5
    n_steps: int = DAYS * 16
    batch_size: int = DAYS * 8
    n_epochs: int = 10
    gamma: float = 0.8
    gae_lambda: float = 0.7
    clip_range: float = 0.005
    clip_range_vf: Optional[float] = 0.03
    normalize_advantage: bool = True
    ent_coef: float = 0.8
    vf_coef: float = 1e-6
    max_grad_norm: float = 0.7
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    stats_window_size: int = 100
    log_std_init: float = 0.25


@dataclass
class SACConfig:
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000  # 1e6
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: Union[int, tuple[int, str]] = 1
    gradient_steps: int = 1
    optimize_memory_usage: bool = False
    ent_coef: Union[str, float] = "auto"
    target_update_interval: int = 1
    target_entropy: Union[str, float] = "auto"
    use_sde: bool = False
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = False


@dataclass
class NetworkConfig:
    hdim: int = 64
    asset_embed_dim: int = 32
    activation: str = "silu"
    conv_dim: int = 16
    policy_dim: int = 64
    value_dim: int = 128
    value_lr_mult: float = 15.0
    init_gain: float = 2.0
    action_proj_init_gain: float = 100.0
    value_proj_init_gain: float = 3.0


@dataclass
class Config:
    run_name: str = "trader"
    data_dir: str = "data"
    seed: int = 1337
    env: str = "eod"
    n_env: int = 8
    device: str = "cpu"

    initial_balance: float = 200000
    nb_stock: int = 10
    nb_days = DAYS
    hist_days: int = 32
    transaction_fee: float = 0.005
    timesteps_per_day: int = 4
    normalize_features: bool = True

    algorithm: str = "ppo_custom"
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    weight_decay: float = 0.0
    adam_beta1: float = 0.5
    adam_beta2: float = 0.99

    train_steps: int = 125000
    eval_episodes: int = 512

    @property
    def out_dir(self):
        return f"models/{self.run_name}"
