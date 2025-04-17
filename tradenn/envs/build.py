import os
import pickle

import polars as pl
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

from tradenn.config import Config


def build_env(config: Config, is_train: bool = True):
    prefix = "eod.train" if is_train else "eod.val"

    df = pl.read_parquet(os.path.join(config.data_dir, f"{prefix}.df.parquet"))
    bid_ask = pl.read_parquet(
        os.path.join(config.data_dir, f"{prefix}.bid_ask.parquet")
    )

    # always use train stats
    with open(os.path.join(config.data_dir, "eod.train.feature_stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    if config.env == "eod":
        from tradenn.envs.env import StockEnv

        env_data = StockEnv.prepare_data(df, bid_ask, stats, config.normalize_features)
        builder = lambda: (  # noqa: E731
            StockEnv(config, env_data, is_train)
        )

    elif config.env == "simple":
        from tradenn.envs.simple import StockEnv

        builder = lambda: StockEnv(  # noqa: E731
            df,
            num_days=config.ppo.n_steps,
            transaction_fee=config.transaction_fee,
            initial_cash=config.initial_balance,
        )
    elif config.env == "simple_multi":
        from tradenn.envs.simple_multi import StockTradingEnv

        builder = lambda: StockTradingEnv(  # noqa: E731
            df,
            num_tickers=config.nb_stock,
        )
    else:
        raise ValueError(f"Unknown environment: {config.env}")

    if config.n_env > 1:
        n_env = config.n_env if is_train else max(1, config.n_env // 2)
        print(f"Spawning {n_env} environments for {'train' if is_train else 'eval'}")
        env = SubprocVecEnv([lambda: builder() for _ in range(n_env)])

    else:
        env = DummyVecEnv([lambda: builder()])

    # train_env = VecNormalize(
    #     train_env,
    #     gamma=config.ppo.gamma if "ppo" in config.algorithm else config.sac.gamma,
    # )
    # eval_env = VecNormalize(
    #     eval_env,
    #     gamma=config.ppo.gamma if "ppo" in config.algorithm else config.sac.gamma,
    # )
    return env
