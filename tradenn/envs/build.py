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


def build_envs(
    config: Config,
) -> tuple[VecEnv, VecEnv]:
    df = pl.read_parquet(os.path.join(config.data_dir, "eod.train.df.parquet"))
    bid_ask = pl.read_parquet(
        os.path.join(config.data_dir, "eod.train.bid_ask.parquet")
    )

    df_eval = pl.read_parquet(os.path.join(config.data_dir, "eod.val.df.parquet"))
    bid_ask_eval = pl.read_parquet(
        os.path.join(config.data_dir, "eod.val.bid_ask.parquet")
    )

    with open(os.path.join(config.data_dir, "eod.feature_stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    if config.env == "v1":
        from tradenn.envs.env import StockEnv

        env_data_train = StockEnv.prepare_data(
            df, bid_ask, stats, config.normalize_features
        )
        env_data_eval = StockEnv.prepare_data(
            df_eval,
            bid_ask_eval,
            stats,
            config.normalize_features,
        )

        builder = lambda train: (  # noqa: E731
            StockEnv(config, env_data_train, True)
            if train
            else StockEnv(config, env_data_eval, False)
        )

    elif config.env == "v2":
        from tradenn.envs.env2 import StockEnv

        env_data_train = StockEnv.prepare_data(
            df, bid_ask, stats, config.normalize_features
        )
        env_data_eval = StockEnv.prepare_data(
            df_eval,
            bid_ask_eval,
            stats,
            config.normalize_features,
        )

        builder = lambda train: (  # noqa: E731
            StockEnv(config, env_data_train, True)
            if train
            else StockEnv(config, env_data_eval, False)
        )

    elif config.env == "simple":
        from tradenn.envs.simple import StockEnv

        builder = lambda train: StockEnv(  # noqa: E731
            df if train else df_eval,
            num_days=config.ppo.n_steps,
            transaction_fee=config.transaction_fee,
            initial_cash=config.initial_balance,
        )
    elif config.env == "simple_multi":
        from tradenn.envs.simple_multi import StockTradingEnv

        builder = lambda train: StockTradingEnv(  # noqa: E731
            df if train else df_eval,
            num_tickers=config.nb_stock,
        )
    else:
        raise ValueError(f"Unknown environment: {config.env}")

    if config.n_env > 1:
        print(
            f"Spawning {config.n_env} train and {max(1, config.n_env // 2)} eval environments"
        )
        train_env = SubprocVecEnv([lambda: builder(True) for _ in range(config.n_env)])
        eval_env = SubprocVecEnv(
            [lambda: builder(False) for _ in range(max(1, config.n_env // 2))]
        )

    else:
        train_env = DummyVecEnv([lambda: builder(True)])
        eval_env = DummyVecEnv([lambda: builder(False)])

    train_env = VecNormalize(
        train_env, True, norm_obs=False, clip_obs=100, gamma=config.ppo.gamma
    )
    eval_env = VecNormalize(
        eval_env, False, norm_obs=False, clip_obs=100, gamma=config.ppo.gamma
    )
    return train_env, eval_env
