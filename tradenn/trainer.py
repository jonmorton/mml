import json
import os
import pickle
import warnings
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import polars as pl
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from torch.utils.tensorboard.writer import SummaryWriter

from tradenn.config import Config
from tradenn.env import StockEnv
from tradenn.networks import AssetTablePolicy


def create_env(config: Config) -> VecEnv:
    df = pl.read_parquet(os.path.join(config.data_dir, "eod.df.parquet"))
    bid_ask = pl.read_parquet(os.path.join(config.data_dir, "eod.bid_ask.parquet"))
    with open(os.path.join(config.data_dir, "eod.feature_stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    def env_factory():
        return StockEnv(config, df, bid_ask, stats, train=True)

    if config.n_env > 1:
        print(f"Spawning {config.n_env} environments")
        env = SubprocVecEnv([lambda: env_factory() for _ in range(config.n_env)])
    else:
        env = DummyVecEnv([lambda: env_factory()])
    return env


def create_agent(config: Config, env: VecEnv) -> OnPolicyAlgorithm:
    if config.algorithm == "ppo":
        agent = SB3_PPO(
            # policy=AssetTablePolicy,
            policy="MlpPolicy",
            env=env,
            device=torch.device(config.device),
            tensorboard_log=f"{config.out_dir}/tb",
            **asdict(config.ppo),
            verbose=1,
        )
    elif config.algorithm == "recurrent_ppo":
        agent = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            device=torch.device(config.device),
            tensorboard_log=f"{config.out_dir}/tb",
            **asdict(config.ppo),
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    return agent


def run_episodes(
    agent, n_episodes=1024, n_steps=364, tb: Optional[SummaryWriter] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a batch of episodes for the given agent.

    Parameters:
    - agent (object): The agent instance being trained.
    - batch_size (int): The number of episodes to run in the batch. Default is 1.
    - n_stpes (int): The maximum number of iterations (steps) allowed for each episode. Default is 10000.
    """

    env: VecEnv = agent.env
    cum_rewards = []
    end_assets = []
    num_trades = []

    if n_episodes % env.num_envs != 0:
        warnings.warn(
            f"Number of episodes ({n_episodes}) is not a multiple of the number of environments ({env.num_envs}). Rouding up to the nearest multiple."
        )
        n_episodes = (n_episodes // env.num_envs + 1) * env.num_envs

    for ep in range(max(1, n_episodes // env.num_envs)):
        s = env.reset()

        a, _ = agent.predict(s, deterministic=True)
        r_ep = 0

        done = np.zeros(env.num_envs, dtype=bool)

        # while not (termination or truncation)
        while not done.all():
            obs, reward, done, infos = env.step(a)

            r_ep += reward
            a, _ = agent.predict(obs, deterministic=True)

        cum_rewards.append(r_ep)
        end_assets.append(np.array([i["assets"] for i in infos]))
        num_trades.append(np.array([i["trades"] for i in infos]))
        returns = np.array([i["returns"] for i in infos])
        cagr = np.array([i["cagr"] for i in infos])

        if tb is not None:
            tb.add_scalar("eval/reward", np.mean(r_ep), ep)
            tb.add_scalar("eval/returns", np.mean(returns), ep)
            tb.add_scalar("eval/cagr", np.mean(cagr), ep)

    rewards = np.concatenate(cum_rewards)
    end_assets = np.concatenate(end_assets)
    num_trades = np.concatenate(num_trades)

    return rewards, end_assets, num_trades


class TBCallback(BaseCallback):
    """
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    """

    def _on_training_start(self):
        self._log_freq = 10  # log every 10 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _on_step(self) -> bool:
        """
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        """
        if self.n_calls % self._log_freq == 0:
            rewards = np.array(self.training_env.get_attr("asset_memory"))

            self.tb_formatter.writer.add_scalar(
                "train/mean_returns",
                np.mean(rewards[:, -1] / rewards[:, 0]),
                self.num_timesteps,
            )
        return True


def train(
    config: Config,
    env: VecEnv,
):
    """
    Run multiple trials for training the agent on the given environment and save the resulting returns
    (and models if a model save path is specified).

    Parameters:
    - agent (PPO): Class of the agent to be trained.
    - env (object): Environment for training the agent.
    - run_name (str): Checkpoint save location
    - cofnig (PPOConfig): Configuration for the PPO algorithm.

    Returns:
    - reward_arr_train (array): Array containing the episodic returns for each episode in the last trial.
    """
    os.makedirs(config.out_dir, exist_ok=True)

    agent = create_agent(config, env)

    print("Training agent...")

    agent.learn(total_timesteps=config.train_steps, callback=TBCallback())

    print("Saving...")
    agent.save(f"{config.out_dir}/agent.pth")

    return agent


def evaluate(config: Config, agent: Any, env: VecEnv):
    tb = SummaryWriter(agent.logger.dir)

    prev_mode = agent.policy.training
    agent.policy.set_training_mode(False)

    print("Testing...")
    # test episode
    rewards, end_assets, num_trades = run_episodes(
        agent, config.eval_episodes, config.ppo.n_steps, tb
    )

    returns = (end_assets.astype(np.float32) / config.initial_balance) * 100
    cagrs = (
        (end_assets.astype(np.float32) / config.initial_balance)
        ** (1 / (config.ppo.n_steps / 252))
        - 1
    ) * 100

    tb.add_histogram("eval/rewards", torch.from_numpy(rewards).flatten())
    tb.add_histogram("eval/returns", torch.from_numpy(returns).flatten())
    tb.add_histogram("eval/cagrs", torch.from_numpy(cagrs).flatten())
    tb.add_histogram("eval/trades", torch.from_numpy(num_trades).flatten())

    print("Eval rewards: ", np.mean(rewards), "±", np.std(rewards))
    print("Eval returns: ", np.mean(returns), "±", np.std(returns))
    print("Eval CAGR: ", np.mean(cagrs), "±", np.std(cagrs))
    print("Eval trades: ", np.mean(num_trades), "±", np.std(num_trades))

    tb.add_scalar("eval/mean_return", np.mean(returns))
    tb.add_scalar("eval/mean_cagr", np.mean(cagrs))
    tb.add_scalar("eval/mean_trades", np.mean(num_trades))

    tb.close()

    agent.policy.set_training_mode(prev_mode)
