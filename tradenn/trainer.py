import json
import os
import pickle
import traceback
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

from tradenn.config import Config, PPOConfig
from tradenn.env import StockEnv
from tradenn.policy import TraderPolicy


def make_lr_schedule(max_lr: float):
    def schedule(progress):
        # linear warmup sqrt decay schedule

        if progress < 0.1:
            return max_lr * progress / 0.1
        elif progress < 0.8:
            return max_lr
        else:
            return max_lr * (1 - (progress - 0.8) / 0.2) ** 2

    return schedule


def create_envs(config: Config) -> tuple[VecEnv, VecEnv]:
    df_train = pl.read_parquet(os.path.join(config.data_dir, "eod.train.df.parquet"))
    bid_ask_train = pl.read_parquet(
        os.path.join(config.data_dir, "eod.train.bid_ask.parquet")
    )

    df_eval = pl.read_parquet(os.path.join(config.data_dir, "eod.val.df.parquet"))
    bid_ask_eval = pl.read_parquet(
        os.path.join(config.data_dir, "eod.val.bid_ask.parquet")
    )

    with open(os.path.join(config.data_dir, "eod.feature_stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    def env_factory(train: bool):
        return StockEnv(
            config,
            df_train if train else df_eval,
            bid_ask_train if train else bid_ask_eval,
            stats,
            train=train,
        )

    if config.n_env > 1:
        print(
            f"Spawning {config.n_env} train and {max(1, config.n_env // 2)} eval environments"
        )
        train_env = SubprocVecEnv(
            [lambda: env_factory(True) for _ in range(config.n_env)]
        )
        eval_env = SubprocVecEnv(
            [lambda: env_factory(False) for _ in range(max(1, config.n_env // 2))]
        )

    else:
        train_env = DummyVecEnv([lambda: env_factory(True)])
        eval_env = DummyVecEnv([lambda: env_factory(False)])

    return train_env, eval_env


def create_agent(config: Config, env: VecEnv) -> OnPolicyAlgorithm:
    ppo_dict = asdict(config.ppo)
    ppo_dict["learning_rate"] = make_lr_schedule(config.ppo.learning_rate)
    if config.algorithm == "ppo":
        agent = SB3_PPO(
            # policy=AssetTablePolicy,
            policy=TraderPolicy,
            env=env,
            device=torch.device(config.device),
            tensorboard_log=f"{config.out_dir}/tb",
            **ppo_dict,
            verbose=1,
            policy_kwargs={
                "optimizer_class": torch.optim.AdamW,
                "optimizer_kwargs": {
                    "betas": (config.adam_beta1, config.adam_beta2),
                    "weight_decay": config.weight_decay,
                },
            },
        )
    elif config.algorithm == "recurrent_ppo":
        agent = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            device=torch.device(config.device),
            tensorboard_log=f"{config.out_dir}/tb",
            **ppo_dict,
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
    agent.save(f"{config.out_dir}/agent_eval_tmp.pth")
    agent = agent.load(f"{config.out_dir}/agent_eval_tmp.pth", env)
    os.remove(f"{config.out_dir}/agent_eval_tmp.pth")

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


def tune(config: Config, train_env: VecEnv, eval_env: VecEnv) -> Config:
    import numpy as np
    import optuna

    original_run_name = config.run_name

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters

        config.ppo = PPOConfig(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            gamma=trial.suggest_float("gamma", 0.8, 0.999, log=True),
            gae_lambda=trial.suggest_float("gae_lambda", 0.8, 1.0),
            clip_range=trial.suggest_float("clip_range", 0.01, 4.0, log=True),
            clip_range_vf=trial.suggest_float("clip_range_vf", 0.01, 4.0, log=True),
            ent_coef=trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
            vf_coef=trial.suggest_float("vf_coef", 0.1, 1.0),
            max_grad_norm=trial.suggest_float("max_grad_norm", 0.1, 1.0),
        )
        config.weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.1, log=True)
        config.adam_beta1 = trial.suggest_float("adam_beta1", 0.5, 0.995, log=True)
        config.adam_beta2 = trial.suggest_float("adam_beta2", 0.8, 0.999, log=True)

        # Create a new agent with the current hyperparameters
        config.run_name = f"{original_run_name}/t{trial.number}"
        agent = create_agent(config, train_env)

        # Train the agent for a short period (tuning objective)
        try:
            agent.learn(total_timesteps=config.train_steps)
        except Exception as _:
            traceback.print_exc()
            trial.report(0, step=0)
            return 0.0

        agent.save(f"{config.out_dir}/agent_tune.pth")
        agent = agent.load(f"{config.out_dir}/agent_tune.pth", eval_env)

        # Run a few episodes for evaluation and obtain mean reward
        rewards, _, _ = run_episodes(agent, n_episodes=64, n_steps=config.ppo.n_steps)
        mean_reward = np.mean(rewards).item()
        return mean_reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # Update the config with the best parameters found
    best_params = best_trial.params

    config.ppo.learning_rate = best_params.get(
        "learning_rate", config.ppo.learning_rate
    )
    config.ppo.gamma = best_params.get("gamma", config.ppo.gamma)
    config.ppo.gae_lambda = best_params.get("gae_lambda", config.ppo.gae_lambda)
    config.ppo.clip_range = best_params.get("clip_range", config.ppo.clip_range)
    config.ppo.clip_range_vf = best_params.get(
        "clip_range_vf", config.ppo.clip_range_vf
    )
    config.ppo.ent_coef = best_params.get("ent_coef", config.ppo.ent_coef)
    config.ppo.vf_coef = best_params.get("vf_coef", config.ppo.vf_coef)
    config.ppo.max_grad_norm = best_params.get(
        "max_grad_norm", config.ppo.max_grad_norm
    )

    return config
