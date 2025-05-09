import copy
import os
import traceback
import warnings
from dataclasses import asdict
from typing import Any, Optional

import haven
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.sac import SAC
from torch.utils.tensorboard.writer import SummaryWriter

from tradenn.config import Config, PPOConfig
from tradenn.envs.build import build_env  # noqa
from tradenn.policy import TraderPolicy


def make_lr_schedule(max_lr: float):
    def schedule(progress):
        progress = min(1.0, max(0, 1 - progress))

        if progress < 0.1:
            return max_lr * progress / 0.1
        elif progress < 0.8:
            return max_lr
        else:
            return max_lr * (1 - (progress - 0.8) / 0.2) ** 2

    return schedule


def create_agent(config: Config, env: VecEnv) -> BaseAlgorithm:
    if config.algorithm == "ppo_custom":
        ppo_dict = asdict(config.ppo)
        log_std_init = ppo_dict.pop("log_std_init")
        ppo_dict["learning_rate"] = make_lr_schedule(config.ppo.learning_rate)
        agent = PPO(
            policy=TraderPolicy,  # type: ignore
            env=env,
            device=torch.device(config.device),
            tensorboard_log=f"{config.out_dir}/tb",
            **ppo_dict,
            seed=config.seed,
            # verbose=1,
            policy_kwargs={
                "log_std_init": log_std_init,
                "network_config": config.network,
                "optimizer_class": torch.optim.AdamW,
                "optimizer_kwargs": {
                    "betas": (config.adam_beta1, config.adam_beta2),
                    "weight_decay": config.weight_decay,
                },
            },
        )
    elif config.algorithm == "ppo":
        ppo_dict = asdict(config.ppo)
        log_std_init = ppo_dict.pop("log_std_init")
        agent = PPO(
            policy="MultiInputPolicy",
            env=env,
            device=torch.device(config.device),
            tensorboard_log=f"{config.out_dir}/tb",
            **ppo_dict,
            seed=config.seed,
            # verbose=1,
            policy_kwargs={
                "log_std_init": log_std_init,
                # "optimizer_class": torch.optim.AdamW,
                # "optimizer_kwargs": {
                #     "betas": (config.adam_beta1, config.adam_beta2),
                #     "weight_decay": config.weight_decay,
                # },
            },
        )
    elif config.algorithm == "recurrent_ppo":
        ppo_dict = asdict(config.ppo)
        log_std_init = ppo_dict.pop("log_std_init")
        ppo_dict["learning_rate"] = make_lr_schedule(config.ppo.learning_rate)
        agent = RecurrentPPO(
            policy=TraderPolicy,
            env=env,
            device=torch.device(config.device),
            tensorboard_log=f"{config.out_dir}/tb",
            **asdict(config.ppo),
            seed=config.seed,
            # verbose=1,
            policy_kwargs={
                "log_std_init": log_std_init,
                "recurrent": True,
                "optimizer_class": torch.optim.AdamW,
                "optimizer_kwargs": {
                    "betas": (config.adam_beta1, config.adam_beta2),
                    "weight_decay": config.weight_decay,
                },
            },
        )
    elif config.algorithm == "sac":
        sac_dict = asdict(config.sac)
        sac_dict["learning_rate"] = make_lr_schedule(config.sac.learning_rate)
        agent = SAC(
            policy="MultiInputPolicy",
            env=env,
            device=torch.device(config.device),
            tensorboard_log=f"{config.out_dir}/tb",
            seed=config.seed,
            **sac_dict,
        )

    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    return agent


def run_episodes(
    agent,
    n_episodes=1024,
    tb: Optional[SummaryWriter] = None,
    tb_prefix: str = "eval",
    deterministic: bool = True,
) -> dict[str, np.ndarray]:
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
    returns = []
    cagrs = []
    sharpe_ratios = []
    max_drawdowns = []
    liquidations = []

    if n_episodes % env.num_envs != 0:
        warnings.warn(
            f"Number of episodes ({n_episodes}) is not a multiple of the number of environments ({env.num_envs}). Rouding up to the nearest multiple."
        )
        n_episodes = (n_episodes // env.num_envs + 1) * env.num_envs

    agent.policy.set_training_mode(False)

    for ep in range(max(1, n_episodes // env.num_envs)):
        r_ep = np.zeros(env.num_envs, dtype=float)
        done_ = np.zeros(env.num_envs, dtype=bool)
        final_infos = [{} for _ in range(env.num_envs)]
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        s = env.reset()
        h = None

        while not done_.all():
            a, h = agent.predict(
                s, h, deterministic=deterministic, episode_start=episode_starts
            )
            s, reward, done, infos = env.step(a)

            r_ep[~done_] += reward[~done_]

            for i in range(env.num_envs):
                if done[i] and not done_[i]:
                    final_infos[i] = infos[i]
                    done_[i] = 1

            episode_starts = done

        end_assets.append(np.array([i["assets"] for i in final_infos]))
        num_trades.append(np.array([i["trades"] for i in final_infos]))
        returns.append(np.array([i["returns"] for i in final_infos]))
        cagrs.append(np.array([i["cagr"] for i in final_infos]))
        sharpe_ratios.append(np.array([i["sharpe_ratio"] for i in final_infos]))
        max_drawdowns.append(np.array([i["max_drawdown"] for i in final_infos]))
        liquidations.append(np.array([i["num_liquidations"] for i in final_infos]))

        cum_rewards.append(r_ep)

        if tb is not None:
            tb.add_scalar(f"{tb_prefix}/reward", np.mean(r_ep), ep)
            tb.add_scalar(f"{tb_prefix}/returns", np.mean(returns[-1]), ep)
            tb.add_scalar(f"{tb_prefix}/cagr", np.mean(cagrs[-1]), ep)
            tb.add_scalar(f"{tb_prefix}/sharpe_ratio", np.mean(sharpe_ratios[-1]), ep)
            tb.add_scalar(f"{tb_prefix}/max_drawdown", np.mean(max_drawdowns[-1]), ep)

    rewards = np.concatenate(cum_rewards)
    end_assets = np.concatenate(end_assets)
    num_trades = np.concatenate(num_trades)
    returns = np.concatenate(returns)
    cagrs = np.concatenate(cagrs)
    sharpe_ratios = np.concatenate(sharpe_ratios)
    max_drawdowns = np.concatenate(max_drawdowns)
    liquidations = np.concatenate(liquidations)

    return {
        "rewards": rewards,
        "end_assets": end_assets,
        "num_trades": num_trades,
        "returns": returns,
        "cagrs": cagrs,
        "sharpe_ratios": sharpe_ratios,
        "max_drawdowns": max_drawdowns,
        "num_liquidations": liquidations,
    }


class TrainCallback(BaseCallback):
    def __init__(self, max_timestep: int):
        super().__init__()
        self.max_timestep = max_timestep

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

        self.training_env.set_attr("training_progress", 0.0)

    def _on_step(self) -> bool:
        """
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        """

        self.training_env.set_attr(
            "training_progress", self.num_timesteps / self.max_timestep
        )

        if self.n_calls % self._log_freq == 0:
            rets = np.array(
                [
                    x[-1] / x[0]
                    for x in self.training_env.get_attr("asset_memory")
                    if len(x) > 1
                ]
            )
            rewards = np.array(
                [
                    x[-1]
                    for x in self.training_env.get_attr("rewards_memory")
                    if len(x) > 0
                ]
            )
            if rets.shape[0] > 1:
                self.tb_formatter.writer.add_scalar(
                    "train/mean_returns",
                    np.mean(rets),
                    self.num_timesteps,
                )
            if rewards.shape[0] > 0:
                self.tb_formatter.writer.add_scalar(
                    "train/mean_rewards",
                    np.mean(rewards),
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
    - config (PPOConfig): Configuration for the PPO algorithm.

    Returns:
    - reward_arr_train (array): Array containing the episodic returns for each episode in the last trial.
    """
    os.makedirs(config.out_dir, exist_ok=True)
    agent = create_agent(config, env)

    env.seed(config.seed)
    env.reset()

    print("Training agent...")
    agent.learn(
        total_timesteps=config.train_steps, callback=TrainCallback(config.train_steps)
    )

    print("Saving...")
    agent.save(f"{config.out_dir}/agent.pth")

    return agent


def evaluate(config: Config, agent: Any, env: VecEnv, deterministic: bool = True):
    prefix = "det" if deterministic else "sto"

    tb = SummaryWriter(agent.logger.dir)

    agent.save(f"{config.out_dir}/agent_eval_tmp.pth")
    agent = agent.load(
        f"{config.out_dir}/agent_eval_tmp.pth", env, device=torch.device(config.device)
    )
    os.remove(f"{config.out_dir}/agent_eval_tmp.pth")

    env.seed(config.seed + 10000)

    print("Testing...")
    info = run_episodes(
        agent,
        config.eval_episodes,
        tb=tb,
        tb_prefix=f"eval_ep_{prefix}",
        deterministic=deterministic,
    )

    tb.add_histogram(
        f"eval_{prefix}/rewards", torch.from_numpy(info["rewards"]).flatten()
    )
    tb.add_histogram(
        f"eval_{prefix}/returns", torch.from_numpy(info["returns"]).flatten()
    )
    tb.add_histogram(f"eval_{prefix}/cagrs", torch.from_numpy(info["cagrs"]).flatten())
    tb.add_histogram(
        f"eval_{prefix}/sharpe_ratios",
        torch.from_numpy(info["sharpe_ratios"]).flatten(),
    )
    tb.add_histogram(
        f"eval_{prefix}/max_drawdowns",
        torch.from_numpy(info["max_drawdowns"]).flatten(),
    )
    tb.add_histogram(
        f"eval_{prefix}/trades", torch.from_numpy(info["num_trades"]).flatten()
    )
    tb.add_histogram(
        f"eval_{prefix}/liquidations",
        torch.from_numpy(info["num_liquidations"]).flatten(),
    )

    print(f"{prefix} evaluation results:")
    print("Eval rewards: ", np.mean(info["rewards"]), "±", np.std(info["rewards"]))
    print("Eval returns: ", np.mean(info["returns"]), "±", np.std(info["returns"]))
    print("Eval CAGR: ", np.mean(info["cagrs"]), "±", np.std(info["cagrs"]))
    print(
        "Eval sharpe: ",
        np.mean(info["sharpe_ratios"]),
        "±",
        np.std(info["sharpe_ratios"]),
    )
    print(
        "Eval max drawdown: ",
        np.mean(info["max_drawdowns"]),
        "±",
        np.std(info["max_drawdowns"]),
    )
    print("Eval trades: ", np.mean(info["num_trades"]), "±", np.std(info["num_trades"]))
    print(
        "Eval liquidations: ",
        np.mean(info["num_liquidations"]),
        "±",
        np.std(info["num_liquidations"]),
    )

    tb.add_scalar(f"eval_{prefix}/mean_return", np.mean(info["returns"]))
    tb.add_scalar(f"eval_{prefix}/mean_cagr", np.mean(info["cagrs"]))
    tb.add_scalar(f"eval_{prefix}/mean_trades", np.mean(info["num_trades"]))
    tb.add_scalar(f"eval_{prefix}/mean_sharpe", np.mean(info["sharpe_ratios"]))
    tb.add_scalar(f"eval_{prefix}/mean_max_drawdown", np.mean(info["max_drawdowns"]))
    tb.add_scalar(f"eval_{prefix}/mean_reward", np.mean(info["rewards"]))
    tb.add_scalar(f"eval_{prefix}/mean_liquidations", np.mean(info["num_liquidations"]))

    tb.close()


def tune(
    config: Config, train_env: VecEnv, eval_env: VecEnv, n_trials: int = 30
) -> Config:
    import optuna

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters

        cfg = copy.deepcopy(config)

        cfg.ppo = PPOConfig(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            n_epochs=trial.suggest_int("n_epochs", 1, 20),
            gamma=trial.suggest_float("gamma", 0.5, 1.0, log=True),
            gae_lambda=trial.suggest_float("gae_lambda", 0.5, 1.0, log=True),
            clip_range=trial.suggest_float("clip_range", 0.001, 4.0, log=True),
            clip_range_vf=trial.suggest_float("clip_range_vf", 0.001, 10.0, log=True),
            ent_coef=trial.suggest_float("ent_coef", 1e-8, 1.0, log=True),
            vf_coef=trial.suggest_float("vf_coef", 1e-8, 1.0, log=True),
            max_grad_norm=trial.suggest_float("max_grad_norm", 0.1, 1.0),
            normalize_advantage=trial.suggest_categorical(
                "normalize_advantage", [True, False]
            ),
            log_std_init=trial.suggest_float("log_std_init", -3.0, 3.0),
            # target_kl=trial.suggest_categorical(
            #     "target_kl", [None, 1e-3, 1e-2, 1e-1, 1.0]
            # ),
        )
        # if cfg.ppo.use_sde:
        #     cfg.policy.squash_output = trial.suggest_categorical(
        #         "squash_output", [False]
        #     )
        #     cfg.policy.full_std = trial.suggest_categorical("full_std", [True, False])
        #     cfg.policy.use_expln = trial.suggest_categorical("use_expln", [True, False])

        # cfg.weight_decay = trial.suggest_float("weight_decay", , 0.1, log=True)
        cfg.adam_beta1 = trial.suggest_float("adam_beta1", 0.5, 0.995, log=True)
        cfg.adam_beta2 = trial.suggest_float("adam_beta2", 0.8, 0.999, log=True)
        # cfg.algorithm = trial.suggest_categorical("algorithm", ["ppo", "ppo_custom"])

        cfg.network.activation = trial.suggest_categorical(
            "activation", ["relu", "tanh", "gelu", "silu", "leaky_relu"]
        )
        cfg.network.asset_embed_dim = trial.suggest_categorical(
            "asset_embed_dim", [16, 32]
        )
        cfg.network.hdim = trial.suggest_categorical("hdim", [64, 128, 192])
        cfg.network.conv_dim = trial.suggest_categorical("conv_dim", [16, 32, 64])
        cfg.network.policy_dim = trial.suggest_categorical("policy_dim", [64, 128, 192])
        cfg.network.value_dim = trial.suggest_categorical("value_dim", [64, 128, 192])
        cfg.network.value_lr_mult = trial.suggest_float(
            "value_lr_mult",
            0.1,
            20.0,
            log=True,
        )
        cfg.network.init_gain = trial.suggest_float(
            "init_gain",
            0.01,
            100.0,
            log=True,
        )
        cfg.network.action_proj_init_gain = trial.suggest_float(
            "action_proj_init_gain",
            0.01,
            100.0,
            log=True,
        )
        cfg.network.value_proj_init_gain = trial.suggest_float(
            "value_proj_init_gain",
            0.01,
            100.0,
            log=True,
        )

        # Create a new agent with the current hyperparameters
        cfg.run_name = os.path.join(config.run_name, f"t{trial.number}")
        agent = create_agent(cfg, train_env)

        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
            f.write(haven.dump(cfg))

        # Train the agent for a short period (tuning objective)
        try:
            train_env.seed(config.seed)
            train_env.reset()
            agent.learn(
                total_timesteps=cfg.train_steps,
                callback=TrainCallback(cfg.train_steps),
            )
        except Exception as _:
            traceback.print_exc()
            return -1000

        tb = SummaryWriter(agent.logger.dir)

        agent.save(f"{cfg.out_dir}/agent_tune.pth")
        agent = agent.load(
            f"{cfg.out_dir}/agent_tune.pth",
            eval_env,
            device=torch.device(cfg.device),
        )
        agent.policy.set_training_mode(False)

        assert agent.env is not None
        agent.env.seed(10000 + config.seed)

        # Run a few episodes for evaluation and obtain mean reward
        infos = run_episodes(
            agent,
            n_episodes=config.eval_episodes,
            tb=tb,
            tb_prefix="eval_ep_det",
            deterministic=True,
        )

        tb.add_scalar("tune/mean_returns", infos["returns"].mean())
        tb.add_scalar("tune/mean_cagr", infos["cagrs"].mean())
        tb.add_scalar("tune/mean_sharpe", infos["sharpe_ratios"].mean())
        tb.add_scalar("tune/mean_max_drawdown", infos["max_drawdowns"].mean())
        tb.add_scalar("tune/mean_trades", infos["num_trades"].mean())
        tb.add_scalar("tune/mean_reward", infos["rewards"].mean())

        tb.close()

        return infos["rewards"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # Update the config with the best parameters found
    best_params = best_trial.params

    config = copy.deepcopy(config)
    config.ppo.learning_rate = best_params.get(
        "learning_rate", config.ppo.learning_rate
    )
    config.ppo.n_epochs = best_params.get("n_epochs", config.ppo.n_epochs)
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
    config.ppo.normalize_advantage = best_params.get(
        "normalize_advantage", config.ppo.normalize_advantage
    )
    config.ppo.use_sde = best_params.get("use_sde", config.ppo.use_sde)
    config.ppo.log_std_init = best_params.get("log_std_init", config.ppo.log_std_init)
    # config.policy.full_std = best_params.get("full_std", config.policy.full_std)
    # config.policy.use_expln = best_params.get("use_expln", config.policy.use_expln)
    # config.policy.squash_output = best_params.get(
    #     "squash_output", config.policy.squash_output
    # )

    config.weight_decay = best_params.get("weight_decay", config.weight_decay)
    config.adam_beta1 = best_params.get("adam_beta1", config.adam_beta1)
    config.adam_beta2 = best_params.get("adam_beta2", config.adam_beta2)
    config.algorithm = best_params.get("algorithm", config.algorithm)

    config.network.activation = best_params.get("activation", config.network.activation)
    config.network.asset_embed_dim = best_params.get(
        "asset_embed_dim", config.network.asset_embed_dim
    )
    config.network.hdim = best_params.get("hdim", config.network.hdim)
    config.network.conv_dim = best_params.get("conv_dim", config.network.conv_dim)
    config.network.policy_dim = best_params.get("policy_dim", config.network.policy_dim)
    config.network.value_dim = best_params.get("value_dim", config.network.value_dim)
    config.network.value_lr_mult = best_params.get(
        "value_lr_mult", config.network.value_lr_mult
    )
    config.network.init_gain = best_params.get("init_gain", config.network.init_gain)
    config.network.action_proj_init_gain = best_params.get(
        "action_proj_init_gain", config.network.action_proj_init_gain
    )
    config.network.value_proj_init_gain = best_params.get(
        "value_proj_init_gain", config.network.value_proj_init_gain
    )

    return config
