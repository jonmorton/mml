import argparse
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

import gymnasium as gym
import haven
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

device = "cpu"


@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    n_steps: int = 384
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 2
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
class Config:
    run_name: str = "trader"

    day: int = 0
    turbulence_threshold: int = 140
    hmax: int = 100
    initial_balance: float = 1000000
    nb_stock: int = 20
    transaction_fee: float = 0.02
    reward_scaling: float = 0.0001
    seed: int = 42

    ppo: PPOConfig = field(default_factory=PPOConfig)

    train_steps: int = 100000
    eval_episodes: int = 100
    n_env: int = 8

    @property
    def out_dir(self):
        return f"models/{self.run_name}"


class StockEnv(gym.Env):
    def __init__(self, config, dataframe, train: bool = False) -> None:
        self.nb_stock = config.nb_stock
        self.initial_balance = config.initial_balance
        self.turbulence_threshold = config.turbulence_threshold
        self.transaction_fee = config.transaction_fee
        self.hmax = config.hmax
        self.reward_scaling = config.reward_scaling
        self.is_train = True

        self.day = 0
        self.dataframe = dataframe.fill_nan(0).fill_null(0)
        self.dates = list(dataframe["date"].unique())

        self.select_tickers()
        self.select_day()

        self.terminal = train

        self.action_space = spaces.Box(-1, 1, (self.nb_stock,))
        self.observation_space = spaces.Box(0, np.inf, (6 * self.nb_stock + 1,))

        self.state = (
            [self.initial_balance]
            + self.data["close"].to_list()[: self.nb_stock]
            + [0] * self.nb_stock
            + self.data["macd"].to_list()[: self.nb_stock]
            + self.data["rsi"].to_list()[: self.nb_stock]
            + self.data["cci"].to_list()[: self.nb_stock]
            + self.data["adx"].to_list()[: self.nb_stock]
        )

        self.reward = 0
        self.cost = 0
        self.turbulence = 0
        self.nb_trades = 0

        self.asset_memory = [self.initial_balance]
        self.reward_mem = []

    def select_day(self):
        self.date = self.dates[self.day]
        self.data = self.dataframe.filter(
            (pl.col("date") == self.date) & pl.col("ticker").is_in(self.tickers)
        )

    def select_tickers(self):
        self.tickers = list(self.dataframe["ticker"].unique())
        random.shuffle(self.tickers)
        self.tickers = self.tickers[: self.nb_stock * 2]

    def _execute_action(self, actions):
        actions = np.clip(actions, -1, 1) * self.hmax
        actions = actions.astype(np.int32)

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        if self.turbulence < self.turbulence_threshold or self.is_train:
            # Sell stocks
            for index in sell_index:
                if actions[index] < 0:
                    amount = min(
                        abs(actions[index]), self.state[index + self.nb_stock + 1]
                    )
                    # print(
                    #     f"SELL {self.tickers[index]} {amount} @ {self.data['close'].values[index]}"
                    # )
                    self.state[0] += (
                        self.state[index + 1] * amount * (1 - self.transaction_fee)
                    )
                    self.state[index + self.nb_stock + 1] -= amount
                    self.cost += self.state[index + 1] * amount * self.transaction_fee
                    self.trades += 1

            # Buy stocks
            for index in buy_index:
                if actions[index] > 0:
                    # print(
                    #     f"BUY {self.tickers[index]} {amount} @ {self.data['close'].values[index]}"
                    # )
                    available_amount = self.state[0] // self.state[index + 1]
                    amount = min(available_amount, actions[index])
                    self.state[0] -= (
                        self.state[index + 1] * amount * (1 + self.transaction_fee)
                    )
                    self.state[index + self.nb_stock + 1] += float(amount)
                    self.cost += self.state[index + 1] * amount * self.transaction_fee
                    self.trades += 1
        else:
            # Sell all stocks
            for index in range(len(actions)):
                amount = self.state[index + self.nb_stock + 1]
                self.state[0] += (
                    self.state[index + 1] * amount * (1 - self.transaction_fee)
                )
                self.state[index + self.nb_stock + 1] = 0
                self.cost += self.state[index + 1] * amount * self.transaction_fee
                self.trades += 1

    def step(self, actions, seed=0):
        self.terminal = self.day >= len(self.dates) - 1
        if not self.terminal:
            begin_total_asset = self.state[0] + sum(
                torch.tensor(self.state[1 : (self.nb_stock + 1)])
                * torch.tensor(
                    self.state[(self.nb_stock + 1) : (self.nb_stock * 2 + 1)]
                )
            )

            self._execute_action(actions)

            self.day += 1
            if self.day >= len(self.dates) - 1:
                self.day = 0
                self.select_tickers()
            self.select_day()

            self.state = (
                [self.state[0]]
                + self.data["close"].to_list()[: self.nb_stock]
                + list(self.state[(self.nb_stock + 1) : (self.nb_stock * 2 + 1)])
                + self.data["macd"].to_list()[: self.nb_stock]
                + self.data["rsi"].to_list()[: self.nb_stock]
                + self.data["cci"].to_list()[: self.nb_stock]
                + self.data["adx"].to_list()[: self.nb_stock]
            )

            end_total_asset = self.state[0] + sum(
                torch.tensor(self.state[1 : (self.nb_stock + 1)])
                * torch.tensor(
                    self.state[(self.nb_stock + 1) : (self.nb_stock * 2 + 1)]
                )
            )
            self.asset_memory.append(end_total_asset)

            self.turbulence = self.data["turbulence"].first()

            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=0):
        self.asset_memory = [self.initial_balance]
        self.day = 0

        self.select_tickers()
        self.select_day()

        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []

        self.state = (
            [self.initial_balance]
            + self.data["close"].to_list()[: self.nb_stock]
            + [0] * self.nb_stock
            + self.data["macd"].to_list()[: self.nb_stock]
            + self.data["rsi"].to_list()[: self.nb_stock]
            + self.data["cci"].to_list()[: self.nb_stock]
            + self.data["adx"].to_list()[: self.nb_stock]
        )
        return self.state, {}


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        """
        Initialize the network and set up the layers.

        Parameters:
        - in_dim (int): Input dimensions.
        - out_dim (int): Output dimensions.
        """
        super().__init__()

        self.layer1 = nn.Linear(in_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(in_dim, 256, bias=False)
        self.layer3 = nn.Linear(256, 64, bias=False)
        self.layer4 = nn.Linear(64, out_dim, bias=False)

        nn.init.trunc_normal_(
            self.layer1.weight,
            mean=0.0,
            std=0.1,
        )
        nn.init.trunc_normal_(self.layer2.weight, mean=0.0, std=0.1)
        nn.init.trunc_normal_(self.layer3.weight, mean=0.0, std=0.1)
        nn.init.trunc_normal_(self.layer4.weight, mean=0.0, std=0.1)

    def forward(self, obs):
        """
        Forward pass of the neural network.

        Parameters:
        - obs (Tensor or ndarray): Input observation.

        Returns:
        - Tensor: Output of the forward pass.
        """

        activation1 = self.layer1(obs) * F.sigmoid(self.layer2(obs))
        activation1 = self.ln1(activation1)
        output = self.layer4(F.relu(self.layer3(activation1)))
        return output


def episode(agent, batch_size=1, n_steps=365) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a batch of episodes for the given agent.

    Parameters:
    - agent (object): The agent instance being trained.
    - batch_size (int): The number of episodes to run in the batch. Default is 1.
    - n_stpes (int): The maximum number of iterations (steps) allowed for each episode. Default is 10000.

    Returns:
    - r_eps (list): List of episodic returns for each episode in the batch.
    """

    for _ in range(batch_size):
        s = agent.env.reset()
        a, _ = agent.predict(s)

        r_eps = []
        assets = []
        r_ep = 0
        t = 0

        # while not (termination or truncation):
        for i in range(n_steps):
            obs, reward, done, _ = agent.env.step(a)
            a, _ = agent.predict(obs)

            r_ep += reward
            assets.append(agent.env.get_attr("asset_memory")[0][-1])
            t += 1

            if done.all():
                break

        r_eps.append(r_ep)

    return np.array(r_eps), np.array(assets)


def run_trials(
    config: Config,
    env,
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
    os.makedirs("models", exist_ok=True)

    agent = SB3_PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        tensorboard_log=f"{config.out_dir}/tb",
        **asdict(config.ppo),
    )

    agent.learn(total_timesteps=config.train_steps)
    agent.save(f"{config.out_dir}/agent.pth")

    all_ep_returns = []
    all_cagr = []

    # test episode
    for ep in range(config.eval_episodes):
        ep_returns, ep_assets = episode(agent, 8, config.ppo.n_steps)

        mean_return = np.mean(ep_returns)
        cagr = (ep_assets[-1] / ep_assets[0]) ** (1 / (len(ep_assets) / 251)) - 1

        all_ep_returns.append(mean_return)
        all_cagr.append(cagr)

        if ep % 10 == 0:
            print(
                f"Episode {ep} - Mean Return: {np.mean(mean_return)}  CAGR: {cagr:.2f}"
            )

    print("Eval returns: ", all_ep_returns)
    print("Eval CAGR: ", all_cagr)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a stock trading agent.")
    argparser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to the configuration file.",
    )
    argparser.add_argument("overrides", nargs="*", help="Config overrides.")
    args = argparser.parse_args()

    if args.config == "":
        config = haven.load(Config, dotlist_overrides=args.overrides)
    else:
        with open(args.config, "r") as f:
            config = haven.load(Config, stream=f, dotlist_overrides=args.overrides)

    run_name = "trader"

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    os.makedirs(config.out_dir, exist_ok=True)

    with open(f"{config.out_dir}/config.yaml", "w") as f:
        f.write(haven.dump(config, "yaml"))

    df = pl.read_parquet("data/eod.parquet")

    env = StockEnv(config, df, train=True)
    if config.n_env > 1:
        print(f"Spawning {config.n_env} environments")
        env = SubprocVecEnv([lambda: env for _ in range(config.n_env)])
    else:
        env = DummyVecEnv([lambda: env])

    run_trials(
        config,
        env,
    )
