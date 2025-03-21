import argparse
import json
import os
import pickle as pkl
import random
from dataclasses import dataclass, field
from doctest import run_docstring_examples

import gymnasium as gym
import haven
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from matplotlib import pyplot as plt
from torch.distributions import MultivariateNormal

device = "cuda"


@dataclass
class PPOConfig:
    learning_rates: float = 0.0003
    gamma: float = 0.99
    clip: float = 0.2
    ent_coef: float = 0.0
    critic_factor: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    n_updates: int = 5
    n_episodes: int = 5
    max_iter: int = 252


@dataclass
class Config:
    day: int = 0
    turbulence_threshold: int = 140
    hmax: int = 100
    initial_balance: float = 1000000
    nb_stock: int = 20
    transaction_fee: float = 0.1
    reward_scaling: float = 0.0001
    seed: int = 42

    ppo: PPOConfig = field(default_factory=PPOConfig)


class StockEnv(gym.Env):
    def __init__(self, config, dataframe, train: bool = False) -> None:
        self.nb_stock = config.nb_stock
        self.initial_balance = config.initial_balance
        self.turbulence_threshold = config.turbulence_threshold
        self.transaction_fee = config.transaction_fee
        self.hmax = config.hmax
        self.reward_scaling = config.reward_scaling
        self.is_train = True

        self.day = config.day
        self.dataframe = dataframe.fillna(0).reset_index(drop=True)
        self.dates = list(dataframe["date"].unique())
        self.select_tickers()
        self.select_day()

        self.terminal = train

        self.action_space = spaces.Box(-1, 1, (self.nb_stock,))
        self.observation_space = spaces.Box(0, np.inf, (6 * self.nb_stock + 1,))

        self.state = (
            [self.initial_balance]
            + self.data["close"].values.tolist()[: self.nb_stock]
            + [0] * self.nb_stock
            + self.data["macd"].values.tolist()[: self.nb_stock]
            + self.data["rsi"].values.tolist()[: self.nb_stock]
            + self.data["cci"].values.tolist()[: self.nb_stock]
            + self.data["adx"].values.tolist()[: self.nb_stock]
        )

        self.reward = 0
        self.cost = 0
        self.turbulence = 0
        self.nb_trades = 0

        self.asset_memory = [self.initial_balance]
        self.reward_mem = []

    def select_day(self):
        self.date = self.dates[self.day]
        self.data = self.dataframe[
            (self.dataframe["date"] == self.date)
            & (self.dataframe["ticker"].isin(self.tickers))
        ]

    def select_tickers(self):
        self.tickers = list(self.dataframe["ticker"].unique())
        random.shuffle(self.tickers)
        self.tickers = self.tickers[: self.nb_stock * 2]

    def _execute_action(self, actions):
        actions = torch.clip(actions, -1, 1) * self.hmax
        actions = actions.to(torch.int)

        argsort_actions = torch.argsort(actions)
        sell_index = argsort_actions[: torch.where(actions < 0)[0].shape[0]]
        buy_index = torch.flip(argsort_actions, dims=[0])[
            : torch.where(actions > 0)[0].shape[0]
        ]

        if self.turbulence < self.turbulence_threshold or self.is_train:
            # Sell stocks
            for index in sell_index:
                if actions[index] < 0:
                    amount = min(
                        abs(actions[index]), self.state[index + self.nb_stock + 1]
                    )
                    # print(
                    #   f"SELL {self.tickers[index]} {amount} @ {self.data['close'].values[index]}"
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

    def step(self, actions):
        self.terminal = self.day >= len(self.dataframe.index.unique()) - 1
        if self.terminal:
            pass

        else:
            begin_total_asset = self.state[0] + sum(
                torch.tensor(self.state[1 : (self.nb_stock + 1)])
                * torch.tensor(
                    self.state[(self.nb_stock + 1) : (self.nb_stock * 2 + 1)]
                )
            )

            self._execute_action(actions)

            self.day += 1
            self.select_day()

            self.state = (
                [self.state[0]]
                + self.data["close"].values.tolist()[: self.nb_stock]
                + list(self.state[(self.nb_stock + 1) : (self.nb_stock * 2 + 1)])
                + self.data["macd"].values.tolist()[: self.nb_stock]
                + self.data["rsi"].values.tolist()[: self.nb_stock]
                + self.data["cci"].values.tolist()[: self.nb_stock]
                + self.data["adx"].values.tolist()[: self.nb_stock]
            )

            end_total_asset = self.state[0] + sum(
                torch.tensor(self.state[1 : (self.nb_stock + 1)])
                * torch.tensor(
                    self.state[(self.nb_stock + 1) : (self.nb_stock * 2 + 1)]
                )
            )
            self.asset_memory.append(end_total_asset)

            self.turbulence = self.data["turbulence"].values[0]

            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, False, {}

    def reset(self):
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
            + self.data["close"].values.tolist()[: self.nb_stock]
            + [0] * self.nb_stock
            + self.data["macd"].values.tolist()[: self.nb_stock]
            + self.data["rsi"].values.tolist()[: self.nb_stock]
            + self.data["cci"].values.tolist()[: self.nb_stock]
            + self.data["adx"].values.tolist()[: self.nb_stock]
        )
        return self.state, {}


class PPO(nn.Module):
    """
    Proximal Policy Optimization (PPO) algorithm implementation.

    Parameters:
    - policy_class (object): Policy class for the actor and critic networks.
    - env (object): Environment for training the agent.
    - lr (float): Learning rate for the optimizer.
    - gamma (float): Discount factor for future rewards.
    - clip (float): Clip parameter for PPO.
    - n_updates (int): Number of updates per episode for PPO.
    """

    def __init__(
        self,
        env,
        lr,
        gamma,
        clip,
        ent_coef,
        critic_factor,
        max_grad_norm,
        gae_lambda,
        n_updates,
    ):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.clip = clip
        self.n_updates = n_updates

        self.env = env
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        self.actor = FeedForwardNN(self.s_dim, self.a_dim)
        self.critic = FeedForwardNN(self.s_dim, 1)

        self.cov_var = nn.Parameter(
            torch.full(size=(self.a_dim,), fill_value=1.0), requires_grad=True
        )
        self.cov_mat = torch.diag(self.cov_var).to(device)

        self.actor_optim = torch.optim.AdamW(
            list(self.actor.parameters()) + [self.cov_var],
            lr=self.lr,
            weight_decay=0.01,
        )
        self.critic_optim = torch.optim.AdamW(
            self.critic.parameters(), lr=self.lr, weight_decay=0.01
        )
        self.max_grad_norm = max_grad_norm

    def select_action(self, s):
        """
        Select action based on the current state.

        Parameters:
        - s (Tensor): Current state.

        Returns:
        - a (ndarray): Selected action.
        - log_prob (Tensor): Log probability of the selected action.
        """
        mean = self.actor(s)
        dist = MultivariateNormal(mean.to(device), self.cov_mat.to(device))

        a = dist.sample()
        log_prob = dist.log_prob(a.to(device, non_blocking=True))

        return a.detach(), log_prob.detach()

    def evaluate(self, batch_s, batch_a):
        """
        Evaluate the policy and value function.

        Parameters:
        - batch_s (Tensor): Batch of states.
        - batch_a (Tensor): Batch of actions.

        Returns:
        - V (Tensor): Value function estimates.
        - log_prob (Tensor): Log probabilities of the actions.
        - entropy (Tensor): Entropy of the action distribution.
        """
        V = self.critic(batch_s).squeeze()
        mean = self.actor(batch_s)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(batch_a.to(device, non_blocking=True))

        return V, log_prob

    def compute_G(self, batch_r, batch_terminal, V):
        """
        Compute the episodic returns.

        Parameters:
        - batch_r (Tensor): Batch of rewards.

        Returns:
        - G (Tensor): Episodic returns.
        - A (Tensor): Advantage estimates.
        """
        G = 0
        batch_G = []

        for r in reversed(batch_r):
            G = r + self.gamma * G
            batch_G.insert(0, G)

        batch_G = torch.tensor(batch_G, dtype=torch.float).to(device, non_blocking=True)

        return batch_G

    # @torch.compile
    def update(self, batch_r, batch_s, batch_a, batch_terminal):
        """
        Perform PPO update step.

        Parameters:
        - batch_r (Tensor): Batch of rewards.
        - batch_s (Tensor): Batch of states.
        - batch_a (Tensor): Batch of actions.
        - batch_terminal (Tensor): Batch of terminal flags.
        """
        batch_a = batch_a.to(device, non_blocking=True)
        batch_s = batch_s.to(device, non_blocking=True)

        V, old_log_prob = self.evaluate(
            batch_s.to(device, non_blocking=True), batch_a.to(device, non_blocking=True)
        )

        old_log_prob = old_log_prob.detach()

        batch_G = self.compute_G(batch_r, batch_terminal, V)

        A = batch_G - V.detach()
        A = (A - A.mean()) / (A.std() + 1e-10)

        for i in range(self.n_updates):
            V, log_prob = self.evaluate(batch_s, batch_a)

            ratios = torch.exp(log_prob - old_log_prob)

            term1 = ratios * A
            term2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A

            actor_loss = (-torch.min(term1, term2)).mean()
            critic_loss = nn.MSELoss()(V, batch_G)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + [self.cov_var], self.max_grad_norm
            )
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            torch.nn.utils.clip_grad_norm_(
                list(self.critic.parameters()), self.max_grad_norm
            )
            critic_loss.backward()
            self.critic_optim.step()


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


def plot_learning_curves(save_path):
    """
    Plot learning curves comparing baseline PPO and our PPO.

    Parameters:
    - save_path (str): Path to the directory containing the saved results.
    """
    import pandas as pd

    ppo_returns = []
    mcppo_returns = []

    for file in os.listdir(save_path):
        file_path = os.path.join(save_path, file)

        if file.endswith(".csv"):
            df = pd.read_csv(file_path, skiprows=1)
            ppo_returns.append(df["r"].tolist())

        elif file.endswith(".json"):
            df = pd.read_json(file_path)
            mcppo_returns.append(df.tolist())

    mcppo_returns = np.array(mcppo_returns).squeeze()
    ppo_returns = np.array(ppo_returns)

    # Calculate mean and standard deviation
    ppo_mean_reward = np.mean(ppo_returns, axis=0)
    ppo_std_reward = np.std(ppo_returns, axis=0)
    mcppo_mean_reward = np.mean(mcppo_returns, axis=0)
    mcppo_std_reward = np.std(mcppo_returns, axis=0)

    # Plot baseline PPO
    plt.plot(ppo_mean_reward, label="Baseline")
    plt.fill_between(
        range(len(ppo_returns)),
        ppo_mean_reward - ppo_std_reward,
        ppo_mean_reward + ppo_std_reward,
        alpha=0.5,
    )

    # Plot our PPO
    plt.plot(mcppo_mean_reward, label="Ours")
    plt.fill_between(
        range(len(mcppo_returns)),
        mcppo_mean_reward - mcppo_std_reward,
        mcppo_mean_reward + mcppo_std_reward,
        alpha=0.5,
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Episodic Return")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def episode(agent, n_batch, max_iter=10000, testing=False) -> torch.Tensor:
    """
    Run a batch of episodes for the given agent.

    Parameters:
    - agent (object): The agent instance being trained.
    - n_batch (int): Number of episodes to run in the batch.
    - max_iter (int): The maximum number of iterations (steps) allowed for each episode. Default is 10000.
    - testing (bool): Whether the episodes are for testing purposes. Default is False.

    Returns:
    - r_eps (list): List of episodic returns for each episode in the batch.
    """
    r_eps = []

    for _ in range(n_batch):
        batch_r, batch_s, batch_a, batch_terminal = [], [], [], []

        s, _ = agent.env.reset()

        termination, truncation = False, False

        s = torch.tensor(s, dtype=torch.float).to("cuda", non_blocking=True)
        s[torch.isnan(s)] = 0
        a, _ = agent.select_action(s)

        r_ep = 0
        t = 0

        # while not (termination or truncation):
        for i in range(max_iter):
            s_prime, r, termination, _, _ = agent.env.step(a)
            s_prime = torch.tensor(s_prime, dtype=torch.float).to(
                "cuda", non_blocking=True
            )
            s_prime[torch.isnan(s_prime)] = 0
            a_prime, _ = agent.select_action(s_prime.to(device, non_blocking=True))

            batch_r.append(r)
            batch_s.append(s)
            batch_a.append(a)
            batch_terminal.append(termination)

            s, a = s_prime, a_prime
            r_ep += r
            t += 1

            if termination:
                break

        r_eps.append(r_ep)

        batch_r = torch.stack(batch_r, dim=0)
        batch_s = torch.stack(batch_s, dim=0)
        batch_a = torch.stack(batch_a, dim=0)
        batch_terminal = torch.tensor(batch_terminal)

        agent.update(batch_r, batch_s, batch_a, batch_terminal)

    return torch.tensor(r_eps)


def run_trials(
    agent_class,
    env,
    run_name,
    config: PPOConfig,
):
    """
    Run multiple trials for training the agent on the given environment and save the resulting returns
    (and models if a model save path is specified).

    Parameters:
    - agent_class (object): Class of the agent to be trained.
    - policy_class (object): Class of the policy used by the agent.
    - env (object): Environment for training the agent.
    - run_save_path (str): Path to save the training results.
    - model_save_path (str): Path to save the trained models. Default is None.
    - model_name (str): Name of the model being trained.
    - learning_rate (float): Learning rate for training the agent.
    - gamma (float): Discount factor for future rewards.
    - clip (float): Clip parameter for PPO.
    - ent_coef (float): Coefficient for the entropy loss.
    - critic_factor (float): Factor for critic loss in PPO.
    - max_grad_norm (float): Maximum gradient norm for PPO.
    - gae_lambda (float): Lambda value for generalized advantage estimation (GAE).
    - n_updates (int): Number of updates per episode for PPO.
    - n_episodes (int): Number of episodes per trial.
    - max_iter (int): Maximum number of iterations (steps) per episode.

    Returns:
    - reward_arr_train (array): Array containing the episodic returns for each episode in the last trial.
    """
    os.makedirs("models", exist_ok=True)

    for run in range(10):
        for _ in range(3):
            reward_arr_train = []
            agent = agent_class(
                env,
                config.learning_rates,
                config.gamma,
                config.clip,
                config.ent_coef,
                config.critic_factor,
                config.max_grad_norm,
                config.gae_lambda,
                config.n_updates,
            ).to(device)

            for ep in range(1000):
                ep_returns: torch.Tensor = episode(
                    agent, config.n_episodes, config.max_iter
                )
                reward_arr_train.extend(ep_returns)

                if ep % 10 == 0:
                    cagr = (agent.env.asset_memory[-1] / agent.env.initial_balance) ** (
                        252 / (ep + 1)
                    ) - 1
                    print(
                        f"Episode {ep} - Mean Return: {torch.mean(ep_returns)}  CAGR: {cagr:.2f}"
                    )

            torch.save(agent.state_dict(), f"models/{run_name}.pt")
            reward_arr_train = np.array(reward_arr_train)

    return reward_arr_train


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

    df = pd.read_parquet("data/eod.parquet")
    env = StockEnv(config, df, train=True)
    env.reset()

    run_name = "trader"

    run_trials(
        PPO,
        env,
        run_name,
        config.ppo,
    )

    with open(f"models/{run_name}.yaml", "w") as f:
        f.write(haven.dump(config, "yaml"))
