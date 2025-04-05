import random
from collections import deque

import numpy as np
import polars as pl
from gymnasium import Env, spaces


class StockTradingEnv(Env):
    """
    A Gymnasium environment for training a daily stock trading agent with multiple tickers.

    Args:
        df (pl.DataFrame): DataFrame with columns 'ticker', 'day' (0-indexed), 'close', 'macd', 'rsi', 'cci', 'adx'.
        N (int): Number of past days' prices in the observation (window size).
        commission (float): Transaction cost as a fraction of the trade value (e.g., 0.001 for 0.1%).
        initial_cash (float): Starting cash balance for the agent.
        num_tickers (int): Number of tickers to trade simultaneously.
    """

    def __init__(self, df, N=7, commission=0.001, initial_cash=10000, num_tickers=2):
        super(StockTradingEnv, self).__init__()

        # Validate and store the DataFrame
        required_columns = ["ticker", "day", "close", "macd", "rsi", "cci", "adx"]
        if not isinstance(df, pl.DataFrame) or not all(
            col in df.columns for col in required_columns
        ):
            raise ValueError(
                f"DataFrame must contain {', '.join(required_columns)} columns."
            )
        self.df = df.clone()

        # Preprocess indicators
        macd = self.df["macd"].to_numpy()
        macd_mean = np.mean(macd)
        macd_std = np.std(macd)

        # Group data by ticker and validate
        self.ticker_data = {}
        for ticker, group in self.df.group_by("ticker"):
            group = group.sort("day")
            if not (group["day"].to_numpy() == np.arange(len(group))).all():
                raise ValueError(f"Ticker {ticker} has non-sequential or missing days.")
            prices = group["close"].to_numpy().astype(np.float32)
            if np.any(prices <= 0):
                raise ValueError(f"Ticker {ticker} has non-positive prices.")

            self.ticker_data[ticker] = {
                "prices": prices,
                "macd": (group["macd"].to_numpy().astype(np.float32) - macd_mean)
                / macd_std,
                "rsi": group["rsi"].to_numpy().astype(np.float32) / 50 - 1.0,
                "cci": group["cci"].to_numpy().astype(np.float32) / 600.0,
                "adx": group["adx"].to_numpy().astype(np.float32) / 50 - 1.0,
            }

        self.tickers = list(self.ticker_data.keys())
        self.N = N
        self.commission = commission
        self.initial_cash = initial_cash
        self.num_tickers = num_tickers

        # Action space: MultiDiscrete for each ticker (0=hold, 1=buy 1, 2=buy 5, 3=sell 1, 4=sell 5)
        self.action_space = spaces.MultiDiscrete([5] * self.num_tickers)

        # Observation space: For each ticker (N prices + 4 indicators + shares) + cash
        self.observation_space = spaces.Box(
            low=0,
            high=1e7,
            shape=(self.num_tickers * (N + 4 + 1) + 1,),
            dtype=np.float32,
        )

        self.M = 251 + self.N  # Episode length assuming 251 trading days + N

    def reset(self, seed=0):
        """
        Reset the environment, selecting a random subset of tickers.

        Returns:
            tuple: (observation, info)
        """
        self.t = self.N - 1
        self.cash = self.initial_cash
        self.asset_memory = [self.initial_cash]
        self.returns = []
        self.num_trades = 0
        self.peak = self.initial_cash
        self.max_drawdown = 0
        self.reward = 0

        # Select a random subset of tickers
        self.selected_tickers = random.sample(self.tickers, self.num_tickers)
        self.prices = [
            self.ticker_data[ticker]["prices"] for ticker in self.selected_tickers
        ]
        self.macd = [
            self.ticker_data[ticker]["macd"] for ticker in self.selected_tickers
        ]
        self.rsi = [self.ticker_data[ticker]["rsi"] for ticker in self.selected_tickers]
        self.cci = [self.ticker_data[ticker]["cci"] for ticker in self.selected_tickers]
        self.adx = [self.ticker_data[ticker]["adx"] for ticker in self.selected_tickers]
        self.P0 = [self.prices[i][0] for i in range(self.num_tickers)]
        self.price_histories = [
            deque(
                [self.prices[i][j] / self.P0[i] for j in range(self.N)], maxlen=self.N
            )
            for i in range(self.num_tickers)
        ]
        self.shares = [0] * self.num_tickers

        # Construct initial observation
        obs = []
        for i in range(self.num_tickers):
            obs.extend(self.price_histories[i])
            obs.extend(
                [
                    self.macd[i][self.t],
                    self.rsi[i][self.t],
                    self.cci[i][self.t],
                    self.adx[i][self.t],
                    self.shares[i],
                ]
            )
        obs.append(self.cash / self.initial_cash)

        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        """
        Take an action and advance the environment by one day.

        Args:
            action (list): List of actions for each ticker.

        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        done = self.t == self.M - 1

        if not done:
            P_t = [self.prices[i][self.t] for i in range(self.num_tickers)]
            V_old = self.cash + sum(
                self.shares[i] * P_t[i] for i in range(self.num_tickers)
            )

            # Process sell actions first
            for i in range(self.num_tickers):
                if action[i] in (3, 4):
                    num_shares = min(self.shares[i], {3: 1, 4: 5}[action[i]])
                    if num_shares > 0:
                        self.shares[i] -= num_shares
                        self.cash += P_t[i] * (1 - self.commission) * num_shares
                        self.num_trades += 1

            # Process buy actions with updated cash
            for i in range(self.num_tickers):
                if action[i] in (1, 2):
                    desired_shares = {1: 1, 2: 5}[action[i]]
                    num_shares = min(
                        desired_shares,
                        int(self.cash / (P_t[i] * (1 + self.commission)))
                        if P_t[i] > 0
                        else 0,
                    )
                    if num_shares > 0:
                        cost = P_t[i] * (1 + self.commission) * num_shares
                        self.shares[i] += num_shares
                        self.cash -= cost
                        self.num_trades += 1

            # Advance time
            self.t += 1

            # Update price histories
            for i in range(self.num_tickers):
                self.price_histories[i].append(self.prices[i][self.t] / self.P0[i])

            # Compute new portfolio value
            V_new = self.cash + sum(
                self.shares[i] * self.prices[i][self.t] for i in range(self.num_tickers)
            )
            self.asset_memory.append(V_new)

            # Calculate daily return and reward
            daily_return = (V_new / V_old) - 1 if V_old > 0 else 0
            self.returns.append(daily_return)
            self.reward = np.log(V_new / V_old) if V_old > 0 else 0

        # Construct new observation
        obs = []
        for i in range(self.num_tickers):
            obs.extend(self.price_histories[i])
            obs.extend(
                [
                    self.macd[i][self.t],
                    self.rsi[i][self.t],
                    self.cci[i][self.t],
                    self.adx[i][self.t],
                    self.shares[i],
                ]
            )
        obs.append(self.cash / self.initial_cash)
        observation = np.array(obs, dtype=np.float32)

        # Update metrics
        current_value = self.asset_memory[-1]
        self.peak = max(self.peak, current_value)
        current_drawdown = (
            (self.peak - current_value) / self.peak if self.peak > 0 else 0
        )
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        returns = (current_value / self.initial_cash) - 1
        cagr = (
            (current_value / self.initial_cash) ** (251 / (self.t - self.N + 1)) - 1
            if self.t > self.N - 1
            else 0
        )
        sharpe_ratio = (
            np.mean(self.returns) / np.std(self.returns, ddof=1)
            if len(self.returns) >= 2 and np.std(self.returns, ddof=1) > 0
            else 0
        )

        info = {
            "tickers": self.selected_tickers,
            "assets": current_value,
            "cash": self.cash,
            "cagr": cagr,
            "returns": returns,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "trades": self.num_trades,
        }

        return observation, self.reward, done, False, info

    @property
    def net_value(self):
        """Calculate the current portfolio value."""
        return self.cash + sum(
            self.shares[i] * self.prices[i][self.t] for i in range(self.num_tickers)
        )
