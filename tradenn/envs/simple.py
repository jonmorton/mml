import random

import numpy as np
import polars as pl
from gymnasium import Env, spaces


class StockEnv(Env):
    def __init__(
        self,
        df,
        num_days=251,
        window_size=14,
        transaction_fee=0.01,
        initial_cash=10000.0,
    ):
        super().__init__()

        # Validate and store the DataFrame
        required_columns = ["ticker", "day", "close", "macd", "rsi", "cci", "adx"]
        if not isinstance(df, pl.DataFrame) or not all(
            col in df.columns for col in required_columns
        ):
            raise ValueError(
                f"DataFrame must contain {', '.join(required_columns)} columns."
            )
        self.df = df.clone()

        macd = self.df["macd"].to_numpy()
        macd_mean = np.mean(macd)
        macd_std = np.std(macd)

        # Group data by ticker and ensure days are sequential and prices are positive
        self.ticker_data = {}
        for ticker, group in self.df.group_by("ticker"):
            group = group.sort("day")
            if not (group["day"].to_numpy() == np.arange(len(group))).all():
                raise ValueError(f"Ticker {ticker} has non-sequential or missing days.")
            prices = group["close"].to_numpy().astype(np.float32)
            if np.any(prices <= 0):
                raise ValueError(f"Ticker {ticker} has non-positive prices.")

            # Store all required data for this ticker
            self.ticker_data[ticker] = {
                "prices": prices,
                "macd": (group["macd"].to_numpy().astype(np.float32) - macd_mean)
                / macd_std,
                "rsi": group["rsi"].to_numpy().astype(np.float32) / 50 - 1.0,
                "cci": group["cci"].to_numpy().astype(np.float32) / 600.0,
                "adx": group["adx"].to_numpy().astype(np.float32) / 50 - 1.0,
            }

        self.tickers = list(self.ticker_data.keys())  # List of unique tickers
        self.num_days = num_days
        self.N = window_size
        self.transaction_fee = transaction_fee  # Trading cost per transaction
        self.initial_cash = initial_cash  # Starting capital

        # Define action space: 0 = hold, 1 = buy one share, 2 = buy 5 shares, 3 = sell one share, 4 = sell 5 shares
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

        # Define observation space: N normalized past prices + 4 indicators + shares + normalized cash
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,  # Large upper bound to accommodate potential growth
            shape=(self.N + 6,),  # N prices + 4 indicators + shares + cash
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=0):
        """
        Reset the environment, selecting a random ticker and normalizing prices and indicators.

        Returns:
            np.array: Initial normalized observation.
        """

        self.shares = 0  # Number of shares held
        self.cash = self.initial_cash  # Current cash balance
        self.asset_memory = [self.cash]
        self.returns = []
        self.num_trades = 0
        self.peak = self.initial_cash
        self.max_drawdown = 0
        self.reward = 0
        self.done = False

        # Select a random ticker
        self.current_ticker = random.choice(self.tickers)
        ticker_data = self.ticker_data[self.current_ticker]
        self.prices = ticker_data["prices"]
        self.macd = ticker_data["macd"]
        self.rsi = ticker_data["rsi"]
        self.cci = ticker_data["cci"]
        self.adx = ticker_data["adx"]

        self.t = random.randint(
            self.N - 1, len(self.prices) - self.num_days
        )  # Current time step
        self.start_t = self.t

        self.P0 = self.prices[self.t - self.N + 1]  # Initial price for normalization

        return self.observation, {}

    @property
    def observation(self):
        # Initial observation: [N normalized prices, indicators, shares, normalized cash]
        observation = [
            *[p / self.P0 for p in self.prices[self.t - self.N + 1 : self.t + 1]],
            self.macd[self.t],
            self.rsi[self.t],
            self.cci[self.t],
            self.adx[self.t],
            self.shares / 20,
            self.cash / self.initial_cash,
        ]
        return np.array(observation, dtype=np.float32)

    @property
    def net_value(self):
        return self.cash + self.shares * self.prices[self.t]

    def step(self, action):
        """
        Take an action and advance the environment by one day.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        if not self.done:
            # Current price at time t
            P_t = self.prices[self.t]

            # Portfolio value before the action
            V_old = self.net_value

            action = int(np.round(np.clip(action, -1, 1) * self.initial_cash * 0.05))

            # Execute the action
            if action > 0:
                num_shares = min(
                    action,
                    int(self.cash / (P_t * (1 + self.transaction_fee))),
                )
                if num_shares > 0:
                    cost = P_t * (1 + self.transaction_fee) * num_shares
                    self.shares += num_shares
                    self.cash -= cost
                    self.num_trades += 1
            elif action < 0:
                num_shares = min(self.shares, -action)
                if num_shares > 0:
                    self.shares -= num_shares
                    self.cash += P_t * (1 - self.transaction_fee) * num_shares
                    self.num_trades += 1

            # Advance time
            self.t += 1

            # Portfolio value after the action and price change
            V_new = self.net_value

            self.asset_memory.append(V_new)

            # Calculate daily return
            daily_return = (V_new / V_old) - 1
            self.returns.append(daily_return)

            # Reward is the change in portfolio value
            self.reward = np.log(V_new / V_old) if V_old > 0 else 0

        # Update peak and max_drawdown
        current_value = self.asset_memory[-1]
        self.peak = max(self.peak, current_value)
        current_drawdown = (
            (self.peak - current_value) / self.peak if self.peak > 0 else 0
        )
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Calculate metrics
        returns = (current_value / self.initial_cash) - 1
        cagr = (
            (current_value / self.initial_cash) ** (251 / (self.t - self.N + 1)) - 1
            if self.t > self.N
            else 0
        )
        if len(self.returns) >= 2:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns, ddof=1)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        self.done = (
            self.t == self.start_t + self.num_days - 1
        )  # Episode ends at the last price

        # Include metrics in info
        info = {
            "ticker": self.current_ticker,
            "assets": self.asset_memory[-1],
            "cash": self.cash,
            "cagr": cagr,
            "returns": returns,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "trades": self.num_trades,
        }

        return self.observation, self.reward, self.done, False, info
