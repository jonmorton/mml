import random
from typing import Optional

import gymnasium as gym
import numpy as np
import polars as pl

from tradenn.config import Config

FEATURES = ["macd", "rsi", "cci", "adx", "ao", "bb", "mf", "vi"]
OHLCV = ["open", "high", "low", "close", "volume"]


class State:
    """
    Manages the environment state as a 2D NumPy array.
    Layout:
    - Internal state:
        [0, :] = cash (identical across assets)
        [1, :] = asset bids
        [2, :] = asset asks
        [3, :] = positions
    - External state:
        [4:4+len(FEATURES)*feat_days, :] = features over feat_days
        [4+len(FEATURES)*feat_days:, :] = OHLCV over ohlcv_days
    """

    def __init__(
        self, normstats, num_assets, initial_balance, feat_days=7, ohlcv_days=30
    ):
        self.num_assets = num_assets
        self.feat_days = feat_days
        self.ohlcv_days = ohlcv_days
        self.initial_balance = initial_balance
        self.normstats = {
            t: {k.name: k.to_numpy().item() for k in x.iter_columns()}
            for t, x in normstats.items()
        }
        self.normstats["means"]["cash"] = self.initial_balance / 2
        self.normstats["stds"]["cash"] = 1000
        self.normstats["means"]["position"] = 0
        self.normstats["stds"]["position"] = 10
        self.reset()

    def reset(self):
        self.array = np.zeros(
            (4 + len(FEATURES) * self.feat_days + 5 * self.ohlcv_days, self.num_assets),
            dtype=np.float32,
        )
        self.array[0, :] = self.initial_balance
        self._normalized_array = None

    @property
    def shape(self):
        return self.array.shape

    def set_prices(self, bids: np.ndarray, asks: np.ndarray):
        self.array[1, :] = bids
        self.array[2, :] = asks
        self._normalized_array = None

    @property
    def normalized_array(self):
        # Normalization is commented out in the original code, so we keep it as is
        return self.array

    def set_features(self, feat_img: np.ndarray, ohlcv_img: np.ndarray):
        """Sets precomputed feature and OHLCV images directly into the state array."""
        self.array[4 : 4 + len(FEATURES) * self.feat_days, :] = feat_img
        self.array[4 + len(FEATURES) * self.feat_days :, :] = ohlcv_img
        self._normalized_array = None

    def sell(
        self,
        assets: np.ndarray,
        amounts: np.ndarray,
        transaction_fee: float = 0.01,
        slippage: float = 0.01,
    ):
        amounts = np.minimum(amounts, self.array[3, assets])
        prices = self.array[1, assets] * (1 - slippage)
        proceeds = prices * amounts * (1 - transaction_fee)
        self.array[0, :] += np.sum(proceeds)
        np.subtract.at(self.array[3, :], assets, amounts)
        self._normalized_array = None

    def buy(
        self,
        assets: np.ndarray,
        amounts: np.ndarray,
        transaction_fee: float = 0.01,
        slippage: float = 0.01,
    ):
        current_cash = self.array[0, 0]  # Initialize scalar cash value

        for i in range(len(assets)):
            idx = assets[i]
            ask = self.array[2, idx]  # Ask price for the asset
            if ask > 0:
                price = ask * (1 + slippage) * (1 + transaction_fee)  # Adjusted price
                available_amount = np.floor(
                    current_cash / price
                )  # Max units affordable
                amounts_to_buy = min(
                    amounts[i], available_amount
                )  # Limit by requested amount
                total_cost = price * amounts_to_buy
            else:
                amounts_to_buy = 0
                total_cost = 0
            current_cash -= total_cost  # Update scalar cash
            self.array[3, idx] += amounts_to_buy  # Update position
        self.array[0, :] = current_cash  # Update cash array once at the end
        self._normalized_array = None

    def net_liq_value(self):
        cash = self.array[0, 0]
        positions = self.array[3, :]
        bids = self.array[1, :]
        asks = self.array[2, :]
        asset_values = np.where(positions > 0, positions * bids, positions * asks)
        return cash + np.sum(asset_values)

    @property
    def cash(self):
        return self.array[0][0]


class StockEnv(gym.Env):
    def __init__(
        self,
        config: Config,
        dataframe: pl.DataFrame,
        bid_ask: pl.DataFrame,
        normstats: dict[str, dict[str, pl.Series]],
        train: bool = False,
    ) -> None:
        self.nb_stock = config.nb_stock
        self.initial_balance = config.initial_balance
        self.turbulence_threshold = config.turbulence_threshold
        self.transaction_fee = config.transaction_fee
        self.slippage = config.slippage
        self.reward_scaling = config.reward_scaling
        self.max_step = config.ppo.n_steps
        self.is_train = train
        self.times_steps_per_day = config.time_steps_per_day
        self.hist_days = 30  # max(feat_days=7, ohlcv_days=30) from State

        # Preprocess data
        self.sorted_days = sorted(dataframe["day"].unique().to_list())
        self.sorted_tickers = sorted(dataframe["ticker"].unique().to_list())
        self.num_days = len(self.sorted_days)
        self.num_tickers = len(self.sorted_tickers)
        self.max_day = int(dataframe["day"].max())
        self.ticker_to_idx = {
            ticker: idx for idx, ticker in enumerate(self.sorted_tickers)
        }

        # Precompute features and OHLCV arrays
        self.features_array = np.full(
            (self.num_days, self.num_tickers, len(FEATURES)), np.nan, dtype=np.float32
        )
        self.ohlcv_array = np.full(
            (self.num_days, self.num_tickers, 5), np.nan, dtype=np.float32
        )
        for row in dataframe.iter_rows(named=True):
            day_idx = self.sorted_days.index(row["day"])
            ticker_idx = self.ticker_to_idx[row["ticker"]]
            self.features_array[day_idx, ticker_idx, :] = [row[f] for f in FEATURES]
            self.ohlcv_array[day_idx, ticker_idx, :] = [row[f] for f in OHLCV]

        # Precompute bid and ask arrays for the last time step
        last_bid_ask = (
            bid_ask.sort(["day", "ticker", "time_step"])
            .group_by(["day", "ticker"])
            .last()
        )
        self.bid_array = np.full(
            (self.num_days, self.num_tickers), np.nan, dtype=np.float32
        )
        self.ask_array = np.full(
            (self.num_days, self.num_tickers), np.nan, dtype=np.float32
        )
        for row in last_bid_ask.iter_rows(named=True):
            day_idx = self.sorted_days.index(row["day"])
            ticker_idx = self.ticker_to_idx[row["ticker"]]
            self.bid_array[day_idx, ticker_idx] = row["bid"]
            self.ask_array[day_idx, ticker_idx] = row["ask"]

        self.state = State(
            normstats, self.nb_stock, self.initial_balance, feat_days=7, ohlcv_days=30
        )

        if config.ppo.n_steps + self.hist_days > self.max_day:
            raise ValueError(
                f"Number of steps ({config.ppo.n_steps}) + history days ({self.hist_days}) exceeds max days ({self.max_day})"
            )

        self.action_space = gym.spaces.Box(-1, 1, (self.state.num_assets,))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.state.shape)
        self.train = train
        self.reset()

    def select_tickers(self):
        """Select random tickers and store their indices."""
        selected_tickers = sorted(random.sample(self.sorted_tickers, self.nb_stock))
        self.ticker_indices = np.array(
            [self.ticker_to_idx[ticker] for ticker in selected_tickers]
        )

    def next_trading_day(self, day: Optional[int] = None):
        """Update the current day without filtering."""
        if day is not None:
            self.day = day
        else:
            self.day += 1
        if self.day >= self.max_day:
            raise ValueError(
                f"Day {self.day} is out of range for max of {self.max_day}"
            )

    def _execute_actions(self, actions):
        actions = np.nan_to_num(actions, nan=0, posinf=1, neginf=-1)
        actions = np.round(np.clip(actions, -1, 1) * self.initial_balance * 0.1).astype(
            np.int32
        )
        sell_mask = actions < 0
        buy_mask = actions > 0
        sell_amounts = -actions[sell_mask]
        sell_indices = np.where(sell_mask)[0]
        self.state.sell(sell_indices, sell_amounts, self.transaction_fee, self.slippage)
        self.num_sells += len(sell_indices)
        buy_amounts = actions[buy_mask]
        buy_indices = np.where(buy_mask)[0]
        self.state.buy(buy_indices, buy_amounts, self.transaction_fee, self.slippage)
        self.num_buys += len(buy_indices)

    @property
    def trades(self):
        return self.num_buys + self.num_sells

    def sample_prices(self):
        """Fetch precomputed bid and ask prices."""
        bids = self.bid_array[self.day, self.ticker_indices]
        asks = self.ask_array[self.day, self.ticker_indices]
        self.state.set_prices(bids, asks)

    def step(self, actions, seed=0):
        self.terminal = self.day >= self.max_step - 1
        if not self.terminal:
            begin_total_asset = self.state.net_liq_value()
            self._execute_actions(actions)
            self.next_trading_day()
            self.sample_prices()
            # Compute feature images directly from precomputed arrays
            feat_img = (
                self.features_array[
                    self.day - self.state.feat_days : self.day, self.ticker_indices, :
                ]
                .transpose(2, 0, 1)
                .reshape(-1, self.nb_stock)
            )
            ohlcv_img = (
                self.ohlcv_array[
                    self.day - self.state.ohlcv_days : self.day, self.ticker_indices, :
                ]
                .transpose(2, 0, 1)
                .reshape(-1, self.nb_stock)
            )
            self.state.set_features(feat_img, ohlcv_img)
            end_total_asset = self.state.net_liq_value()
            self.asset_memory.append(end_total_asset)
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling / self.initial_balance
        return (
            self.state.normalized_array,
            self.reward,
            self.terminal,
            False,
            {
                "returns": (self.state.net_liq_value() / self.initial_balance) * 100,
                "cagr": (max(1, self.state.net_liq_value()) / self.initial_balance)
                ** (252 / (self.day - self.hist_days + 1)),
                "trades": self.trades,
                "assets": self.state.net_liq_value(),
            },
        )

    def reset(self, seed=0):
        self.cost = 0
        self.terminal = False
        self.rewards_memory = []
        self.reward = 0
        self.turbulence = 0
        self.asset_memory = [self.initial_balance]
        self.num_buys = 0
        self.num_sells = 0
        self.liquidations = 0
        self.state.reset()
        self.select_tickers()
        self.next_trading_day(day=self.hist_days)
        self.sample_prices()
        feat_img = (
            self.features_array[
                self.day - self.state.feat_days : self.day, self.ticker_indices, :
            ]
            .transpose(2, 0, 1)
            .reshape(-1, self.nb_stock)
        )
        ohlcv_img = (
            self.ohlcv_array[
                self.day - self.state.ohlcv_days : self.day, self.ticker_indices, :
            ]
            .transpose(2, 0, 1)
            .reshape(-1, self.nb_stock)
        )
        self.state.set_features(feat_img, ohlcv_img)
        return self.state.normalized_array, {}
