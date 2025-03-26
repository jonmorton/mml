import random
from encodings import big5
from typing import Optional

import gymnasium as gym
import numpy as np
import polars as pl

from tradenn.config import Config

FEATURES = ["macd", "rsi"]
OHLCV = ["open", "high", "low", "close", "volume"]


class State:
    """
    Conveniences for tracking the env state as a 2-d numpy array

    Layout:

    Internal state:
        state[0, :num_assets] = cash (repeated identical value for all assets)
        state[1, :num_assets] = asset bid
        state[2, :num_assets] = asset ask
        state[3, :num_assets] = position in each asset

    External state (features):
        state[4:4+len(FEATURES)*feat_days, :num_assets] = features for each stock (macd, rsi, etc.) for each past `feat_days` das
        state[4+len(FEATURES)*feat_days:4+len(FEATURES)*feat_days+5*ohlcv_days, :num_assets] = OHLCV for each asset for each past `ohlcv_days` days

    """

    def __init__(self, num_assets, initial_balance, feat_days=7, ohlcv_days=30):
        self.num_assets = num_assets
        self.feat_days = feat_days
        self.ohlcv_days = ohlcv_days
        self.initial_balance = initial_balance
        self.reset()

    @property
    def flat_array(self):
        return self.array.flatten()

    def reset(self):
        self.array = np.zeros(
            (4 + len(FEATURES) * self.feat_days + 5 * self.ohlcv_days, self.num_assets),
            dtype=np.float32,
        )
        self.array[0, :] = self.initial_balance

    @property
    def shape(self):
        return self.array.shape

    def set_prices(self, bids: np.ndarray, asks: np.ndarray):
        self.array[1, :] = bids
        self.array[2, :] = asks

    def set_df(self, df: pl.DataFrame, day: int):
        feat_img = (
            df.filter(pl.col("day").is_between(day - self.feat_days, day - 1))
            .pivot("day", index="ticker", values=FEATURES)
            .drop("ticker")
            .to_numpy()
            .T
        )
        ohlcv_img = (
            df.filter(pl.col("day").is_between(day - self.ohlcv_days, day - 1))
            .pivot("day", index="ticker", values=OHLCV)
            .drop("ticker")
            .to_numpy()
            .T
        )

        self.array[4 : 4 + len(FEATURES) * self.feat_days, :] = feat_img
        self.array[4 + len(FEATURES) * self.feat_days :, :] = ohlcv_img

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

    def buy(
        self,
        assets: np.ndarray,
        amounts: np.ndarray,
        transaction_fee: float = 0.01,
        slippage: float = 0.01,
    ):
        prices = self.array[2, assets] * (1 + slippage) * (1 + transaction_fee)
        available_amounts = np.floor(self.cash / prices)
        amounts = np.minimum(available_amounts, amounts)

        total_cost = prices * amounts
        self.array[0, :] -= np.sum(total_cost)
        np.add.at(self.array[3, :], assets, amounts)

    def net_liq_value(self):
        cash = self.array[0, 0]
        positions = self.array[3, :]
        bids = self.array[1, :]
        asks = self.array[2, :]
        asset_values = np.where(positions > 0, positions * bids, positions * asks)
        total_value = cash + np.sum(asset_values)
        return total_value

    @property
    def cash(self):
        return self.array[0][0]


class StockEnv(gym.Env):
    def __init__(
        self,
        config: Config,
        dataframe: pl.DataFrame,
        bid_ask: pl.DataFrame,
        train: bool = False,
    ) -> None:
        self.nb_stock = config.nb_stock
        self.initial_balance = config.initial_balance
        self.turbulence_threshold = config.turbulence_threshold
        self.transaction_fee = config.transaction_fee
        self.slippage = config.slippage
        self.hmax = config.hmax
        self.reward_scaling = config.reward_scaling
        self.max_step = config.ppo.n_steps
        self.is_train = train
        self.times_steps_per_day = config.time_steps_per_day

        self.dataframe = dataframe
        self.bid_ask = bid_ask
        self.tickers = self.dataframe["ticker"].unique().to_list()
        self.max_day = int(self.dataframe["day"].max())  # type: ignore
        self.state = State(
            self.nb_stock, self.initial_balance, feat_days=7, ohlcv_days=30
        )
        self.hist_days = max(self.state.feat_days, self.state.ohlcv_days)

        if config.ppo.n_steps + self.hist_days > self.max_day:
            raise ValueError(
                f"Number of steps ({config.ppo.n_steps}) + history days ({self.hist_days}) is greater than the number of days ({self.max_day})"
            )

        self.action_space = gym.spaces.Box(-1, 1, (self.state.num_assets,))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.state.shape)
        print(
            "Action shape:",
            self.action_space.shape,
            " Observation shape:",
            self.observation_space.shape,
        )

        self.train = train
        self.reset()

    def next_trading_day(self, day: Optional[int] = None):
        if day is not None:
            self.day = day
        else:
            self.day += 1
        if self.day >= self.max_day:
            raise ValueError(
                f"Day {self.day} is out of range for max of {self.max_day}"
            )

        self.data = self.dataframe.filter(
            (pl.col("day").is_between(self.day - self.hist_days, self.day))
            & pl.col("ticker").is_in(self.tickers)
        )

    def select_tickers(self):
        self.tickers = random.sample(self.tickers, self.nb_stock)

    def _execute_actions(self, actions):
        actions = np.round(np.clip(actions, -1, 1) * self.hmax).astype(np.int32)

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

    def step(self, actions, seed=0):
        self.terminal = self.day >= self.max_step - 1

        if not self.terminal:
            # execute actions

            begin_total_asset = self.state.net_liq_value()

            self._execute_actions(actions)

            # next day
            self.next_trading_day()
            self.sample_prices()
            self.state.set_df(self.data, self.day)

            end_total_asset = self.state.net_liq_value()

            self.asset_memory.append(end_total_asset)
            # self.turbulence = self.data["turbulence"].first()
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling / self.initial_balance

        return (
            self.state.array,
            self.reward,
            self.terminal,
            False,
            {},
        )

    def sample_prices(self):
        # Combine all filtering conditions into a single call to reduce overhead
        condition = (
            (pl.col("day") == self.day)
            & (pl.col("ticker").is_in(self.tickers))
            & (pl.col("time_step") == self.times_steps_per_day - 1)
        )
        latest_ba = self.bid_ask.filter(condition)
        # Select bids and asks in a vectorized way
        bids = latest_ba.select("bid").to_numpy().flatten()
        asks = latest_ba.select("ask").to_numpy().flatten()

        self.state.set_prices(bids, asks)

    def reset(self, seed=0):
        self.cost = 0
        self.terminal = False
        self.rewards_memory = []
        self.reward = 0
        self.cost = 0
        self.turbulence = 0
        self.asset_memory = [self.initial_balance]

        self.num_buys = 0
        self.num_sells = 0
        self.liquidations = 0

        self.state.reset()
        self.select_tickers()
        self.next_trading_day(day=self.hist_days)
        self.sample_prices()
        self.state.set_df(self.data, self.day)

        return self.state.array, {}
