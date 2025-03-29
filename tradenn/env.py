import math
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

        if self._normalized_array is None:
            self._normalized_array = self.array.copy()

            self._normalized_array[0, :] = (
                (self.array[0, :] - self.initial_balance / 2)
                / (self.initial_balance / 10)
            ).astype(np.float32)

            self._normalized_array[1, :] = (
                (self.array[1, :] - self.normstats["means"]["close"])
                / self.normstats["stds"]["close"]
            ).astype(np.float32)

            self._normalized_array[2, :] = (
                (self.array[2, :] - self.normstats["means"]["close"])
                / self.normstats["stds"]["close"]
            ).astype(np.float32)

            self._normalized_array[3, :] = (self.array[3, :] / 100).astype(np.float32)

            # self._normalized_array[3, :] = (
            #     (self.array[3, :] - self.normstats["means"]["position"])
            #     / self.normstats["stds"]["position"]
            # ).astype(np.float32)

            cols = sum([[x] * self.feat_days for x in FEATURES], []) + sum(
                [[x] * self.ohlcv_days for x in OHLCV], []
            )
            for i in range(5, self.array.shape[0]):
                feature_name = cols[i - 4]
                mean = self.normstats["means"][feature_name]
                std = self.normstats["stds"][feature_name]
                self._normalized_array[i, :] = ((self.array[i, :] - mean) / std).astype(
                    np.float32
                )

        return self._normalized_array

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
        trade_fee: float = 0.05,
    ):
        # Cap amounts to current holdings (for long positions)
        amounts = np.minimum(amounts, np.maximum(self.array[3, assets], 0))
        prices = self.array[1, assets] * (1 - slippage)  # Bid price for selling
        proceeds = prices * amounts * (1 - transaction_fee)
        total_proceeds = np.sum(proceeds)

        # Simulate the trade to check margin
        new_positions = self.array[3, :].copy()
        np.subtract.at(new_positions, assets, amounts)
        short_mask = new_positions < 0
        liabilities = (
            -new_positions[short_mask] * self.array[2, short_mask]
        )  # Cost to cover shorts
        long_mask = new_positions > 0
        long_value = np.sum(new_positions[long_mask] * self.array[1, long_mask])

        k = 2
        simulated_cash = self.array[0, 0] + total_proceeds
        simulated_net_value = simulated_cash + np.sum(
            np.where(
                new_positions > 0,
                new_positions * self.array[1, :],
                new_positions * self.array[2, :],
            )
        )
        if np.sum(liabilities) > k * simulated_net_value:
            return

        # Execute the trade
        self.array[0, :] += total_proceeds
        np.subtract.at(self.array[3, :], assets, amounts)
        self._normalized_array = None

    def buy(
        self,
        assets: np.ndarray,
        amounts: np.ndarray,
        transaction_fee: float = 0.01,
        slippage: float = 0.01,
        trade_fee: float = 0.05,
    ):
        current_cash = self.array[0, 0]
        for i in range(len(assets)):
            idx = assets[i]
            ask = self.array[2, idx]
            if ask > 0:
                price = ask * (1 + slippage) * (1 + transaction_fee)
                available_amount = np.floor(current_cash / price)
                amounts_to_buy = min(amounts[i], available_amount)
                total_cost = price * amounts_to_buy
            else:
                amounts_to_buy = 0
                total_cost = 0
            current_cash -= total_cost
            self.array[3, idx] += amounts_to_buy
        self.array[0, :] = current_cash
        self._normalized_array = None

    def apply_interest_charge(self, lending_rate: float):
        short_positions = np.where(self.array[3, :] < 0, -self.array[3, :], 0)
        prices = (self.array[2, :] + self.array[1, :]) / 2.0
        interest_charge = np.sum(lending_rate * short_positions * prices)
        self.array[0, :] -= interest_charge

        self._normalized_array = None

    def maybe_liquidate(self, transaction_fee: float = 0.01, slippage: float = 0.01):
        net_value = self.net_liq_value()
        if net_value <= 0:
            return  # Let step handle termination
        if self.cash < 0:
            # Liquidate long positions to cover cash shortfall
            long_positions = np.where(self.array[3, :] > 0)[0]
            for asset in long_positions:
                amount_to_sell = min(
                    self.array[3, asset],
                    math.ceil(
                        -self.cash
                        / (
                            self.array[1, asset]
                            * (1 - slippage)
                            * (1 - transaction_fee)
                        )
                    ),
                )
                self.sell(
                    np.array([asset]),
                    np.array([amount_to_sell]),
                    transaction_fee,
                    slippage,
                )
                if self.cash >= 0:
                    break
        # Check margin after shorts
        short_mask = self.array[3, :] < 0
        liabilities = -self.array[3, short_mask] * self.array[2, short_mask]
        long_mask = self.array[3, :] > 0
        long_value = np.sum(self.array[3, long_mask] * self.array[1, long_mask])
        if np.sum(liabilities) > 2 * (self.cash + long_value):
            # Cover shorts by buying back
            short_positions = np.where(self.array[3, :] < 0)[0]
            for asset in short_positions:
                amount_to_cover = min(
                    -self.array[3, asset],
                    math.floor(
                        self.cash
                        / (
                            self.array[2, asset]
                            * (1 + slippage)
                            * (1 + transaction_fee)
                        )
                    ),
                )
                self.buy(
                    np.array([asset]),
                    np.array([amount_to_cover]),
                    transaction_fee,
                    slippage,
                )
                if self.net_liq_value() > 0 and np.sum(liabilities) <= 2 * (
                    self.cash + long_value
                ):
                    break

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
        normalize_features: bool = False,
    ) -> None:
        self.nb_stock = config.nb_stock
        self.initial_balance = config.initial_balance
        self.transaction_fee = config.transaction_fee
        self.slippage = config.slippage
        self.trade_fee = 0.02
        self.reward_scaling = config.reward_scaling
        self.max_step = config.ppo.n_steps
        self.is_train = train
        self.times_steps_per_day = config.time_steps_per_day
        self.drawdown_penalty = config.drawdown_penalty

        self.normalize_features = normalize_features
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

        if config.ppo.n_steps + self.hist_days >= self.max_day:
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
        actions = np.clip(actions, -1, 1)  # Keep range [-1, 1]

        # Sell actions
        sell_mask = actions < 0
        sell_indices = np.where(sell_mask)[0]
        sell_fractions = -actions[sell_mask]  # Fraction to sell (0 to 1)
        sell_amounts = np.zeros_like(sell_fractions)
        for i, idx in enumerate(sell_indices):
            if self.state.array[3, idx] > 0:  # Long position
                sell_amounts[i] = sell_fractions[i] * self.state.array[3, idx]
            else:  # Shorting or increasing short
                cash = self.state.cash * 0.6
                ask = self.state.array[2, idx]
                max_short = (2 * cash) / (
                    ask * (1 + self.slippage) * (1 + self.transaction_fee)
                )  # Margin limit
                sell_amounts[i] = sell_fractions[i] * max_short
        sell_amounts = np.floor(sell_amounts).astype(np.int32)
        self.state.sell(
            sell_indices,
            sell_amounts,
            self.transaction_fee,
            self.slippage,
            self.trade_fee,
        )
        self.num_sells += len(sell_indices)

        # Buy actions
        buy_mask = actions > 0
        buy_indices = np.where(buy_mask)[0]
        buy_fractions = actions[buy_mask]  # Fraction of cash to spend (0 to 1)
        buy_amounts = np.zeros_like(buy_fractions)
        cash = self.state.cash
        for i, idx in enumerate(buy_indices):
            ask = self.state.array[2, idx]
            if ask > 0:
                price = ask * (1 + self.slippage) * (1 + self.transaction_fee)
                buy_amounts[i] = buy_fractions[i] * (cash * 0.8 / price)
        buy_amounts = np.floor(buy_amounts).astype(np.int32)
        self.state.buy(
            buy_indices,
            buy_amounts,
            self.transaction_fee,
            self.slippage,
            self.trade_fee,
        )
        self.num_buys += len(buy_indices)

    @property
    def trades(self):
        return self.num_buys + self.num_sells

    def sample_prices(self):
        """Fetch precomputed bid and ask prices."""
        bids = self.bid_array[self.day, self.ticker_indices]
        asks = self.ask_array[self.day, self.ticker_indices]
        if np.any(bids > 1e5) or np.any(asks > 1e5):
            print(f"Warning: High prices at day {self.day}: bids={bids}, asks={asks}")
        self.state.set_prices(bids, asks)

    def step(self, actions, seed=0):
        sharpe_ratio = 0.0

        if not self.terminal:
            begin_total_asset = self.state.net_liq_value()
            self._execute_actions(actions)
            self.state.apply_interest_charge(0.05 / 252)
            self.state.maybe_liquidate(self.transaction_fee, self.slippage)
            self.next_trading_day()
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

            end_total_asset = self.state.net_liq_value()

            if end_total_asset <= 0:
                self.terminal = True
                self.reward = -1000
                log_return = -1000
                end_total_asset = 0
            else:
                # Update peak and drawdown
                self.peak_value = max(self.peak_value, end_total_asset)
                drawdown_t = (
                    (self.peak_value - end_total_asset) / self.peak_value
                    if self.peak_value > 0
                    else 0
                )
                self.max_drawdown = max(self.max_drawdown, drawdown_t)
                self.drawdowns.append(drawdown_t)

                # Compute reward with logarithmic penalty
                if begin_total_asset > 0 and end_total_asset > 0:
                    log_return = np.log(end_total_asset / begin_total_asset)
                else:
                    log_return = 0

                if len(self.drawdowns) > 1:
                    # penalty = (
                    #     self.drawdowns[-2] - self.drawdowns[-1]
                    # ) * self.drawdown_penalty
                    penalty = 0
                else:
                    penalty = 0

                self.reward = np.clip(log_return - penalty, -100, 100)

                self.asset_memory.append(end_total_asset)
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * self.reward_scaling

                self.returns.append(
                    (end_total_asset - begin_total_asset) / begin_total_asset
                )

        # Compute Sharpe Ratio
        if len(self.returns) > 1:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns, ddof=1)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

        self.terminal = self.day >= self.max_step + self.hist_days

        if self.terminal:
            print(end_total_asset)

        return (
            self.state.normalized_array
            if self.normalize_features
            else self.state.array,
            self.reward,
            self.terminal,
            False,
            {
                "returns": (
                    (self.state.net_liq_value() - self.initial_balance)
                    / self.initial_balance
                )
                * 100,
                "cagr": (
                    (max(1, self.state.net_liq_value()) / self.initial_balance)
                    ** ((self.day - self.hist_days + 1) / 252)
                    - 1.0
                )
                * 100,
                "trades": self.trades,
                "assets": self.state.net_liq_value(),
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": self.max_drawdown * 100,
            },
        )

    def reset(self, seed=0):
        self.cost = 0
        self.terminal = False
        self.rewards_memory = []
        self.returns = []
        self.drawdowns = []
        self.reward = 0
        self.asset_memory = [self.initial_balance]
        self.max_drawdown = 0
        self.num_buys = 0
        self.num_sells = 0
        self.peak_value = self.initial_balance
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
        return (
            self.state.normalized_array
            if self.normalize_features
            else self.state.array,
            {},
        )
