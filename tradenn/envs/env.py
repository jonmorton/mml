import math
import random
from typing import Optional

import gymnasium as gym
import numpy as np
import polars as pl
from attr import dataclass

from tradenn.config import Config

FEATURES = ["macd", "rsi", "cci", "adx", "ao", "bb", "mf", "vi"]
OHLCV = ["open", "high", "low", "close", "volume"]

LONG_ONLY = False
ACTION_SCALE = 0.1 / 50  # 5% of initial balance
SHORT_K = 0.15


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

    def __init__(self, num_assets, initial_balance, feat_days=7, ohlcv_days=14):
        self.num_assets = num_assets
        self.feat_days = feat_days
        self.ohlcv_days = ohlcv_days
        self.initial_balance = initial_balance
        self.bidask_mean = 0
        self.bidask_std = 1
        self.reset()

    def set_normstats(self, bidask_mean: float, bidask_std: float):
        self.bidask_mean = bidask_mean
        self.bidask_std = bidask_std

    def reset(self):
        self.array = np.zeros(
            (5 + len(FEATURES) * self.feat_days + 5 * self.ohlcv_days, self.num_assets),
            dtype=np.float32,
        )
        self.array[0, :] = self.initial_balance
        self.total_interest_expense = 0
        self.num_sells = 0
        self.num_buys = 0

    @property
    def shape(self):
        return self.array.shape

    def set_prices(self, bids: np.ndarray, asks: np.ndarray):
        self.array[1, :] = bids
        self.array[2, :] = asks

    def set_features(self, feat_img: np.ndarray, ohlcv_img: np.ndarray):
        """Sets precomputed feature and OHLCV images directly into the state array."""

        self.array[4 : 4 + len(FEATURES) * self.feat_days, :] = feat_img
        self.array[
            4 + len(FEATURES) * self.feat_days : 4
            + len(FEATURES) * self.feat_days
            + len(OHLCV) * self.ohlcv_days,
            :,
        ] = ohlcv_img

    def sell(
        self,
        assets: np.ndarray,
        amounts: np.ndarray,
        transaction_fee: float = 0.01,
        slippage: float = 0.01,
    ):
        if LONG_ONLY:
            amounts = np.minimum(amounts, self.array[3, assets])

        amounts = np.maximum(amounts, 0)
        prices = self.array[1, assets] * (1 - slippage)  # Bid price for selling
        proceeds = prices * amounts * (1 - transaction_fee)
        total_proceeds = np.sum(proceeds)

        new_positions = self.array[3, :].copy()
        new_positions[assets] -= amounts

        if not LONG_ONLY and (new_positions[assets] < 0).any():
            # Simulate the trade to check margin

            short_mask = new_positions < 0
            liabilities = np.sum(-self.array[3, short_mask] * self.array[2, short_mask])
            long_mask = new_positions > 0
            long_value = np.sum(self.array[3, long_mask] * self.array[1, long_mask])

            if liabilities > SHORT_K * (self.cash + long_value):
                return 0.0

        self.num_sells += (amounts > 0).sum()
        # Execute the trade
        self.array[0, :] += total_proceeds
        np.subtract.at(self.array[3, :], assets, amounts)

        return total_proceeds

    def buy(
        self,
        assets: np.ndarray,
        amounts: np.ndarray,
        transaction_fee: float = 0.01,
        slippage: float = 0.01,
    ):
        current_cash = self.array[0, 0]
        tot = 0
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
            tot += total_cost
            self.array[3, idx] += amounts_to_buy
            if amounts_to_buy > 0:
                self.num_buys += 1
            if current_cash <= 1:
                break
        self.array[0, :] = current_cash
        return tot

    def apply_interest_charge(self, lending_rate: float):
        short_positions = np.where(self.array[3, :] < 0, -self.array[3, :], 0)
        prices = (self.array[2, :] + self.array[1, :]) / 2.0
        interest_charge = np.sum(lending_rate * short_positions * prices)
        self.array[0, :] -= interest_charge
        self.total_interest_expense += interest_charge

    def maybe_liquidate(self, transaction_fee: float = 0.01, slippage: float = 0.01):
        if self.net_liq_value <= 0:
            print(f"*** NET LIQ < 0  (interest: {self.total_interest_expense})")
            return 0.0  # Let step handle termination

        total_liq = 0.0

        if self.cash < 0:
            print(
                f"Liquidate {self.cash} {self.total_interest_expense} *******************************"
            )
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
                total_liq += self.sell(
                    np.array([asset]),
                    np.array([amount_to_sell]),
                    transaction_fee,
                    slippage,
                )
                if self.cash >= 0:
                    break

        # Check margin after shorts
        short_mask = self.array[3, :] < 0
        liabilities = np.sum(-self.array[3, short_mask] * self.array[2, short_mask])
        long_mask = self.array[3, :] > 0
        long_value = np.sum(self.array[3, long_mask] * self.array[1, long_mask])
        if liabilities > SHORT_K * (self.cash + long_value):
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
                if amount_to_cover <= 0:
                    continue
                total_liq += self.buy(
                    np.array([asset]),
                    np.array([amount_to_cover]),
                    transaction_fee,
                    slippage,
                )
                if self.net_liq_value > 0 and liabilities <= SHORT_K * (
                    self.cash + long_value
                ):
                    break

        return total_liq

    @property
    def net_liq_value(self):
        cash = self.array[0, 0]
        positions = self.array[3, :]
        bids = self.array[1, :]
        asks = self.array[2, :]
        asset_values = np.where(positions > 0, positions * bids, positions * asks)
        return (cash + np.sum(asset_values)).item()

    @property
    def cash(self):
        return self.array[0][0]

    @property
    def observation(self):
        array = self.array.copy()
        array[0, :] /= self.initial_balance
        array[1, :] = (array[1, :] - self.bidask_mean) / self.bidask_std
        array[2, :] = (array[2, :] - self.bidask_mean) / self.bidask_std
        array[3, :] /= ACTION_SCALE * self.initial_balance

        short_mask = self.array[3, :] < 0
        liabilities = np.sum(-self.array[3, short_mask] * self.array[2, short_mask])
        long_mask = self.array[3, :] > 0
        long_value = np.sum(self.array[3, long_mask] * self.array[1, long_mask])
        array[-1, :] = liabilities / (self.cash + long_value)

        return array


@dataclass
class StockEnvData:
    df: pl.DataFrame
    bid_ask: pl.DataFrame
    stats: dict[str, dict[str, pl.Series]]

    max_day: int
    num_days: int
    num_tickers: int
    ticker_to_idx: dict[str, int]

    features: np.ndarray
    ohlcv: np.ndarray

    bid_arrays: np.ndarray
    ask_arrays: np.ndarray


class StockEnv(gym.Env):
    def __init__(
        self,
        config: Config,
        data: StockEnvData,
        train: bool = False,
    ) -> None:
        self.nb_stock = config.nb_stock
        self.initial_balance = config.initial_balance
        self.transaction_fee = config.transaction_fee
        self.slippage = config.slippage
        self.max_step = config.ppo.n_steps
        self.is_train = train
        self.drawdown_penalty = config.drawdown_penalty

        self.normalize_features = config.normalize_features
        self.hist_days = 31  # max(feat_days=7, ohlcv_days=30) from State

        self.state = State(
            self.nb_stock, self.initial_balance, feat_days=5, ohlcv_days=15
        )
        self.state.set_normstats(
            bidask_mean=data.stats["means"]["close"].to_numpy().item(),
            bidask_std=data.stats["stds"]["close"].to_numpy().item(),
        )

        self.features_array = data.features
        self.ohlcv_array = data.ohlcv
        self.bid_arrays = data.bid_arrays
        self.ask_arrays = data.ask_arrays
        self.max_day = data.max_day
        self.sorted_tickers = data.df["ticker"].unique().to_list()
        self.ticker_to_idx = data.ticker_to_idx

        if config.ppo.n_steps + self.hist_days >= self.max_day:
            raise ValueError(
                f"Number of steps ({config.ppo.n_steps}) + history days ({self.hist_days}) exceeds max days ({self.max_day})"
            )

        self.action_space = gym.spaces.Box(-1, 1, (self.state.num_assets,))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.state.shape)
        self.train = train
        self.training_progress = 0.0
        self.reset()

    @classmethod
    def prepare_data(
        cls,
        df: pl.DataFrame,
        bid_ask: pl.DataFrame,
        stats: dict[str, dict[str, pl.Series]],
        normalize: bool = False,
    ) -> StockEnvData:
        # Preprocess data
        dataframe = df.fill_nan(0).fill_null(0)
        sorted_days = sorted(dataframe["day"].unique().to_list())
        sorted_tickers = sorted(dataframe["ticker"].unique().to_list())
        num_days = len(sorted_days)
        num_tickers = len(sorted_tickers)
        max_day = int(dataframe["day"].max())  # type: ignore
        ticker_to_idx = {ticker: idx for idx, ticker in enumerate(sorted_tickers)}

        # Precompute features and OHLCV arrays
        features_array = np.full(
            (num_days, num_tickers, len(FEATURES)), np.nan, dtype=np.float32
        )
        ohlcv_array = np.full((num_days, num_tickers, 5), np.nan, dtype=np.float32)
        for row in dataframe.iter_rows(named=True):
            day_idx = sorted_days.index(row["day"])
            ticker_idx = ticker_to_idx[row["ticker"]]
            features_array[day_idx, ticker_idx, :] = [row[f] for f in FEATURES]
            ohlcv_array[day_idx, ticker_idx, :] = [row[f] for f in OHLCV]

        if normalize:
            for i, f in enumerate(FEATURES):
                features_array[:, :, i] -= stats["means"][f].to_numpy().item()
                features_array[:, :, i] /= stats["stds"][f].to_numpy().item()

        # Precompute bid and ask arrays for all time steps
        bid_ask = bid_ask.sort(["day", "ticker", "time_step"])

        # Create 3D arrays for bid/ask prices [day, time_step, ticker]
        bid_arrays = np.full(
            (num_days, num_tickers),
            np.nan,
            dtype=np.float32,
        )
        ask_arrays = np.full(
            (num_days, num_tickers),
            np.nan,
            dtype=np.float32,
        )

        # Fill the arrays with bid/ask data
        for row in bid_ask.filter(time_step=0).iter_rows(named=True):
            day_idx = sorted_days.index(row["day"])
            ticker_idx = ticker_to_idx[row["ticker"]]

            bid_arrays[day_idx, ticker_idx] = row["bid"]
            ask_arrays[day_idx, ticker_idx] = row["ask"]

        return StockEnvData(
            df=dataframe,
            bid_ask=bid_ask,
            stats=stats,
            max_day=max_day,
            num_days=num_days,
            num_tickers=num_tickers,
            ticker_to_idx=ticker_to_idx,
            features=features_array,
            ohlcv=ohlcv_array,
            bid_arrays=bid_arrays,
            ask_arrays=ask_arrays,
        )

    def select_tickers(self):
        """Select random tickers and store their indices."""
        selected_tickers = random.sample(self.sorted_tickers, self.nb_stock)
        random.shuffle(selected_tickers)
        self.ticker_indices = np.array(
            [self.ticker_to_idx[ticker] for ticker in selected_tickers]
        )

    def next_trading_day(self, day: Optional[int] = None):
        """Update the current day and reset time step."""
        if day is not None:
            self.day = day
        else:
            self.day += 1
        if self.day >= self.max_day:
            raise ValueError(
                f"Day {self.day} is out of range for max of {self.max_day}"
            )

    def _execute_actions(self, actions):
        penalty = 0.0

        training_progress = self.training_progress if self.is_train else 1.0
        scale = min(1.0, training_progress * 2)
        actions = np.nan_to_num(actions, nan=0, posinf=1, neginf=-1)
        actions = np.round(
            np.tanh(actions) * self.initial_balance * ACTION_SCALE * scale
        )

        # Sell actions
        sell_mask = actions < 0
        sell_indices = np.where(sell_mask)[0]
        sell_amounts = -actions[sell_mask]

        self.state.sell(
            sell_indices,
            sell_amounts,
            self.transaction_fee,
            self.slippage,
        )

        # Buy actions
        buy_mask = actions > 0
        buy_indices = np.where(buy_mask)[0]
        buy_amounts = actions[buy_mask]
        self.state.buy(
            buy_indices,
            buy_amounts,
            self.transaction_fee,
            self.slippage,
        )

        return penalty

    @property
    def trades(self):
        return self.state.num_buys + self.state.num_sells

    def sample_prices(self):
        """Fetch bid and ask prices for the current day and time step."""
        bids = self.bid_arrays[self.day, self.ticker_indices]
        asks = self.ask_arrays[self.day, self.ticker_indices]
        if np.any(bids > 1e5) or np.any(asks > 1e5):
            print(f"Warning: High prices at day {self.day}: bids={bids}, asks={asks}")
        self.state.set_prices(bids, asks)

    def update_features(self):
        feat_img = (
            self.features_array[
                self.day - self.state.feat_days : self.day, self.ticker_indices, :
            ]
            .transpose(2, 0, 1)
            .reshape(-1, self.nb_stock)
        )
        P0 = np.stack(
            [
                self.ohlcv_array[
                    self.day - self.state.ohlcv_days - 1, self.ticker_indices, 3
                ]
            ]
            * 4
            + [
                self.ohlcv_array[
                    self.day - self.state.ohlcv_days - 1,
                    self.ticker_indices,
                    -1,
                ]
            ],
            axis=-1,
        )
        if (P0 == 0.0).any():
            raise ValueError("Bad P0 values")
        ohlcv_img = (
            (
                (
                    self.ohlcv_array[
                        self.day - self.state.ohlcv_days : self.day,
                        self.ticker_indices,
                        :,
                    ]
                )
                / P0[None, :]
                - 1.0
            )
            .transpose(2, 0, 1)
            .reshape(-1, self.nb_stock)
        )
        self.state.set_features(feat_img, ohlcv_img)

    def step(self, actions, seed=0):
        sharpe_ratio = 0.0
        penalty = 0.0

        if not self.terminal:
            begin_total_asset = self.state.net_liq_value

            # Execute actions at this time step
            penalty += self._execute_actions(actions)

            # Apply interest charges at the end of the day
            self.state.apply_interest_charge(0.01 / 251)

            total_liq = self.state.maybe_liquidate(self.transaction_fee, self.slippage)

            if total_liq > 0:
                liq_penalty = total_liq / self.initial_balance * 10 + 0.5
                print(f"Liquidation penalty {liq_penalty} for {total_liq}")
                penalty += liq_penalty

            # Move to the next day
            self.next_trading_day()

            self.sample_prices()
            self.update_features()

            end_total_asset = self.state.net_liq_value

            if end_total_asset <= 0:
                print("END TOTAL ASSET < 0")
                self.terminal = True
                self.reward = -1.1
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
                self.returns.append(
                    (end_total_asset - begin_total_asset) / begin_total_asset
                )

                # Compute reward with logarithmic penalty
                if begin_total_asset > 0 and end_total_asset > 0:
                    log_return = np.log(max(1, end_total_asset) / begin_total_asset)
                else:
                    log_return = 0

                self.reward = np.clip(log_return - penalty, -7, 7) / 7

            self.asset_memory.append(end_total_asset)
            self.rewards_memory.append(self.reward)
            self.reward = self.reward

        else:
            end_total_asset = max(0, self.state.net_liq_value)

        # Compute Sharpe Ratio only if we have returns data
        if len(self.returns) > 1:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns, ddof=1)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

        # Check if we've reached the terminal state (max steps)
        self.terminal = self.terminal or self.day >= self.max_step + self.start_day - 1

        return (
            self.state.observation,
            self.reward,
            self.terminal,
            self.trunc,
            {
                "returns": (end_total_asset / self.initial_balance),
                "cagr": (
                    (max(1, end_total_asset) / self.initial_balance)
                    ** ((self.day - self.start_day + 1) / 252)
                    - 1.0
                )
                * 100,
                "trades": self.trades,
                "assets": end_total_asset,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "day": self.day,
            },
        )

    def reset(self, seed=0):
        self.cost = 0
        self.terminal = False
        self.trunc = False
        self.rewards_memory = []
        self.returns = []
        self.drawdowns = []
        self.reward = 0
        self.asset_memory = [self.initial_balance]
        self.max_drawdown = 0
        self.peak_value = self.initial_balance
        self.state.reset()
        self.select_tickers()
        self.start_day = random.randint(
            self.hist_days, self.max_day - self.max_step - 1
        )
        self.next_trading_day(day=self.start_day)
        self.sample_prices()
        self.update_features()
        return (
            self.state.observation if self.normalize_features else self.state.array,
            {},
        )
