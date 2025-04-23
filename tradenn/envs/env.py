import math
from typing import Optional

import gymnasium as gym
import numpy as np
import polars as pl
import torch
from attr import dataclass

from tradenn.config import Config

FEATURES = [
    "rsi",
    "macd",
    "cci",
    "adx",
    "bb",
    "logvol_7d",
]
OHLCV = ["open", "high", "low", "close", "volume"]

LONG_ONLY = True
MAINT_MARGIN = 0.5
BUFFER = 0.001


class State:
    """
    Manages the environment state as three numpy arrays for different types of features
    (global, per-asset, and per-day-per-asset).

    a_f (global) layout:
        [0] cash
        [1] exposure (long value / net liquidation value)

    a_fa (per-asset) layout:
        [0] bid prices
        [1] ask prices
        [2] asset positions (long/short)

    a_dfa (per-day-per-asset) layout:
        [:len(FEATURES)] features (e.g., MACD, RSI, etc.)
        [len(FEATURES):len(FEATURES) + len(OHLCV)] OHLCV data (open, high, low, close, volume)

    """

    def __init__(self, num_assets, num_asset_feats, initial_balance, hist_days=7):
        self.num_assets = num_assets
        self.hist_days = hist_days
        self.initial_balance = initial_balance
        self.bidask_mean = 0
        self.bidask_std = 1
        self.num_asset_feats = num_asset_feats
        self.reset()

    def set_day(self, date):
        self.a_f[1] = date.timetuple().tm_yday / 365.0 * 2.0 - 1.0
        self.a_f[2] = date.timetuple().tm_wday / 7.0 * 2.0 - 1.0

    def set_normstats(self, bidask_mean: float, bidask_std: float):
        self.bidask_mean = bidask_mean
        self.bidask_std = bidask_std

    def reset(self):
        # feature, feature-asset, day-feature-asset
        self.a_f = np.zeros((3 + (1 if not LONG_ONLY else 0),), dtype=np.float32)
        self.a_fa = np.zeros(
            (3 + self.num_asset_feats, self.num_assets), dtype=np.float32
        )
        self.a_dfa = np.zeros(
            (self.hist_days, len(FEATURES) + len(OHLCV), self.num_assets),
            dtype=np.float32,
        )
        self.a_f[0] = self.initial_balance
        self.total_interest_expense = 0
        self.num_sells = 0
        self.num_buys = 0

    @property
    def space(self):
        return gym.spaces.Dict(
            {
                "feats": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.a_f.shape,
                    dtype=np.float32,
                ),
                "asset_feats": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.a_fa.shape,
                    dtype=np.float32,
                ),
                "day_asset_feats": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.a_dfa.shape,
                    dtype=np.float32,
                ),
            }
        )

    def set_prices(self, bids: np.ndarray, asks: np.ndarray):
        self.a_fa[0, :] = bids
        self.a_fa[1, :] = asks

    def set_features(
        self, day_feats: np.ndarray, ohlcv: np.ndarray, asset_feats: np.ndarray
    ):
        """Sets precomputed feature and OHLCV images directly into the state array."""

        # feat_img.fill(0)
        # ohlcv_img.fill(0)

        self.a_dfa[:, : len(FEATURES), :] = day_feats
        self.a_dfa[:, len(FEATURES) :, :] = ohlcv
        self.a_fa[3:, :] = asset_feats

    def sell(
        self,
        assets: np.ndarray,
        amounts: np.ndarray,
        transaction_fee: float = 0.01,
    ):
        if (amounts < 0).any():
            raise ValueError(f"Negative amounts to sell: {amounts}")

        if LONG_ONLY:
            amounts = np.minimum(amounts, self.a_fa[2, assets])

        amounts = np.maximum(amounts, 0)
        prices = self.a_fa[0, assets]
        proceeds = prices * amounts * (1 - transaction_fee)
        total_proceeds = np.sum(proceeds)

        self.num_sells += (amounts > 0).sum()
        self.a_f[0] += total_proceeds
        np.subtract.at(self.a_fa[2, :], assets, amounts)

        return total_proceeds

    def buy(
        self,
        assets: np.ndarray,
        amounts: np.ndarray,
        transaction_fee: float = 0.01,
    ):
        current_cash = self.cash
        tot = 0
        for i in range(len(assets)):
            idx = assets[i]
            ask = self.a_fa[1, idx]
            if ask > 0:
                price = ask * (1 + transaction_fee)
                available_amount = np.floor(current_cash / (price * (1 + BUFFER)))
                amounts_to_buy = max(0, min(amounts[i], available_amount))
                total_cost = price * amounts_to_buy
            else:
                raise ValueError(f"Invalid ask price for asset {idx}: {ask}")
            if LONG_ONLY and total_cost >= current_cash:
                amounts_to_buy = max(0, amounts_to_buy - 1)
                total_cost = price * amounts_to_buy
            if amounts_to_buy <= 0:
                continue
            current_cash -= total_cost
            tot += total_cost
            self.a_fa[2, idx] += amounts_to_buy
            self.num_buys += 1
        self.a_f[0] = current_cash
        return tot

    def apply_interest_charge(self, lending_rate: float):
        short_positions = np.where(self.a_fa[2, :] < 0, -self.a_fa[2, :], 0)
        prices = (self.a_fa[1, :] + self.a_fa[0, :]) / 2.0
        interest_charge = np.sum(lending_rate * short_positions * prices)
        if interest_charge > 0:
            print(f"Interest charge: {interest_charge}")
        self.a_f[0] -= interest_charge
        self.total_interest_expense += interest_charge

    # buy method remains unchanged

    def maybe_liquidate(self, transaction_fee: float = 0.01):
        if self.cash > 0 and self.margin_shortfall <= 0:
            return 0.0

        total_liq = 0.0
        did_something = True

        print(
            f"Margin shortfall {self.margin_shortfall} exists, attempting liquidation"
        )

        while (self.cash < 0 or self.margin_shortfall > 0) and did_something:
            did_something = False

            if self.cash < 0:
                # Amount of buys to liquidate to raise cash
                liquidation_ratio = min(
                    1.0,
                    1.005
                    * (
                        -self.cash
                        / np.sum(
                            np.where(
                                self.a_fa[2, :] > 0,
                                self.a_fa[2, :]
                                * self.a_fa[0, :]
                                * (1 - transaction_fee),
                                0,
                            )
                        )
                    ),
                )
                print(
                    f"Raising cash={self.cash} by liquidating {liquidation_ratio} of buys"
                )
            else:
                liquidation_ratio = min(
                    1.0,
                    1.005
                    * (1 + transaction_fee)
                    * (1 - self.net_liq_value / (MAINT_MARGIN * self.exposure)),
                )

            for asset in range(self.num_assets):
                position = self.a_fa[2, asset]
                if position == 0:
                    continue
                if position < 0 and self.cash < 0:
                    continue

                amount = np.ceil(np.abs(position) * liquidation_ratio)

                if amount == 0:
                    continue

                if position > 0:
                    liq = self.sell(
                        np.array([asset]),
                        np.array([amount]),
                        transaction_fee,
                    )
                else:
                    price = self.a_fa[1, asset] * (1 + transaction_fee)
                    total_cost = price * amount
                    self.a_f[0] -= total_cost
                    self.a_fa[2, asset] += amount
                    liq = total_cost

                if liq > 0:
                    total_liq += liq
                    did_something = True

        if self.margin_shortfall > 0:
            print(
                f"Warning: Margin shortfall {self.margin_shortfall} exists after liquidating"
            )
            print(self.positions)

        if total_liq > 0:
            if self.cash < 0:
                print(f"Warning: Cash is negative after liquidation {self.cash}")

        return total_liq

    @property
    def net_liq_value(self):
        positions = self.a_fa[2, :]
        bids = self.a_fa[0, :]
        asks = self.a_fa[1, :]
        asset_values = np.where(positions > 0, positions * bids, positions * asks)
        return (self.cash + np.sum(asset_values)).item()

    @property
    def cash(self):
        return self.a_f[0]

    @property
    def observation(self):
        f = self.a_f.copy()
        f[0] = f[0] / self.initial_balance * 2.0 - 1.0
        if not LONG_ONLY:
            f[-1] = self.exposure * MAINT_MARGIN / self.net_liq_value

        fa = self.a_fa.copy()
        fa[0, :] = (fa[0, :] - self.bidask_mean) / self.bidask_std
        fa[1, :] = (fa[1, :] - self.bidask_mean) / self.bidask_std
        fa[2, :] = fa[2, :] * self.a_fa[0, :] / self.initial_balance * 10 - 5

        return {
            "feats": f,
            "asset_feats": fa,
            "day_asset_feats": self.a_dfa,
        }

    @property
    def long_value(self):
        return np.sum(
            np.where(self.a_fa[2, :] > 0, self.a_fa[2, :] * self.a_fa[0, :], 0)
        )

    @property
    def margin_shortfall(self):
        return self.exposure * MAINT_MARGIN - self.net_liq_value

    @property
    def exposure(self):
        return np.sum(
            np.where(
                self.a_fa[2, :] > 0,
                self.a_fa[2, :] * self.a_fa[0, :],
                -self.a_fa[2, :] * self.a_fa[1, :],
            )
        )

    @property
    def positions(self):
        return self.a_fa[2, :]


@dataclass
class StockEnvData:
    df: pl.DataFrame
    bid_ask: pl.DataFrame
    stats: dict[str, dict[str, pl.Series]]
    tickers: list[str]
    max_day: int
    num_days: int
    num_tickers: int
    ticker_to_idx: dict[str, int]
    features: np.ndarray
    ohlcv: np.ndarray
    dates: np.ndarray
    bid_arrays: np.ndarray
    ask_arrays: np.ndarray
    bidask_mean: float
    bidask_std: float
    asset_features: np.ndarray


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
        self.max_step = config.nb_days
        self.is_train = train

        self.normalize_features = config.normalize_features
        self.hist_days = config.hist_days + 1

        self.state = State(
            self.nb_stock, 1, self.initial_balance, hist_days=config.hist_days
        )
        self.state.set_normstats(data.bidask_mean, data.bidask_std)

        self.asset_features = data.asset_features
        self.features_array = data.features
        self.ohlcv_array = data.ohlcv

        self.date_array = data.dates
        self.bid_arrays = data.bid_arrays
        self.ask_arrays = data.ask_arrays
        self.max_day = data.max_day
        self.sorted_tickers = data.tickers
        self.ticker_to_idx = data.ticker_to_idx

        if config.nb_days + self.hist_days >= self.max_day:
            raise ValueError(
                f"Number of steps ({config.ppo.n_steps}) + history days ({self.hist_days}) exceeds max days ({self.max_day})"
            )

        # Increase action space size by 1 for cash allocation
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.state.num_assets,))
        self.observation_space = self.state.space
        self.train = train
        self.training_progress = 0.0

        # keep a rolling AR(1) noise buffer
        self.feat_noise = np.zeros(
            (config.hist_days, len(FEATURES), self.nb_stock), dtype=np.float32
        )
        self.ohlcv_noise = np.zeros(
            (config.hist_days, len(OHLCV), self.nb_stock), dtype=np.float32
        )

        self.reset(seed=0)

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
        sorted_days = sorted(dataframe["day_idx"].unique().to_list())
        sorted_tickers = sorted(dataframe["ticker"].unique().to_list())
        num_days = len(sorted_days)
        num_tickers = len(sorted_tickers)
        max_day = int(dataframe["day_idx"].max())  # type: ignore
        ticker_to_idx = {ticker: idx for idx, ticker in enumerate(sorted_tickers)}

        # Precompute features and OHLCV arrays
        features_array = np.full(
            (num_days, num_tickers, len(FEATURES)), np.nan, dtype=np.float32
        )
        ohlcv_array = np.full((num_days, num_tickers, 5), np.nan, dtype=np.float32)
        date_array = np.full((num_days,), np.nan, dtype=object)
        asset_feat_array = np.full((num_days, num_tickers, 1), np.nan, dtype=np.float32)
        for row in dataframe.iter_rows(named=True):
            day_idx = sorted_days.index(row["day_idx"])
            ticker_idx = ticker_to_idx[row["ticker"]]
            features_array[day_idx, ticker_idx, :] = [row[f] for f in FEATURES]
            ohlcv_array[day_idx, ticker_idx, :] = [row[f] for f in OHLCV]
            date_array[day_idx] = row["date"]
            asset_feat_array[day_idx, ticker_idx, 0] = row["logmarketcap"]

        if normalize:
            for i, f in enumerate(FEATURES):
                features_array[:, :, i] = (
                    features_array[:, :, i] - stats["means"][f].item()
                ) / stats["stds"][f].item()
            asset_feat_array[:, :, 0] = (
                asset_feat_array[:, :, 0] - stats["means"]["logmarketcap"].item()
            ) / stats["stds"]["logmarketcap"].item()

        # Precompute bid and ask arrays for all time steps
        bid_ask = bid_ask.sort(["day_idx", "ticker", "time_step"])

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
        max_ts = int(bid_ask["time_step"].max())  # type: ignore
        for row in bid_ask.filter(time_step=max_ts).iter_rows(named=True):
            day_idx = row["day_idx"]
            ticker_idx = ticker_to_idx[row["ticker"]]

            bid_arrays[day_idx, ticker_idx] = row["bid"]
            ask_arrays[day_idx, ticker_idx] = row["ask"]

        # for i in range(bid_arrays.shape[0]):
        #     for j in range(bid_arrays.shape[1]):
        #         if np.isnan(bid_arrays[i, j]):
        #             print(
        #                 f"Missing bid price at day {i}, ticker {j}: {bid_arrays[i, j]}"
        #             )
        #         if np.isnan(ask_arrays[i, j]):
        #             print(
        #                 f"Missing ask price at day {i}, ticker {j}: {ask_arrays[i, j]}"
        #             )

        return StockEnvData(
            df=dataframe,
            bid_ask=bid_ask,
            stats=stats,
            tickers=sorted_tickers,
            max_day=max_day,
            num_days=num_days,
            num_tickers=num_tickers,
            ticker_to_idx=ticker_to_idx,
            features=features_array,
            ohlcv=ohlcv_array,
            dates=date_array,
            bid_arrays=bid_arrays,
            ask_arrays=ask_arrays,
            bidask_mean=float(stats["means"]["close"].to_numpy().item()),
            bidask_std=float(stats["stds"]["close"].to_numpy().item()),
            asset_features=asset_feat_array,
        )

    def select_tickers(self):
        """Select random tickers and store their indices."""
        selected_tickers = self.random.choice(self.sorted_tickers, self.nb_stock)
        self.random.shuffle(selected_tickers)
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

        self.state.set_day(self.date_array[self.day])

    def _execute_actions(self, actions):
        """Execute buy/sell actions based on the provided action array."""
        assets = np.arange(self.state.num_assets)
        # SELL: action < 0 → sell fraction of current holdings
        sell_mask = actions < 0
        if np.any(sell_mask):
            sell_assets = assets[sell_mask]
            pos = self.state.a_fa[2, sell_mask]
            sell_amounts = np.floor(pos * (-actions[sell_mask])).astype(int)
            self.state.sell(sell_assets, sell_amounts, self.transaction_fee)
        # BUY: action > 0 → allocate fraction of cash to each asset
        buy_mask = actions > 0
        if np.any(buy_mask):
            buy_assets = assets[buy_mask]
            ask_prices = self.state.a_fa[1, buy_mask]
            alloc = actions[buy_mask]
            # number of shares = floor( fraction * cash / (ask*(1+fee)) )
            buy_amounts = np.floor(
                self.state.cash * alloc / (ask_prices * (1 + self.transaction_fee))
            ).astype(int)
            self.state.buy(buy_assets, buy_amounts, self.transaction_fee)
            if (self.state.a_fa[2, :] < 0).any():
                raise ValueError(
                    f"After Buy Negative positions detected: {self.state.a_fa[2, :]}"
                )
        return 0.0

    @property
    def trades(self):
        return self.state.num_buys + self.state.num_sells

    def sample_prices(self):
        """Fetch bid and ask prices for the current day and time step."""
        bids = self.bid_arrays[self.day, self.ticker_indices]
        asks = self.ask_arrays[self.day, self.ticker_indices]
        if np.any(bids > 1e5) or np.any(asks > 1e5):
            print(f"Warning: High prices at day {self.day}: bids={bids}, asks={asks}")
        if np.any(np.isnan(bids)) or np.any(np.isnan(asks)):
            print(self.day, self.ticker_indices)
            raise ValueError(
                f"NaN values in bid/ask prices at day {self.day}: bids={bids}, asks={asks}"
            )
        self.state.set_prices(bids, asks)

    def update_features(self):
        day_feats = self.features_array[
            self.day - self.state.hist_days : self.day, self.ticker_indices, :
        ].transpose(0, 2, 1)
        P0 = np.stack(
            [
                self.ohlcv_array[
                    self.day - self.state.hist_days - 1, self.ticker_indices, 3
                ]
            ]
            * 4
            + [
                self.ohlcv_array[
                    self.day - self.state.hist_days - 1,
                    self.ticker_indices,
                    -1,
                ]
            ],
            axis=-1,
        )
        if (P0 == 0.0).any():
            raise ValueError("Bad P0 values")
        ohlcv = (
            (
                self.ohlcv_array[
                    self.day - self.state.hist_days : self.day,
                    self.ticker_indices,
                    :,
                ]
            )
            / P0[None, :]
            - 1.0
        ).transpose(0, 2, 1)

        asset_feats = self.asset_features[self.day, self.ticker_indices, :].T

        # if self.train:
        #     # AR(1) parameters
        #     phi_feat, sigma_feat = 0.8, 0.03
        #     phi_ohlcv, sigma_ohlcv = 0.8, 0.01

        #     # generate one new noise slice: (feats, stocks)
        #     new_fn = phi_feat * self.feat_noise[-1] + sigma_feat * self.random.normal(
        #         size=self.feat_noise[-1].shape
        #     )
        #     # shift window and append
        #     self.feat_noise = np.roll(self.feat_noise, -1, axis=0)
        #     self.feat_noise[-1] = new_fn
        #     # add per-day noise from the buffer
        #     day_feats = day_feats + self.feat_noise

        #     new_on = phi_ohlcv * self.ohlcv_noise[
        #         -1
        #     ] + sigma_ohlcv * self.random.normal(size=self.ohlcv_noise[-1].shape)
        #     self.ohlcv_noise = np.roll(self.ohlcv_noise, -1, axis=0)
        #     self.ohlcv_noise[-1] = new_on
        #     ohlcv = ohlcv + self.ohlcv_noise

        self.state.set_features(day_feats, ohlcv, asset_feats)

    def step(self, actions):
        sharpe_ratio = 0.0
        penalty = 0.0

        if not self.terminal:
            begin_total_asset = self.state.net_liq_value

            # Execute actions
            penalty += self._execute_actions(actions)

            if (self.state.a_fa[2, :] < 0).any():
                print(f"Warning: Negative positions detected: {self.state.a_fa[2, :]}")

            if not LONG_ONLY:
                # Apply interest charges
                self.state.apply_interest_charge(0.05 / 251)

                # Attempt liquidation if needed
                total_liq = self.state.maybe_liquidate(self.transaction_fee)
            else:
                total_liq = 0.0

            # Check for insolvency
            if (
                self.state.net_liq_value <= 0
                or self.state.cash < 0
                or self.state.margin_shortfall > 0
            ):
                self.terminal = True
                self.reward = -10.0  # Severe penalty for insolvency
                end_total_asset = 0.0
                print(
                    f"Terminated: Cash < 0 or margin shortfall > 0, cash={self.state.cash}, margin_shortfall={self.state.margin_shortfall} net_liq_value={self.state.net_liq_value}"
                )
            else:
                if total_liq > 0:
                    self.num_liquidations += 1
                    liq_penalty = total_liq / self.initial_balance * 4
                    penalty += liq_penalty
                    # print(
                    #     f"Liquidation penalty {liq_penalty} for {total_liq}  eq={self.state.net_liq_value} exposure={self.state.exposure} cash={self.state.cash}",
                    #     self.state.net_liq_value,
                    #     self.asset_memory[-1],
                    #     self.asset_memory[-2],
                    # )

                # if self.state.cash < self.initial_balance * 0.05:
                #     cash_penalty = (
                #         1 - self.state.cash / (self.initial_balance * 0.05)
                #     ) ** 2.0
                #     penalty += cash_penalty
                # print(f"Cash penalty {cash_penalty}  cash={self.state.cash}")

                # Move to next day
                self.next_trading_day()
                self.sample_prices()
                self.update_features()

                end_total_asset = self.state.net_liq_value

                if end_total_asset <= 0:
                    self.terminal = True
                    self.reward = -10.0
                    print("END TOTAL ASSET <= 0 ", begin_total_asset, end_total_asset)
                    print("Actions: ", actions)
                    # print("State before: ", state_before)
                    print("State after: ", self.state.observation)
                    # print("assets before:", assets_before)
                    print("assets after:", self.state.a_fa[2, :])

                    end_total_asset = 0.0
                else:
                    # Update peak and drawdown
                    self.peak_value = max(self.peak_value, end_total_asset)
                    drawdown_t = (
                        (self.peak_value - end_total_asset) / self.peak_value
                        if self.peak_value > 0
                        else 0
                    )
                    delta_drawdown = drawdown_t - self.max_drawdown
                    if delta_drawdown > 0:
                        penalty += delta_drawdown
                    self.max_drawdown = max(self.max_drawdown, drawdown_t)
                    self.returns.append(
                        (end_total_asset - begin_total_asset) / begin_total_asset
                    )

                    # if len(self.returns) >= 4:
                    #     penalty += 0.01 * np.std(self.returns[-4:])

                    self.reward = (
                        np.clip(np.log(end_total_asset / begin_total_asset) * 5, -1, 1)
                        - penalty
                    )

            self.asset_memory.append(end_total_asset)
            self.rewards_memory.append(self.reward)

        else:
            end_total_asset = max(0, self.state.net_liq_value)

        # Compute Sharpe Ratio
        if len(self.returns) > 1:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns, ddof=1)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

        self.terminal = self.terminal or self.day >= self.max_step + self.start_day - 1

        # print(
        #     f"reward={self.reward:.3f} prog={self.training_progress:.2f} day={self.day} returns={self.returns[-1]:.3f} penalty={penalty:.3f}"
        # )
        # print(actions)
        # if self.terminal:
        #     print(
        #         f"Cumulative returns: {np.sum(self.returns):.3f}  total return: {end_total_asset / self.initial_balance:.3f}  cum reward: {np.sum(self.rewards_memory):.3f}  sharpe: {sharpe_ratio:.3f}  max drawdown: {self.max_drawdown:.3f}"
        #     )

        cagr = (end_total_asset / self.initial_balance) ** (
            252 / (self.day - self.start_day + 1)
        ) - 1.0

        return (
            self.state.observation,
            self.reward,
            self.terminal,
            self.trunc,
            {
                "returns": (
                    (end_total_asset - self.initial_balance) / self.initial_balance
                ),
                "cagr": cagr,
                "trades": self.trades,
                "assets": end_total_asset,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "day": self.day,
                "num_liquidations": self.num_liquidations,
            },
        )

    def reset(self, seed=None):
        if seed is not None:
            self.random = np.random.RandomState(seed)

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
        self.num_liquidations = 0

        self.state.reset()
        self.select_tickers()
        self.start_day = self.random.randint(
            self.hist_days, self.max_day - self.max_step - 1
        )
        self.next_trading_day(day=self.start_day)
        self.sample_prices()
        self.update_features()

        return self.state.observation, {}
