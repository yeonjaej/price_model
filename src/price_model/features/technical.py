"""Technical features computed from OHLCV alone.

Each feature is a small, stock-agnostic function of past prices. They're cheap and
weak individually; the model combines them.

All features are .over("ticker") so computation is isolated per ticker — no cross-stock
leakage even before the cross-sectional normalization step.
"""

from __future__ import annotations

import polars as pl

from price_model.features.base import Feature, register


@register
class Return5d(Feature):
    """5-day log return. Captures short-term momentum / mean reversion."""

    name = "return_5d"
    inputs = ("adj_close",)
    lookback_days = 5

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        return panel.with_columns(
            (pl.col("adj_close").log() - pl.col("adj_close").log().shift(5).over("ticker"))
            .alias(self.name)
        )


@register
class Momentum60(Feature):
    """60-day return (excluding the last 5d, to reduce overlap with short-term reversal)."""

    name = "momentum_60"
    inputs = ("adj_close",)
    lookback_days = 65

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        # Classic "momentum minus reversal": return from t-65 to t-5.
        c = pl.col("adj_close")
        return panel.with_columns(
            (c.shift(5).log() - c.shift(65).log()).over("ticker").alias(self.name)
        )


@register
class Vol20(Feature):
    """20-day realized volatility of daily log returns."""

    name = "vol_20"
    inputs = ("adj_close",)
    lookback_days = 21

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        log_ret = (c.log() - c.log().shift(1)).over("ticker")
        return panel.with_columns(
            log_ret.rolling_std(window_size=20).over("ticker").alias(self.name)
        )


@register
class Rsi14(Feature):
    """14-day RSI (Wilder's smoothing approximated with simple rolling means)."""

    name = "rsi_14"
    inputs = ("adj_close",)
    lookback_days = 15

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        delta = (c - c.shift(1)).over("ticker")
        gain = pl.when(delta > 0).then(delta).otherwise(0.0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
        avg_gain = gain.rolling_mean(window_size=14).over("ticker")
        avg_loss = loss.rolling_mean(window_size=14).over("ticker")
        rs = avg_gain / pl.when(avg_loss == 0).then(1e-12).otherwise(avg_loss)
        return panel.with_columns((100.0 - 100.0 / (1.0 + rs)).alias(self.name))


@register
class DistanceMA200(Feature):
    """(price - 200d MA) / 200d MA. Long-term trend deviation."""

    name = "distance_ma_200"
    inputs = ("adj_close",)
    lookback_days = 200

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        ma = c.rolling_mean(window_size=200).over("ticker")
        return panel.with_columns(((c - ma) / ma).alias(self.name))
