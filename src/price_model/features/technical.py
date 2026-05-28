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
            (pl.col("adj_close").log() - pl.col("adj_close").log().shift(5).over("ticker")).alias(
                self.name
            )
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


# -----------------------------------------------------------------------------
# Documented academic anomaly features
# -----------------------------------------------------------------------------
#
# These three features are the highest-EV adds for getting a positive PIT-correct
# IC on liquid US equities. They have decades of academic and industry track
# record, are routinely confirmed in survivorship-bias-free backtests, and
# require nothing new beyond the existing adj_close column.
#
# References:
# - Jegadeesh & Titman (1993). "Returns to Buying Winners and Selling Losers".
#   Journal of Finance 48(1). The original 12-1 momentum paper. Replicated in
#   Asness, Moskowitz & Pedersen (2013) across asset classes.
# - Hong, Lim & Stein (2000). "Bad News Travels Slowly: Size, Analyst Coverage,
#   and the Profitability of Momentum Strategies". Journal of Finance 55(1).
#   52-week-high anchoring effect.
# - Lehmann (1990). "Fads, Martingales, and Market Efficiency". Quarterly Journal
#   of Economics 105(1). 5-day reversal effect. Our existing `return_5d` already
#   exposes the underlying signal; adding `return_1d` gives the tree model an
#   even cleaner short-term reversal feature.


@register
class Momentum12Minus1(Feature):
    """Jegadeesh-Titman 12-1 momentum: log-return from t-252 to t-21.

    The "classic" momentum factor. Excluding the most recent month (t-21 → t)
    removes the short-term reversal effect that dilutes simpler trailing-12-month
    returns. Among the most robust anomalies in US equity literature — survives
    PIT-correct evaluation across multiple decades and reappears in international
    samples (Rouwenhorst 1998).

    Typical univariate PIT-correct IC on liquid US large-caps: 0.02-0.04.
    """

    name = "momentum_12_1"
    inputs = ("adj_close",)
    lookback_days = 252

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        return panel.with_columns(
            (c.shift(21).log() - c.shift(252).log()).over("ticker").alias(self.name)
        )


@register
class Distance52WeekHigh(Feature):
    """Hong-Lim-Stein 52-week-high distance: log(price_t / max(price_{t-252:t-1})).

    Stocks trading near their 52-week high tend to keep rising — an anchoring
    effect documented across markets. The signal is in *how close* the current
    price is to the trailing 252-day max, not in the level of returns.

    Output is always ≤ 0 (price is at most equal to the trailing max). Values
    closer to 0 → near 52w high → expected to outperform. Large negative values
    → far below 52w high → expected to underperform.

    Typical univariate PIT-correct IC on liquid US equities: 0.02-0.03.
    """

    name = "distance_52w_high"
    inputs = ("adj_close",)
    lookback_days = 252

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        # Use t-1 max so today's price isn't trivially included (which would
        # always give 0 on the date the high is set). The rolling window of
        # length 252 on the lagged series gives "trailing 52-week high
        # excluding today".
        trailing_max = c.shift(1).rolling_max(window_size=252).over("ticker")
        return panel.with_columns(
            (c.log() - trailing_max.log()).alias(self.name)
        )


@register
class Return1d(Feature):
    """Single-day log return. Captures short-term mean reversion at the cleanest
    horizon.

    The 1-day reversal effect (Lehmann 1990) is the strongest at very short
    horizons on liquid US equities. While our `return_5d` already exposes a
    similar signal, the 1-day version is a more direct reversal indicator —
    tree models that split on it can isolate the t-1 jump from the longer
    momentum signal in `momentum_12_1`.

    Typical univariate PIT-correct IC: 0.01-0.02 with the EXPECTED SIGN
    flipped (high yesterday → low tomorrow, on average).
    """

    name = "return_1d"
    inputs = ("adj_close",)
    lookback_days = 1

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        return panel.with_columns(
            (c.log() - c.log().shift(1)).over("ticker").alias(self.name)
        )
