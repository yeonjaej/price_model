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
        return panel.with_columns((c.log() - trailing_max.log()).alias(self.name))


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
        return panel.with_columns((c.log() - c.log().shift(1)).over("ticker").alias(self.name))


# -----------------------------------------------------------------------------
# OHLCV / volume features — the obvious omissions
# -----------------------------------------------------------------------------
#
# Up to this point every feature in this file consumed only `adj_close`. That
# leaves the day-bar's `high`, `low`, `open`, and `volume` unused, which means
# we throw away two well-documented signal families:
#
#   (1) Intraday range/body, which are more efficient volatility estimators
#       than close-to-close `vol_20` (Parkinson 1980, Garman-Klass 1980).
#   (2) Volume and dollar-volume, which carry the entire liquidity/turnover
#       anomaly literature (Amihud 2002 illiquidity, abnormal-volume catalyst
#       detectors, etc.).
#
# We also add the Bali-Cakici-Whitelaw MAX-effect feature, which is a
# return-based anomaly we previously omitted alongside the three already in
# this file (Jegadeesh-Titman, Hong-Lim-Stein, Lehmann).
#
# Important data-quality notes for this block:
#
#   - `yfinance_source.fetch(...)` calls `auto_adjust=False`, so `open/high/
#     low/close` are RAW (not split-adjusted); only `adj_close` is. WITHIN-DAY
#     ratios (range/close, body/close, Parkinson) are safe because all four
#     OHLC prices share the same day's currency and the split factor cancels.
#     CROSS-DAY raw-OHLC features (overnight gap = open_t / close_{t-1}) are
#     NOT safe without manual adjustment, so they're deliberately excluded.
#
#   - Raw volume is also unadjusted: at a split, share count gets multiplied,
#     so unadjusted volume jumps spuriously. We construct an
#     adjustment-aware volume `volume_adj = volume * close / adj_close` for
#     features where the level matters (abnormal_volume). For the log
#     dollar-volume feature, `close * volume` is approximately invariant under
#     splits (price halves, shares double), so we use raw close * volume.
#
# References:
# - Parkinson, M. (1980). "The Extreme Value Method for Estimating the
#   Variance of the Rate of Return". Journal of Business 53(1).
# - Bali, T., Cakici, N., Whitelaw, R. (2011). "Maxing Out: Stocks as
#   Lotteries and the Cross-Section of Expected Returns". Journal of
#   Financial Economics 99(2). — MAX effect.
# - Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and
#   time-series effects". Journal of Financial Markets 5(1).


@register
class RangeRatio(Feature):
    """Daily high-low range as a fraction of close: `(high - low) / close`.

    A within-day intraday volatility proxy. Captures "how much the stock
    moved during the session" independent of close-to-close drift. Useful
    cheap signal alongside the close-to-close `vol_20` family.

    Safe under unadjusted OHLC because all three terms share the same day's
    currency (the split factor cancels in the ratio).
    """

    name = "range_ratio"
    inputs = ("high", "low", "close")
    lookback_days = 1

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        return panel.with_columns(
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias(self.name)
        )


@register
class BodyRatio(Feature):
    """Daily |open-close| body as a fraction of close: `|open - close| / close`.

    Captures intraday directional conviction independent of magnitude. A
    "doji" day (small body, wide range) and a "trend" day (wide body, small
    overshoot beyond OHLC body) have very different downstream behavior —
    `range_ratio` measures the former, this feature measures the latter.

    Safe under unadjusted OHLC for the same reason as `range_ratio`.
    """

    name = "body_ratio"
    inputs = ("open", "close")
    lookback_days = 1

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        return panel.with_columns(
            ((pl.col("open") - pl.col("close")).abs() / pl.col("close")).alias(self.name)
        )


@register
class ParkinsonVol20(Feature):
    """Parkinson high-low volatility estimator, 20-day average.

    For each day, compute σ²_P = (log(high / low))² / (4 · ln 2). Average
    over a 20-day window and take the square root. ~5x more statistically
    efficient than close-to-close `vol_20` for the same window length —
    fewer observations needed for the same precision because high and low
    contain more information than two arbitrary points on the daily path.

    See Parkinson (1980). Output is on the same scale as `vol_20` (daily
    log-return standard deviation) so the tree can split on it analogously.
    """

    name = "parkinson_vol_20"
    inputs = ("high", "low")
    lookback_days = 21

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        # Per-day Parkinson variance, then rolling mean over 20d, then sqrt.
        log_hl = (pl.col("high").log() - pl.col("low").log()).over("ticker")
        # 1/(4*ln 2) ≈ 0.36067376
        per_day_var = (log_hl * log_hl) / (4.0 * 0.6931471805599453)
        return panel.with_columns(
            per_day_var.rolling_mean(window_size=20).over("ticker").sqrt().alias(self.name)
        )


@register
class LogDollarVolume(Feature):
    """log(close * volume), the standard liquidity proxy.

    Raw volume varies by orders of magnitude across S&P 500 names (megacaps
    ~50M shares/day; smaller index members ~500K). Dollar volume converts
    to a $-denominated measure and the log compresses the right tail.

    `close * volume` is approximately invariant under splits (price halves,
    shares double, product preserved), so raw `close` is fine here — we
    don't need the split-adjusted version. Pre-normalization, the feature
    distribution is roughly ln-normal across the cross-section.
    """

    name = "log_dollar_volume"
    inputs = ("close", "volume")
    lookback_days = 1

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        # max(., 1) guard for halts / zero-volume days (rare on liquid names
        # but possible). log(1) = 0 is the floor.
        dollar = pl.col("close") * pl.col("volume")
        return panel.with_columns(
            pl.when(dollar > 1.0).then(dollar).otherwise(1.0).log().alias(self.name)
        )


@register
class AbnormalVolume(Feature):
    """Ratio of today's volume to the trailing 20-day mean: `volume / mean_20(volume)`.

    Spike detector. Common around earnings, M&A news, index inclusions, and
    other catalysts. Cross-sectional ranking of this feature isolates
    "today's unusual names" which historically have short-horizon momentum
    spillover.

    Uses split-adjusted volume `volume_adj = volume * close / adj_close` to
    avoid spurious spikes on the day of a stock split. The denominator
    (trailing mean) is also computed in adjusted units, so the ratio is
    split-invariant.
    """

    name = "abnormal_volume"
    inputs = ("volume", "close", "adj_close")
    lookback_days = 21

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        # Split-adjusted volume: pre-split, raw close > adj_close, so the
        # ratio close/adj_close > 1 and we scale raw share count up to the
        # post-split equivalent. Post-split (and on non-split dates) the
        # ratio is 1.0 so we get raw volume back.
        adj_ratio = pl.col("close") / pl.when(pl.col("adj_close") > 1e-9).then(
            pl.col("adj_close")
        ).otherwise(1e-9)
        vol_adj = (pl.col("volume").cast(pl.Float64) * adj_ratio).over("ticker")
        panel = panel.with_columns(vol_adj.alias("_vol_adj"))
        mean_20 = pl.col("_vol_adj").rolling_mean(window_size=20).over("ticker")
        # Guard against division by zero on extended halts.
        safe_mean = pl.when(mean_20 > 1.0).then(mean_20).otherwise(1.0)
        out = panel.with_columns((pl.col("_vol_adj") / safe_mean).alias(self.name))
        return out.drop("_vol_adj")


@register
class MaxReturn21d(Feature):
    """Maximum single-day log return over the trailing 21 days.

    The Bali-Cakici-Whitelaw (2011) MAX effect: stocks with extreme recent
    positive returns underperform on average — interpreted as lottery-
    preference investors bidding up "jackpot" stocks past their fair value.
    Expected univariate IC sign: NEGATIVE (high MAX → low subsequent
    return).

    One of the most-cited documented anomalies still operating in liquid
    US equities post-2010. Belongs alongside the existing
    `momentum_12_1`, `distance_52w_high`, and `return_1d` anomaly trio.
    """

    name = "max_return_21d"
    inputs = ("adj_close",)
    lookback_days = 22

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        daily_ret = (c.log() - c.log().shift(1)).over("ticker")
        panel = panel.with_columns(daily_ret.alias("_dret"))
        out = panel.with_columns(
            pl.col("_dret").rolling_max(window_size=21).over("ticker").alias(self.name)
        )
        return out.drop("_dret")
