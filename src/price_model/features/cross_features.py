"""Features that depend on cross-sectional context (sector, market return).

These can't be computed from a single ticker's history alone — they need the panel
of all tickers at each date. That's why they live separately from the pure technicals
in technical.py.

Leakage discipline: every operation here uses .over("date") (cross-sectional) or
.over("ticker") (time-series within ticker) — never a window that mixes the two
in a way that lets future data leak into the present.
"""

from __future__ import annotations

import polars as pl

from price_model.features.base import Feature, register


@register
class MomentumSectorRelative(Feature):
    """60-day momentum minus the sector median momentum on the same date.

    Strips out sector-wide moves so the signal is "did this stock outperform its
    sector peers" rather than "did the sector rally". Requires a `sector` column
    on the panel (attached by load_panel).
    """

    name = "momentum_60_sector_rel"
    inputs = ("adj_close", "sector")
    lookback_days = 65

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        if "sector" not in panel.columns:
            raise ValueError(
                f"{self.name} requires a 'sector' column on the panel "
                "(call data.sectors.attach_sector first or use load_panel)."
            )
        c = pl.col("adj_close")
        mom = (c.shift(5).log() - c.shift(65).log()).over("ticker")
        panel = panel.with_columns(mom.alias("_tmp_mom60"))
        out = panel.with_columns(
            (pl.col("_tmp_mom60") - pl.col("_tmp_mom60").median().over(["date", "sector"])).alias(
                self.name
            )
        )
        return out.drop("_tmp_mom60")


@register
class IdioVol20(Feature):
    """20-day rolling std of residuals from a 60-day rolling market-beta regression.

    "Market" is the cross-sectional mean log return on each date (a crude proxy
    for the equal-weighted universe return). For each ticker we compute:

        beta_t  = rolling_cov(r_ticker, r_market, 60) / rolling_var(r_market, 60)
        eps_t   = r_ticker - beta_t * r_market           # residual on date t
        idio_t  = rolling_std(eps_t, 20)

    All windows are right-aligned (no future data). The rolling beta uses cov via
    E[xy] - E[x]E[y] over a 60-day window.
    """

    name = "idio_vol_20"
    inputs = ("adj_close",)
    lookback_days = 81  # 1 for diff + 60 for beta + 20 for vol

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        log_ret = (c.log() - c.log().shift(1)).over("ticker")
        panel = panel.with_columns(log_ret.alias("_ret"))

        # Cross-sectional mean return per date (the "market") — uses only same-date data
        panel = panel.with_columns(pl.col("_ret").mean().over("date").alias("_mkt_ret"))

        # 60-day rolling beta via cov / var, per ticker
        mean_x = pl.col("_ret").rolling_mean(window_size=60).over("ticker")
        mean_y = pl.col("_mkt_ret").rolling_mean(window_size=60).over("ticker")
        mean_xy = (pl.col("_ret") * pl.col("_mkt_ret")).rolling_mean(window_size=60).over("ticker")
        mean_y2 = (
            (pl.col("_mkt_ret") * pl.col("_mkt_ret")).rolling_mean(window_size=60).over("ticker")
        )
        cov_xy = mean_xy - mean_x * mean_y
        var_y = mean_y2 - mean_y * mean_y
        beta = pl.when(var_y > 1e-12).then(cov_xy / var_y).otherwise(0.0)
        panel = panel.with_columns(beta.alias("_beta"))

        # Residual = ticker_return - beta * market_return (alpha absorbed into noise)
        panel = panel.with_columns(
            (pl.col("_ret") - pl.col("_beta") * pl.col("_mkt_ret")).alias("_resid")
        )

        # 20-day rolling std of the residual
        panel = panel.with_columns(
            pl.col("_resid").rolling_std(window_size=20).over("ticker").alias(self.name)
        )

        return panel.drop(["_ret", "_mkt_ret", "_beta", "_resid"])


def _rank_in_date(expr: pl.Expr) -> pl.Expr:
    """Uniform rank in [0, 1] across the cross-section of each date.

    A standalone helper so the rank features stay readable.
    """
    return expr.rank("average").over("date") / expr.count().over("date")


@register
class RollingBeta60(Feature):
    """Rolling 60-day market beta of the ticker's daily log returns.

    Market is proxied by the cross-sectional mean daily log return per date.
    Beta = cov(r_ticker, r_market) / var(r_market) over the trailing 60 days,
    computed via `mean(xy) - mean(x)·mean(y)` rolling expectations.

    Diagnostic basis: cross-sectional correlation between ARIMA's prediction
    and rolling beta_60 was -0.067 (full sample) — the project's "beta tilt"
    alternative explanation for ARIMA's IC is empirically falsified. This
    feature is included primarily as a CONTROL in regularized linear models
    (Han-He-Rapach-Zhou-style E-LASSO): include β as a covariate so the
    coefficients on momentum / vol features represent alpha *after*
    controlling for market exposure.
    """

    name = "beta_60"
    inputs = ("adj_close",)
    lookback_days = 61  # 1 for diff + 60 for rolling beta window

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        log_ret = (c.log() - c.log().shift(1)).over("ticker")
        panel = panel.with_columns(log_ret.alias("_ret"))
        # Cross-sectional mean return per date (equal-weight universe proxy for market)
        panel = panel.with_columns(pl.col("_ret").mean().over("date").alias("_mkt_ret"))

        mean_x = pl.col("_ret").rolling_mean(window_size=60).over("ticker")
        mean_y = pl.col("_mkt_ret").rolling_mean(window_size=60).over("ticker")
        mean_xy = (pl.col("_ret") * pl.col("_mkt_ret")).rolling_mean(window_size=60).over("ticker")
        mean_y2 = (
            (pl.col("_mkt_ret") * pl.col("_mkt_ret")).rolling_mean(window_size=60).over("ticker")
        )
        cov_xy = mean_xy - mean_x * mean_y
        var_y = mean_y2 - mean_y * mean_y
        beta = pl.when(var_y > 1e-12).then(cov_xy / var_y).otherwise(None)
        out = panel.with_columns(beta.alias(self.name))
        return out.drop(["_ret", "_mkt_ret"])


@register
class Momentum60Rank(Feature):
    """Cross-sectional rank (within date) of momentum_60. Robust to fat tails."""

    name = "momentum_60_rank"
    inputs = ("adj_close",)
    lookback_days = 65

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        mom = (c.shift(5).log() - c.shift(65).log()).over("ticker")
        panel = panel.with_columns(mom.alias("_tmp"))
        out = panel.with_columns(_rank_in_date(pl.col("_tmp")).alias(self.name))
        return out.drop("_tmp")


@register
class Vol20Rank(Feature):
    """Cross-sectional rank of 20-day realized vol. High rank = high relative risk."""

    name = "vol_20_rank"
    inputs = ("adj_close",)
    lookback_days = 21

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        log_ret = (c.log() - c.log().shift(1)).over("ticker")
        vol = log_ret.rolling_std(window_size=20).over("ticker")
        panel = panel.with_columns(vol.alias("_tmp"))
        out = panel.with_columns(_rank_in_date(pl.col("_tmp")).alias(self.name))
        return out.drop("_tmp")


@register
class DistanceMA200Rank(Feature):
    """Cross-sectional rank of distance from the 200-day moving average."""

    name = "distance_ma_200_rank"
    inputs = ("adj_close",)
    lookback_days = 200

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        ma = c.rolling_mean(window_size=200).over("ticker")
        dist = (c - ma) / ma
        panel = panel.with_columns(dist.alias("_tmp"))
        out = panel.with_columns(_rank_in_date(pl.col("_tmp")).alias(self.name))
        return out.drop("_tmp")


# -----------------------------------------------------------------------------
# v2 features — promoted from notebook prototypes (notebooks/01_diagnostics.ipynb).
# Univariate quick_ic on the panel showed:
#   - vol_ewm_20:           IC +0.023, t-stat +4.47   (volatility risk premium)
#   - vol_20_cs_min_dist:   IC +0.022, t-stat +4.00   (volatility regime)
#   - mom60_minus_dist_ma200: IC -0.013, t-stat -3.25 (overbought / stretched)
# -----------------------------------------------------------------------------


@register
class VolEwm20(Feature):
    """EWM volatility with span=20 of daily log returns.

    Exponentially-weighted estimate adapts faster to recent volatility regime
    changes than `vol_20` (flat window). On this universe, the strongest single
    feature found in the diagnostic notebook (t-stat 4.47 univariate).
    """

    name = "vol_ewm_20"
    inputs = ("adj_close",)
    lookback_days = 100  # generous for EWM convergence

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        log_ret = (c.log() - c.log().shift(1)).over("ticker")
        return panel.with_columns(log_ret.ewm_std(span=20).over("ticker").alias(self.name))


@register
class Vol20CsMinDist(Feature):
    """vol_20 minus the cross-sectional minimum vol_20 on the same date.

    Captures "how much more volatile is this stock than the calmest name today."
    Cross-sectionally monotone but robust to fat tails in a different way than
    ranks (preserves the magnitude of the gap, not just ordering). Univariate
    t-stat 4.00 in the diagnostic.
    """

    name = "vol_20_cs_min_dist"
    inputs = ("adj_close",)
    lookback_days = 21

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        log_ret = (c.log() - c.log().shift(1)).over("ticker")
        vol_20 = log_ret.rolling_std(window_size=20).over("ticker")
        panel = panel.with_columns(vol_20.alias("_v"))
        out = panel.with_columns((pl.col("_v") - pl.col("_v").min().over("date")).alias(self.name))
        return out.drop("_v")


@register
class Mom60MinusDistMA200(Feature):
    """60-day momentum minus distance-from-200d-MA.

    Captures "trend disagreement": when short-to-medium momentum is high but
    the stock isn't far above its long-term MA (or vice versa), there's tension
    that tends to resolve. Negative univariate IC (t-stat -3.25) suggests this
    is a mean-reversion / overbought signal.
    """

    name = "mom60_minus_dist_ma200"
    inputs = ("adj_close",)
    lookback_days = 200

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        # 60-day momentum (skipping last 5 to avoid overlap with return_5d)
        mom60 = (c.shift(5).log() - c.shift(65).log()).over("ticker")
        # Distance from 200d MA
        ma200 = c.rolling_mean(window_size=200).over("ticker")
        dist_ma200 = (c - ma200) / ma200
        return panel.with_columns((mom60 - dist_ma200).alias(self.name))


# -----------------------------------------------------------------------------
# Cross-sectional reversal features for 5-day-horizon prediction.
#
# Raw `return_5d` as a reversal factor on a daily-rebalanced strategy on
# liquid US large-caps suffers from two contamination sources:
#
#   1. Market-direction component: when the whole market rallies over 5 days,
#      every ticker has a positive return; raw reversal mechanically shorts
#      the highest-beta names regardless of their idiosyncratic positioning.
#      Beta-residualizing the return strips this out.
#
#   2. Sector-rotation component: when an entire sector rallies, every name
#      in it has a positive 5-day return; raw reversal mechanically shorts
#      the whole sector. Sector-demeaning produces within-sector dispersion.
#
# Both `residual_return_5d` and `sector_relative_return_5d` are the RAW
# (un-sign-flipped) cleaned 5-day return. The MomentumFactor model class
# with `sign=-1` produces the reversal direction; this matches the
# convention used by `reversal_1d_factor` / `reversal_5d_factor` for the
# raw `return_1d` / `return_5d` features.
# -----------------------------------------------------------------------------


@register
class ResidualReturn5d(Feature):
    """5-day log return residualized against beta·market-return.

    For each (date, ticker), compute:
        beta_60 = 60-day rolling regression of ticker daily return on
                  cross-sectional mean daily return (the project's
                  equal-weighted market proxy)
        r_5d   = log(adj_close_t / adj_close_{t-5})
        mkt_5d = sum of cross-sectional mean daily returns over the past 5 days
        residual_return_5d = r_5d - beta_60 · mkt_5d

    Sign-flipped (via `MomentumFactor(feature_name='residual_return_5d',
    sign=-1)`) this is the canonical "residual reversal" baseline from the
    short-horizon cross-sectional literature: cleaner than raw reversal
    because the market-direction component (which is a substantial portion
    of `return_5d` in directional regimes) is removed before the cross-
    sectional ranking step.

    Computes beta_60 inline rather than depending on the `beta_60` feature
    being registered first, so the feature ordering in `feature_names`
    doesn't matter.
    """

    name = "residual_return_5d"
    inputs = ("adj_close",)
    lookback_days = 65  # 1 for return diff + 60 for beta window + 5 for r_5d

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        log_ret = (c.log() - c.log().shift(1)).over("ticker")
        panel = panel.with_columns(log_ret.alias("_ret"))
        # Cross-sectional mean daily return per date (market proxy)
        panel = panel.with_columns(pl.col("_ret").mean().over("date").alias("_mkt_ret"))

        # 60-day rolling beta via cov / var, per ticker
        mean_x = pl.col("_ret").rolling_mean(window_size=60).over("ticker")
        mean_y = pl.col("_mkt_ret").rolling_mean(window_size=60).over("ticker")
        mean_xy = (pl.col("_ret") * pl.col("_mkt_ret")).rolling_mean(window_size=60).over("ticker")
        mean_y2 = (
            (pl.col("_mkt_ret") * pl.col("_mkt_ret")).rolling_mean(window_size=60).over("ticker")
        )
        cov_xy = mean_xy - mean_x * mean_y
        var_y = mean_y2 - mean_y * mean_y
        # Use None (not 0.0) for the otherwise branch so warmup rows propagate
        # nulls into residual; a fake-zero beta during warmup would leak
        # uncorrected ret_5d into residual_return_5d.
        beta = pl.when(var_y > 1e-12).then(cov_xy / var_y).otherwise(None)
        panel = panel.with_columns(beta.alias("_beta"))

        # 5-day return per ticker and 5-day market return
        ret_5d = c.log() - c.log().shift(5)
        mkt_5d = pl.col("_mkt_ret").rolling_sum(window_size=5).over("ticker")
        panel = panel.with_columns(ret_5d.over("ticker").alias("_ret_5d"))
        panel = panel.with_columns(mkt_5d.alias("_mkt_5d"))

        # Residualized 5-day return: r_5d - beta · mkt_5d.
        # Null in any component (beta during warmup, ret_5d / mkt_5d during
        # warmup) propagates into a null residual, as required.
        residual = pl.col("_ret_5d") - pl.col("_beta") * pl.col("_mkt_5d")
        out = panel.with_columns(residual.alias(self.name))
        return out.drop(["_ret", "_mkt_ret", "_beta", "_ret_5d", "_mkt_5d"])


@register
class SectorRelativeReturn5d(Feature):
    """5-day log return minus the sector median 5-day log return on the same date.

    For each (date, ticker) with sector `s`:
        r_5d                       = log(adj_close_t / adj_close_{t-5})
        sector_median_5d           = median r_5d over tickers with sector == s on date t
        sector_relative_return_5d  = r_5d - sector_median_5d

    Sign-flipped via `MomentumFactor(feature_name='sector_relative_return_5d',
    sign=-1)` this is the within-sector reversal factor: idiosyncratic
    positioning of a ticker relative to its sector peers, with the sector-
    rotation component removed before ranking. Reduces the mechanical-short-
    the-whole-sector behavior of raw reversal during sector-rotation regimes.

    Requires a `sector` column on the panel (attached by `load_panel`).
    Median (not mean) for robustness to fat tails in cross-sectional return
    distributions on small sector buckets.
    """

    name = "sector_relative_return_5d"
    inputs = ("adj_close", "sector")
    lookback_days = 5

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        if "sector" not in panel.columns:
            raise ValueError(
                f"{self.name} requires a 'sector' column on the panel "
                "(call data.sectors.attach_sector first or use load_panel)."
            )
        c = pl.col("adj_close")
        r_5d = (c.log() - c.log().shift(5)).over("ticker")
        panel = panel.with_columns(r_5d.alias("_r_5d"))
        out = panel.with_columns(
            (pl.col("_r_5d") - pl.col("_r_5d").median().over(["date", "sector"])).alias(self.name)
        )
        return out.drop("_r_5d")


@register
class CsReturnDispersion20(Feature):
    """20-day smoothed cross-sectional return dispersion — a regime indicator.

    For each date, the cross-sectional standard deviation of daily log returns
    across the universe measures how decorrelated tickers are. High dispersion
    indicates a regime where ticker-specific factors dominate (drawdowns,
    sector rotations, AI-bull-like leadership shifts); low dispersion
    indicates a regime where the market moves in lockstep.

    Smoothed by a 20-day rolling mean to reduce day-to-day noise while
    preserving regime-shift signals.

    The feature is broadcast: every ticker on a given date receives the same
    value (it's a date-level statistic, not a ticker-level one). The point
    of including it in a feature panel is to give the model a
    contemporaneously-observable conditioning variable. A tree can then
    learn rules like "when cs_return_dispersion_20 is high, weight
    sector_relative_reversal_5d more; when low, weight momentum_12_1 more"
    without ever seeing a labeled regime.

    This is the lookahead-safe alternative to training separate models on
    hindsight-labeled regimes (AI-bull, post-AI, etc.) — the regime
    boundary is detected from data available at prediction time, not from
    post-hoc inspection of the time series.

    Empirically on this universe, high cs_return_dispersion_20 periods
    correspond to: late 2021 drawdown, 2022 Fed-pivot reversal, periodic
    flash-crashes and sector rotations. Low dispersion periods correspond
    to the strong-momentum 2024+ regime.
    """

    name = "cs_return_dispersion_20"
    inputs = ("adj_close",)
    lookback_days = 21  # 1 for diff + 20 for rolling mean

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        c = pl.col("adj_close")
        log_ret = (c.log() - c.log().shift(1)).over("ticker")
        panel = panel.with_columns(log_ret.alias("_ret"))
        # Cross-sectional std per date — same value for every ticker on that date.
        cs_std = pl.col("_ret").std().over("date")
        panel = panel.with_columns(cs_std.alias("_cs_std"))
        # 20-day time-series smoothing per ticker. Since _cs_std is identical
        # across tickers on each date, the per-ticker rolling mean produces
        # identical results — but using .over("ticker") keeps polars happy
        # about partitioning.
        out = panel.with_columns(
            pl.col("_cs_std").rolling_mean(window_size=20).over("ticker").alias(self.name)
        )
        return out.drop(["_ret", "_cs_std"])
