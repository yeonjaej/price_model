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
