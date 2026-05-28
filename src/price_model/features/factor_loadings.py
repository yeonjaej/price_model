"""Rolling Fama-French factor-loading features.

For each (ticker, date) we compute the OLS slope (beta) of the ticker's daily log
return on each Fama-French factor over a trailing N-day window. These betas are
then exposed as features that the LightGBM model can consume alongside the
technical features.

Why this is the right place to put them:
- Same Feature contract as everything else (computed once, lives in the panel).
- Joins KF factor returns onto the panel by date, computes rolling regressions
  per ticker using the same cov/var trick used by IdioVol20, and drops the joined
  columns at the end so downstream features don't see them.

Leakage discipline: for date t, the beta uses only returns and factor returns
strictly up to date t — never t+h. The cov/var rolling formulas are right-aligned
(polars `rolling_*` defaults), and we use `.over("ticker")` so each ticker's
window is computed independently.

Computation per ticker:
    cov_t  = E[r_ticker * f]_window - E[r_ticker]_window * E[f]_window
    var_t  = E[f * f]_window         - E[f]_window^2
    beta_t = cov_t / var_t           (guarded against var ≈ 0)

We expose individual factor betas plus a "factor exposure score" that combines them
(useful as a single feature when you don't want to pay for 5 columns).

Cache key: the KF data is downloaded on first use of any factor feature; subsequent
features in the same process re-use the in-memory cache via @lru_cache.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import ClassVar

import polars as pl

from price_model.data.sources import fama_french
from price_model.features.base import Feature, register

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_kf_factors() -> pl.DataFrame:
    """Lazy-load the KF 5-factor daily series, cached per process.

    Returning a polars frame with columns (date, MKT_RF, SMB, HML, RMW, CMA, RF).
    """
    return fama_french.fetch()


def _attach_factor_returns(panel: pl.DataFrame, factors: list[str]) -> pl.DataFrame:
    """Add `_ret` (ticker log return) and the requested factor columns to `panel`.

    Joins on `date` left → some rows may be null for factor cols if the KF feed
    doesn't yet cover a date (e.g. the most recent few trading days before the
    monthly KF refresh). Those rows produce null betas, which is the right
    behavior — the model will see them as null and tree models handle that.
    """
    c = pl.col("adj_close")
    log_ret = (c.log() - c.log().shift(1)).over("ticker")
    panel = panel.with_columns(log_ret.alias("_ret"))

    kf = _load_kf_factors().select(["date", *factors])
    return panel.join(kf, on="date", how="left")


def _rolling_beta(ret: str, factor: str, window: int) -> pl.Expr:
    """Polars expression for the rolling per-ticker beta of `ret` on `factor`.

    Caller must include `.alias(...)` and ensure the panel is sorted by (ticker, date).
    """
    r = pl.col(ret)
    f = pl.col(factor)
    mean_r = r.rolling_mean(window_size=window).over("ticker")
    mean_f = f.rolling_mean(window_size=window).over("ticker")
    mean_rf = (r * f).rolling_mean(window_size=window).over("ticker")
    mean_f2 = (f * f).rolling_mean(window_size=window).over("ticker")
    cov = mean_rf - mean_r * mean_f
    var = mean_f2 - mean_f * mean_f
    return pl.when(var > 1e-12).then(cov / var).otherwise(None)


# ---------------------------------------------------------------------------
# Per-factor beta features
# ---------------------------------------------------------------------------


class _RollingBetaBase(Feature):
    """Shared implementation for individual rolling-beta features.

    Subclasses set `factor` (KF column name) and `window`. The `name` and
    `lookback_days` class attrs follow the convention `{factor.lower()}_beta_{window}`.
    """

    factor: ClassVar[str]
    window: ClassVar[int] = 60
    inputs: ClassVar[tuple[str, ...]] = ("adj_close",)

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        panel = _attach_factor_returns(panel, factors=[self.factor])
        out = panel.with_columns(_rolling_beta("_ret", self.factor, self.window).alias(self.name))
        return out.drop(["_ret", self.factor])


@register
class MktBeta60(_RollingBetaBase):
    """60-day rolling beta on the market excess return (CAPM beta with KF MKT-RF)."""

    name = "mkt_beta_60"
    factor = "MKT_RF"
    window = 60
    lookback_days = 61  # 1 for diff + 60 for window


@register
class SmbBeta60(_RollingBetaBase):
    """60-day rolling beta on SMB (size factor). Positive = behaves like a small-cap."""

    name = "smb_beta_60"
    factor = "SMB"
    window = 60
    lookback_days = 61


@register
class HmlBeta60(_RollingBetaBase):
    """60-day rolling beta on HML (value factor). Positive = behaves like a value stock."""

    name = "hml_beta_60"
    factor = "HML"
    window = 60
    lookback_days = 61


@register
class RmwBeta60(_RollingBetaBase):
    """60-day rolling beta on RMW (profitability factor)."""

    name = "rmw_beta_60"
    factor = "RMW"
    window = 60
    lookback_days = 61


@register
class CmaBeta60(_RollingBetaBase):
    """60-day rolling beta on CMA (investment factor: conservative minus aggressive)."""

    name = "cma_beta_60"
    factor = "CMA"
    window = 60
    lookback_days = 61


# ---------------------------------------------------------------------------
# Factor-spread / regime-interaction features
# ---------------------------------------------------------------------------
#
# Why these exist:
# - On a 20-name liquid mega-cap universe, naked rolling betas are weak features:
#   the cross-section of mkt_beta is compressed (everyone ~1.0), SMB/HML betas
#   cluster near zero, and the per-date z-score normalization further squashes
#   the dispersion. The first round of FF features (mkt_beta_60 etc.) produced
#   IC = -0.022 on this universe — i.e. the model overfit to noise.
#
# - The features below pair a per-ticker exposure with a recent factor *regime*.
#   E.g. smb_beta_x_smb_20d = ticker's SMB beta × cumulative SMB return over the
#   last 20 trading days. Cross-sectional variation per date still comes from
#   the beta (regime is constant across tickers on a given date), but the *sign*
#   of the product flips when the regime flips — so after per-date z-scoring
#   the model gets `sign(regime) * z(beta)`, which is exactly the conditional
#   pattern a tree can exploit ("when SMB is up, prefer small-leaning names").
#
# Caveat: z-score per date neutralizes the *magnitude* of a date-constant factor.
# Only the sign of the regime survives normalization. That's fine for capturing
# regime *direction*, but it means we don't communicate "the factor is +0.5% vs
# +0.05%" to the model — only "+ vs −". That's an acceptable trade for fitting
# inside the existing normalize pipeline; a future feature could carry magnitude
# via a non-normalized track if it proves valuable.


def _factor_cumulative_return(factor: str, window: int) -> pl.Expr:
    """Helper: cumulative factor return over `window` days, joined onto the panel.

    Assumes the factor column has already been joined onto the panel (we use
    `over("ticker")` so the rolling sum runs within-ticker, but since the joined
    factor column is constant across tickers per date, every ticker sees the
    same value — which is the intended behavior).
    """
    return pl.col(factor).rolling_sum(window_size=window).over("ticker")


@register
class MktBetaXsDispersion(Feature):
    """Cross-sectional demean of mkt_beta_60.

    Pure cross-sectional differentiation: ticker's mkt beta minus the cross-sectional
    median on the same date. On a mega-cap universe where betas cluster near 1.0,
    the absolute beta is uninformative (everyone is "the market"), but the *relative*
    position within the cross-section can carry signal.
    """

    name = "mkt_beta_xs_dispersion"
    inputs = ("adj_close",)
    lookback_days = 61

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        panel = _attach_factor_returns(panel, factors=["MKT_RF"])
        beta = _rolling_beta("_ret", "MKT_RF", 60)
        panel = panel.with_columns(beta.alias("_beta_tmp"))
        out = panel.with_columns(
            (pl.col("_beta_tmp") - pl.col("_beta_tmp").median().over("date")).alias(self.name)
        )
        return out.drop(["_ret", "MKT_RF", "_beta_tmp"])


class _BetaRegimeInteraction(Feature):
    """Shared implementation for `beta × cumulative factor return` features.

    Subclasses set `factor` (KF column) and `regime_window` (days of cumulative
    factor return). The beta uses the standard 60-day rolling window.
    """

    factor: ClassVar[str]
    regime_window: ClassVar[int] = 20
    beta_window: ClassVar[int] = 60
    inputs: ClassVar[tuple[str, ...]] = ("adj_close",)

    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        panel = _attach_factor_returns(panel, factors=[self.factor])
        beta = _rolling_beta("_ret", self.factor, self.beta_window)
        regime = _factor_cumulative_return(self.factor, self.regime_window)
        panel = panel.with_columns(
            beta.alias("_beta_tmp"),
            regime.alias("_regime_tmp"),
        )
        out = panel.with_columns((pl.col("_beta_tmp") * pl.col("_regime_tmp")).alias(self.name))
        return out.drop(["_ret", self.factor, "_beta_tmp", "_regime_tmp"])


@register
class SmbBetaXSmb20d(_BetaRegimeInteraction):
    """SMB 60-day beta × cumulative SMB return over last 20 trading days.

    Captures "small-cap regime interacted with ticker's small-cap exposure". After
    per-date z-score, reduces to `sign(SMB_20d) × z(smb_beta)` — sign-conditional
    ranking on the size factor's recent direction.
    """

    name = "smb_beta_x_smb_20d"
    factor = "SMB"
    regime_window = 20
    lookback_days = 61  # 60-day beta dominates the requirement


@register
class HmlBetaXHml20d(_BetaRegimeInteraction):
    """HML 60-day beta × cumulative HML return over last 20 trading days.

    Same idea as smb_beta_x_smb_20d but for the value factor.
    """

    name = "hml_beta_x_hml_20d"
    factor = "HML"
    regime_window = 20
    lookback_days = 61


@register
class MktBetaXMkt20d(_BetaRegimeInteraction):
    """Market 60-day beta × cumulative MKT-RF return over last 20 trading days.

    Captures momentum/regime interaction with market beta. In a recent bull tape
    (positive MKT_RF_20d), high-beta names should out-rank low-beta names; after
    z-score, the sign of the recent market move flips the ranking direction.
    """

    name = "mkt_beta_x_mkt_20d"
    factor = "MKT_RF"
    regime_window = 20
    lookback_days = 61
