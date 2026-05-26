"""Evaluation metrics — same definitions used across all models.

Conventions:
- All metrics are computed on `(prediction, realized)` pairs that share (date, ticker).
- Cross-sectional metrics are computed per-date and then averaged.
- "Realized" should be the same target the model was trained on (forward excess return).
- Where a metric isn't well-defined (e.g. all-constant predictions on a date), we return
  NaN for that date and skip it in the average.

The functions accept a polars DataFrame to keep the call sites simple. Any caller can
pass: SELECT prediction_date AS date, ticker, prediction, realized FROM ....
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr


@dataclass
class MetricSummary:
    """Bundle of metrics for one (model, evaluation window)."""

    n_observations: int
    n_dates: int
    information_coefficient: float  # avg per-date Spearman
    ic_t_stat: float  # mean(IC) / (std(IC)/sqrt(n_dates))
    hit_rate: float  # fraction of correct sign predictions
    mae: float
    rmse: float
    long_short_sharpe: float  # annualized Sharpe of decile L/S portfolio

    def as_dict(self) -> dict[str, float]:
        return {
            "n_observations": self.n_observations,
            "n_dates": self.n_dates,
            "information_coefficient": self.information_coefficient,
            "ic_t_stat": self.ic_t_stat,
            "hit_rate": self.hit_rate,
            "mae": self.mae,
            "rmse": self.rmse,
            "long_short_sharpe": self.long_short_sharpe,
        }


def _per_date_ic(df: pl.DataFrame) -> pl.DataFrame:
    """Spearman IC per date. Returns (date, ic)."""
    rows = []
    for d, grp in df.group_by("date"):
        # Need at least 5 valid observations to compute a Spearman
        sub = grp.drop_nulls(subset=["prediction", "realized"])
        if sub.height < 5:
            continue
        # spearmanr returns nan if one side is constant
        rho, _ = spearmanr(sub["prediction"].to_numpy(), sub["realized"].to_numpy())
        if rho is not None and not math.isnan(rho):
            rows.append({"date": d[0] if isinstance(d, tuple) else d, "ic": float(rho)})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"date": pl.Date, "ic": pl.Float64})


def _long_short_returns(df: pl.DataFrame, top_frac: float = 0.2) -> pl.DataFrame:
    """Construct a daily decile long-short portfolio return series.

    For each date: take the top `top_frac` predicted names, equally long; bottom
    `top_frac`, equally short; return is mean(top.realized) - mean(bottom.realized).
    """
    rows = []
    for d, grp in df.group_by("date"):
        sub = grp.drop_nulls(subset=["prediction", "realized"])
        n = sub.height
        if n < 10:
            continue
        k = max(1, round(n * top_frac))
        sorted_sub = sub.sort("prediction")
        bot = sorted_sub.head(k)["realized"].to_numpy()
        top = sorted_sub.tail(k)["realized"].to_numpy()
        ret = float(top.mean() - bot.mean())
        rows.append({"date": d[0] if isinstance(d, tuple) else d, "ret": ret})
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"date": pl.Date, "ret": pl.Float64})


def summarize(df: pl.DataFrame, horizon_days: int = 5) -> MetricSummary:
    """Compute all standard metrics on a (date, ticker, prediction, realized) frame."""
    valid = df.drop_nulls(subset=["prediction", "realized"])
    if valid.height == 0:
        return MetricSummary(
            0, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        )

    err = (valid["prediction"] - valid["realized"]).to_numpy()
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    hit_rate = float(
        np.mean(np.sign(valid["prediction"].to_numpy()) == np.sign(valid["realized"].to_numpy()))
    )

    ic_df = _per_date_ic(valid)
    if ic_df.height >= 2:
        ic_mean = float(ic_df["ic"].mean())
        ic_std = float(ic_df["ic"].std())
        t_stat = (ic_mean * math.sqrt(ic_df.height)) / ic_std if ic_std > 0 else float("nan")
    else:
        ic_mean = float("nan")
        t_stat = float("nan")

    ls = _long_short_returns(valid)
    if ls.height >= 20:
        per_day = float(ls["ret"].mean())
        per_day_std = float(ls["ret"].std())
        # Scale from per-horizon return to annualized Sharpe.
        # The realized side is a horizon-day forward return; daily refresh implies
        # we hold roughly horizon_days per signal, but for a rough comparison we
        # annualize by sqrt(252 / horizon_days).
        ann_factor = math.sqrt(252 / max(horizon_days, 1))
        sharpe = (per_day / per_day_std) * ann_factor if per_day_std > 0 else float("nan")
    else:
        sharpe = float("nan")

    return MetricSummary(
        n_observations=valid.height,
        n_dates=valid["date"].n_unique(),
        information_coefficient=ic_mean,
        ic_t_stat=t_stat,
        hit_rate=hit_rate,
        mae=mae,
        rmse=rmse,
        long_short_sharpe=sharpe,
    )


def compare_models(
    df: pl.DataFrame,
    model_ids: Iterable[str] | None = None,
    horizon_days: int = 5,
) -> pl.DataFrame:
    """Run `summarize` once per model_id. Returns a long-form comparison table."""
    if "model_id" not in df.columns:
        raise ValueError("DataFrame must have a model_id column for comparison")
    ids = list(model_ids) if model_ids else sorted(df["model_id"].unique().to_list())
    rows = []
    for mid in ids:
        sub = df.filter(pl.col("model_id") == mid).select(
            "date", "ticker", "prediction", "realized"
        )
        summary = summarize(sub, horizon_days=horizon_days).as_dict()
        summary["model_id"] = mid
        rows.append(summary)
    return pl.DataFrame(rows).select(
        "model_id",
        "n_observations",
        "n_dates",
        "information_coefficient",
        "ic_t_stat",
        "hit_rate",
        "mae",
        "rmse",
        "long_short_sharpe",
    )
