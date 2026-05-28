"""Robustness checks for predictions.

Three families of check, all designed to make the headline IC defensible in an
interview / due-diligence conversation:

1. **Bootstrap IC confidence interval** — resample dates with replacement,
   recompute mean IC each time, report the 5th/50th/95th percentile band.
   Lets you say "IC = +0.0075 [+0.0030, +0.0120]" instead of just a point
   estimate. Catches the failure mode "one good year of luck inflates IC."

2. **Decile bucket returns** — bin per-date predictions into N quantiles,
   average realized returns within each bucket. Monotonic ascending steps =
   real ranking ability. Flat middle + tails-only signal = the model only
   works at the extremes (still useful but a different story than uniform
   ranking).

3. **Time-split evaluation** — split the walk-forward predictions by date,
   compute metrics on each half, compare. Walk-forward predictions are
   already out-of-sample by construction, so this is a *filter*, not a
   re-fit. The compare exposes regime dependence: if IC is +0.02 in
   2017-2021 and 0.00 in 2022-2026, the edge is decaying.

All three accept the same `(date, ticker, prediction, realized)` polars
DataFrame the rest of `eval/` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from price_model.eval.metrics import MetricSummary, _per_date_ic, summarize

# ---------------------------------------------------------------------------
# Bootstrap IC confidence interval
# ---------------------------------------------------------------------------


@dataclass
class BootstrapICResult:
    """5th/50th/95th percentile band on bootstrapped mean IC."""

    point_estimate: float  # raw mean IC, no resampling
    p05: float
    p50: float  # bootstrap median (sanity check — should track point estimate)
    p95: float
    n_dates: int  # number of unique dates that contributed an IC value
    n_bootstrap: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point_estimate": self.point_estimate,
            "p05": self.p05,
            "p50": self.p50,
            "p95": self.p95,
            "n_dates": self.n_dates,
            "n_bootstrap": self.n_bootstrap,
        }

    @property
    def excludes_zero(self) -> bool:
        """True iff the 5th-95th interval is strictly on one side of zero.

        The bootstrap-CI equivalent of "the t-stat clears the 5% bar."
        """
        return (self.p05 > 0.0) or (self.p95 < 0.0)


def bootstrap_ic_ci(
    df: pl.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> BootstrapICResult:
    """Bootstrap the mean Spearman IC by resampling dates with replacement.

    The underlying ICs are computed per date once; the bootstrap then samples
    those daily values (not the underlying ticker rows) with replacement to
    estimate the sampling distribution of the mean. This is the right unit
    because each date's IC is the natural independent observation in a
    cross-sectional walk-forward setting — ticker observations within a date
    are correlated and shouldn't be resampled independently.
    """
    valid = df.drop_nulls(subset=["prediction", "realized"])
    ic_per_date = _per_date_ic(valid)
    n = ic_per_date.height
    if n < 2:
        return BootstrapICResult(
            point_estimate=float("nan"),
            p05=float("nan"),
            p50=float("nan"),
            p95=float("nan"),
            n_dates=n,
            n_bootstrap=0,
        )

    ic_values = ic_per_date["ic"].to_numpy()
    point = float(ic_values.mean())

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = ic_values[idx].mean()

    p05, p50, p95 = np.percentile(boot_means, [5, 50, 95])
    return BootstrapICResult(
        point_estimate=point,
        p05=float(p05),
        p50=float(p50),
        p95=float(p95),
        n_dates=n,
        n_bootstrap=n_bootstrap,
    )


# ---------------------------------------------------------------------------
# Decile bucket returns
# ---------------------------------------------------------------------------


def decile_returns(
    df: pl.DataFrame,
    n_buckets: int = 10,
) -> pl.DataFrame:
    """Mean realized return per prediction-bucket, averaged across dates.

    For each date:
      1. Rank predictions ascending.
      2. Split tickers into `n_buckets` equal-size groups (1 = lowest predicted,
         n_buckets = highest predicted).
      3. Compute mean realized return per bucket.

    Then average those per-date bucket means across all dates. Returns a frame
    with one row per bucket and the cross-date stats.

    A "good" plot of `mean_realized` vs `bucket` is a monotonic ascending
    staircase. A flat middle with rising tails = the model only differentiates
    the extremes. A non-monotonic line = no real cross-sectional ranking.
    """
    valid = df.drop_nulls(subset=["prediction", "realized"])
    if valid.height == 0:
        return pl.DataFrame(
            schema={
                "bucket": pl.Int64,
                "mean_realized": pl.Float64,
                "n_dates": pl.Int64,
            }
        )

    bucketed = (
        valid.with_columns(
            (pl.col("prediction").rank("ordinal").over("date") - 1).alias("_rank"),
            pl.col("date").count().over("date").alias("_n_on_date"),
        )
        .with_columns(
            # Floor-divide ranks into n_buckets equal-size groups. Edge case:
            # tiny days (< n_buckets tickers) get fewer effective buckets.
            ((pl.col("_rank") * n_buckets) // pl.col("_n_on_date") + 1).alias("bucket")
        )
        .filter(pl.col("_n_on_date") >= n_buckets)
    )

    if bucketed.height == 0:
        return pl.DataFrame(
            schema={
                "bucket": pl.Int64,
                "mean_realized": pl.Float64,
                "n_dates": pl.Int64,
            }
        )

    per_date = bucketed.group_by(["date", "bucket"]).agg(
        pl.col("realized").mean().alias("bucket_mean")
    )
    across = per_date.group_by("bucket").agg(
        pl.col("bucket_mean").mean().alias("mean_realized"),
        pl.col("bucket_mean").std().alias("std_realized"),
        pl.col("date").n_unique().alias("n_dates"),
    )
    return across.sort("bucket")


# ---------------------------------------------------------------------------
# Time-split evaluation
# ---------------------------------------------------------------------------


@dataclass
class TimeSplitResult:
    """Side-by-side metrics for two date windows."""

    window_a: tuple[str, str]
    window_b: tuple[str, str]
    metrics_a: MetricSummary
    metrics_b: MetricSummary

    def as_frame(self) -> pl.DataFrame:
        rows = [
            {"window": f"{self.window_a[0]} → {self.window_a[1]}", **self.metrics_a.as_dict()},
            {"window": f"{self.window_b[0]} → {self.window_b[1]}", **self.metrics_b.as_dict()},
        ]
        return pl.DataFrame(rows)


def time_split_evaluate(
    df: pl.DataFrame,
    cutoff: str | date,
    horizon_days: int = 5,
) -> TimeSplitResult:
    """Compute metrics separately on dates < cutoff and dates >= cutoff.

    The harness already produces out-of-sample predictions via walk-forward, so
    this is a *partition*, not a re-fit. The two windows share the same model
    and the same evaluation logic — only the date filter changes. Use this to
    answer "does the IC hold up in the second half of the sample?"

    `cutoff` is inclusive of the second window: predictions on `cutoff` go to
    window B.
    """
    from datetime import datetime as _dt

    cutoff_d = cutoff if isinstance(cutoff, date) else _dt.fromisoformat(cutoff).date()

    a = df.filter(pl.col("date") < pl.lit(cutoff_d).cast(pl.Date))
    b = df.filter(pl.col("date") >= pl.lit(cutoff_d).cast(pl.Date))

    metrics_a = summarize(a.select("date", "ticker", "prediction", "realized"), horizon_days)
    metrics_b = summarize(b.select("date", "ticker", "prediction", "realized"), horizon_days)

    # Format window labels from the actual data range, not just the cutoff
    a_start = str(a["date"].min()) if a.height else "—"
    a_end = str(a["date"].max()) if a.height else "—"
    b_start = str(b["date"].min()) if b.height else "—"
    b_end = str(b["date"].max()) if b.height else "—"

    return TimeSplitResult(
        window_a=(a_start, a_end),
        window_b=(b_start, b_end),
        metrics_a=metrics_a,
        metrics_b=metrics_b,
    )


# ---------------------------------------------------------------------------
# Convenience: one-call robustness panel
# ---------------------------------------------------------------------------


def robustness_panel(
    df: pl.DataFrame,
    *,
    n_bootstrap: int = 1000,
    n_buckets: int = 10,
    time_cutoff: str | date | None = None,
    horizon_days: int = 5,
) -> dict[str, object]:
    """Run all three robustness checks at once. Returns a dict for easy printing.

    Useful from a notebook:

        from price_model.eval.robustness import robustness_panel
        panel = robustness_panel(eval_df, time_cutoff="2023-01-01")
        print(panel["bootstrap"])
        panel["deciles"]  # → DataFrame
        panel["time_split"].as_frame()  # → DataFrame
    """
    out: dict[str, object] = {
        "bootstrap": bootstrap_ic_ci(df, n_bootstrap=n_bootstrap),
        "deciles": decile_returns(df, n_buckets=n_buckets),
    }
    if time_cutoff is not None:
        out["time_split"] = time_split_evaluate(df, cutoff=time_cutoff, horizon_days=horizon_days)
    # `summarize` is the headline metrics — included so the panel is self-contained.
    out["headline"] = summarize(df, horizon_days=horizon_days)
    return out


__all__ = [
    "BootstrapICResult",
    "TimeSplitResult",
    "bootstrap_ic_ci",
    "decile_returns",
    "robustness_panel",
    "time_split_evaluate",
]
