"""Deflated Sharpe Ratio (Bailey-López de Prado 2014) across all project models.

The project has run roughly N = 25 model variants across the apples-to-apples
comparison framework. Reporting an observed Sharpe ratio without correcting for
this multiple-testing search inflates the apparent significance. The Deflated
Sharpe Ratio addresses this by computing the probability that the true Sharpe
is positive given:
  - observed Sharpe,
  - number of trials attempted,
  - sample length (number of independent return periods),
  - skewness and kurtosis of the strategy's per-period returns.

This script computes DSR for every model in the canonical comparison on the
apples-to-apples slice (full sample + post-Oct-2022 regime). The output is a
column added to the comparison table indicating whether each Sharpe survives
multi-test correction.

Reference: Bailey & López de Prado (2014). JPM 40(5).

Usage:
    PYTHONPATH=src python scripts/deflated_sharpe_audit.py
    PYTHONPATH=src python scripts/deflated_sharpe_audit.py --since 2022-10-01
    PYTHONPATH=src python scripts/deflated_sharpe_audit.py --top-frac 0.1
"""

from __future__ import annotations

import argparse
import math
from datetime import date

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr  # noqa: F401  (loaded for env parity)

from price_model.data.loaders import load_panel
from price_model.eval.metrics import deflated_sharpe_ratio
from price_model.features.targets import add_forward_excess_return
from price_model.serving.store import PredictionStore

# Same MODEL_IDS as scripts/compare_apples_to_apples.py — kept in sync.
MODEL_IDS = [
    "arima_classical_pit_v1",
    "gbm_classical_pit_v1",
    "mom_12_1_factor",
    "mom_378_factor",
    "mom_504_factor",
    "mom_756_factor",
    "reversal_1d_factor",
    "reversal_5d_factor",
    "overnight_continuation_factor",
    "intraday_reversal_factor",
    "residual_reversal_5d_factor",
    "sector_relative_reversal_5d_factor",
    "amihud_illiquidity_factor",
    "lasso_pit_v1",
    "lasso_pit_v2",
    "ridge_pit_v1",
    "lasso_elasso_pit_v1",
    "lightgbm_kaggle_v2_ohlcv",
    "lightgbm_kaggle_v2_ohlcv_pared",
    "lightgbm_kaggle_v3_curated",
    "ff_factor_pit_v1",
]


def _load_predictions(store: PredictionStore, model_ids: list[str]) -> pl.DataFrame:
    ids = ", ".join(f"'{m}'" for m in model_ids)
    sql = f"""
        SELECT prediction_date AS date, ticker, model_id, prediction
        FROM predictions
        WHERE model_id IN ({ids})
    """
    return store.query(sql)


def _long_short_returns(df: pl.DataFrame, top_frac: float) -> np.ndarray:
    """Build the L/S daily return series for a single model."""
    rets = []
    for _d, grp in df.group_by("date"):
        sub = grp.drop_nulls(["prediction", "realized"])
        n = sub.height
        if n < 10:
            continue
        k = max(1, round(n * top_frac))
        sorted_sub = sub.sort("prediction")
        bot = sorted_sub.head(k)["realized"].mean()
        top = sorted_sub.tail(k)["realized"].mean()
        if bot is None or top is None:
            continue
        rets.append(float(top - bot))
    return np.array(rets, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--since", default=None,
        help="Inclusive lower bound (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--until", default=None,
        help="Exclusive upper bound (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--top-frac", type=float, default=0.2,
        help="Top quintile (0.2) by default; 0.1 for decile L/S.",
    )
    ap.add_argument(
        "--horizon", type=int, default=5,
        help="Forward horizon for realized excess return.",
    )
    args = ap.parse_args()
    console = Console()

    cutoff_since = date.fromisoformat(args.since) if args.since else None
    cutoff_until = date.fromisoformat(args.until) if args.until else None

    # ----- Load + dedup + intersect -----
    with PredictionStore(read_only=True) as store:
        available = set(store.list_models())
        target_ids = [m for m in MODEL_IDS if m in available]
        missing = [m for m in MODEL_IDS if m not in available]
        if missing:
            console.print(f"[yellow]Missing from store:[/yellow] {missing}")
        if not target_ids:
            console.print("[red]No target models found.")
            return
        preds = _load_predictions(store, target_ids)
    preds = preds.group_by(["date", "ticker", "model_id"]).agg(pl.col("prediction").mean())

    # Date intersection
    by_model = preds.group_by("model_id").agg(pl.col("date").unique().alias("dates"))
    per_model_dates = [set(row["dates"]) for row in by_model.iter_rows(named=True)]
    common = set.intersection(*per_model_dates) if per_model_dates else set()
    if cutoff_since:
        common = {d for d in common if d >= cutoff_since}
    if cutoff_until:
        common = {d for d in common if d < cutoff_until}
    common_sorted = sorted(common)
    if not common_sorted:
        console.print("[red]Intersection empty.")
        return
    console.print(
        f"Apples-to-apples slice: {len(common_sorted):,} dates "
        f"[{common_sorted[0]} → {common_sorted[-1]}]"
    )

    # Realized
    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)
    panel = add_forward_excess_return(panel, horizon_days=args.horizon, target_col="y")
    realized = panel.select("date", "ticker", pl.col("y").alias("realized")).filter(
        pl.col("date").is_in(list(common))
    )
    preds = preds.filter(pl.col("date").is_in(list(common)))
    eval_df = preds.join(realized, on=["date", "ticker"], how="inner")

    # ----- DSR per model -----
    n_trials = len(target_ids)  # the number of variants attempted
    horizon = args.horizon
    ann_factor = math.sqrt(252 / max(horizon, 1))

    rows = []
    for mid in target_ids:
        sub = eval_df.filter(pl.col("model_id") == mid).select(
            "date", "ticker", "prediction", "realized"
        )
        if sub.height == 0:
            continue
        ls_returns = _long_short_returns(sub, top_frac=args.top_frac)
        if ls_returns.size < 20:
            continue
        # Per-period Sharpe (annualized to match the project's L/S Sharpe convention)
        per_day_mean = float(ls_returns.mean())
        per_day_std = float(ls_returns.std(ddof=1))
        if per_day_std <= 0:
            continue
        sharpe_obs = (per_day_mean / per_day_std) * ann_factor
        # Higher moments of the per-period return series
        from scipy.stats import kurtosis as scipy_kurtosis
        from scipy.stats import skew as scipy_skew

        skew = float(scipy_skew(ls_returns))
        # Use NON-excess kurtosis (gaussian = 3)
        kurt = float(scipy_kurtosis(ls_returns, fisher=False))
        # Deflated Sharpe
        dsr = deflated_sharpe_ratio(
            sharpe_obs=sharpe_obs,
            n_trials=n_trials,
            n_periods=ls_returns.size,
            skew=skew,
            kurt=kurt,
        )
        rows.append(
            {
                "model_id": mid,
                "n_periods": ls_returns.size,
                "sharpe_obs": sharpe_obs,
                "expected_max_under_null": dsr["expected_max_sharpe_under_null"],
                "test_statistic": dsr["test_statistic"],
                "p(true_sharpe>0)": dsr["probability_true_sharpe_positive"],
                "skew": skew,
                "kurt": kurt,
            }
        )

    if not rows:
        console.print("[red]No rows produced.")
        return
    result = pl.DataFrame(rows).sort("sharpe_obs", descending=True)

    window_parts = []
    if args.since:
        window_parts.append(f"since {args.since}")
    if args.until:
        window_parts.append(f"until {args.until}")
    window_desc = ", ".join(window_parts) if window_parts else "full sample"
    leg_desc = "decile L/S" if args.top_frac == 0.1 else "quintile L/S"
    console.rule(
        f"[bold green]Deflated Sharpe (N_trials={n_trials}, {leg_desc}, {window_desc})"
    )
    table = Table()
    cols = [
        "model_id",
        "n_periods",
        "sharpe_obs",
        "expected_max_under_null",
        "test_statistic",
        "p(true_sharpe>0)",
        "skew",
        "kurt",
    ]
    for c in cols:
        table.add_column(c)
    for row in result.iter_rows(named=True):
        cells = []
        for c in cols:
            v = row.get(c)
            if isinstance(v, float):
                if c == "p(true_sharpe>0)":
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(f"{v:+.4f}" if abs(v) < 100 else f"{v:.1f}")
            else:
                cells.append(str(v))
        table.add_row(*cells)
    console.print(table)
    console.print()
    console.print(
        f"[dim]Interpretation: a model passes the deflated-Sharpe test if "
        f"p(true_sharpe>0) > 0.95 (i.e., > 95% probability the true Sharpe is positive "
        f"under multi-test correction for {n_trials} trials).[/dim]"
    )


if __name__ == "__main__":
    main()
