"""Strict apples-to-apples comparison of all the project's models on the same
date intersection.

The CLI's `run` command evaluates each model on whatever evaluation slice
results from THAT experiment's feature warmup. Different experiments have
different warmups (LightGBM v2_ohlcv: ~21 days; ARIMA classical_pit: 252 days
min_history; momentum_factor_pit with mom_756 added: 756 days). Comparing IC
across those is NOT apples-to-apples — broader slices include earlier dates
that may carry different signal strength.

This script:
  1. Reads predictions for a fixed set of model_ids from the DuckDB store.
  2. Computes the date intersection across all of them — the slice where
     every model has at least one prediction.
  3. Loads the panel and computes realized 5-day forward excess returns,
     restricted to that intersection.
  4. Joins each model's predictions to realized on (date, ticker).
  5. Runs the standard `summarize` metrics on each model.
  6. Prints a clean comparison table.

Usage:
    PYTHONPATH=src python scripts/compare_apples_to_apples.py
    # optionally restrict to a regime:
    PYTHONPATH=src python scripts/compare_apples_to_apples.py --since 2022-10-01
    # the complement of the cut (pre-regime):
    PYTHONPATH=src python scripts/compare_apples_to_apples.py --until 2022-10-01
    # both flags together → an inner window:
    PYTHONPATH=src python scripts/compare_apples_to_apples.py --since 2022-10-01 --until 2024-01-01
"""

from __future__ import annotations

import argparse
from datetime import date

import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.eval.metrics import summarize
from price_model.features.targets import add_forward_excess_return
from price_model.serving.store import PredictionStore

# The model ids we want to compare. These must match the model_id strings
# written to the prediction store by each experiment's YAML.
MODEL_IDS = [
    # Classical baselines (classical_pit_v1)
    "arima_classical_pit_v1",
    "gbm_classical_pit_v1",
    # Direct momentum factors (momentum_factor_pit_v1)
    "mom_12_1_factor",
    "mom_378_factor",
    "mom_504_factor",
    "mom_756_factor",
    # Tier-1 canonical baseline for daily horizon (Lehmann 1990 reversal)
    "reversal_1d_factor",
    "reversal_5d_factor",
    # Microstructure factors (Lou-Polk-Skouras, Amihud, sector/beta-cleaned reversal)
    "overnight_continuation_factor",
    "intraday_reversal_factor",
    "residual_reversal_5d_factor",
    "sector_relative_reversal_5d_factor",
    "amihud_illiquidity_factor",
    # Linear (lasso_pit_v1, lasso_pit_v2, and the L2 alternative ridge_pit_v1)
    "lasso_pit_v1",
    "lasso_pit_v2",
    "ridge_pit_v1",
    # ML (extended_kaggle_v2_ohlcv + the pared variant + audit-curated v3 +
    # Optuna-tuned LightGBM + XGBoost + CatBoost on v3 panel +
    # the held-out HP-free version of the tuned LightGBM where Optuna only
    # saw dates <= 2023-12-31, so the 2024+ slice is a clean HP-free OOS test)
    "lightgbm_kaggle_v2_ohlcv",
    "lightgbm_kaggle_v2_ohlcv_pared",
    "lightgbm_kaggle_v3_curated",
    "lightgbm_kaggle_v3_curated_tuned",
    "lightgbm_kaggle_v3_curated_hp_pre20231231",
    "lightgbm_kaggle_v3_curated_hp_pre20241231",
    "xgboost_v3_curated",
    "catboost_v3_curated",
    # Factor model — Fama-French 5-factor (PIT). DROPPED from MODEL_IDS
    # because Kenneth French's data library lags ~6 weeks behind today;
    # the YAML's hard end-date (e.g. '2026-03-31') would cap the entire
    # apples-to-apples intersection. FF is reported as a separate
    # baseline row on its own date window in the README rather than
    # being forced into the intersection. To re-include for older
    # comparisons, uncomment the line below and bump the FF YAML end.
    # "ff_factor_pit_v1",
    # Han-He-Rapach-Zhou E-LASSO multi-anomaly baseline
    "lasso_elasso_pit_v1",
    # Pure-momentum linear blends: Lasso (L1) and ElasticNet (L1+L2) on
    # {mom_12_1, mom_378, mom_504, mom_756}. Tests whether learned linear
    # combinations of correlated momentum factors beat the best single
    # constituent (mom_756) and the equal-weight ensemble of the same set.
    "momentum_lasso_pit_v1",
    "momentum_elasticnet_pit_v1",
    "momentum_lasso_pit_h21",
    "momentum_elasticnet_pit_h21",
    # 21-day-horizon variants — test whether the 5-day result generalizes
    # to monthly horizon. Run with `--horizon 21` to evaluate. The h21
    # variants will produce different predictions than the 5-day versions
    # for any model that's actually fit to the target (Lasso, LightGBM);
    # momentum factors will be identical since they return the feature
    # value directly.
    "mom_12_1_factor_h21",
    "mom_378_factor_h21",
    "mom_504_factor_h21",
    "mom_756_factor_h21",
    "lasso_elasso_pit_h21",
    "lightgbm_kaggle_v3_curated_h21",
    # Held-out Optuna sweep at 21-day target (HPs only saw <= 2024-12-31),
    # the fair apples-to-apples ML counterpart to lasso_elasso_pit_h21.
    "lightgbm_kaggle_v3_curated_h21_hp_pre20241231",
]


def _load_predictions(store: PredictionStore, model_ids: list[str]) -> pl.DataFrame:
    """Pull (date, ticker, model_id, prediction) for the given model ids."""
    ids = ", ".join(f"'{m}'" for m in model_ids)
    sql = f"""
        SELECT prediction_date AS date, ticker, model_id, prediction
        FROM predictions
        WHERE model_id IN ({ids})
    """
    return store.query(sql)


def _date_intersection(preds: pl.DataFrame) -> set[date]:
    """The set of dates that every model in `preds` has at least one prediction on."""
    by_model = preds.group_by("model_id").agg(pl.col("date").unique().alias("dates"))
    per_model_dates = [set(row["dates"]) for row in by_model.iter_rows(named=True)]
    if not per_model_dates:
        return set()
    return set.intersection(*per_model_dates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--since",
        default=None,
        help="Optional inclusive lower bound (YYYY-MM-DD). Only evaluate on dates >= this.",
    )
    ap.add_argument(
        "--until",
        default=None,
        help="Optional exclusive upper bound (YYYY-MM-DD). Only evaluate on dates < this.",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward horizon (days) used for realized target.",
    )
    args = ap.parse_args()
    cutoff_since: date | None = date.fromisoformat(args.since) if args.since else None
    cutoff_until: date | None = date.fromisoformat(args.until) if args.until else None

    console = Console()

    # 1. Pull predictions for all models from the store.
    with PredictionStore(read_only=True) as store:
        available = set(store.list_models())
        target_ids = [m for m in MODEL_IDS if m in available]
        missing = [m for m in MODEL_IDS if m not in available]
        if missing:
            console.print(f"[yellow]Missing from store (skipping):[/yellow] {missing}")
        if not target_ids:
            console.print("[red]No target model predictions found in the store.")
            return
        preds = _load_predictions(store, target_ids)
    console.print(f"Loaded {preds.height:,} prediction rows across {len(target_ids)} models.")

    # 2. Compute the date intersection across all loaded models.
    # Diagnostic: print per-model max date first so we can see which model
    # is the binding constraint capping the intersection's upper end.
    per_model_max = (
        preds.group_by("model_id")
        .agg(pl.col("date").max().alias("max_date"))
        .sort("max_date")
    )
    earliest_max = per_model_max.row(0)
    console.print(
        f"[dim]Binding model on end-date: {earliest_max[0]} stops at "
        f"{earliest_max[1]}. The 2nd-earliest is {per_model_max.row(1)[0]} "
        f"at {per_model_max.row(1)[1]}.[/dim]"
    )

    common_dates = _date_intersection(preds)
    if cutoff_since:
        common_dates = {d for d in common_dates if d >= cutoff_since}
    if cutoff_until:
        common_dates = {d for d in common_dates if d < cutoff_until}
    if not common_dates:
        console.print("[red]Date intersection is empty.")
        return
    common_dates_sorted = sorted(common_dates)
    console.print(
        f"Apples-to-apples slice: {len(common_dates_sorted):,} dates "
        f"[{common_dates_sorted[0]} → {common_dates_sorted[-1]}]"
    )

    # 3. Load the panel and compute realized 5-day forward excess returns.
    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)
    panel = add_forward_excess_return(panel, horizon_days=args.horizon, target_col="y")
    realized = panel.select(["date", "ticker", pl.col("y").alias("realized")])
    realized = realized.filter(pl.col("date").is_in(list(common_dates)))

    # 4. Filter predictions to the intersection slice and join to realized.
    preds = preds.filter(pl.col("date").is_in(list(common_dates)))
    eval_df = preds.join(realized, on=["date", "ticker"], how="inner")
    console.print(
        f"After joining to realized targets: {eval_df.height:,} rows across "
        f"{eval_df['date'].n_unique():,} dates."
    )

    # 5. Summarize per-model on the apples-to-apples slice.
    rows = []
    for mid in target_ids:
        sub = eval_df.filter(pl.col("model_id") == mid).select(
            "date", "ticker", "prediction", "realized"
        )
        if sub.height == 0:
            console.print(f"[dim]Skipping {mid}: no rows after join.[/dim]")
            continue
        summary = summarize(sub, horizon_days=args.horizon).as_dict()
        summary["model_id"] = mid
        rows.append(summary)

    if not rows:
        console.print("[red]No summaries produced.")
        return

    result = (
        pl.DataFrame(rows)
        .select(
            "model_id",
            "n_observations",
            "n_dates",
            "information_coefficient",
            "ic_t_stat",
            "hit_rate",
            "long_short_sharpe",
        )
        .sort("information_coefficient", descending=True)
    )

    # 6. Print.
    window_parts = []
    if args.since:
        window_parts.append(f"since {args.since}")
    if args.until:
        window_parts.append(f"until {args.until}")
    window_desc = ", ".join(window_parts) if window_parts else "full sample"
    rule_label = (
        f"[bold green]Apples-to-apples comparison "
        f"(intersection of {len(target_ids)} models, {window_desc})"
    )
    console.rule(rule_label)
    table = Table()
    for col in result.columns:
        table.add_column(col)
    for row in result.iter_rows():
        table.add_row(
            *[f"{v:+.4f}" if isinstance(v, float) else str(v) for v in row]
        )
    console.print(table)


if __name__ == "__main__":
    main()
