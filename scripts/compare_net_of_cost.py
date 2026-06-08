"""Strict apples-to-apples net-of-cost comparison across all project models.

Parallel to scripts/compare_apples_to_apples.py, but uses
`compute_turnover_and_costs` from `eval/turnover.py` to produce: gross IC +
t-stat, gross long-short Sharpe, daily and annual turnover, and net-of-cost
Sharpe at 3 bp, 10 bp, and 20 bp transaction-cost levels.

Why this matters: gross IC measures cross-sectional ranking quality, which
is what `compare_apples_to_apples.py` reports. Net-of-cost Sharpe is what
actually determines deployment viability. A model with high gross IC but
200x annual turnover (LightGBM v2_ohlcv) can be a worse deployment than a
model with moderate gross IC and 2x annual turnover (ARIMA at annual
refit), even though the gross-IC ranking puts them close. This script
surfaces that distinction explicitly.

Usage:
    PYTHONPATH=src python scripts/compare_net_of_cost.py
    PYTHONPATH=src python scripts/compare_net_of_cost.py --since 2022-10-01
    PYTHONPATH=src python scripts/compare_net_of_cost.py --until 2022-10-01
    PYTHONPATH=src python scripts/compare_net_of_cost.py --since 2024-01-01
"""

from __future__ import annotations

import argparse
from datetime import date

import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.eval.turnover import compare_models_costs
from price_model.features.targets import add_forward_excess_return
from price_model.serving.store import PredictionStore

# Same model ID list as scripts/compare_apples_to_apples.py — keep in sync.
MODEL_IDS = [
    # Classical baselines
    "arima_classical_pit_v1",
    "gbm_classical_pit_v1",
    # Direct momentum factors
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
    # Linear / Ridge
    "lasso_pit_v1",
    "lasso_pit_v2",
    "ridge_pit_v1",
    "ridge_pit_h21",
    "lasso_elasso_pit_v1",
    # Pure-momentum linear blends: Lasso (L1) and ElasticNet (L1+L2) on
    # {mom_12_1, mom_378, mom_504, mom_756}.
    "momentum_lasso_pit_v1",
    "momentum_elasticnet_pit_v1",
    "momentum_lasso_pit_h21",
    "momentum_elasticnet_pit_h21",
    # 21-day-horizon variants — test whether 5-day result generalizes
    # to monthly horizon. Run with `--horizon 21`. The h21 momentum
    # variants are identical to their 5-day counterparts in prediction
    # (just the feature value); Lasso and LightGBM h21 will differ from
    # their 5-day siblings because they're fit to a different target.
    "mom_12_1_factor_h21",
    "mom_378_factor_h21",
    "mom_504_factor_h21",
    "mom_756_factor_h21",
    "lasso_elasso_pit_h21",
    "lightgbm_kaggle_v3_curated_h21",
    # Held-out Optuna at 21-day target (HPs only saw <= 2024-12-31).
    "lightgbm_kaggle_v3_curated_h21_hp_pre20241231",
    # ML (extended_kaggle_v2_ohlcv + pared variant + audit-curated v3 +
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
    # the YAML's hard end-date would cap the entire apples-to-apples
    # intersection. FF is reported as a separate baseline row on its
    # own date window in the README rather than being forced into the
    # intersection. To re-include, uncomment + bump the FF YAML end.
    # "ff_factor_pit_v1",
]


def _load_predictions(store: PredictionStore, model_ids: list[str]) -> pl.DataFrame:
    ids = ", ".join(f"'{m}'" for m in model_ids)
    sql = f"""
        SELECT prediction_date AS date, ticker, model_id, prediction
        FROM predictions
        WHERE model_id IN ({ids})
    """
    return store.query(sql)


def _date_intersection(preds: pl.DataFrame) -> set[date]:
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
        help="Optional inclusive lower bound (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--until",
        default=None,
        help="Optional exclusive upper bound (YYYY-MM-DD).",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward horizon (days) used for realized target.",
    )
    ap.add_argument(
        "--top-frac",
        type=float,
        default=0.2,
        help="Top quintile (0.2) by default; use 0.1 for decile L/S.",
    )
    args = ap.parse_args()
    cutoff_since: date | None = date.fromisoformat(args.since) if args.since else None
    cutoff_until: date | None = date.fromisoformat(args.until) if args.until else None

    console = Console()

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

    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)
    panel = add_forward_excess_return(panel, horizon_days=args.horizon, target_col="y")
    realized = panel.select(["date", "ticker", pl.col("y").alias("realized")])
    realized = realized.filter(pl.col("date").is_in(list(common_dates)))

    preds = preds.filter(pl.col("date").is_in(list(common_dates)))
    eval_df = preds.join(realized, on=["date", "ticker"], how="inner")
    console.print(
        f"After joining to realized targets: {eval_df.height:,} rows across "
        f"{eval_df['date'].n_unique():,} dates."
    )

    result = compare_models_costs(
        eval_df,
        model_ids=target_ids,
        top_frac=args.top_frac,
        cost_bps=(3, 10, 20),
        horizon_days=args.horizon,
    )

    window_parts = []
    if args.since:
        window_parts.append(f"since {args.since}")
    if args.until:
        window_parts.append(f"until {args.until}")
    window_desc = ", ".join(window_parts) if window_parts else "full sample"
    leg_desc = f"top_frac={args.top_frac:.1f}"
    rule_label = (
        f"[bold green]Net-of-cost comparison "
        f"(intersection of {len(target_ids)} models, {window_desc}, {leg_desc})"
    )
    console.rule(rule_label)
    table = Table()
    for col in result.columns:
        table.add_column(col)
    for row in result.iter_rows():
        formatted = []
        for v in row:
            if isinstance(v, float):
                formatted.append(f"{v:+.4f}" if abs(v) < 100 else f"{v:.1f}")
            else:
                formatted.append(str(v))
        table.add_row(*formatted)
    console.print(table)


if __name__ == "__main__":
    main()
