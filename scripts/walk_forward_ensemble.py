"""Walk-forward regime-conditional ensemble.

Variant of `regime_conditional_ensemble.py` that addresses the regime
non-stationarity failure mode documented by the static-fit variant: when
the training half and evaluation half land in different regime mixes, a
single fitted weight vector cannot generalize.

This script instead recomputes weights at each evaluation date `t` using
a rolling window of the most recent `--rolling-window` days ending at
`t ---embargo-days`. As the regime mix in recent history shifts, the
ensemble's weights adapt: if `mom_12_1` starts working in the most recent
6 months, its rolling-window IC rises and softmax weight follows.

Lookahead-safety
----------------
1. The regime indicator `cs_return_dispersion_20` uses only past returns
   (verified by the universal truncation-invariance leakage test).
2. The dispersion bucket thresholds are computed once from a FIXED initial
   training window (the first --train-window days of the apples-to-apples
   slice) — these define what counts as "low / mid / high dispersion"
   throughout the project. They do NOT use future data.
3. The per-bucket weights at date t are computed from the rolling window
   `[t -rolling-window, t -embargo]`, which contains only past dates.
4. Eval begins after a burn-in of (train_window + rolling_window) days so
   the first evaluation date has a full rolling window of training-fold-
   defined-dispersion observations to fit weights on.

This is the textbook walk-forward design from de Prado's "Advances in
Financial Machine Learning" Ch. 7 applied to ensemble weighting.

Usage:
    PYTHONPATH=src python scripts/walk_forward_ensemble.py
    PYTHONPATH=src python scripts/walk_forward_ensemble.py \
        --rolling-window 252 --temperature 50 --n-buckets 3
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

from price_model.data.loaders import load_panel
from price_model.eval.metrics import summarize
from price_model.eval.turnover import compute_turnover_and_costs
from price_model.features.targets import add_forward_excess_return
from price_model.serving.store import PredictionStore

BASE_MODELS = [
    "lightgbm_kaggle_v2_ohlcv",
    "lightgbm_kaggle_v3_curated",
    "lightgbm_kaggle_v3_curated_tuned",
    "mom_12_1_factor",
    "sector_relative_reversal_5d_factor",
    "arima_classical_pit_v1",
]


def _load_predictions(store: PredictionStore, model_ids: list[str]) -> pl.DataFrame:
    ids = ", ".join(f"'{m}'" for m in model_ids)
    sql = f"""
        SELECT prediction_date AS date, ticker, model_id, prediction
        FROM predictions
        WHERE model_id IN ({ids})
    """
    return store.query(sql)


def _date_intersection(preds: pl.DataFrame) -> list[date]:
    by_model = preds.group_by("model_id").agg(pl.col("date").unique().alias("dates"))
    per_model_dates = [set(row["dates"]) for row in by_model.iter_rows(named=True)]
    if not per_model_dates:
        return []
    return sorted(set.intersection(*per_model_dates))


def _compute_dispersion_indicator(panel: pl.DataFrame) -> pl.DataFrame:
    sorted_panel = panel.sort(["ticker", "date"])
    c = pl.col("adj_close")
    log_ret = (c.log() - c.log().shift(1)).over("ticker")
    sorted_panel = sorted_panel.with_columns(log_ret.alias("_ret"))
    cs_std = sorted_panel.group_by("date").agg(pl.col("_ret").std().alias("_cs_std")).sort("date")
    cs_std = cs_std.with_columns(
        pl.col("_cs_std").rolling_mean(window_size=20).alias("dispersion")
    )
    return cs_std.select("date", "dispersion").drop_nulls("dispersion")


def _cross_sectional_zscore(df: pl.DataFrame, col: str) -> pl.DataFrame:
    mean = pl.col(col).mean().over("date")
    std = pl.col(col).std().over("date")
    return df.with_columns(
        ((pl.col(col) - mean) / pl.when(std == 0).then(1.0).otherwise(std)).alias(col)
    )


def _per_date_model_ic(df: pl.DataFrame) -> pl.DataFrame:
    """Per (date, model_id), compute Spearman IC. Returns (date, model_id, ic)."""
    rows = []
    for d, by_date in df.group_by("date"):
        date_value = d[0] if isinstance(d, tuple) else d
        for m, sub in by_date.group_by("model_id"):
            model_id = m[0] if isinstance(m, tuple) else m
            valid = sub.drop_nulls(["prediction", "realized"])
            if valid.height < 5:
                continue
            rho, _ = spearmanr(valid["prediction"].to_numpy(), valid["realized"].to_numpy())
            if rho is None or np.isnan(rho):
                continue
            rows.append({"date": date_value, "model_id": model_id, "ic": float(rho)})
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"date": pl.Date, "model_id": pl.Utf8, "ic": pl.Float64}
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-window", type=int, default=504,
        help="Burn-in window (days) used to fix dispersion bucket thresholds.",
    )
    ap.add_argument(
        "--rolling-window", type=int, default=252,
        help="Rolling weight-fit window (days) used at each eval date.",
    )
    ap.add_argument(
        "--embargo-days", type=int, default=6,
        help="Gap between rolling-window end and prediction date (matches harness).",
    )
    ap.add_argument(
        "--n-buckets", type=int, default=3,
        help="Number of dispersion regime buckets.",
    )
    ap.add_argument(
        "--temperature", type=float, default=20.0,
        help="Softmax temperature for converting per-bucket ICs to weights.",
    )
    ap.add_argument(
        "--horizon", type=int, default=5,
        help="Forward horizon (days) for realized excess return.",
    )
    args = ap.parse_args()
    console = Console()

    # ----- 1. Load predictions -----
    with PredictionStore(read_only=True) as store:
        available = set(store.list_models())
        target_ids = [m for m in BASE_MODELS if m in available]
        missing = [m for m in BASE_MODELS if m not in available]
        if missing:
            console.print(f"[yellow]Missing from store:[/yellow] {missing}")
        if len(target_ids) < 2:
            console.print("[red]Need at least 2 base models.")
            return
        preds = _load_predictions(store, target_ids)
    console.print(f"Loaded predictions for {len(target_ids)} base models.")

    # ----- 2. Apples-to-apples date intersection -----
    common_dates = _date_intersection(preds)
    if not common_dates:
        console.print("[red]Date intersection empty.")
        return
    console.print(
        f"Apples-to-apples slice: {len(common_dates):,} dates "
        f"[{common_dates[0]} → {common_dates[-1]}]"
    )

    # ----- 3. Load panel, realized, dispersion -----
    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)
    panel = add_forward_excess_return(panel, horizon_days=args.horizon, target_col="y")
    realized = panel.select("date", "ticker", pl.col("y").alias("realized")).filter(
        pl.col("date").is_in(common_dates)
    )
    dispersion = _compute_dispersion_indicator(panel).filter(
        pl.col("date").is_in(common_dates)
    )

    # ----- 4. Join + dedup + z-score predictions -----
    preds = preds.filter(pl.col("date").is_in(common_dates))
    # Deduplicate (same store-duplication issue handled in scripts/regime_conditional_ensemble.py)
    preds = preds.group_by(["date", "ticker", "model_id"]).agg(
        pl.col("prediction").mean()
    )
    eval_df = preds.join(realized, on=["date", "ticker"], how="inner")
    eval_df = eval_df.join(dispersion, on="date", how="inner")
    eval_df = _cross_sectional_zscore(eval_df, "prediction")
    console.print(
        f"Eval frame: {eval_df.height:,} rows across {eval_df['date'].n_unique():,} dates."
    )

    # ----- 5. Bucket thresholds from initial training window ONLY -----
    initial_train_dates = common_dates[: args.train_window]
    if len(initial_train_dates) < args.train_window:
        console.print(
            f"[red]Insufficient dates for train_window={args.train_window}; "
            f"got only {len(initial_train_dates)}."
        )
        return
    initial_disp = (
        eval_df.filter(pl.col("date").is_in(initial_train_dates))
        .select("date", "dispersion")
        .unique()
        .sort("date")
    )
    quantiles = np.linspace(0, 1, args.n_buckets + 1)[1:-1]
    thresholds = [float(initial_disp["dispersion"].quantile(q)) for q in quantiles]
    console.print(
        f"Fixed bucket thresholds from first {args.train_window} dates: "
        f"{[round(t, 5) for t in thresholds]}"
    )

    def assign_bucket(d: float) -> int:
        for i, t in enumerate(thresholds):
            if d < t:
                return i
        return args.n_buckets - 1

    eval_df = eval_df.with_columns(
        pl.col("dispersion").map_elements(assign_bucket, return_dtype=pl.Int32).alias("bucket")
    )

    # ----- 6. Precompute per-date per-model IC over ALL dates ONCE -----
    console.print("[dim]Computing per-date IC for each model (one pass)...[/dim]")
    per_date_ic = _per_date_model_ic(
        eval_df.select("date", "ticker", "model_id", "prediction", "realized")
    )
    # Attach bucket
    bucket_map = eval_df.select("date", "bucket").unique()
    per_date_ic = per_date_ic.join(bucket_map, on="date", how="inner").sort("date")

    # ----- 7. Walk-forward weight estimation -----
    # First evaluation date is the date AFTER (train_window + rolling_window + embargo)
    # to ensure the rolling window contains only pre-prediction data.
    burn_in = args.train_window + args.rolling_window + args.embargo_days
    if burn_in >= len(common_dates):
        console.print(
            f"[red]burn_in ({burn_in}) >= total apples-to-apples dates "
            f"({len(common_dates)})."
        )
        return
    eval_dates = common_dates[burn_in:]
    console.print(
        f"Burn-in: {burn_in} dates. Walk-forward eval: {len(eval_dates):,} dates "
        f"[{eval_dates[0]} → {eval_dates[-1]}]"
    )

    # Build a per-date date-index map for fast slicing.
    date_to_idx = {d: i for i, d in enumerate(common_dates)}

    # Pre-aggregate per-bucket per-model IC as numpy for fast rolling lookups.
    # Sort by date for monotone indexing.
    per_date_ic_sorted = per_date_ic.sort("date")

    tau = args.temperature
    n_models = len(target_ids)
    model_to_idx = {m: i for i, m in enumerate(target_ids)}

    # We'll iterate eval dates and for each, slice the per-date IC by the
    # rolling window dates, group by (bucket, model_id), compute mean IC,
    # softmax to weights, and apply to the prediction row.
    ensemble_rows: list[dict] = []

    # Pivot eval predictions to wide form once
    pivot = eval_df.pivot(
        on="model_id",
        index=["date", "ticker", "realized", "bucket"],
        values="prediction",
    )
    # Cache per-date prediction rows for fast iteration
    pivot = pivot.sort("date")

    for i, t in enumerate(eval_dates):
        # Rolling window: [t - rolling_window - embargo, t - embargo)
        start_idx = date_to_idx[t] - args.rolling_window - args.embargo_days
        end_idx = date_to_idx[t] - args.embargo_days
        if start_idx < 0:
            continue
        window_dates = common_dates[start_idx:end_idx]
        window_ic = per_date_ic_sorted.filter(pl.col("date").is_in(window_dates))
        if window_ic.height == 0:
            continue

        # Per-bucket per-model mean IC in this window
        per_bucket_ic = window_ic.group_by(["bucket", "model_id"]).agg(
            pl.col("ic").mean().alias("mean_ic")
        )
        # Build weight tensor: bucket -> model -> weight
        weights = np.full((args.n_buckets, n_models), 1.0 / n_models)
        for b in range(args.n_buckets):
            bucket_rows = per_bucket_ic.filter(pl.col("bucket") == b)
            if bucket_rows.height == 0:
                continue
            ic_map = {row["model_id"]: row["mean_ic"] for row in bucket_rows.iter_rows(named=True)}
            ic_arr = np.array([ic_map.get(m, 0.0) for m in target_ids])
            scaled = tau * ic_arr
            scaled -= scaled.max()
            exp = np.exp(scaled)
            weights[b] = exp / exp.sum()

        # Apply weights to date-t predictions
        date_rows = pivot.filter(pl.col("date") == t)
        if date_rows.height == 0:
            continue
        buckets_arr = date_rows["bucket"].to_numpy()
        ensemble_pred = np.zeros(date_rows.height, dtype=np.float64)
        for m in target_ids:
            if m not in date_rows.columns:
                continue
            col = date_rows[m].fill_null(0.0).to_numpy()
            m_idx = model_to_idx[m]
            w_arr = weights[buckets_arr, m_idx]
            ensemble_pred += col * w_arr

        for row, pred_val in zip(date_rows.iter_rows(named=True), ensemble_pred, strict=True):
            ensemble_rows.append(
                {
                    "date": row["date"],
                    "ticker": row["ticker"],
                    "realized": row["realized"],
                    "prediction": float(pred_val),
                }
            )
        if (i + 1) % 100 == 0:
            console.print(f"  Progress: {i + 1:,} / {len(eval_dates):,} eval dates done")

    if not ensemble_rows:
        console.print("[red]No ensemble predictions produced.")
        return
    ensemble_df = pl.DataFrame(ensemble_rows)
    console.print(
        f"[green]Walk-forward ensemble produced {ensemble_df.height:,} predictions "
        f"across {ensemble_df['date'].n_unique():,} dates.[/green]"
    )

    # ----- 8. Score ensemble + base models on the SAME eval dates -----
    console.rule("[bold green]Walk-forward ensemble evaluation")
    eval_date_set = set(ensemble_df["date"].to_list())
    summary_rows: list[dict] = []

    def score_one(name: str, df: pl.DataFrame) -> dict:
        m = summarize(df, horizon_days=args.horizon).as_dict()
        m["model_id"] = name
        cost_summary = compute_turnover_and_costs(
            df, top_frac=0.2, cost_bps=(3, 10, 20), horizon_days=args.horizon
        ).as_dict()
        for k, v in cost_summary.items():
            if k not in m and k not in ("n_observations", "n_dates"):
                m[k] = v
        return m

    summary_rows.append(score_one("ENSEMBLE_WF", ensemble_df))

    # Score each base model on the same eval-date subset
    base_subset = eval_df.filter(pl.col("date").is_in(eval_date_set))
    for m in target_ids:
        sub = base_subset.filter(pl.col("model_id") == m).select(
            "date", "ticker", "prediction", "realized"
        )
        if sub.height == 0:
            continue
        summary_rows.append(score_one(m, sub))

    table = Table()
    cols_to_show = [
        "model_id",
        "information_coefficient",
        "ic_t_stat",
        "long_short_sharpe",
        "annual_turnover",
        "after_cost_sharpe_3bp",
        "after_cost_sharpe_10bp",
        "after_cost_sharpe_20bp",
    ]
    for c in cols_to_show:
        table.add_column(c)
    summary_rows.sort(key=lambda r: -(r.get("long_short_sharpe") or 0.0))
    for row in summary_rows:
        is_ens = row["model_id"] == "ENSEMBLE_WF"
        cells = []
        for c in cols_to_show:
            v = row.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("nan")
            elif isinstance(v, float):
                cells.append(f"{v:+.4f}" if abs(v) < 100 else f"{v:.1f}")
            else:
                cells.append(str(v))
        if is_ens:
            cells = [f"[bold green]{c}[/bold green]" for c in cells]
        table.add_row(*cells)
    console.print(table)


if __name__ == "__main__":
    main()
