"""Regime-conditional ensemble of cross-sectional return predictors.

Combines five complementary base models — each strong in a specific regime —
into a single ensemble whose per-date weights are conditioned on the
contemporaneously-observable regime indicator `cs_return_dispersion_20`:

    - lightgbm_kaggle_v2_ohlcv          (peak-regime ML, 21-feature panel)
    - lightgbm_kaggle_v3_curated        (regime-robust ML, 14-feature curated)
    - mom_12_1_factor                   (JT canonical momentum, slow)
    - sector_relative_reversal_5d_factor (regime-complementary reversal)
    - arima_classical_pit_v1            (per-ticker MLE drift, regime-robust)

Lookahead-safety design
-----------------------
1. The regime indicator is computed from past returns only (already verified
   by the universal leakage-invariance test).
2. The dispersion-bucket boundaries are fixed using only the TRAINING half
   of the apples-to-apples evaluation slice.
3. The per-regime model weights are fitted on the TRAINING half ONLY.
4. The fitted weights are applied to the held-out EVALUATION half.

No information from the evaluation half is used to set boundaries or weights.
This is the standard "out-of-sample validation" structure used in cross-
sectional ensemble papers (e.g., Gu-Kelly-Xiu 2020 §4.4).

Weight scheme
-------------
For each regime bucket b ∈ {low, mid, high} dispersion:
    1. Compute each base model's mean per-date Spearman IC on training-half
       dates that fall in bucket b.
    2. Convert IC values to weights via softmax with temperature τ:
           w_m(b) = exp(τ · IC_m(b)) / Σ_j exp(τ · IC_j(b))
       Defaults to τ = 200, which produces moderately concentrated weights
       (the strongest model in a regime gets ~50-70% weight).
    3. Apply weights at prediction time based on the date's current bucket.

Predictions are cross-sectionally z-scored per date before combining, so
the ensemble weights operate on a common scale across models with
different raw prediction magnitudes.

Usage:
    PYTHONPATH=src python scripts/regime_conditional_ensemble.py
    PYTHONPATH=src python scripts/regime_conditional_ensemble.py --temperature 100
    PYTHONPATH=src python scripts/regime_conditional_ensemble.py --n-buckets 2
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.eval.metrics import summarize
from price_model.eval.turnover import compute_turnover_and_costs
from price_model.features.targets import add_forward_excess_return
from price_model.serving.store import PredictionStore

# Five base models — complementary by design
BASE_MODELS = [
    "lightgbm_kaggle_v2_ohlcv",
    "lightgbm_kaggle_v3_curated",
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
    """Return a (date, dispersion) frame: 20-day rolling mean of per-date
    cross-sectional std of daily log returns. Same value broadcast across
    tickers on any given date, so we collapse to one row per date."""
    sorted_panel = panel.sort(["ticker", "date"])
    c = pl.col("adj_close")
    log_ret = (c.log() - c.log().shift(1)).over("ticker")
    sorted_panel = sorted_panel.with_columns(log_ret.alias("_ret"))
    # Cross-sectional std per date
    cs_std = sorted_panel.group_by("date").agg(pl.col("_ret").std().alias("_cs_std")).sort("date")
    # 20-day rolling mean
    cs_std = cs_std.with_columns(
        pl.col("_cs_std").rolling_mean(window_size=20).alias("dispersion")
    )
    return cs_std.select("date", "dispersion").drop_nulls("dispersion")


def _cross_sectional_zscore(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Z-score a column cross-sectionally per date."""
    mean = pl.col(col).mean().over("date")
    std = pl.col(col).std().over("date")
    return df.with_columns(
        ((pl.col(col) - mean) / pl.when(std == 0).then(1.0).otherwise(std)).alias(col)
    )


def _per_date_ic_by_model(df: pl.DataFrame) -> pl.DataFrame:
    """Per (date, model_id), compute Spearman IC of prediction vs realized.
    Returns long-form (date, model_id, ic)."""
    from scipy.stats import spearmanr

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
        "--n-buckets", type=int, default=3,
        help="Number of dispersion regime buckets (2 or 3 typical).",
    )
    ap.add_argument(
        "--temperature", type=float, default=200.0,
        help="Softmax temperature for converting per-bucket ICs to weights. "
        "Higher → more concentrated weights on the strongest model.",
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
            console.print("[red]Need at least 2 base models in the store to build an ensemble.")
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

    # ----- 3. Load panel, compute realized + dispersion -----
    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)
    panel = add_forward_excess_return(panel, horizon_days=args.horizon, target_col="y")
    realized = panel.select("date", "ticker", pl.col("y").alias("realized"))
    realized = realized.filter(pl.col("date").is_in(common_dates))

    dispersion = _compute_dispersion_indicator(panel)
    dispersion = dispersion.filter(pl.col("date").is_in(common_dates))

    # ----- 4. Join predictions + realized + dispersion -----
    preds = preds.filter(pl.col("date").is_in(common_dates))
    # Deduplicate: the prediction store accumulates predictions across overlapping
    # refit windows, producing multiple rows per (date, ticker, model_id). For
    # the per-date Spearman IC this is harmless (duplicate identical rows don't
    # change the rank correlation), but pivot requires uniqueness. Average
    # across duplicate rows — if any model has multiple predictions for a
    # (date, ticker) we take the mean.
    preds = preds.group_by(["date", "ticker", "model_id"]).agg(
        pl.col("prediction").mean()
    )
    eval_df = preds.join(realized, on=["date", "ticker"], how="inner")
    eval_df = eval_df.join(dispersion, on="date", how="inner")
    console.print(
        f"Eval frame: {eval_df.height:,} rows across "
        f"{eval_df['date'].n_unique():,} dates."
    )

    # Z-score predictions cross-sectionally per date so weights operate on
    # a common scale across model classes.
    eval_df = _cross_sectional_zscore(eval_df, "prediction")

    # ----- 5. Split apples-to-apples slice into TRAIN / EVAL halves -----
    midpoint = common_dates[len(common_dates) // 2]
    train_df = eval_df.filter(pl.col("date") < midpoint)
    test_df = eval_df.filter(pl.col("date") >= midpoint)
    console.print(
        f"Train half: {train_df['date'].n_unique():,} dates "
        f"[{common_dates[0]} → {midpoint}]"
    )
    console.print(
        f"Eval half : {test_df['date'].n_unique():,} dates "
        f"[{midpoint} → {common_dates[-1]}]"
    )

    # ----- 6. Define dispersion-bucket boundaries from TRAIN half ONLY -----
    train_dates_disp = train_df.select("date", "dispersion").unique().sort("date")
    quantiles = np.linspace(0, 1, args.n_buckets + 1)[1:-1]
    thresholds = [
        float(train_dates_disp["dispersion"].quantile(q)) for q in quantiles
    ]
    console.print(
        f"Regime bucket thresholds (from train-half dispersion quantiles): "
        f"{[round(t, 5) for t in thresholds]}"
    )

    def assign_bucket(d: float) -> int:
        for i, t in enumerate(thresholds):
            if d < t:
                return i
        return args.n_buckets - 1

    train_df = train_df.with_columns(
        pl.col("dispersion").map_elements(assign_bucket, return_dtype=pl.Int32).alias("bucket")
    )
    test_df = test_df.with_columns(
        pl.col("dispersion").map_elements(assign_bucket, return_dtype=pl.Int32).alias("bucket")
    )

    # ----- 7. Compute per-bucket per-model train IC -----
    train_ic = _per_date_ic_by_model(train_df.select("date", "ticker", "model_id", "prediction", "realized"))
    bucket_map = train_df.select("date", "bucket").unique()
    train_ic = train_ic.join(bucket_map, on="date", how="inner")

    per_bucket_mean_ic = (
        train_ic.group_by(["bucket", "model_id"])
        .agg(pl.col("ic").mean().alias("mean_ic"), pl.col("ic").len().alias("n"))
        .sort(["bucket", "mean_ic"], descending=[False, True])
    )

    console.rule("[bold]Per-regime training-half mean IC")
    table = Table()
    table.add_column("bucket")
    table.add_column("model")
    table.add_column("train mean IC")
    table.add_column("n dates")
    for row in per_bucket_mean_ic.iter_rows(named=True):
        table.add_row(
            str(row["bucket"]),
            row["model_id"],
            f"{row['mean_ic']:+.4f}",
            str(row["n"]),
        )
    console.print(table)

    # ----- 8. Softmax weights per bucket -----
    tau = args.temperature
    weights: dict[int, dict[str, float]] = {}
    for b in range(args.n_buckets):
        bucket_rows = per_bucket_mean_ic.filter(pl.col("bucket") == b)
        if bucket_rows.height == 0:
            # No training data in this bucket — fall back to uniform.
            weights[b] = {m: 1.0 / len(target_ids) for m in target_ids}
            continue
        ic_map = {row["model_id"]: row["mean_ic"] for row in bucket_rows.iter_rows(named=True)}
        for m in target_ids:
            ic_map.setdefault(m, 0.0)
        ic_arr = np.array([ic_map[m] for m in target_ids])
        scaled = tau * ic_arr
        # Numerical stability: subtract max
        scaled -= scaled.max()
        exp = np.exp(scaled)
        w = exp / exp.sum()
        weights[b] = {m: float(w_i) for m, w_i in zip(target_ids, w, strict=True)}

    console.rule(f"[bold]Per-regime ensemble weights (softmax τ = {tau})")
    table = Table()
    table.add_column("bucket")
    for m in target_ids:
        table.add_column(m, justify="right")
    for b in range(args.n_buckets):
        row = [str(b)]
        for m in target_ids:
            row.append(f"{weights[b][m]:.3f}")
        table.add_row(*row)
    console.print(table)

    # ----- 9. Apply weights to EVAL half to produce ensemble predictions -----
    # Pivot test_df to wide form: one column per model_id with z-scored prediction.
    pivot = test_df.pivot(
        on="model_id",
        index=["date", "ticker", "realized", "bucket"],
        values="prediction",
    )
    # Apply per-row regime weights
    ensemble_pred = pl.Series(np.zeros(pivot.height, dtype=np.float64))
    buckets_arr = pivot["bucket"].to_numpy()
    for m in target_ids:
        if m not in pivot.columns:
            continue
        col = pivot[m].fill_null(0.0).to_numpy()
        w_arr = np.array([weights[int(b)][m] for b in buckets_arr])
        ensemble_pred = ensemble_pred + pl.Series(col * w_arr)
    pivot = pivot.with_columns(ensemble_pred.alias("ensemble_prediction"))

    # Drop rows where any base model's prediction is null
    pivot = pivot.drop_nulls(subset=target_ids)

    # ----- 10. Score ensemble + each individual model on EVAL half -----
    console.rule("[bold green]Out-of-sample evaluation (EVAL half)")

    summary_rows: list[dict] = []

    def score_one(
        name: str, df: pl.DataFrame, pred_col: str
    ) -> dict:
        score_df = df.select(
            "date", "ticker", pl.col(pred_col).alias("prediction"), "realized"
        )
        m = summarize(score_df, horizon_days=args.horizon).as_dict()
        m["model_id"] = name
        # Net-of-cost
        cost_summary = compute_turnover_and_costs(
            score_df, top_frac=0.2, cost_bps=(3, 10, 20), horizon_days=args.horizon
        ).as_dict()
        for k, v in cost_summary.items():
            if k not in m and k != "n_observations" and k != "n_dates":
                m[k] = v
        return m

    summary_rows.append(score_one("ENSEMBLE", pivot, "ensemble_prediction"))
    for m in target_ids:
        if m in pivot.columns:
            summary_rows.append(score_one(m, pivot, m))

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
    # Sort by long_short_sharpe descending for stable display
    summary_rows.sort(key=lambda r: -(r.get("long_short_sharpe") or 0.0))
    for row in summary_rows:
        is_ens = row["model_id"] == "ENSEMBLE"
        cells = []
        for c in cols_to_show:
            v = row.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("nan")
            elif isinstance(v, float):
                if abs(v) < 100:
                    cells.append(f"{v:+.4f}")
                else:
                    cells.append(f"{v:.1f}")
            else:
                cells.append(str(v))
        # Highlight ensemble row
        if is_ens:
            cells = [f"[bold green]{c}[/bold green]" for c in cells]
        table.add_row(*cells)
    console.print(table)


if __name__ == "__main__":
    main()
