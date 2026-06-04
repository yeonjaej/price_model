"""Non-Negative Least Squares (NNLS) stacked ensemble of cross-sectional
return predictors.

Companion to `scripts/ridge_stack_ensemble.py`. Same structure, same lookahead-
safety guarantees, same evaluation; only the second-stage stacker changes:

  Ridge stacker (other script): RidgeCV with α via cross-validation, coefficients
    unconstrained (can go negative).
  NNLS stacker (this script):    LinearRegression with positive=True, coefficients
    constrained to ≥ 0.

The constraint matters in cross-sectional return prediction because base-model
predictions are usually highly correlated (they all predict the same target).
An unconstrained stacker can pick up correlation patterns in the train half
that don't generalize OOS, including assigning negative weights that act as
"contrasts" against other base models. The contrast pattern is fragile to
regime change: a model that was contrarian in the train-half regime may flip
to being correlated in the eval-half regime, and the negative weight then
becomes actively harmful.

The non-negativity constraint eliminates that failure mode. Every weight is
"how much does this base model contribute," with zero as the minimum (the
optimizer drops models that don't help). This is the standard Kaggle stacking
move for cross-sectional return prediction problems, supported by the broader
asset-pricing-ML literature on stacked ensembles (Gu-Kelly-Xiu 2020 use
non-negativity-constrained linear stacking).

Mathematically:

  Ridge:  minimize ||X·β − y||² + α·||β||²              (β unconstrained)
  NNLS:   minimize ||X·β − y||²            subject to β ≥ 0

In sklearn:

  Ridge:  RidgeCV(alphas=[…], cv=K)
  NNLS:   LinearRegression(positive=True, fit_intercept=True)

The intercept is left unconstrained (it can be any real number); only the
slope coefficients are constrained to be ≥ 0.

Usage:
    PYTHONPATH=src python scripts/nnls_stack_ensemble.py
    PYTHONPATH=src python scripts/nnls_stack_ensemble.py --regime-conditional
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


def _fit_nnls_and_apply(
    train_X: np.ndarray,
    train_y: np.ndarray,
    eval_X: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit NNLS (LinearRegression with positive=True) on (train_X, train_y),
    apply to eval_X. Returns (eval_predictions, diagnostic_dict).

    Equivalent to scipy.optimize.nnls but with an unconstrained intercept term
    handled by sklearn's LinearRegression. The intercept can be any real
    number; only the slope coefficients are constrained to be ≥ 0.
    """
    from sklearn.linear_model import LinearRegression

    model = LinearRegression(positive=True, fit_intercept=True)
    model.fit(train_X, train_y)
    eval_pred = model.predict(eval_X)
    diagnostics = {
        "intercept": float(model.intercept_),
        "coefficients": [float(c) for c in model.coef_],
        "n_nonzero_weights": int(np.sum(np.abs(model.coef_) > 1e-12)),
    }
    return eval_pred, diagnostics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--regime-conditional", action="store_true",
        help="Fit a separate NNLS per dispersion bucket instead of one global NNLS.",
    )
    ap.add_argument(
        "--n-buckets", type=int, default=3,
        help="Number of dispersion regime buckets (only used with --regime-conditional).",
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

    # ----- 3. Realized + dispersion -----
    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)
    panel = add_forward_excess_return(panel, horizon_days=args.horizon, target_col="y")
    realized = panel.select("date", "ticker", pl.col("y").alias("realized")).filter(
        pl.col("date").is_in(common_dates)
    )
    dispersion = _compute_dispersion_indicator(panel).filter(
        pl.col("date").is_in(common_dates)
    )

    # ----- 4. Join + dedup + z-score -----
    preds = preds.filter(pl.col("date").is_in(common_dates))
    preds = preds.group_by(["date", "ticker", "model_id"]).agg(
        pl.col("prediction").mean()
    )
    eval_df = preds.join(realized, on=["date", "ticker"], how="inner")
    eval_df = eval_df.join(dispersion, on="date", how="inner")
    eval_df = _cross_sectional_zscore(eval_df, "prediction")

    # ----- 5. Pivot to wide form: one column per base model prediction -----
    pivot = eval_df.pivot(
        on="model_id",
        index=["date", "ticker", "realized", "dispersion"],
        values="prediction",
    )
    pivot = pivot.drop_nulls(subset=[*target_ids, "realized"])
    console.print(
        f"Pivot frame: {pivot.height:,} rows × {len(target_ids)} model columns "
        f"after dropping null realized + null predictions."
    )

    # ----- 6. Train / eval split at midpoint of apples-to-apples slice -----
    midpoint = common_dates[len(common_dates) // 2]
    train_pivot = pivot.filter(pl.col("date") < midpoint)
    test_pivot = pivot.filter(pl.col("date") >= midpoint)
    console.print(
        f"Train half: {train_pivot['date'].n_unique():,} dates "
        f"[{common_dates[0]} → {midpoint}]"
    )
    console.print(
        f"Eval half:  {test_pivot['date'].n_unique():,} dates "
        f"[{midpoint} → {common_dates[-1]}]"
    )

    # ----- 7. Fit NNLS stacking -----
    if not args.regime_conditional:
        # ----- 7a. Global NNLS stack -----
        train_X = train_pivot.select(target_ids).to_numpy()
        train_y = train_pivot["realized"].to_numpy()
        eval_X = test_pivot.select(target_ids).to_numpy()

        eval_pred, diag = _fit_nnls_and_apply(train_X, train_y, eval_X)
        console.rule("[bold]NNLS stacking (global)")
        table = Table()
        table.add_column("base model")
        table.add_column("weight (NNLS coefficient)", justify="right")
        table.add_column("normalized (% of total)", justify="right")
        total_weight = sum(diag["coefficients"])
        for m, c in zip(target_ids, diag["coefficients"], strict=True):
            normalized = (c / total_weight * 100.0) if total_weight > 1e-12 else 0.0
            table.add_row(m, f"{c:+.4f}", f"{normalized:5.1f}%")
        table.add_row("intercept", f"{diag['intercept']:+.6f}", "—")
        table.add_row(
            "non-zero weight count", str(diag["n_nonzero_weights"]), "—"
        )
        console.print(table)

        test_pivot = test_pivot.with_columns(
            pl.Series("ensemble_prediction", eval_pred)
        )

    else:
        # ----- 7b. Regime-conditional NNLS (one model per bucket) -----
        train_disp = train_pivot["dispersion"].to_numpy()
        quantiles = np.linspace(0, 1, args.n_buckets + 1)[1:-1]
        thresholds = [float(np.quantile(train_disp, q)) for q in quantiles]
        console.print(
            f"Regime bucket thresholds (train half quantiles): "
            f"{[round(t, 5) for t in thresholds]}"
        )

        def assign_bucket(d: float) -> int:
            for i, t in enumerate(thresholds):
                if d < t:
                    return i
            return args.n_buckets - 1

        train_pivot = train_pivot.with_columns(
            pl.col("dispersion").map_elements(
                assign_bucket, return_dtype=pl.Int32
            ).alias("bucket")
        )
        test_pivot = test_pivot.with_columns(
            pl.col("dispersion").map_elements(
                assign_bucket, return_dtype=pl.Int32
            ).alias("bucket")
        )

        eval_pred = np.zeros(test_pivot.height, dtype=np.float64)
        console.rule("[bold]NNLS stacking (per regime bucket)")
        table = Table()
        table.add_column("bucket")
        for m in target_ids:
            table.add_column(m, justify="right")
        table.add_column("intercept", justify="right")
        table.add_column("n_nonzero", justify="right")

        for b in range(args.n_buckets):
            train_b = train_pivot.filter(pl.col("bucket") == b)
            test_b = test_pivot.filter(pl.col("bucket") == b)
            if train_b.height < 100:
                console.print(
                    f"[yellow]Bucket {b}: only {train_b.height} train rows; "
                    f"falling back to global NNLS weights.[/yellow]"
                )
                train_X_b = train_pivot.select(target_ids).to_numpy()
                train_y_b = train_pivot["realized"].to_numpy()
            else:
                train_X_b = train_b.select(target_ids).to_numpy()
                train_y_b = train_b["realized"].to_numpy()
            test_X_b = test_b.select(target_ids).to_numpy()
            if test_X_b.shape[0] == 0:
                continue
            pred_b, diag_b = _fit_nnls_and_apply(train_X_b, train_y_b, test_X_b)
            test_b_index = test_pivot["bucket"].to_numpy() == b
            eval_pred[test_b_index] = pred_b
            cells = [str(b)] + [f"{c:+.4f}" for c in diag_b["coefficients"]]
            cells.append(f"{diag_b['intercept']:+.6f}")
            cells.append(str(diag_b["n_nonzero_weights"]))
            table.add_row(*cells)
        console.print(table)
        test_pivot = test_pivot.with_columns(
            pl.Series("ensemble_prediction", eval_pred)
        )

    # ----- 8. Evaluate ensemble + base models on eval half -----
    console.rule("[bold green]Out-of-sample evaluation (eval half)")

    summary_rows: list[dict] = []

    def score_one(name: str, df: pl.DataFrame, pred_col: str) -> dict:
        score_df = df.select(
            "date", "ticker", pl.col(pred_col).alias("prediction"), "realized"
        )
        m = summarize(score_df, horizon_days=args.horizon).as_dict()
        m["model_id"] = name
        cost_summary = compute_turnover_and_costs(
            score_df, top_frac=0.2, cost_bps=(3, 10, 20), horizon_days=args.horizon
        ).as_dict()
        for k, v in cost_summary.items():
            if k not in m and k not in ("n_observations", "n_dates"):
                m[k] = v
        return m

    label = "ENSEMBLE_NNLS_REGIME" if args.regime_conditional else "ENSEMBLE_NNLS"
    summary_rows.append(score_one(label, test_pivot, "ensemble_prediction"))
    for m in target_ids:
        if m in test_pivot.columns:
            summary_rows.append(score_one(m, test_pivot, m))

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
    summary_rows.sort(key=lambda r: -(r.get("information_coefficient") or 0.0))
    for row in summary_rows:
        is_ens = row["model_id"].startswith("ENSEMBLE_")
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
