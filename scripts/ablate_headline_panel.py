"""Panel-ablation: does pruning the collinear vol cluster beat the 9-feature panel?

Hypothesis (from the vol-cluster cancellation finding): the volatility cluster
{vol_ewm_20, idio_vol_20, max_return_21d, beta_60} mostly cancels under L1 in the
2025+ refits, contributing little net signal. If so, dropping the redundant
members should hold IC (or improve it via lower variance) and cut turnover.

Runs the identical walk-forward recipe (rank-normalized, h=21, embargo 22d,
refit 252d, min_train 504d) for several panel variants, evaluated on the matched
2025-01-02+ OOS slice. All variants share identical splits, so the comparison
isolates the panel.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/ablate_headline_panel.py
"""

from __future__ import annotations

from datetime import date

import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.eval.metrics import summarize
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.pipeline.walk_forward import join_with_realized, run_walk_forward

FULL9 = [
    "momentum_12_1", "momentum_756", "return_1d",
    "vol_ewm_20", "idio_vol_20", "max_return_21d",
    "distance_52w_high", "log_dollar_volume", "beta_60",
]

# Variants: name -> feature list. Vol cluster = {vol_ewm_20, idio_vol_20, max_return_21d, beta_60}.
VARIANTS: dict[str, list[str]] = {
    "9 full (headline)": FULL9,
    "8 drop idio_vol": [f for f in FULL9 if f != "idio_vol_20"],
    "7 drop idio_vol+max_ret": [f for f in FULL9 if f not in {"idio_vol_20", "max_return_21d"}],
    "6 vol->vol_ewm only": [f for f in FULL9 if f not in {"idio_vol_20", "max_return_21d", "beta_60"}],
    "5 drop vol cluster": [f for f in FULL9 if f not in {"vol_ewm_20", "idio_vol_20", "max_return_21d", "beta_60"}],
}

OOS_START = date(2025, 1, 2)


def run_variant(matrix_full: pl.DataFrame, feats: list[str], target_full: pl.DataFrame) -> dict:
    # Re-normalize on just this feature subset so rank scaling matches a real run.
    sub = matrix_full.select(["date", "ticker", "adj_close", *feats]).drop_nulls(subset=feats)
    model = build_model(
        "LassoCrossSectional",
        ModelConfig(model_id="ablate", feature_cols=feats, params={"cv": 3}),
    )
    preds = run_walk_forward(
        sub.join(target_full, on=["date", "ticker"], how="left"),
        model=model,
        feature_cols=feats,
        target_col="y",
        experiment_id="ablate",
        horizon_days=21,
        refit_freq_days=252,
        embargo_days=22,
        min_train_days=504,
    )
    joined = join_with_realized(preds, target_full).filter(pl.col("date") >= OOS_START)
    m = summarize(joined, horizon_days=21)
    return {
        "n_dates": m.n_dates,
        "ic": m.information_coefficient,
        "t": m.ic_t_stat,
        "sharpe": m.long_short_sharpe,
    }


def main() -> None:
    console = Console()
    console.print("Loading panel...")
    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)

    # Build each variant by re-normalizing its own feature subset (rank, per date),
    # so the rank transform is computed within the subset exactly as a real config would.
    table = Table(title=f"Headline-panel ablation — OOS {OOS_START}+ (matched splits)")
    table.add_column("variant", style="bold")
    table.add_column("k", justify="right")
    table.add_column("n_dates", justify="right")
    table.add_column("IC", justify="right")
    table.add_column("t-stat", justify="right")
    table.add_column("L/S Sharpe", justify="right")

    for name, feats in VARIANTS.items():
        matrix = build_feature_matrix(panel, feature_names=feats, normalize_kind="rank", target_horizon=21)
        matrix = drop_warmup_rows(matrix, feats).sort(["ticker", "date"])
        target = matrix.select("date", "ticker", "y")
        sub = matrix.select(["date", "ticker", *feats])
        model = build_model(
            "LassoCrossSectional",
            ModelConfig(model_id="ablate", feature_cols=feats, params={"cv": 3}),
        )
        preds = run_walk_forward(
            matrix, model=model, feature_cols=feats, target_col="y",
            experiment_id="ablate", horizon_days=21,
            refit_freq_days=252, embargo_days=22, min_train_days=504,
        )
        joined = join_with_realized(preds, target).filter(pl.col("date") >= OOS_START)
        m = summarize(joined, horizon_days=21)
        table.add_row(
            name, str(len(feats)), str(m.n_dates),
            f"{m.information_coefficient:+.4f}", f"{m.ic_t_stat:+.2f}", f"{m.long_short_sharpe:+.2f}",
        )
        console.print(f"  done: {name}")

    console.print(table)


if __name__ == "__main__":
    main()
