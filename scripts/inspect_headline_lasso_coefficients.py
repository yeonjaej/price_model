"""Inspect headline 9-feature L1 (Lasso) coefficients across walk-forward refits.

Question this answers
---------------------
The headline panel (lasso_elasso_pit_h21) contains a strongly collinear
volatility cluster — vol_ewm_20 / idio_vol_20 / max_return_21d / beta_60 at
pairwise Spearman 0.48-0.79 (notebook 05). That is exactly the structure that
triggers the documented L1-cancellation failure mode (cf.
inspect_momentum_lasso_coefficients.py). This script checks how the headline
Lasso actually treats that cluster: does sparsity drop all-but-one (benign), or
does L1 split the cluster into large offsetting signs (cancellation)?

It reproduces the experiment's walk-forward refits (rank-normalized, 21-day
horizon, embargo 22d, refit every 252d, min_train 504d) and prints the fitted
coefficient on every feature at each refit, flagging the vol cluster.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/inspect_headline_lasso_coefficients.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.data.splits import slice_train, walk_forward_splits
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig

FEATS = [
    "momentum_12_1",
    "momentum_756",
    "return_1d",
    "vol_ewm_20",
    "idio_vol_20",
    "max_return_21d",
    "distance_52w_high",
    "log_dollar_volume",
    "beta_60",
]
VOL_CLUSTER = {"vol_ewm_20", "idio_vol_20", "max_return_21d", "beta_60"}


def main() -> None:
    console = Console()

    console.print("Loading panel + building rank-normalized 9-feature matrix (h=21)...")
    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)
    matrix = build_feature_matrix(
        panel, feature_names=FEATS, normalize_kind="rank", target_horizon=21
    )
    matrix = drop_warmup_rows(matrix, FEATS).sort(["ticker", "date"])

    start = matrix["date"].min()
    end = matrix["date"].max()
    splits = list(
        walk_forward_splits(
            start=start, end=end, refit_freq_days=252, embargo_days=22, min_train_days=504
        )
    )
    console.print(f"{len(splits)} walk-forward refits over [{start} -> {end}]\n")

    header = f"{'refit':12s} {'alpha':>9s}  " + "  ".join(
        f"{('*'+f if f in VOL_CLUSTER else f)[:14]:>14s}" for f in FEATS
    )
    console.print(header)
    console.print("-" * len(header))

    cluster_rows: list[list[float]] = []
    for split in splits:
        train = slice_train(matrix, split).drop_nulls(subset=["y", *FEATS])
        if train.height == 0:
            continue
        model = build_model(
            "LassoCrossSectional",
            ModelConfig(model_id="headline", feature_cols=FEATS, params={"cv": 3}),
        )
        model.fit(train)
        coefs = model.feature_importance()
        alpha = model.selected_alpha()
        line = f"{str(split.refit_date):12s} {alpha:9.2e}  " + "  ".join(
            f"{coefs[f]:+14.4f}" for f in FEATS
        )
        console.print(line)
        cluster_rows.append([coefs[f] for f in sorted(VOL_CLUSTER)])

    # Cancellation diagnostic on the vol cluster.
    arr = np.array(cluster_rows)
    console.print("\n[bold]Vol-cluster cancellation check[/bold] "
                  f"(features: {sorted(VOL_CLUSTER)})")
    for i, name in enumerate(sorted(VOL_CLUSTER)):
        col = arr[:, i]
        nz = np.mean(np.abs(col) > 1e-6)
        console.print(f"  {name:18s}  mean={col.mean():+.3f}  "
                      f"|mean|abs={np.abs(col).mean():.3f}  nonzero in {nz*100:.0f}% of refits")
    signed_sum = arr.sum(axis=1)
    abs_sum = np.abs(arr).sum(axis=1)
    ratio = np.where(abs_sum > 1e-9, np.abs(signed_sum) / abs_sum, 1.0)
    console.print(
        f"\n  cancellation ratio |Σcoef|/Σ|coef| per refit "
        f"(1.0 = no cancellation, ~0 = full cancellation):\n  "
        + "  ".join(f"{r:.2f}" for r in ratio)
    )
    console.print(f"  mean ratio = {ratio.mean():.2f}")


if __name__ == "__main__":
    main()
