"""Diagnose the pure-momentum-Lasso/ElasticNet IC failure.

The pure-momentum L1 / L1+L2 experiments (momentum_lasso_pit, momentum_elasticnet_pit)
produced statistically significant NEGATIVE IC (~-0.019, t=-3.0) on the 2025+ slice.
The hypothesis is the canonical Lasso failure mode on collinear features:
L1 splits weight between mom_504 and mom_756 (correlation ~0.85+) into cancelling
signs of large magnitude. This script verifies that hypothesis by fitting
LassoCrossSectional and ElasticNetCrossSectional on a fixed train window
matching the walk-forward setup, then printing the actual fitted coefficients.

If the hypothesis is correct, we'll see:
  - mom_504 and mom_756 with OPPOSITE signs (cancellation)
  - mom_12_1 and mom_378 near zero
  - Sum of coefficients near zero (the cancellation)
  - High |coefficient| values despite the cancellation

If we see all-positive small coefficients summing to a sensible weighted average,
the failure must be elsewhere (e.g., walk-forward refit instability, prediction
sign error, etc.).

Usage:
    PYTHONPATH=src python scripts/inspect_momentum_lasso_coefficients.py
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig

FEATS = ["momentum_12_1", "momentum_378", "momentum_504", "momentum_756"]


def main() -> None:
    console = Console()

    # Load the same data the experiments used
    console.print("Loading panel...")
    panel = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=True)

    # Build the rank-normalized feature matrix matching the experiment config
    matrix = build_feature_matrix(
        panel,
        feature_names=FEATS,
        normalize_kind="rank",
        target_horizon=5,
    )
    matrix = drop_warmup_rows(matrix, FEATS).drop_nulls("y").sort("date")
    console.print(
        f"Feature matrix: {matrix.height:,} rows × {matrix['date'].n_unique():,} dates "
        f"[{matrix['date'].min()} → {matrix['date'].max()}]"
    )

    # Use a fixed train window: everything before 2024-01-01.
    # This approximates the LAST walk-forward refit before the 2025+ eval slice.
    train_end = date(2024, 1, 1)
    train = matrix.filter(pl.col("date") < train_end)
    console.print(
        f"\nTrain window: {train.height:,} rows, "
        f"{train['date'].min()} → {train['date'].max()}"
    )

    # ---------- Inspect the feature correlation matrix ----------
    console.print("\n[bold]Feature correlation matrix (within train window):[/bold]")
    feat_corr_table = Table()
    feat_corr_table.add_column("")
    for f in FEATS:
        feat_corr_table.add_column(f)
    arr = train.select(FEATS).drop_nulls().to_numpy()
    corr = np.corrcoef(arr.T)
    for i, f1 in enumerate(FEATS):
        row = [f1] + [f"{corr[i, j]:+.3f}" for j in range(len(FEATS))]
        feat_corr_table.add_row(*row)
    console.print(feat_corr_table)

    # Pairwise correlations > 0.7 indicate Lasso cancellation risk
    console.print("\n[bold]Pairwise feature correlations > 0.7 (cancellation risk):[/bold]")
    for i in range(len(FEATS)):
        for j in range(i + 1, len(FEATS)):
            if abs(corr[i, j]) > 0.7:
                console.print(
                    f"  {FEATS[i]} vs {FEATS[j]}: r = {corr[i, j]:+.3f} "
                    f"[red]<-- collinear[/red]"
                )

    # ---------- Fit LassoCV ----------
    console.print("\n[bold]Fitting LassoCrossSectional on train window...[/bold]")
    lasso_cfg = ModelConfig(
        model_id="momentum_lasso_diag",
        feature_cols=tuple(FEATS),
        params={"cv": 5, "max_iter": 5000},
    )
    lasso = build_model("LassoCrossSectional", lasso_cfg)
    lasso.fit(train)
    lasso_coefs = lasso.feature_importance()
    lasso_alpha = lasso.selected_alpha()

    console.print(f"  CV-selected α: {lasso_alpha:.6g}")
    console.print(f"  Sum of coefficients: {sum(lasso_coefs.values()):+.4f}")
    console.print(f"  Sum of |coefficients|: {sum(abs(v) for v in lasso_coefs.values()):.4f}")
    lasso_table = Table()
    lasso_table.add_column("feature")
    lasso_table.add_column("coefficient", justify="right")
    lasso_table.add_column("|coefficient|", justify="right")
    for f, c in lasso_coefs.items():
        lasso_table.add_row(f, f"{c:+.4f}", f"{abs(c):.4f}")
    console.print(lasso_table)

    # ---------- Fit ElasticNetCV ----------
    console.print("\n[bold]Fitting ElasticNetCrossSectional on train window...[/bold]")
    enet_cfg = ModelConfig(
        model_id="momentum_elasticnet_diag",
        feature_cols=tuple(FEATS),
        params={"cv": 5, "max_iter": 5000},
    )
    enet = build_model("ElasticNetCrossSectional", enet_cfg)
    enet.fit(train)
    enet_coefs = enet.feature_importance()
    enet_alpha = enet.selected_alpha()
    enet_l1_ratio = enet.selected_l1_ratio()

    console.print(f"  CV-selected α: {enet_alpha:.6g}")
    console.print(f"  CV-selected l1_ratio: {enet_l1_ratio:.3f}")
    console.print(f"  Sum of coefficients: {sum(enet_coefs.values()):+.4f}")
    console.print(f"  Sum of |coefficients|: {sum(abs(v) for v in enet_coefs.values()):.4f}")
    enet_table = Table()
    enet_table.add_column("feature")
    enet_table.add_column("coefficient", justify="right")
    enet_table.add_column("|coefficient|", justify="right")
    for f, c in enet_coefs.items():
        enet_table.add_row(f, f"{c:+.4f}", f"{abs(c):.4f}")
    console.print(enet_table)

    # ---------- Verdict ----------
    console.rule("[bold green]Verdict")
    pos_lasso = [f for f, c in lasso_coefs.items() if c > 0]
    neg_lasso = [f for f, c in lasso_coefs.items() if c < 0]
    if pos_lasso and neg_lasso:
        console.print(
            f"[red]CANCELLATION CONFIRMED for Lasso:[/red] "
            f"positive coefficients on {pos_lasso}, "
            f"negative coefficients on {neg_lasso}. "
            f"Net sum is small but |coefficients| are large — "
            f"the classical L1-on-collinear-features pathology."
        )
    else:
        console.print(
            f"[yellow]No sign cancellation observed in Lasso.[/yellow] "
            f"All coefficients same sign. Failure must be elsewhere "
            f"(walk-forward refit instability, sign error, etc.)."
        )

    pos_enet = [f for f, c in enet_coefs.items() if c > 0]
    neg_enet = [f for f, c in enet_coefs.items() if c < 0]
    if pos_enet and neg_enet:
        console.print(
            f"[red]CANCELLATION CONFIRMED for ElasticNet:[/red] "
            f"positive coefficients on {pos_enet}, "
            f"negative coefficients on {neg_enet}. "
            f"L2 stabilization did not prevent sign cancellation at the CV-selected l1_ratio."
        )
    else:
        console.print(
            f"[yellow]No sign cancellation observed in ElasticNet.[/yellow] "
            f"All coefficients same sign."
        )


if __name__ == "__main__":
    main()
