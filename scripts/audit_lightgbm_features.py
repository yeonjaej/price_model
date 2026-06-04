"""Feature audit for the v2_ohlcv LightGBM panel.

Two diagnostics on the existing 22-feature panel:

1. FEATURE IMPORTANCE (gain) — trains a fresh LightGBM with v2_ohlcv hyper-
   parameters on the full apples-to-apples slice, then extracts the
   gain-based feature importance averaged across the multi-seed boosters.
   Identifies which features the tree relies on most for splitting and
   which contribute negligible information.

2. CROSS-FEATURE CORRELATION — computes the cross-feature Pearson
   correlation matrix on the (post-warmup, post-normalization) feature
   matrix, identifies pairs with |corr| > 0.7, and groups them into
   redundancy clusters. Within each cluster, keeping the highest-
   importance representative and dropping the rest typically maintains
   gross IC while reducing turnover and noise.

Output is two ranked tables printed to the console, plus a summary
section identifying the most likely candidates for removal.

Usage:
    PYTHONPATH=src python scripts/audit_lightgbm_features.py
    PYTHONPATH=src python scripts/audit_lightgbm_features.py \
        --experiment extended_kaggle_v2_ohlcv \
        --corr-threshold 0.7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig


def _cluster_high_correlation_pairs(
    pairs: list[tuple[str, str, float]],
) -> list[set[str]]:
    """Union-find on the correlation graph: features connected by |corr| > τ
    get grouped into clusters."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), x)
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, _ in pairs:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    clusters: dict[str, set[str]] = {}
    for node in parent:
        root = find(node)
        clusters.setdefault(root, set()).add(node)
    return [c for c in clusters.values() if len(c) > 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment",
        default="extended_kaggle_v2_ohlcv",
        help="Experiment YAML to audit (without .yaml suffix).",
    )
    ap.add_argument(
        "--corr-threshold",
        type=float,
        default=0.7,
        help="|Pearson r| threshold for flagging feature redundancy.",
    )
    args = ap.parse_args()

    console = Console()
    cfg_path = Path("config/experiments") / f"{args.experiment}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    console.print(f"Auditing experiment: [bold]{cfg['experiment_id']}[/bold]")
    console.print(f"Feature panel size: {len(cfg['features'])} features")

    # ----- Load panel and build feature matrix -----
    panel = load_panel(
        universe=cfg["data"]["universe"],
        start=cfg["data"]["start"],
        end=cfg["data"].get("end"),
        pit_filter=cfg["data"].get("pit_filter", False),
    )
    matrix = build_feature_matrix(
        panel,
        feature_names=cfg["features"],
        normalize_kind=cfg.get("normalize_kind", "zscore"),
        target_horizon=cfg["target_horizon"],
    )
    matrix = drop_warmup_rows(matrix, cfg["features"]).drop_nulls("y")
    console.print(
        f"Training rows after warmup + target dropna: {matrix.height:,} | "
        f"unique dates: {matrix['date'].n_unique():,}"
    )

    # ----- Train LightGBM with experiment hyperparameters -----
    lgbm_specs = [m for m in cfg["models"] if m["class"] == "LightGBMModel"]
    if not lgbm_specs:
        classes_in_cfg = sorted({m["class"] for m in cfg["models"]})
        console.print(
            f"[red]This audit script requires a LightGBMModel in the experiment "
            f"config[/red] (gain-based feature importance is a tree-model concept). "
            f"Found classes: {classes_in_cfg}. "
            f"For Lasso / Ridge experiments, inspect `feature_importance()` "
            f"directly via `scripts/inspect_lasso.py`."
        )
        return
    lgbm_spec = lgbm_specs[0]
    model_cfg = ModelConfig(
        model_id=lgbm_spec["id"],
        feature_cols=tuple(cfg["features"]),
        target_col="y",
        params=lgbm_spec.get("params", {}),
    )
    console.print("[dim]Training LightGBM on full panel (this can take 2-5 min)...[/dim]")
    model = build_model("LightGBMModel", model_cfg)
    model.fit(matrix)

    # ----- (1) Feature importance (gain) -----
    importance = model.feature_importance()  # dict[feature -> gain]
    importance_sorted = sorted(importance.items(), key=lambda kv: -kv[1])
    total_gain = sum(importance.values()) or 1.0

    console.rule("[bold green]Feature importance (gain), sorted descending")
    table = Table()
    table.add_column("rank")
    table.add_column("feature")
    table.add_column("gain")
    table.add_column("share")
    for i, (name, gain) in enumerate(importance_sorted):
        share = gain / total_gain
        table.add_row(f"{i + 1}", name, f"{gain:,.0f}", f"{share:.1%}")
    console.print(table)

    # Identify the 5 lowest-importance features
    bottom_n = 5
    bottom = importance_sorted[-bottom_n:]
    console.print()
    console.print(
        f"[bold]Bottom {bottom_n} features by gain[/bold] (removal candidates):"
    )
    for name, gain in bottom:
        share = gain / total_gain
        console.print(f"  - {name:30s} gain={gain:,.0f}  share={share:.2%}")

    # ----- (2) Cross-feature correlation matrix -----
    feat_arr = matrix.select(cfg["features"]).to_numpy()
    # The matrix is post-normalization (zscore per date), so raw corr is fine.
    corr = np.corrcoef(feat_arr, rowvar=False)
    n = len(cfg["features"])

    threshold = args.corr_threshold
    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if not np.isnan(r) and abs(r) > threshold:
                pairs.append((cfg["features"][i], cfg["features"][j], float(r)))
    pairs.sort(key=lambda x: -abs(x[2]))

    console.rule(
        f"[bold green]Feature pairs with |corr| > {threshold:.2f} "
        f"({len(pairs)} pairs)"
    )
    if pairs:
        table = Table()
        table.add_column("feature A")
        table.add_column("feature B")
        table.add_column("corr")
        for a, b, r in pairs:
            table.add_row(a, b, f"{r:+.3f}")
        console.print(table)
    else:
        console.print(
            f"[dim]No feature pairs exceed |corr| > {threshold:.2f}. "
            "Panel is well-decorrelated.[/dim]"
        )

    # ----- Cluster correlated features and identify suggested drops -----
    clusters = _cluster_high_correlation_pairs(pairs)
    if clusters:
        console.rule("[bold green]Redundancy clusters (drop all but highest-importance)")
        for ci, cluster in enumerate(clusters):
            # Sort cluster members by gain importance, descending
            ranked = sorted(cluster, key=lambda f: -importance.get(f, 0.0))
            keep = ranked[0]
            drop = ranked[1:]
            console.print(
                f"  Cluster {ci + 1} ({len(cluster)} features): "
                f"[bold green]keep[/bold green] {keep} "
                f"(gain={importance[keep]:,.0f})  "
                f"[bold red]drop[/bold red] {drop}"
            )
    else:
        console.print()
        console.print(
            "[dim]No redundancy clusters detected at the chosen threshold.[/dim]"
        )

    # ----- Combined removal candidates -----
    bottom_set = {name for name, _ in bottom}
    drop_from_clusters: set[str] = set()
    for cluster in clusters:
        ranked = sorted(cluster, key=lambda f: -importance.get(f, 0.0))
        drop_from_clusters.update(ranked[1:])
    suggested_drops = sorted(bottom_set | drop_from_clusters)

    console.rule("[bold green]Suggested drops (union of bottom-5 and cluster-redundant)")
    if suggested_drops:
        for name in suggested_drops:
            reasons = []
            if name in bottom_set:
                reasons.append(f"bottom {bottom_n} by gain")
            if name in drop_from_clusters:
                reasons.append("cluster-redundant")
            console.print(
                f"  [bold red]drop[/bold red] {name:30s} "
                f"({'; '.join(reasons)})"
            )
        keep_count = len(cfg["features"]) - len(suggested_drops)
        console.print()
        console.print(
            f"Suggested curated panel size: {keep_count} features "
            f"(from {len(cfg['features'])})"
        )
    else:
        console.print(
            "[dim]No removal candidates flagged — current panel is reasonably curated.[/dim]"
        )


if __name__ == "__main__":
    main()
