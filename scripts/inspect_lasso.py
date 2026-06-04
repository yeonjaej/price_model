"""Print LassoCV's selected alpha and feature coefficients for lasso_pit.

The walk-forward harness doesn't checkpoint per-refit model state to disk —
it only writes predictions to the DuckDB store. So to inspect the fitted
Lasso we re-run the LAST training fold inline here and print its internals.

Usage:
    python scripts/inspect_lasso.py
    # or with a different config:
    python scripts/inspect_lasso.py --experiment lasso_pit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from price_model.data.loaders import load_panel
from price_model.data.splits import slice_train, walk_forward_splits
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.models.linear import LassoCrossSectional


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="lasso_pit")
    args = ap.parse_args()

    cfg_path = Path("config/experiments") / f"{args.experiment}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

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
    matrix = drop_warmup_rows(matrix, cfg["features"])
    matrix = matrix.sort(["ticker", "date"])

    # Find the LassoCrossSectional model spec in the config
    lasso_spec = next(m for m in cfg["models"] if m["class"] == "LassoCrossSectional")
    feature_cols = lasso_spec.get("features", cfg["features"])

    # Build the same walk-forward splits the harness uses and grab the LAST one
    start = matrix["date"].min()
    end = matrix["date"].max()
    splits = list(
        walk_forward_splits(
            start=start,
            end=end,
            refit_freq_days=cfg["walk_forward"]["refit_freq_days"],
            embargo_days=cfg["walk_forward"]["embargo_days"],
            min_train_days=cfg["walk_forward"]["min_train_days"],
        )
    )
    if not splits:
        print("No splits produced — check walk_forward config.")
        return
    last_split = splits[-1]
    print(f"Inspecting LAST refit: train ends {last_split.train_end}")
    print(f"({len(splits)} total refits over the run)")

    train = slice_train(matrix, last_split).drop_nulls(subset=["y", *feature_cols])
    print(f"Training rows: {train.height:,}   |   features: {len(feature_cols)}")

    config = ModelConfig(
        model_id=lasso_spec["id"],
        feature_cols=tuple(feature_cols),
        target_col="y",
        params=lasso_spec.get("params", {}),
    )
    model: LassoCrossSectional = build_model("LassoCrossSectional", config)  # type: ignore[assignment]
    model.fit(train)

    print()
    print(f"selected alpha:  {model.selected_alpha():.6g}")
    print()
    print("Feature coefficients (sorted by |coef|):")
    fi = sorted(model.feature_importance().items(), key=lambda kv: -abs(kv[1]))
    for name, coef in fi:
        marker = "  " if coef != 0.0 else " (zero)"
        print(f"  {name:25s} {coef:+.6f}{marker}")

    n_nonzero = sum(1 for _, v in fi if v != 0.0)
    print()
    print(f"Non-zero coefficients: {n_nonzero} / {len(fi)}")
    if n_nonzero == 0:
        print("→ L1 zeroed every feature: model is the cross-sectional mean.")
        print("  This is the universe telling you no feature in the panel has a")
        print("  stable enough cross-sectional relationship with forward 5-day")
        print("  excess returns to survive cross-validated regularization.")
    else:
        # Show the prediction's empirical spread on a recent slice
        recent = train.tail(min(50_000, train.height))
        preds = model.predict(recent)["prediction"]
        print()
        print("In-sample prediction stats on tail of training fold:")
        print(f"  std:    {float(preds.std()):.6f}")
        print(f"  min:    {float(preds.min()):.6f}")
        print(f"  max:    {float(preds.max()):.6f}")
        print(f"  mean:   {float(preds.mean()):.6f}")
        print(
            f"  approximate dispersion vs target std ({float(recent['y'].std()):.4f}): "
            f"{float(preds.std()) / float(recent['y'].std()):.4f}x"
        )


if __name__ == "__main__":
    main()
