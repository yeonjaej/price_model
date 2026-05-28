"""Compose registered features into a feature matrix.

Usage from a config:

    feature_names = ["return_5d", "momentum_60", "vol_20", "rsi_14", "distance_ma_200"]
    panel = build_feature_matrix(
        raw_panel,
        feature_names=feature_names,
        normalize_kind="zscore",
        target_horizon=5,
    )

The output panel has one row per (ticker, date) with: input columns + each feature
column (cross-sectionally normalized) + `y` (forward excess return).
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

import price_model.features.cross_features
import price_model.features.factor_loadings
import price_model.features.technical  # noqa: F401  trigger registration
from price_model.features.base import get_feature
from price_model.features.cross_sectional import NormKind, normalize
from price_model.features.targets import add_forward_excess_return


def build_feature_matrix(
    panel: pl.DataFrame,
    feature_names: Sequence[str],
    normalize_kind: NormKind = "zscore",
    target_horizon: int = 5,
    target_col: str = "y",
) -> pl.DataFrame:
    """Run each registered feature in order, normalize, attach target."""
    out = panel.sort(["ticker", "date"])
    for name in feature_names:
        out = get_feature(name).compute(out)
    out = normalize(out, feature_cols=list(feature_names), kind=normalize_kind)
    out = add_forward_excess_return(out, horizon_days=target_horizon, target_col=target_col)
    return out


def drop_warmup_rows(
    panel: pl.DataFrame,
    feature_names: Sequence[str],
) -> pl.DataFrame:
    """Drop rows where any feature is null (warmup window).

    No-op when `feature_names` is empty — used by feature-free models like Chronos
    that operate on raw prices.
    """
    if not feature_names:
        return panel
    return panel.drop_nulls(subset=list(feature_names))
