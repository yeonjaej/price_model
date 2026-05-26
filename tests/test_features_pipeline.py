"""Smoke tests for the feature pipeline end-to-end."""

from __future__ import annotations

import polars as pl

from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows


def test_pipeline_produces_expected_columns(synthetic_panel):
    feats = ["return_5d", "momentum_60", "vol_20", "rsi_14", "distance_ma_200"]
    matrix = build_feature_matrix(synthetic_panel, feature_names=feats, target_horizon=5)
    for c in [*feats, "y", "date", "ticker"]:
        assert c in matrix.columns


def test_cross_sectional_zscore_has_zero_mean_per_date(synthetic_panel):
    feats = ["return_5d", "momentum_60"]
    matrix = build_feature_matrix(synthetic_panel, feature_names=feats, target_horizon=5)
    matrix = drop_warmup_rows(matrix, feats)
    # Mean of normalized feature per date should be ~0
    by_date = matrix.group_by("date").agg(
        pl.col("return_5d").mean().alias("m_ret"),
        pl.col("momentum_60").mean().alias("m_mom"),
    )
    assert by_date["m_ret"].abs().max() < 1e-9
    assert by_date["m_mom"].abs().max() < 1e-9


def test_target_is_excess_zero_mean(synthetic_panel):
    feats = ["return_5d"]
    matrix = build_feature_matrix(synthetic_panel, feature_names=feats, target_horizon=5)
    by_date = matrix.drop_nulls("y").group_by("date").agg(pl.col("y").mean().alias("m"))
    assert by_date["m"].abs().max() < 1e-9
