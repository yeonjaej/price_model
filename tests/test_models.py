"""Smoke tests for the Model layer: fit, predict, save, load round-trip."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig


FEATS = ["return_5d", "momentum_60", "vol_20", "rsi_14", "distance_ma_200"]


def _matrix(synthetic_panel):
    m = build_feature_matrix(synthetic_panel, feature_names=FEATS, target_horizon=5)
    return drop_warmup_rows(m, FEATS).drop_nulls("y")


def test_zero_predictor(synthetic_panel):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(model_id="zero_test", feature_cols=tuple(FEATS))
    model = build_model("ZeroPredictor", cfg)
    model.fit(m)
    preds = model.predict(m.head(50))
    assert preds.height == 50
    assert (preds["prediction"] == 0.0).all()


def test_last_return_predictor(synthetic_panel):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(model_id="lr_test", feature_cols=tuple(FEATS))
    model = build_model("LastReturnPredictor", cfg)
    model.fit(m)
    preds = model.predict(m.head(50))
    assert preds.height == 50


def test_lightgbm_round_trip(synthetic_panel, tmp_path: Path):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="lgbm_test",
        feature_cols=tuple(FEATS),
        params={"n_estimators": 20, "num_leaves": 7, "min_data_in_leaf": 5},
    )
    model = build_model("LightGBMModel", cfg)
    model.fit(m)
    preds = model.predict(m.head(50))
    assert preds.height == 50
    assert preds["prediction"].is_not_null().all()

    model.save(tmp_path / "lgbm_test")
    from price_model.models.boosting import LightGBMModel
    loaded = LightGBMModel.load(tmp_path / "lgbm_test")
    preds2 = loaded.predict(m.head(50))
    # Round-trip predictions should be identical
    diffs = (preds["prediction"] - preds2["prediction"]).abs()
    assert diffs.max() < 1e-9
