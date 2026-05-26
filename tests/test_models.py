"""Smoke tests for the Model layer: fit, predict, save, load round-trip."""

from __future__ import annotations

from pathlib import Path

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


def test_lightgbm_multiseed_averages_predictions(synthetic_panel, tmp_path: Path):
    """Multi-seed: train 3 boosters, verify save/load preserves all of them
    and predictions are the average across seeds."""
    import numpy as np

    from price_model.models.boosting import LightGBMModel

    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="lgbm_multiseed",
        feature_cols=tuple(FEATS),
        params={
            "n_estimators": 15,
            "num_leaves": 7,
            "min_data_in_leaf": 5,
            "seeds": [1, 2, 3],
        },
    )
    model = build_model("LightGBMModel", cfg)
    model.fit(m)
    assert len(model._boosters) == 3

    preds = model.predict(m.head(20))
    # Compute the per-seed average manually and verify it matches
    X = m.head(20).select(FEATS).fill_null(0.0).to_numpy()
    expected = np.mean([b.predict(X) for b in model._boosters], axis=0)
    assert (preds["prediction"].to_numpy() - expected).max() < 1e-12

    # Save/load preserves all boosters
    model.save(tmp_path / "lgbm_multiseed")
    loaded = LightGBMModel.load(tmp_path / "lgbm_multiseed")
    assert len(loaded._boosters) == 3
    preds_loaded = loaded.predict(m.head(20))
    assert (preds["prediction"] - preds_loaded["prediction"]).abs().max() < 1e-9


def test_lightgbm_early_stopping(synthetic_panel):
    """With val_fraction + early_stopping_rounds, fit completes cleanly and
    produces a usable booster (best_iteration < n_estimators in practice)."""
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="lgbm_es",
        feature_cols=tuple(FEATS),
        params={
            "n_estimators": 200,
            "num_leaves": 15,
            "min_data_in_leaf": 10,
            "val_fraction": 0.15,
            "early_stopping_rounds": 20,
        },
    )
    model = build_model("LightGBMModel", cfg)
    model.fit(m)
    preds = model.predict(m.head(30))
    assert preds.height == 30
    assert preds["prediction"].is_not_null().all()
