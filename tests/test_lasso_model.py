"""Tests for the regularized linear cross-sectional model (LassoCrossSectional).

Coverage:
  - registry: model class is registered under its name
  - fit/predict round-trip on the synthetic panel
  - save/load round-trip preserves coefficients and predictions
  - feature_importance() returns one entry per feature with finite floats
  - selected_alpha() exposes the CV-chosen regularization strength
  - L1 sparsity: at least one feature gets a zero coefficient under heavy
    regularization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.models.linear import LassoCrossSectional

# A small but signal-rich feature set — the same one used in test_models.py for
# the LightGBM round-trip tests. Lasso doesn't need a large grid to exercise
# its fit / predict / save / load path.
FEATS = ["return_5d", "momentum_60", "vol_20", "rsi_14", "distance_ma_200"]


def _matrix(panel):
    m = build_feature_matrix(panel, feature_names=FEATS, target_horizon=5)
    return drop_warmup_rows(m, FEATS).drop_nulls("y")


def test_lasso_registered_in_model_registry():
    """`LassoCrossSectional` must be reachable via build_model()."""
    from price_model.models import MODEL_REGISTRY

    assert "LassoCrossSectional" in MODEL_REGISTRY
    assert MODEL_REGISTRY["LassoCrossSectional"] is LassoCrossSectional


def test_lasso_fit_predict_round_trip(synthetic_panel):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="lasso_test",
        feature_cols=tuple(FEATS),
        params={"cv": 3, "max_iter": 2000},
    )
    model = build_model("LassoCrossSectional", cfg)
    model.fit(m)

    preds = model.predict(m.head(50))
    assert preds.height == 50
    assert {"date", "ticker", "prediction"}.issubset(preds.columns)
    assert preds["prediction"].is_not_null().all()
    # Predictions should be finite numbers
    arr = preds["prediction"].to_numpy()
    assert np.isfinite(arr).all()


def test_lasso_save_load_preserves_predictions(synthetic_panel, tmp_path: Path):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="lasso_save_load",
        feature_cols=tuple(FEATS),
        params={"cv": 3, "max_iter": 2000},
    )
    model = build_model("LassoCrossSectional", cfg)
    model.fit(m)
    preds_before = model.predict(m.head(50))["prediction"].to_numpy()

    save_dir = tmp_path / "lasso_save"
    model.save(save_dir)
    loaded = LassoCrossSectional.load(save_dir)
    preds_after = loaded.predict(m.head(50))["prediction"].to_numpy()

    # Round-trip predictions should be identical to numerical noise.
    assert np.max(np.abs(preds_before - preds_after)) < 1e-12


def test_lasso_feature_importance_shape_and_finite(synthetic_panel):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="lasso_fi",
        feature_cols=tuple(FEATS),
        params={"cv": 3, "max_iter": 2000},
    )
    model = build_model("LassoCrossSectional", cfg)
    model.fit(m)

    fi = model.feature_importance()
    assert set(fi.keys()) == set(FEATS)
    assert all(np.isfinite(v) for v in fi.values())


def test_lasso_selected_alpha_is_positive(synthetic_panel):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="lasso_alpha",
        feature_cols=tuple(FEATS),
        params={"cv": 3, "max_iter": 2000},
    )
    model = build_model("LassoCrossSectional", cfg)
    model.fit(m)

    alpha = model.selected_alpha()
    assert alpha is not None
    assert alpha > 0.0


def test_lasso_heavy_regularization_zeros_some_features(synthetic_panel):
    """A grid forcing a large alpha should drive some coefficients to exactly 0.

    L1 is supposed to do feature selection. If forcing the only allowed alpha to be
    large doesn't zero any of the 5 coefficients, the model isn't actually
    applying the penalty.
    """
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="lasso_heavy",
        feature_cols=tuple(FEATS),
        # Only one alpha value to force LassoCV to pick it. 1.0 is large relative to
        # cross-sectional excess returns (which are O(0.01)).
        params={"alphas": [1.0], "cv": 3, "max_iter": 2000},
    )
    model = build_model("LassoCrossSectional", cfg)
    model.fit(m)
    fi = model.feature_importance()
    n_zero = sum(1 for v in fi.values() if v == 0.0)
    assert n_zero >= 1, f"expected at least one zero-coefficient under heavy L1, got {fi}"
