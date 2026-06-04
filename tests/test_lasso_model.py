"""Tests for the regularized linear cross-sectional models (Lasso + Ridge).

Coverage:
  Lasso:
    - registry: model class is registered under its name
    - fit/predict round-trip on the synthetic panel
    - save/load round-trip preserves coefficients and predictions
    - feature_importance() returns one entry per feature with finite floats
    - selected_alpha() exposes the CV-chosen regularization strength
    - L1 sparsity: at least one feature gets a zero coefficient under heavy
      regularization

  Ridge:
    - registry: model class is registered under its name
    - fit/predict round-trip on the synthetic panel
    - save/load round-trip preserves coefficients and predictions
    - feature_importance() returns one entry per feature with finite floats
    - L2 dense: NO coefficients are exactly zero (the defining contrast vs Lasso)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.models.linear import (
    ElasticNetCrossSectional,
    LassoCrossSectional,
    RidgeCrossSectional,
)

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


# -----------------------------------------------------------------------------
# Ridge tests (L2 regularization — should never zero coefficients exactly)
# -----------------------------------------------------------------------------


def test_ridge_registered_in_model_registry():
    """`RidgeCrossSectional` must be reachable via build_model()."""
    from price_model.models import MODEL_REGISTRY

    assert "RidgeCrossSectional" in MODEL_REGISTRY
    assert MODEL_REGISTRY["RidgeCrossSectional"] is RidgeCrossSectional


def test_ridge_fit_predict_round_trip(synthetic_panel):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="ridge_test",
        feature_cols=tuple(FEATS),
        params={"cv": 3},
    )
    model = build_model("RidgeCrossSectional", cfg)
    model.fit(m)

    preds = model.predict(m.head(50))
    assert preds.height == 50
    assert {"date", "ticker", "prediction"}.issubset(preds.columns)
    assert preds["prediction"].is_not_null().all()
    arr = preds["prediction"].to_numpy()
    assert np.isfinite(arr).all()


def test_ridge_save_load_preserves_predictions(synthetic_panel, tmp_path: Path):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="ridge_save_load",
        feature_cols=tuple(FEATS),
        params={"cv": 3},
    )
    model = build_model("RidgeCrossSectional", cfg)
    model.fit(m)
    preds_before = model.predict(m.head(50))["prediction"].to_numpy()

    save_dir = tmp_path / "ridge_save"
    model.save(save_dir)
    loaded = RidgeCrossSectional.load(save_dir)
    preds_after = loaded.predict(m.head(50))["prediction"].to_numpy()

    assert np.max(np.abs(preds_before - preds_after)) < 1e-12


def test_ridge_feature_importance_and_alpha(synthetic_panel):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="ridge_fi",
        feature_cols=tuple(FEATS),
        params={"cv": 3},
    )
    model = build_model("RidgeCrossSectional", cfg)
    model.fit(m)

    fi = model.feature_importance()
    assert set(fi.keys()) == set(FEATS)
    assert all(np.isfinite(v) for v in fi.values())

    alpha = model.selected_alpha()
    assert alpha is not None
    assert alpha > 0.0


def test_ridge_never_zeros_coefficients_under_heavy_regularization(synthetic_panel):
    """The defining contrast with Lasso: even at high alpha, Ridge shrinks all
    coefficients toward zero but keeps every one of them strictly non-zero.
    """
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="ridge_heavy",
        feature_cols=tuple(FEATS),
        # A single large alpha; L2 shrinks but never zeros.
        params={"alphas": [1000.0], "cv": 3},
    )
    model = build_model("RidgeCrossSectional", cfg)
    model.fit(m)
    fi = model.feature_importance()
    n_exact_zero = sum(1 for v in fi.values() if v == 0.0)
    # L2 should keep every coefficient strictly non-zero, even at alpha=1000.
    assert n_exact_zero == 0, f"expected no exact zeros under L2, got {fi}"


# -----------------------------------------------------------------------------
# ElasticNet tests (L1 + L2 hybrid — interpolates between Lasso and Ridge)
# -----------------------------------------------------------------------------


def test_elasticnet_registered_in_model_registry():
    """`ElasticNetCrossSectional` must be reachable via build_model()."""
    from price_model.models import MODEL_REGISTRY

    assert "ElasticNetCrossSectional" in MODEL_REGISTRY
    assert MODEL_REGISTRY["ElasticNetCrossSectional"] is ElasticNetCrossSectional


def test_elasticnet_fit_predict_round_trip(synthetic_panel):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="elasticnet_test",
        feature_cols=tuple(FEATS),
        params={"cv": 3, "max_iter": 2000},
    )
    model = build_model("ElasticNetCrossSectional", cfg)
    model.fit(m)

    preds = model.predict(m.head(50))
    assert preds.height == 50
    assert {"date", "ticker", "prediction"}.issubset(preds.columns)
    assert preds["prediction"].is_not_null().all()
    arr = preds["prediction"].to_numpy()
    assert np.isfinite(arr).all()


def test_elasticnet_save_load_preserves_predictions(synthetic_panel, tmp_path: Path):
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="elasticnet_save_load",
        feature_cols=tuple(FEATS),
        params={"cv": 3, "max_iter": 2000},
    )
    model = build_model("ElasticNetCrossSectional", cfg)
    model.fit(m)
    preds_before = model.predict(m.head(50))["prediction"].to_numpy()

    save_dir = tmp_path / "elasticnet_save"
    model.save(save_dir)
    loaded = ElasticNetCrossSectional.load(save_dir)
    preds_after = loaded.predict(m.head(50))["prediction"].to_numpy()

    assert np.max(np.abs(preds_before - preds_after)) < 1e-12


def test_elasticnet_feature_importance_alpha_l1_ratio(synthetic_panel):
    """ElasticNet exposes both selected_alpha() and selected_l1_ratio()."""
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="elasticnet_fi",
        feature_cols=tuple(FEATS),
        params={"cv": 3, "max_iter": 2000},
    )
    model = build_model("ElasticNetCrossSectional", cfg)
    model.fit(m)

    fi = model.feature_importance()
    assert set(fi.keys()) == set(FEATS)
    assert all(np.isfinite(v) for v in fi.values())

    alpha = model.selected_alpha()
    assert alpha is not None
    assert alpha > 0.0

    l1_ratio = model.selected_l1_ratio()
    assert l1_ratio is not None
    assert 0.0 <= l1_ratio <= 1.0


def test_elasticnet_pure_l1_matches_lasso_behavior_qualitatively(synthetic_panel):
    """With l1_ratios=[1.0] forced, ElasticNet should behave like Lasso.

    Concretely: under heavy regularization with l1_ratio=1.0, at least one
    coefficient should be exactly zero (Lasso sparsity property).
    """
    m = _matrix(synthetic_panel)
    cfg = ModelConfig(
        model_id="elasticnet_pure_l1",
        feature_cols=tuple(FEATS),
        params={
            "l1_ratios": [1.0],
            "alphas": [1.0],
            "cv": 3,
            "max_iter": 2000,
        },
    )
    model = build_model("ElasticNetCrossSectional", cfg)
    model.fit(m)

    assert model.selected_l1_ratio() == 1.0
    fi = model.feature_importance()
    n_zero = sum(1 for v in fi.values() if v == 0.0)
    assert n_zero >= 1, f"expected at least one zero under heavy L1, got {fi}"


def test_elasticnet_sparsity_decreases_as_l1_ratio_decreases(synthetic_panel):
    """L1-vs-L2 interpolation: more L1 weight should produce MORE zero coefficients.

    Why not "no zeros at l1_ratio≈0":
        sklearn's ElasticNet applies coordinate-descent soft-thresholding for ANY
        l1_ratio > 0. At l1_ratio=0.01 the L1 weight is small but non-zero, so
        small coefficients still get soft-thresholded to exactly zero. The only
        way to get strictly-no-zeros behavior is l1_ratio=0, which sklearn
        explicitly forbids (it recommends RidgeCV for that case).

    Instead this test verifies the meaningful relative property: at fixed alpha,
    sweeping l1_ratio from 1.0 (pure Lasso) down to 0.01 (near-Ridge) should
    produce a MONOTONE decrease in the zero-coefficient count. This is the
    practical "L2 mixed in → less sparsity" property that motivates using
    ElasticNet over pure Lasso on collinear panels.
    """
    m = _matrix(synthetic_panel)

    def _n_zeros_at(l1_ratio: float) -> int:
        cfg = ModelConfig(
            model_id=f"elasticnet_l1ratio_{l1_ratio}",
            feature_cols=tuple(FEATS),
            params={
                "l1_ratios": [l1_ratio],
                "alphas": [0.001],  # small fixed alpha - soft enough that not everything zeros
                "cv": 3,
                "max_iter": 5000,
            },
        )
        model = build_model("ElasticNetCrossSectional", cfg)
        model.fit(m)
        return sum(1 for v in model.feature_importance().values() if v == 0.0)

    n_zero_pure_lasso = _n_zeros_at(1.0)
    n_zero_near_ridge = _n_zeros_at(0.01)
    assert n_zero_pure_lasso >= n_zero_near_ridge, (
        f"Pure-Lasso ElasticNet (l1_ratio=1.0) should produce at least as many "
        f"zero coefficients as near-Ridge ElasticNet (l1_ratio=0.01). "
        f"Got pure-Lasso n_zero={n_zero_pure_lasso}, near-Ridge n_zero={n_zero_near_ridge}."
    )
