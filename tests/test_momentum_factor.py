"""Tests for the MomentumFactor baseline model class.

MomentumFactor is the literature-standard "rank by a single documented
factor" baseline for cross-sectional ML evaluation. It returns the named
feature's value as the prediction — no fit, no hyperparameters. The
walk-forward harness then ranks tickers by the prediction and computes IC
the same way it does for any other model.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from price_model.models import MODEL_REGISTRY, MomentumFactor, build_model
from price_model.models.base import ModelConfig


def _config(feature_cols=("momentum_504",), feature_name="momentum_504"):
    return ModelConfig(
        model_id="mom_factor",
        feature_cols=tuple(feature_cols),
        target_col="y",
        params={"feature_name": feature_name},
    )


def test_registered_in_model_registry():
    assert "MomentumFactor" in MODEL_REGISTRY
    assert MODEL_REGISTRY["MomentumFactor"] is MomentumFactor


def test_build_model_returns_momentum_factor_instance():
    m = build_model("MomentumFactor", _config())
    assert isinstance(m, MomentumFactor)


def test_predict_returns_feature_value_unchanged():
    m = MomentumFactor(_config())
    m.fit(pl.DataFrame({"momentum_504": [0.1, 0.2, 0.3], "y": [0.0, 0.0, 0.0]}))
    panel = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "ticker": ["AAA", "BBB", "AAA"],
            "momentum_504": [0.5, -0.3, 0.8],
        }
    ).with_columns(pl.col("date").str.to_date())
    preds = m.predict(panel)
    assert preds["prediction"].to_list() == [0.5, -0.3, 0.8]


def test_fit_raises_when_feature_name_missing_from_params():
    """Without feature_name in params, fit() must raise."""
    cfg = ModelConfig(model_id="x", feature_cols=("momentum_504",), target_col="y", params={})
    m = MomentumFactor(cfg)
    with pytest.raises(ValueError, match="feature_name"):
        m.fit(pl.DataFrame({"momentum_504": [0.1]}))


def test_fit_raises_when_feature_name_not_in_feature_cols():
    """feature_name must appear in feature_cols so the harness pipes it through."""
    cfg = ModelConfig(
        model_id="x",
        feature_cols=("momentum_60",),  # different from params.feature_name
        target_col="y",
        params={"feature_name": "momentum_504"},
    )
    m = MomentumFactor(cfg)
    with pytest.raises(ValueError, match="feature_name"):
        m.fit(pl.DataFrame({"momentum_60": [0.1]}))


def test_predict_before_fit_raises():
    m = MomentumFactor(_config())
    with pytest.raises(RuntimeError):
        m.predict(pl.DataFrame({"momentum_504": [0.1], "date": ["2024-01-01"], "ticker": ["AAA"]}))


def test_predict_preserves_nan_in_feature():
    """If the feature is null (warmup window), the prediction is null too."""
    m = MomentumFactor(_config())
    m.fit(pl.DataFrame({"momentum_504": [0.1], "y": [0.0]}))
    panel = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["AAA", "BBB"],
            "momentum_504": [None, 0.2],
        }
    ).with_columns(pl.col("date").str.to_date())
    preds = m.predict(panel)
    vals = preds["prediction"].to_list()
    # First row's prediction is NaN/None propagated from the feature
    assert vals[0] is None or (isinstance(vals[0], float) and np.isnan(vals[0]))
    assert vals[1] == 0.2


def test_works_with_any_named_feature():
    """The class is parametric — name any feature; it returns that one."""
    cfg = ModelConfig(
        model_id="mom_378",
        feature_cols=("momentum_378",),
        target_col="y",
        params={"feature_name": "momentum_378"},
    )
    m = MomentumFactor(cfg)
    m.fit(pl.DataFrame({"momentum_378": [0.1], "y": [0.0]}))
    panel = pl.DataFrame(
        {
            "date": ["2024-01-01"],
            "ticker": ["AAA"],
            "momentum_378": [0.42],
        }
    ).with_columns(pl.col("date").str.to_date())
    assert m.predict(panel)["prediction"].to_list() == [0.42]


def test_negative_sign_negates_predictions():
    """With sign=-1, predictions are the negation of the feature value — the
    Lehmann short-term reversal convention applied to return_1d / return_5d."""
    cfg = ModelConfig(
        model_id="reversal_test",
        feature_cols=("return_1d",),
        target_col="y",
        params={"feature_name": "return_1d", "sign": -1},
    )
    m = MomentumFactor(cfg)
    m.fit(pl.DataFrame({"return_1d": [0.01, -0.02], "y": [0.0, 0.0]}))
    panel = pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "ticker": ["AAA", "AAA"],
            "return_1d": [0.05, -0.03],
        }
    ).with_columns(pl.col("date").str.to_date())
    preds = m.predict(panel)["prediction"].to_list()
    # sign=-1 → preds = -feature
    assert preds == [-0.05, 0.03]


def test_positive_sign_default_matches_no_sign_param():
    """The default sign (+1) must produce identical predictions to a fit with
    no sign param at all — backwards compatibility guarantee."""
    cfg_no_sign = ModelConfig(
        model_id="x",
        feature_cols=("return_1d",),
        target_col="y",
        params={"feature_name": "return_1d"},
    )
    cfg_explicit = ModelConfig(
        model_id="x",
        feature_cols=("return_1d",),
        target_col="y",
        params={"feature_name": "return_1d", "sign": 1},
    )
    m1 = MomentumFactor(cfg_no_sign)
    m2 = MomentumFactor(cfg_explicit)
    m1.fit(pl.DataFrame({"return_1d": [0.0], "y": [0.0]}))
    m2.fit(pl.DataFrame({"return_1d": [0.0], "y": [0.0]}))
    panel = pl.DataFrame(
        {
            "date": ["2024-01-01"],
            "ticker": ["AAA"],
            "return_1d": [0.07],
        }
    ).with_columns(pl.col("date").str.to_date())
    assert m1.predict(panel)["prediction"].to_list() == m2.predict(panel)["prediction"].to_list()


def test_invalid_sign_raises():
    """Any sign other than ±1 must raise at fit time."""
    cfg = ModelConfig(
        model_id="x",
        feature_cols=("return_1d",),
        target_col="y",
        params={"feature_name": "return_1d", "sign": 2},
    )
    m = MomentumFactor(cfg)
    with pytest.raises(ValueError, match="sign"):
        m.fit(pl.DataFrame({"return_1d": [0.0], "y": [0.0]}))
