"""Chronos model tests.

Two layers:
- Always-run: registry registration + clean ImportError when the optional deps
  aren't installed. These run on every CI build.
- Skip-if-missing: a tiny real inference test that runs only when `chronos-forecasting`
  and `torch` are importable. Useful on a dev machine where the extras are installed.
"""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta

import polars as pl
import pytest

from price_model.models import MODEL_REGISTRY, build_model
from price_model.models.base import ModelConfig
from price_model.models.foundation import ChronosZeroShot

CHRONOS_AVAILABLE = (
    importlib.util.find_spec("chronos") is not None
    and importlib.util.find_spec("torch") is not None
)


def test_chronos_is_registered():
    assert "ChronosZeroShot" in MODEL_REGISTRY
    assert MODEL_REGISTRY["ChronosZeroShot"] is ChronosZeroShot


def test_chronos_fit_does_not_require_torch():
    """fit() only captures history; should work even without the optional deps."""
    cfg = ModelConfig(model_id="chronos_test", feature_cols=())
    model = build_model("ChronosZeroShot", cfg)
    panel = pl.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "ticker": ["AAPL", "AAPL"],
            "adj_close": [100.0, 101.0],
        }
    )
    model.fit(panel)  # must not raise
    assert model._fitted


def test_chronos_predict_raises_clean_import_error_when_missing():
    """If chronos/torch are not installed, predict() must fail with a helpful
    message — not a cryptic ModuleNotFoundError from somewhere deep in the import."""
    if CHRONOS_AVAILABLE:
        pytest.skip("chronos installed — can't test the missing-dep path here")
    cfg = ModelConfig(model_id="chronos_test", feature_cols=())
    model = build_model("ChronosZeroShot", cfg)
    panel = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)],
            "ticker": ["AAPL"],
            "adj_close": [100.0],
        }
    )
    model.fit(panel)
    with pytest.raises(ImportError, match="optional dependency"):
        model.predict(panel)


@pytest.mark.skipif(not CHRONOS_AVAILABLE, reason="chronos-forecasting not installed")
def test_chronos_round_trip_small_real():
    """End-to-end smoke: tiny model, 2 tickers, ~60 days history."""
    cfg = ModelConfig(
        model_id="chronos_smoke",
        feature_cols=(),
        params={
            "model_name": "amazon/chronos-t5-tiny",
            "num_samples": 5,
            "context_length": 60,
            "prediction_length": 5,
            "device": "cpu",
        },
    )
    model = build_model("ChronosZeroShot", cfg)
    start = date(2024, 1, 1)
    rows = []
    for t, base in [("AAA", 100.0), ("BBB", 50.0)]:
        for i in range(80):
            rows.append(
                {
                    "date": start + timedelta(days=i),
                    "ticker": t,
                    "adj_close": base * (1 + 0.001 * i),
                }
            )
    panel = pl.DataFrame(rows)
    model.fit(panel.head(140))  # train on first 70 days per ticker
    preds = model.predict(panel.tail(20))
    assert preds.height >= 1
    assert "prediction" in preds.columns
    assert "pred_lower" in preds.columns
    assert "pred_upper" in preds.columns
