"""Tests for classical baseline models.

Three layers:
- Always-run: registry registration + clean ImportError when extras are missing.
- Skip-if-missing: end-to-end fit/predict tests against synthetic data, only when
  the relevant optional dep is importable.

GBM has no optional deps so its real fit/predict test runs unconditionally.
"""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from price_model.models import (
    MODEL_REGISTRY,
    ArimaPerTicker,
    GarchVolForecaster,
    GbmMaximumLikelihood,
    build_model,
)
from price_model.models.base import ModelConfig

STATSMODELS_AVAILABLE = importlib.util.find_spec("statsmodels") is not None
ARCH_AVAILABLE = importlib.util.find_spec("arch") is not None


def _two_ticker_panel(n_days: int = 400, seed: int = 7) -> pl.DataFrame:
    """Small two-ticker synthetic panel — enough history for fits to converge."""
    rng = np.random.default_rng(seed)
    start = date(2022, 1, 3)
    rows = []
    for ticker, drift in [("AAA", 0.0003), ("BBB", -0.0001)]:
        log_p = np.cumsum(rng.normal(drift, 0.015, size=n_days))
        prices = 100.0 * np.exp(log_p)
        for i, p in enumerate(prices):
            rows.append(
                {
                    "date": start + timedelta(days=i),
                    "ticker": ticker,
                    "adj_close": float(p),
                }
            )
    return pl.DataFrame(rows).sort(["ticker", "date"])


# --- Registry --------------------------------------------------------------


def test_classical_registered():
    for name in ("ArimaPerTicker", "GarchVolForecaster", "GbmMaximumLikelihood"):
        assert name in MODEL_REGISTRY


# --- ARIMA ----------------------------------------------------------------


def test_arima_fit_raises_clean_import_error_when_missing():
    if STATSMODELS_AVAILABLE:
        pytest.skip("statsmodels installed — can't exercise missing-dep path here")
    cfg = ModelConfig(model_id="arima_test", feature_cols=())
    model = build_model("ArimaPerTicker", cfg)
    with pytest.raises(ImportError, match="statsmodels"):
        model.fit(_two_ticker_panel())


@pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="statsmodels not installed")
def test_arima_round_trip(tmp_path):
    cfg = ModelConfig(
        model_id="arima_test",
        feature_cols=(),
        params={"order": (1, 0, 1), "min_history": 100, "horizon_days": 5},
    )
    model = build_model("ArimaPerTicker", cfg)
    panel = _two_ticker_panel(n_days=300)
    model.fit(panel)
    assert len(model._fits) >= 1

    preds = model.predict(panel.tail(100))
    assert preds.height >= 1
    assert "prediction" in preds.columns

    # Cross-sectional mean should be ~0 per date (excess-return convention)
    by_date_mean = preds.group_by("date").agg(pl.col("prediction").mean().alias("m"))
    assert by_date_mean["m"].abs().max() < 1e-9

    model.save(tmp_path / "arima_test")
    loaded = ArimaPerTicker.load(tmp_path / "arima_test")
    assert len(loaded._fits) == len(model._fits)


# --- GARCH ----------------------------------------------------------------


def test_garch_fit_raises_clean_import_error_when_missing():
    if ARCH_AVAILABLE:
        pytest.skip("arch installed — can't exercise missing-dep path here")
    cfg = ModelConfig(model_id="garch_test", feature_cols=())
    model = build_model("GarchVolForecaster", cfg)
    with pytest.raises(ImportError, match="arch"):
        model.fit(_two_ticker_panel())


@pytest.mark.skipif(not ARCH_AVAILABLE, reason="arch not installed")
def test_garch_round_trip(tmp_path):
    cfg = ModelConfig(
        model_id="garch_test",
        feature_cols=(),
        params={"min_history": 100, "horizon_days": 5},
    )
    model = build_model("GarchVolForecaster", cfg)
    panel = _two_ticker_panel(n_days=300)
    model.fit(panel)
    assert len(model._fits) >= 1

    preds = model.predict(panel.tail(100))
    assert preds.height >= 1
    # GARCH point prediction is by design 0; value is in the interval
    assert (preds["prediction"] == 0.0).all()
    assert (preds["pred_upper"] - preds["pred_lower"] > 0).all()

    model.save(tmp_path / "garch_test")
    loaded = GarchVolForecaster.load(tmp_path / "garch_test")
    assert len(loaded._fits) == len(model._fits)


# --- GBM (no optional deps, always runs) ---------------------------------


def test_gbm_round_trip(tmp_path):
    cfg = ModelConfig(
        model_id="gbm_test",
        feature_cols=(),
        params={"min_history": 100, "horizon_days": 5},
    )
    model = build_model("GbmMaximumLikelihood", cfg)
    panel = _two_ticker_panel(n_days=300)
    model.fit(panel)
    assert len(model._fits) == 2

    preds = model.predict(panel.tail(100))
    assert preds.height >= 1
    # Cross-sectional excess: mean prediction per date is ~0
    by_date_mean = preds.group_by("date").agg(pl.col("prediction").mean().alias("m"))
    assert by_date_mean["m"].abs().max() < 1e-9

    # Save + load preserves the (mu, sigma) pairs
    model.save(tmp_path / "gbm_test")
    loaded = GbmMaximumLikelihood.load(tmp_path / "gbm_test")
    assert loaded._fits == model._fits
