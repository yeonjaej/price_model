"""Prediction store round-trip tests."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from price_model.serving.store import PredictionStore


def test_store_write_read(tmp_path: Path):
    store = PredictionStore(tmp_path / "preds.duckdb")
    df = pl.DataFrame({
        "date": [date(2024, 1, 5), date(2024, 1, 5)],
        "ticker": ["AAPL", "MSFT"],
        "prediction": [0.01, -0.005],
    })
    n = store.write(
        df,
        model_id="lightgbm_v1",
        experiment_id="test",
        horizon_days=5,
        generated_at=datetime(2024, 1, 5, 23, 0, 0),
    )
    assert n == 2

    read = store.query("SELECT * FROM predictions ORDER BY ticker")
    assert read.height == 2
    assert read["target_date"][0] == date(2024, 1, 10)
    assert "prediction_kind" in read.columns
    store.close()


def test_latest_predictions(tmp_path: Path):
    store = PredictionStore(tmp_path / "preds.duckdb")
    store.write(
        pl.DataFrame({"date": [date(2024, 1, 1)], "ticker": ["AAPL"], "prediction": [0.0]}),
        model_id="m1", experiment_id="e", horizon_days=5,
    )
    store.write(
        pl.DataFrame({"date": [date(2024, 2, 1)], "ticker": ["AAPL"], "prediction": [0.02]}),
        model_id="m1", experiment_id="e", horizon_days=5,
    )
    latest = store.latest_predictions(model_ids=["m1"])
    assert latest.height == 1
    assert latest["prediction_date"][0] == date(2024, 2, 1)
    store.close()
