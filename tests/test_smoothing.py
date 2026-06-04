"""Tests for the per-ticker EWM signal smoothing helper."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from price_model.eval.smoothing import smooth_predictions
from price_model.eval.turnover import compute_turnover_and_costs


def _make_panel(
    n_dates: int = 50,
    n_tickers: int = 30,
    pred_volatility: float = 1.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Synthetic (date, ticker, prediction, realized) panel.

    Predictions vary across both dates and tickers; realized correlates
    weakly with prediction. Per-ticker prediction has its own random walk
    plus daily noise so that smoothing has something to smooth.
    """
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    rows = []
    for tk in tickers:
        # Ticker-specific drift + daily i.i.d. shock; smoothing should
        # collapse the daily shocks while preserving the drift.
        trend = rng.normal(0.0, 0.3, 1)[0]
        for d in dates:
            raw_pred = trend + rng.normal(0.0, pred_volatility, 1)[0]
            realized = trend * 0.01 + rng.normal(0.0, 0.02, 1)[0]
            rows.append(
                {
                    "date": d,
                    "ticker": tk,
                    "prediction": float(raw_pred),
                    "realized": float(realized),
                }
            )
    return pl.DataFrame(rows)


def test_smoothing_preserves_shape_and_columns():
    df = _make_panel(n_dates=30, n_tickers=20, seed=1)
    out = smooth_predictions(df, halflife_days=5)
    assert out.height == df.height
    assert set(out.columns) == set(df.columns)


def test_smoothing_reduces_cross_sectional_dispersion_of_prediction():
    """An EWM of i.i.d. noisy predictions should have smaller daily std."""
    df = _make_panel(n_dates=60, n_tickers=40, pred_volatility=1.0, seed=2)
    raw_std = float(df["prediction"].std())
    out = smooth_predictions(df, halflife_days=10)
    smoothed_std = float(out["prediction"].std())
    assert smoothed_std < raw_std


def test_smoothing_reduces_turnover_when_predictions_are_noisy():
    """The diagnostic-grade claim: smoothing collapses turnover on a noisy signal."""
    df = _make_panel(n_dates=80, n_tickers=40, pred_volatility=1.0, seed=3)
    raw_metrics = compute_turnover_and_costs(df, cost_bps=(5,))
    smoothed = smooth_predictions(df, halflife_days=21)
    smoothed_metrics = compute_turnover_and_costs(smoothed, cost_bps=(5,))
    assert smoothed_metrics.daily_turnover_mean < raw_metrics.daily_turnover_mean


def test_smoothing_is_causal_no_future_leakage():
    """Smoothed value at date t depends only on data ≤ t for that ticker.

    Verified by manipulating only the LAST date for one ticker and confirming
    that all earlier smoothed values for that ticker (and all other tickers)
    are unchanged.
    """
    df = _make_panel(n_dates=15, n_tickers=10, seed=4)
    out_orig = smooth_predictions(df, halflife_days=5).sort(["ticker", "date"])

    # Mutate only the LAST date for ticker T000
    last_date = sorted(df["date"].unique().to_list())[-1]
    perturbed = df.with_columns(
        pl.when((pl.col("ticker") == "T000") & (pl.col("date") == last_date))
        .then(pl.col("prediction") + 99.0)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )
    out_perturbed = smooth_predictions(perturbed, halflife_days=5).sort(["ticker", "date"])

    # All earlier rows for T000 must match exactly
    earlier_orig = out_orig.filter((pl.col("ticker") == "T000") & (pl.col("date") < last_date))[
        "prediction"
    ].to_list()
    earlier_perturbed = out_perturbed.filter(
        (pl.col("ticker") == "T000") & (pl.col("date") < last_date)
    )["prediction"].to_list()
    assert earlier_orig == earlier_perturbed

    # All rows for other tickers must match exactly
    other_orig = out_orig.filter(pl.col("ticker") != "T000")["prediction"].to_list()
    other_perturbed = out_perturbed.filter(pl.col("ticker") != "T000")["prediction"].to_list()
    assert other_orig == other_perturbed


def test_smoothing_per_ticker_isolation():
    """Perturbing AAA's predictions must not affect BBB's smoothed predictions."""
    df = _make_panel(n_dates=20, n_tickers=10, seed=5)
    out_orig = smooth_predictions(df, halflife_days=5)
    perturbed = df.with_columns(
        pl.when(pl.col("ticker") == "T000")
        .then(pl.col("prediction") * 100.0)
        .otherwise(pl.col("prediction"))
        .alias("prediction")
    )
    out_perturbed = smooth_predictions(perturbed, halflife_days=5)

    bbb_orig = out_orig.filter(pl.col("ticker") == "T001").sort("date")["prediction"].to_list()
    bbb_perturbed = (
        out_perturbed.filter(pl.col("ticker") == "T001").sort("date")["prediction"].to_list()
    )
    assert bbb_orig == bbb_perturbed


def test_smoothing_rejects_invalid_halflife():
    df = _make_panel(n_dates=10, n_tickers=5, seed=6)
    with pytest.raises(ValueError, match="halflife_days must be positive"):
        smooth_predictions(df, halflife_days=0)
    with pytest.raises(ValueError, match="halflife_days must be positive"):
        smooth_predictions(df, halflife_days=-3)


def test_smoothing_empty_input_returns_empty():
    df = pl.DataFrame(
        schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "prediction": pl.Float64,
            "realized": pl.Float64,
        }
    )
    out = smooth_predictions(df, halflife_days=5)
    assert out.height == 0


def test_smoothing_missing_required_columns_raises():
    df = pl.DataFrame({"date": [date(2024, 1, 1)], "prediction": [0.5]})
    with pytest.raises(ValueError, match="missing"):
        smooth_predictions(df, halflife_days=5)


def test_first_observation_per_ticker_equals_raw():
    """EWM with adjust=False starts at the first observation, so the smoothed
    value at the earliest date for each ticker equals the raw prediction."""
    df = _make_panel(n_dates=30, n_tickers=10, seed=7).sort(["ticker", "date"])
    out = smooth_predictions(df, halflife_days=10).sort(["ticker", "date"])
    for tk in df["ticker"].unique():
        first_raw = (
            df.filter(pl.col("ticker") == tk).sort("date").head(1)["prediction"].to_list()[0]
        )
        first_smoothed = (
            out.filter(pl.col("ticker") == tk).sort("date").head(1)["prediction"].to_list()[0]
        )
        assert abs(first_raw - first_smoothed) < 1e-12
