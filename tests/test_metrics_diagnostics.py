"""Tests for the diagnostic helpers in eval.metrics: shuffle_null_ic
and compute_ic_neutralized.

These functions are not part of the standard summary path; they exist for
significance testing (shuffle null) and factor-exposure isolation
(neutralization). Tests here exercise:

  shuffle_null_ic:
    - returns roughly zero-mean null distribution on uncorrelated random data
    - returns large positive null-z when prediction = realized exactly
    - handles empty / under-populated frames gracefully

  compute_ic_neutralized:
    - returns ic_gross == ic_neutralized when neutralize_cols is empty
    - reduces ic_gross when the predictor is purely a linear function of the
      neutralization column (the signal IS the factor exposure)
    - preserves ic_gross when prediction is independent of the neutralization
      column (no exposure to remove)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from price_model.eval.metrics import (
    ShuffleNullResult,
    compute_ic_neutralized,
    deflated_sharpe_ratio,
    shuffle_null_ic,
)


def _random_frame(n_dates: int = 40, n_tickers: int = 50, seed: int = 1) -> pl.DataFrame:
    """A (date, ticker, prediction, realized) frame with uncorrelated random
    prediction and realized columns — the canonical null setup."""
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    rows = []
    for i in range(n_dates):
        d = start + timedelta(days=i)
        for t in range(n_tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{t:03d}",
                    "prediction": float(rng.normal()),
                    "realized": float(rng.normal()),
                }
            )
    return pl.DataFrame(rows)


# -----------------------------------------------------------------------------
# shuffle_null_ic
# -----------------------------------------------------------------------------


def test_shuffle_null_ic_returns_shuffle_null_result():
    df = _random_frame()
    result = shuffle_null_ic(df, n_iterations=50, seed=7)
    assert isinstance(result, ShuffleNullResult)
    assert result.n_iterations == 50
    assert result.null_mean_ics.shape == (50,)


def test_shuffle_null_ic_mean_near_zero_under_random_data():
    """On uncorrelated random data, the null distribution mean should be ~ 0
    (the no-information point) within statistical tolerance."""
    df = _random_frame(n_dates=60, n_tickers=80, seed=11)
    result = shuffle_null_ic(df, n_iterations=200, seed=11)
    # The null mean should be within ~3 standard errors of zero.
    se = result.null_std / np.sqrt(result.n_iterations)
    assert abs(result.null_mean) < 5 * se, (
        f"null mean {result.null_mean:.4f} far from 0 (se={se:.4f})"
    )


def test_shuffle_null_ic_high_z_when_prediction_equals_realized():
    """If predictions perfectly match realized, observed mean IC = 1.0, far
    above any shuffled value → p_value ~ 0, z_score very large."""
    rng = np.random.default_rng(3)
    start = date(2024, 1, 1)
    rows = []
    for i in range(30):
        d = start + timedelta(days=i)
        for t in range(40):
            v = float(rng.normal())
            rows.append({"date": d, "ticker": f"T{t}", "prediction": v, "realized": v})
    df = pl.DataFrame(rows)
    result = shuffle_null_ic(df, n_iterations=100, seed=3)
    assert result.observed_ic > 0.95, "perfect alignment should produce IC near 1"
    assert result.p_value_one_sided <= 0.01, "perfect alignment should be p ≤ 0.01"
    assert result.z_score() > 10, f"z_score = {result.z_score():.2f}, expected >> 10"


def test_shuffle_null_ic_handles_empty_frame():
    """Empty frame returns a NaN-bearing result without raising."""
    df = pl.DataFrame(
        schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "prediction": pl.Float64,
            "realized": pl.Float64,
        }
    )
    result = shuffle_null_ic(df, n_iterations=10)
    assert result.n_iterations == 0
    assert np.isnan(result.observed_ic)


# -----------------------------------------------------------------------------
# compute_ic_neutralized
# -----------------------------------------------------------------------------


def test_compute_ic_neutralized_empty_cols_matches_gross():
    """With no neutralization columns, neutralized IC must equal gross IC."""
    df = _random_frame(n_dates=30, n_tickers=40, seed=5)
    result = compute_ic_neutralized(df, neutralize_cols=[])
    assert result["ic_gross"] == result["ic_neutralized"]
    assert result["delta"] == 0.0


def test_compute_ic_neutralized_kills_signal_when_prediction_is_the_factor():
    """If prediction is a pure linear function of the neutralization column
    (so prediction's only information IS the factor exposure), residualizing
    should leave near-zero residual prediction → near-zero neutralized IC.

    Construct: realized = factor + noise; prediction = factor (exactly).
    Gross IC is high (prediction perfectly captures the factor signal in
    realized). Neutralized IC against the same factor should drop to ~0.
    """
    rng = np.random.default_rng(13)
    start = date(2024, 1, 1)
    n_dates, n_tickers = 40, 60
    rows = []
    for i in range(n_dates):
        d = start + timedelta(days=i)
        # Per-date factor exposures and idiosyncratic noise.
        factor_vals = rng.normal(0, 1.0, n_tickers)
        noise = rng.normal(0, 0.5, n_tickers)
        for t in range(n_tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{t}",
                    "prediction": float(factor_vals[t]),     # prediction = factor
                    "realized": float(factor_vals[t] + noise[t]),  # realized = factor + noise
                    "factor": float(factor_vals[t]),
                }
            )
    df = pl.DataFrame(rows)

    result = compute_ic_neutralized(df, neutralize_cols=["factor"])
    # Gross IC should be high — prediction captures the factor that drives realized.
    assert result["ic_gross"] > 0.5, f"gross IC = {result['ic_gross']:.3f}, expected > 0.5"
    # Neutralized: residualizing prediction against the factor should leave ~0,
    # so neutralized IC collapses toward zero.
    assert abs(result["ic_neutralized"]) < 0.1, (
        f"neutralized IC = {result['ic_neutralized']:.3f}, expected near 0"
    )
    # Delta is positive (gross > neutralized; the factor exposure was real).
    assert result["delta"] > 0.4


def test_compute_ic_neutralized_preserves_signal_when_independent_of_factor():
    """If prediction is independent of the neutralization column, neutralized
    IC should be approximately equal to gross IC."""
    rng = np.random.default_rng(17)
    start = date(2024, 1, 1)
    n_dates, n_tickers = 40, 60
    rows = []
    for i in range(n_dates):
        d = start + timedelta(days=i)
        signal = rng.normal(0, 1.0, n_tickers)
        independent_factor = rng.normal(0, 1.0, n_tickers)
        noise = rng.normal(0, 0.3, n_tickers)
        for t in range(n_tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{t}",
                    "prediction": float(signal[t]),
                    "realized": float(signal[t] + noise[t]),
                    "factor": float(independent_factor[t]),
                }
            )
    df = pl.DataFrame(rows)

    result = compute_ic_neutralized(df, neutralize_cols=["factor"])
    assert result["ic_gross"] > 0.5
    # Independent factor → neutralization doesn't change IC much.
    assert abs(result["ic_gross"] - result["ic_neutralized"]) < 0.1, (
        f"gross={result['ic_gross']:.3f}, neutralized={result['ic_neutralized']:.3f}, "
        f"delta={result['delta']:.3f} — expected near zero"
    )


# -----------------------------------------------------------------------------
# deflated_sharpe_ratio (Bailey-López de Prado 2014)
# -----------------------------------------------------------------------------


def test_deflated_sharpe_more_trials_lowers_probability():
    """With more trials attempted, the same observed Sharpe should produce a
    LOWER probability that the true Sharpe is positive (more multiple-test
    deflation). Use a borderline Sharpe (0.3) so neither case saturates at p=1.
    """
    r_few = deflated_sharpe_ratio(
        sharpe_obs=0.3, n_trials=5, n_periods=100, skew=0.0, kurt=3.0
    )
    r_many = deflated_sharpe_ratio(
        sharpe_obs=0.3, n_trials=500, n_periods=100, skew=0.0, kurt=3.0
    )
    assert r_few["probability_true_sharpe_positive"] > r_many["probability_true_sharpe_positive"]
    # More trials → higher expected max under null
    assert r_many["expected_max_sharpe_under_null"] > r_few["expected_max_sharpe_under_null"]


def test_deflated_sharpe_long_history_raises_probability():
    """With more periods of observation, the same Sharpe / trial count should
    produce a HIGHER probability that the true Sharpe is positive. Borderline
    Sharpe (0.2) so the test doesn't saturate.
    """
    r_short = deflated_sharpe_ratio(
        sharpe_obs=0.2, n_trials=200, n_periods=50, skew=0.0, kurt=3.0
    )
    r_long = deflated_sharpe_ratio(
        sharpe_obs=0.2, n_trials=200, n_periods=500, skew=0.0, kurt=3.0
    )
    assert r_long["probability_true_sharpe_positive"] > r_short["probability_true_sharpe_positive"]


def test_deflated_sharpe_handles_degenerate_input():
    """n_periods < 2 should return NaN cleanly without raising."""
    r = deflated_sharpe_ratio(
        sharpe_obs=1.0, n_trials=10, n_periods=1, skew=0.0, kurt=3.0
    )
    assert np.isnan(r["probability_true_sharpe_positive"])
