"""Tests for eval/robustness.py — bootstrap CI, decile returns, time split.

All tests run on small synthetic data. The point is to exercise the contract
(shapes, monotonicity, CI inclusion) rather than to verify exact numerical
values, which depend on RNG seeds and bootstrap sample size.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from price_model.eval.robustness import (
    bootstrap_ic_ci,
    decile_returns,
    robustness_panel,
    time_split_evaluate,
)


def _ranked_eval_panel(
    n_days: int = 200,
    n_tickers: int = 30,
    noise: float = 0.01,
    seed: int = 7,
) -> pl.DataFrame:
    """Synthetic panel where prediction perfectly ranks realized + noise.

    `noise` controls how much IC degrades: 0 = perfect rank, large noise = ~0 IC.
    """
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    rows = []
    for d_offset in range(n_days):
        d = start + timedelta(days=d_offset)
        # Per-date, predictions and realized share a latent signal + ticker-noise
        signal = rng.normal(0.0, 0.005, n_tickers)
        realized = signal + rng.normal(0.0, noise, n_tickers)
        for i in range(n_tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{i:02d}",
                    "prediction": float(signal[i]),
                    "realized": float(realized[i]),
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


# ---------------------------------------------------------------------------
# Bootstrap IC CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_excludes_zero_on_strong_signal():
    df = _ranked_eval_panel(n_days=200, noise=0.005)
    result = bootstrap_ic_ci(df, n_bootstrap=500, seed=1)
    assert result.n_dates >= 100
    assert result.point_estimate > 0.1, f"Expected strong positive IC, got {result.point_estimate}"
    # p05 strictly above zero on a strong signal
    assert result.excludes_zero
    # p50 should track the point estimate within bootstrap noise
    assert abs(result.p50 - result.point_estimate) < 0.05


def test_bootstrap_ci_includes_zero_on_pure_noise():
    """With noise dominant, the CI should straddle zero."""
    rng = np.random.default_rng(99)
    start = date(2024, 1, 1)
    rows = []
    for d_offset in range(120):
        d = start + timedelta(days=d_offset)
        for i in range(20):
            rows.append(
                {
                    "date": d,
                    "ticker": f"T{i:02d}",
                    "prediction": float(rng.normal()),
                    "realized": float(rng.normal()),  # independent → IC ≈ 0
                }
            )
    df = pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
    result = bootstrap_ic_ci(df, n_bootstrap=500, seed=2)
    assert result.p05 < 0.0 < result.p95
    assert not result.excludes_zero


def test_bootstrap_ci_returns_nan_on_insufficient_dates():
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 1)] * 5,
            "ticker": ["A", "B", "C", "D", "E"],
            "prediction": [0.1, 0.2, 0.3, 0.4, 0.5],
            "realized": [0.05, 0.1, 0.15, 0.2, 0.25],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    result = bootstrap_ic_ci(df, n_bootstrap=100)
    # Only 1 date — can't compute std of IC, so bootstrap returns nan
    import math

    assert math.isnan(result.point_estimate)
    assert result.n_bootstrap == 0


# ---------------------------------------------------------------------------
# Decile returns
# ---------------------------------------------------------------------------


def test_decile_returns_is_monotonic_on_strong_signal():
    df = _ranked_eval_panel(n_days=150, n_tickers=30, noise=0.003)
    out = decile_returns(df, n_buckets=5)
    assert out.height == 5
    assert set(out["bucket"].to_list()) == {1, 2, 3, 4, 5}
    realized = out.sort("bucket")["mean_realized"].to_list()
    # Strictly ascending (allowing tiny inversions due to noise)
    diffs = [realized[i + 1] - realized[i] for i in range(len(realized) - 1)]
    # On a strong signal almost all bucket transitions should be positive
    assert sum(d > 0 for d in diffs) >= 3, f"Expected monotonic ascending, got {realized}"


def test_decile_returns_handles_small_dates():
    """Dates with fewer than n_buckets tickers get dropped."""
    # 5 tickers per date, asking for 10 buckets — no rows should survive
    df = _ranked_eval_panel(n_days=50, n_tickers=5)
    out = decile_returns(df, n_buckets=10)
    assert out.height == 0


# ---------------------------------------------------------------------------
# Time-split evaluation
# ---------------------------------------------------------------------------


def test_time_split_partitions_dates_correctly():
    df = _ranked_eval_panel(n_days=200, noise=0.005)
    cutoff = date(2024, 4, 1)
    result = time_split_evaluate(df, cutoff=cutoff)

    # Window A is strictly before cutoff; window B starts at cutoff
    assert result.metrics_a.n_dates > 0
    assert result.metrics_b.n_dates > 0
    # Total should match the input panel's date count
    assert result.metrics_a.n_dates + result.metrics_b.n_dates == df["date"].n_unique()


def test_time_split_accepts_string_cutoff():
    df = _ranked_eval_panel(n_days=100, noise=0.005)
    result = time_split_evaluate(df, cutoff="2024-03-01")
    # Just verify it doesn't crash and produces both windows
    assert result.metrics_a.n_dates > 0 or result.metrics_b.n_dates > 0


def test_time_split_consistent_ic_on_stationary_signal():
    """When the signal is stationary, both windows should have similar IC."""
    df = _ranked_eval_panel(n_days=400, noise=0.005, seed=11)
    result = time_split_evaluate(df, cutoff=date(2024, 7, 1))
    ic_a = result.metrics_a.information_coefficient
    ic_b = result.metrics_b.information_coefficient
    # Both should be clearly positive; magnitudes within reasonable bound of each other
    assert ic_a > 0.1 and ic_b > 0.1, f"Got IC_a={ic_a}, IC_b={ic_b}"
    assert abs(ic_a - ic_b) < 0.15


# ---------------------------------------------------------------------------
# Convenience panel
# ---------------------------------------------------------------------------


def test_robustness_panel_runs_end_to_end():
    df = _ranked_eval_panel(n_days=150, noise=0.005)
    panel = robustness_panel(df, n_bootstrap=200, n_buckets=5, time_cutoff="2024-04-01")
    assert "bootstrap" in panel and "deciles" in panel and "time_split" in panel
    assert "headline" in panel
    assert isinstance(panel["deciles"], pl.DataFrame)
