"""Truncation-invariance leakage test.

For every registered feature: compute it on the full panel, then on a panel truncated
at date T. The feature value for date T must be identical between the two — otherwise
the feature is using data from after T (look-ahead bias).

This is the single most important test in the project. If a new feature breaks it,
that feature MUST NOT ship until fixed.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_model.features.cross_features
import price_model.features.technical  # noqa: F401  trigger registration
from price_model.features.base import FEATURE_REGISTRY


@pytest.mark.parametrize("feature_name", sorted(FEATURE_REGISTRY))
def test_feature_is_truncation_invariant(feature_name, synthetic_panel):
    """For each date T in the test window, the feature value computed on the
    full panel must equal the value computed when the panel is truncated at T."""
    feature = FEATURE_REGISTRY[feature_name]

    # Compute on the full panel
    full = feature.compute(synthetic_panel.sort(["ticker", "date"]))

    # Pick a few interior dates to truncate at — avoid the warmup region
    all_dates = sorted(synthetic_panel["date"].unique().to_list())
    sample_dates = all_dates[
        max(feature.lookback_days + 10, 50) :: 100  # every ~100 days
    ][:5]
    assert sample_dates, "Need at least one test date past warmup"

    for cutoff in sample_dates:
        truncated_panel = synthetic_panel.filter(pl.col("date") <= cutoff)
        truncated = feature.compute(truncated_panel.sort(["ticker", "date"]))

        for ticker in synthetic_panel["ticker"].unique():
            full_val = full.filter((pl.col("date") == cutoff) & (pl.col("ticker") == ticker))[
                feature_name
            ]
            trunc_val = truncated.filter((pl.col("date") == cutoff) & (pl.col("ticker") == ticker))[
                feature_name
            ]
            assert full_val.len() == 1 and trunc_val.len() == 1, (
                f"Missing row for {ticker} on {cutoff}"
            )
            a, b = full_val[0], trunc_val[0]
            if a is None and b is None:
                continue
            assert a is not None and b is not None, (
                f"{feature_name}: null mismatch at {cutoff}/{ticker} (full={a}, trunc={b})"
            )
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (
                f"{feature_name} leaks: at {cutoff}/{ticker} full={a} truncated={b}"
            )
