"""Truncation-invariance leakage test.

For every registered feature: compute it on the full panel, then on a panel truncated
at date T. The feature value for date T must be identical between the two — otherwise
the feature is using data from after T (look-ahead bias).

This is the single most important test in the project. If a new feature breaks it,
that feature MUST NOT ship until fixed.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

import price_model.features.cross_features
import price_model.features.factor_loadings as _factor_loadings
import price_model.features.technical  # noqa: F401  trigger registration
from price_model.data.sources import fama_french
from price_model.features.base import FEATURE_REGISTRY


@pytest.fixture(autouse=True)
def _patch_kf_offline(monkeypatch, synthetic_panel):
    """Auto-patch the Ken French download so factor-loading features work offline.

    The synthetic_panel covers a known date range; we synthesize KF factors over
    the same range with the same RNG seed so the leakage test is deterministic.
    """
    dates = sorted(synthetic_panel["date"].unique().to_list())
    # Extend a few days past either end so the join is always fully covered.
    pad = 30
    if dates:
        first = dates[0] - timedelta(days=pad)
        last = dates[-1] + timedelta(days=pad)
        n = (last - first).days + 1
        kf_dates = [first + timedelta(days=i) for i in range(n)]
    else:  # pragma: no cover
        kf_dates = []
    rng = np.random.default_rng(seed=4242)
    fake = pl.DataFrame(
        {
            "date": kf_dates,
            "MKT_RF": rng.normal(0.0003, 0.011, len(kf_dates)),
            "SMB": rng.normal(0.0, 0.005, len(kf_dates)),
            "HML": rng.normal(0.0, 0.005, len(kf_dates)),
            "RMW": rng.normal(0.0, 0.004, len(kf_dates)),
            "CMA": rng.normal(0.0, 0.004, len(kf_dates)),
            "RF": np.full(len(kf_dates), 1e-4),
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    monkeypatch.setattr(fama_french, "fetch", lambda *a, **kw: fake)
    _factor_loadings._load_kf_factors.cache_clear()
    yield
    _factor_loadings._load_kf_factors.cache_clear()


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
