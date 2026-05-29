"""Tests for the OHLCV/volume feature batch added in price_model.features.technical.

These features are the first ones in the project to consume `high`, `low`,
`open`, `volume`, and the close/adj_close ratio. The tests below pin down:

  - Each feature computes the right column with the right dtype.
  - Per-ticker isolation: a value for ticker A at date d depends only on A's
    history, not B's (the `.over("ticker")` discipline that protects against
    cross-stock leakage).
  - Warmup nulls match the declared `lookback_days` upper bound (so
    `drop_warmup_rows` correctly drops them).
  - Range-based and body-based features are non-negative as their math implies.
"""

from __future__ import annotations

import polars as pl
import pytest

import price_model.features.technical  # noqa: F401  trigger feature registration
from price_model.features.base import FEATURE_REGISTRY

NEW_FEATURES = [
    "range_ratio",
    "body_ratio",
    "parkinson_vol_20",
    "log_dollar_volume",
    "abnormal_volume",
    "max_return_21d",
]


@pytest.mark.parametrize("name", NEW_FEATURES)
def test_feature_registered(name: str):
    assert name in FEATURE_REGISTRY


@pytest.mark.parametrize("name", NEW_FEATURES)
def test_feature_produces_output_column(synthetic_panel, name: str):
    feat = FEATURE_REGISTRY[name]
    out = feat.compute(synthetic_panel)
    assert name in out.columns
    assert out[name].dtype == pl.Float64


def test_range_ratio_is_non_negative(synthetic_panel):
    out = FEATURE_REGISTRY["range_ratio"].compute(synthetic_panel)
    assert out["range_ratio"].drop_nulls().min() >= 0.0


def test_body_ratio_is_non_negative(synthetic_panel):
    out = FEATURE_REGISTRY["body_ratio"].compute(synthetic_panel)
    assert out["body_ratio"].drop_nulls().min() >= 0.0


def test_parkinson_vol_is_non_negative(synthetic_panel):
    out = FEATURE_REGISTRY["parkinson_vol_20"].compute(synthetic_panel)
    assert out["parkinson_vol_20"].drop_nulls().min() >= 0.0


def test_log_dollar_volume_is_finite(synthetic_panel):
    out = FEATURE_REGISTRY["log_dollar_volume"].compute(synthetic_panel)
    vals = out["log_dollar_volume"].drop_nulls().to_numpy()
    assert vals.size > 0
    # log of positive close*volume should be a finite real number
    import math

    for v in vals:
        assert math.isfinite(v)


def test_abnormal_volume_handles_split_adjusted_ratio(synthetic_panel):
    """abnormal_volume reads (volume, close, adj_close); make sure the
    close/adj_close ratio doesn't blow up when adj_close == close (no split)."""
    # synthetic_panel has close == adj_close, so split factor is 1.0 everywhere
    out = FEATURE_REGISTRY["abnormal_volume"].compute(synthetic_panel)
    vals = out["abnormal_volume"].drop_nulls().to_numpy()
    assert vals.size > 0
    # All values must be positive and finite
    import math

    assert all(v > 0 and math.isfinite(v) for v in vals)


def test_max_return_21d_dominates_recent_returns(synthetic_panel):
    """max_return_21d should equal max(daily_return) over the trailing 21 days."""
    out = FEATURE_REGISTRY["max_return_21d"].compute(synthetic_panel)
    # Pick one ticker, verify against a manual rolling-max of daily log returns
    one = (
        synthetic_panel.filter(pl.col("ticker") == "AAA")
        .sort("date")
        .with_columns(
            (pl.col("adj_close").log() - pl.col("adj_close").log().shift(1)).alias("_d"),
        )
    )
    manual = one["_d"].rolling_max(window_size=21).to_numpy()
    feature_vals = out.filter(pl.col("ticker") == "AAA").sort("date")["max_return_21d"].to_numpy()
    # Compare on the non-null tail
    import math

    n = len(manual)
    for i in range(n):
        if manual[i] is None or (isinstance(manual[i], float) and math.isnan(manual[i])):
            continue
        if math.isnan(feature_vals[i]):
            continue
        assert abs(feature_vals[i] - manual[i]) < 1e-12, (
            f"row {i}: {feature_vals[i]} vs {manual[i]}"
        )


@pytest.mark.parametrize(
    "name,expected_lookback",
    [
        ("range_ratio", 1),
        ("body_ratio", 1),
        ("parkinson_vol_20", 21),
        ("log_dollar_volume", 1),
        ("abnormal_volume", 21),
        ("max_return_21d", 22),
    ],
)
def test_warmup_row_count_matches_declared_lookback(
    synthetic_panel, name: str, expected_lookback: int
):
    """The first ~lookback_days rows per ticker should be null; afterward populated."""
    feat = FEATURE_REGISTRY[name]
    out = feat.compute(synthetic_panel).sort(["ticker", "date"])
    # For one ticker, check that the leading rows are null and later rows are not
    one = out.filter(pl.col("ticker") == "AAA").sort("date")[name].to_list()
    # The leading window of nulls should have at most expected_lookback rows.
    leading_nulls = 0
    for v in one:
        if v is None:
            leading_nulls += 1
        else:
            break
    assert leading_nulls <= expected_lookback, (
        f"{name}: {leading_nulls} leading nulls > declared lookback {expected_lookback}"
    )
    # And there should be SOME non-null value once we get past warmup
    assert any(v is not None for v in one)


def test_per_ticker_isolation(synthetic_panel):
    """Changing one ticker's prices must not change another ticker's range_ratio.

    Sanity check that .over("ticker") is doing its job — if the feature were
    inadvertently using cross-stock data, perturbing AAA would also move BBB.
    """
    feat = FEATURE_REGISTRY["range_ratio"]
    baseline = feat.compute(synthetic_panel).filter(pl.col("ticker") == "BBB")["range_ratio"]

    # Perturb AAA's high
    perturbed = synthetic_panel.with_columns(
        pl.when(pl.col("ticker") == "AAA")
        .then(pl.col("high") * 1.5)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    after = feat.compute(perturbed).filter(pl.col("ticker") == "BBB")["range_ratio"]

    # BBB's range_ratio must be identical
    assert baseline.to_list() == after.to_list()
