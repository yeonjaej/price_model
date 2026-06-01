"""Tests for the long-horizon momentum features (momentum_378, momentum_504, momentum_756).

These features were added as the diagnostic-grade response to the empirical
finding that the cross-sectional momentum signal on this PIT universe lives
at 12-24 month lookback windows. They give LightGBM and the Lasso explicit
access to the signal an annual-refit ARIMA implicitly computes via its drift
term.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_model.features.technical  # noqa: F401  trigger registration
from price_model.features.base import FEATURE_REGISTRY

NEW_FEATURES = ["momentum_378", "momentum_504", "momentum_756"]


@pytest.mark.parametrize("name", NEW_FEATURES)
def test_feature_registered(name: str):
    assert name in FEATURE_REGISTRY


@pytest.mark.parametrize("name", NEW_FEATURES)
def test_feature_produces_output_column(synthetic_panel, name: str):
    feat = FEATURE_REGISTRY[name]
    out = feat.compute(synthetic_panel)
    assert name in out.columns
    assert out[name].dtype == pl.Float64


@pytest.mark.parametrize(
    "name,lookback",
    [("momentum_378", 378), ("momentum_504", 504), ("momentum_756", 756)],
)
def test_warmup_nulls_match_lookback(synthetic_panel, name: str, lookback: int):
    """The first `lookback` rows per ticker should be null (no history yet)."""
    feat = FEATURE_REGISTRY[name]
    out = feat.compute(synthetic_panel).sort(["ticker", "date"])
    one = out.filter(pl.col("ticker") == "AAA").sort("date")[name].to_list()
    leading_nulls = 0
    for v in one:
        if v is None:
            leading_nulls += 1
        else:
            break
    # EXACTLY `lookback` nulls (shift(n) puts nulls in the first n rows)
    assert leading_nulls == lookback, (
        f"{name}: expected {lookback} leading nulls, got {leading_nulls}"
    )


def test_momentum_504_equals_log_ratio(synthetic_panel):
    """Verify the math: momentum_504(t) = log(adj_close_t / adj_close_{t-504})."""
    feat = FEATURE_REGISTRY["momentum_504"]
    out = feat.compute(synthetic_panel).sort(["ticker", "date"])
    one = out.filter(pl.col("ticker") == "AAA").sort("date")
    # Pick an arbitrary populated row (index 510 = 504 + buffer for safety)
    row = one.row(510, named=True)
    expected = math.log(row["adj_close"]) - math.log(one.row(510 - 504, named=True)["adj_close"])
    assert abs(row["momentum_504"] - expected) < 1e-12


def test_momentum_378_equals_log_ratio(synthetic_panel):
    """Verify the math: momentum_378(t) = log(adj_close_t / adj_close_{t-378})."""
    feat = FEATURE_REGISTRY["momentum_378"]
    out = feat.compute(synthetic_panel).sort(["ticker", "date"])
    one = out.filter(pl.col("ticker") == "AAA").sort("date")
    row = one.row(400, named=True)
    expected = math.log(row["adj_close"]) - math.log(one.row(400 - 378, named=True)["adj_close"])
    assert abs(row["momentum_378"] - expected) < 1e-12


def test_momentum_756_equals_log_ratio(synthetic_panel):
    """Verify the math: momentum_756(t) = log(adj_close_t / adj_close_{t-756})."""
    feat = FEATURE_REGISTRY["momentum_756"]
    out = feat.compute(synthetic_panel).sort(["ticker", "date"])
    one = out.filter(pl.col("ticker") == "AAA").sort("date")
    row = one.row(800, named=True)
    expected = math.log(row["adj_close"]) - math.log(one.row(800 - 756, named=True)["adj_close"])
    assert abs(row["momentum_756"] - expected) < 1e-12


@pytest.mark.parametrize("name", NEW_FEATURES)
def test_per_ticker_isolation(synthetic_panel, name: str):
    """Perturbing AAA's prices must not change BBB's momentum values."""
    feat = FEATURE_REGISTRY[name]
    baseline = feat.compute(synthetic_panel).filter(pl.col("ticker") == "BBB")[name]
    perturbed = synthetic_panel.with_columns(
        pl.when(pl.col("ticker") == "AAA")
        .then(pl.col("adj_close") * 2.0)
        .otherwise(pl.col("adj_close"))
        .alias("adj_close")
    )
    after = feat.compute(perturbed).filter(pl.col("ticker") == "BBB")[name]
    assert baseline.to_list() == after.to_list()


@pytest.mark.parametrize("name", NEW_FEATURES)
def test_constant_prices_produce_zero_momentum(synthetic_panel, name: str):
    """A flat price series → zero log-return over any window."""
    feat = FEATURE_REGISTRY[name]
    flat = synthetic_panel.with_columns(pl.lit(100.0).alias("adj_close"))
    out = feat.compute(flat).drop_nulls(name)
    if out.height > 0:
        # All non-null values must be ~0
        assert all(abs(v) < 1e-12 for v in out[name].to_list())
