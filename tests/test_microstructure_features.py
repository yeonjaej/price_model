"""Tests for the microstructure feature additions:
    - overnight_return_1d / intraday_return_1d (Lou-Polk-Skouras 2019 decomposition)
    - amihud_illiquidity_20 (Amihud 2002 price-impact-per-dollar)
    - residual_return_5d (beta-residualized 5-day return)
    - sector_relative_return_5d (sector-demeaned 5-day return)

Each test covers: registration, dtype, warmup behavior, math correctness on a
constructed panel, per-ticker isolation where applicable.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import price_model.features.cross_features  # trigger registration
import price_model.features.technical  # noqa: F401  trigger registration
from price_model.features.base import FEATURE_REGISTRY

MICRO_FEATURES = [
    "overnight_return_1d",
    "intraday_return_1d",
    "amihud_illiquidity_20",
    "residual_return_5d",
    "sector_relative_return_5d",
    "cs_return_dispersion_20",
]


@pytest.mark.parametrize("name", MICRO_FEATURES)
def test_feature_registered(name: str):
    assert name in FEATURE_REGISTRY


@pytest.mark.parametrize("name", MICRO_FEATURES)
def test_feature_produces_output_column(synthetic_panel, name: str):
    feat = FEATURE_REGISTRY[name]
    out = feat.compute(synthetic_panel)
    assert name in out.columns
    assert out[name].dtype == pl.Float64


# -----------------------------------------------------------------------------
# Overnight + intraday decomposition: math correctness
# -----------------------------------------------------------------------------


def test_overnight_plus_intraday_equals_close_to_close_return():
    """For each (date, ticker), overnight_return_1d + intraday_return_1d must
    equal the standard close-to-close log return log(adj_close_t/adj_close_{t-1}).
    This is the defining identity of the decomposition.
    """
    rng = np.random.default_rng(42)
    n_days = 100
    start = date(2024, 1, 1)
    rows = []
    for t_idx, ticker in enumerate(["AAA", "BBB"]):
        log_p = np.cumsum(rng.normal(0, 0.02, n_days))
        prices = 100.0 * np.exp(log_p)
        # Synthesize open != close so overnight and intraday are non-trivial
        open_factor = rng.normal(1.0, 0.005, n_days)
        opens = prices * open_factor
        for i, d in enumerate(start + timedelta(days=j) for j in range(n_days)):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": float(opens[i]),
                    "close": float(prices[i]),
                    "adj_close": float(prices[i]),  # no split → adj = raw
                    "high": float(prices[i]) * 1.01,
                    "low": float(prices[i]) * 0.99,
                    "volume": int(rng.integers(1_000_000, 10_000_000)),
                    "sector": "Tech",
                }
            )
        _ = t_idx  # tickers iterate; ticker var is loop scoped
    panel = pl.DataFrame(rows).sort(["ticker", "date"])

    overnight = FEATURE_REGISTRY["overnight_return_1d"].compute(panel)
    intraday = FEATURE_REGISTRY["intraday_return_1d"].compute(overnight)
    # Add the close-to-close return for comparison
    c = pl.col("adj_close")
    cc_return = (c.log() - c.log().shift(1)).over("ticker")
    intraday = intraday.with_columns(cc_return.alias("_cc_return"))

    valid = intraday.drop_nulls(["overnight_return_1d", "intraday_return_1d", "_cc_return"])
    assert valid.height > 0
    decomp_sum = (valid["overnight_return_1d"] + valid["intraday_return_1d"]).to_numpy()
    cc = valid["_cc_return"].to_numpy()
    assert np.allclose(decomp_sum, cc, rtol=1e-10, atol=1e-12)


def test_overnight_first_row_per_ticker_is_null():
    """Overnight return needs the previous day's close — first row per
    ticker must be null."""
    feat = FEATURE_REGISTRY["overnight_return_1d"]
    panel = pl.DataFrame(
        [
            {
                "date": date(2024, 1, 2),
                "ticker": "AAA",
                "open": 101.0,
                "close": 102.0,
                "adj_close": 102.0,
                "high": 103.0,
                "low": 100.0,
                "volume": 1000000,
                "sector": "Tech",
            },
            {
                "date": date(2024, 1, 3),
                "ticker": "AAA",
                "open": 102.5,
                "close": 103.0,
                "adj_close": 103.0,
                "high": 104.0,
                "low": 102.0,
                "volume": 1000000,
                "sector": "Tech",
            },
        ]
    ).sort(["ticker", "date"])
    out = feat.compute(panel).sort(["ticker", "date"])
    vals = out["overnight_return_1d"].to_list()
    assert vals[0] is None
    expected = math.log(102.5 / 102.0)
    assert abs(vals[1] - expected) < 1e-10


def test_intraday_first_row_has_value():
    """Intraday return is same-day (open → close), no shift needed —
    first row should NOT be null."""
    feat = FEATURE_REGISTRY["intraday_return_1d"]
    panel = pl.DataFrame(
        [
            {
                "date": date(2024, 1, 2),
                "ticker": "AAA",
                "open": 100.0,
                "close": 102.0,
                "adj_close": 102.0,
                "high": 103.0,
                "low": 99.0,
                "volume": 1000000,
                "sector": "Tech",
            }
        ]
    )
    out = feat.compute(panel)
    expected = math.log(102.0 / 100.0)
    assert abs(out["intraday_return_1d"][0] - expected) < 1e-10


# -----------------------------------------------------------------------------
# Amihud illiquidity
# -----------------------------------------------------------------------------


def test_amihud_warmup(synthetic_panel):
    """First 20 rows per ticker should be null (1 for diff + 19 for window)."""
    feat = FEATURE_REGISTRY["amihud_illiquidity_20"]
    out = feat.compute(synthetic_panel).sort(["ticker", "date"])
    one = out.filter(pl.col("ticker") == "AAA").sort("date")["amihud_illiquidity_20"].to_list()
    leading_nulls = 0
    for v in one:
        if v is None:
            leading_nulls += 1
        else:
            break
    # 20 from rolling_mean; the first daily return is also null but rolling_mean
    # tolerates that, so warmup is exactly 20 (the rolling window size).
    assert 19 <= leading_nulls <= 21, f"got {leading_nulls} leading nulls"


def test_amihud_is_non_negative(synthetic_panel):
    """|return| / dollar_volume is non-negative by construction."""
    feat = FEATURE_REGISTRY["amihud_illiquidity_20"]
    out = feat.compute(synthetic_panel).drop_nulls("amihud_illiquidity_20")
    assert (out["amihud_illiquidity_20"] >= 0).all()


# -----------------------------------------------------------------------------
# Residual return 5d
# -----------------------------------------------------------------------------


def test_residual_return_warmup(synthetic_panel):
    """Beta needs 60d of returns + 5d for r_5d + 1 for diff → ~65 leading nulls."""
    feat = FEATURE_REGISTRY["residual_return_5d"]
    out = feat.compute(synthetic_panel).sort(["ticker", "date"])
    one = out.filter(pl.col("ticker") == "AAA").sort("date")["residual_return_5d"].to_list()
    leading_nulls = 0
    for v in one:
        if v is None:
            leading_nulls += 1
        else:
            break
    # Tight window: 60 for beta rolling window + 1 for diff. The rolling_mean
    # in beta computation pulls warmup to ~60.
    assert leading_nulls >= 60, f"expected >= 60 leading nulls, got {leading_nulls}"


def test_residual_return_collapses_to_zero_for_pure_market_movers():
    """Construct a panel where every ticker follows the market perfectly
    (beta = 1). The residual return must be approximately zero by definition.
    """
    rng = np.random.default_rng(7)
    n_days = 200
    start = date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    common_log_ret = rng.normal(0.0, 0.015, n_days)
    rows = []
    for ticker in ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]:
        log_p = np.cumsum(common_log_ret)
        prices = 100.0 * np.exp(log_p)
        for d, p in zip(dates, prices, strict=True):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": float(p),
                    "close": float(p),
                    "adj_close": float(p),
                    "high": float(p) * 1.005,
                    "low": float(p) * 0.995,
                    "volume": 1_000_000,
                    "sector": "Tech",
                }
            )
    panel = pl.DataFrame(rows).sort(["ticker", "date"])
    out = FEATURE_REGISTRY["residual_return_5d"].compute(panel)
    valid = out.drop_nulls("residual_return_5d")
    # All tickers move identically → beta = 1, residual ≈ 0
    max_abs = float(valid["residual_return_5d"].abs().max())
    assert max_abs < 1e-9, f"max |residual| = {max_abs} on pure-market mover panel"


# -----------------------------------------------------------------------------
# Sector-relative return 5d
# -----------------------------------------------------------------------------


def test_sector_relative_requires_sector_column():
    """Computing without a 'sector' column should raise ValueError."""
    feat = FEATURE_REGISTRY["sector_relative_return_5d"]
    panel = pl.DataFrame(
        [
            {
                "date": date(2024, 1, 1),
                "ticker": "AAA",
                "adj_close": 100.0,
            }
        ]
    )
    with pytest.raises(ValueError, match="sector"):
        feat.compute(panel)


def test_cs_return_dispersion_20_is_constant_across_tickers_per_date(synthetic_panel):
    """The dispersion feature is a date-level statistic — every ticker on a
    given date must receive the same value."""
    feat = FEATURE_REGISTRY["cs_return_dispersion_20"]
    out = feat.compute(synthetic_panel).drop_nulls("cs_return_dispersion_20")
    # For each date in the output, all tickers should have identical dispersion
    per_date = out.group_by("date").agg(
        pl.col("cs_return_dispersion_20").std().alias("std_across_tickers")
    )
    # Std across tickers on the same date should be ~0 (broadcast value)
    max_std = float(per_date["std_across_tickers"].max())
    assert max_std < 1e-12, f"cs_dispersion not broadcast: max std across tickers = {max_std}"


def test_cs_return_dispersion_20_is_positive(synthetic_panel):
    """Cross-sectional std of returns is non-negative, and the smoothing
    preserves non-negativity."""
    feat = FEATURE_REGISTRY["cs_return_dispersion_20"]
    out = feat.compute(synthetic_panel).drop_nulls("cs_return_dispersion_20")
    assert (out["cs_return_dispersion_20"] >= 0).all()


def test_sector_relative_zero_when_ticker_equals_sector_median():
    """If a ticker's 5-day return exactly equals the sector median, its
    sector-relative return must be zero."""
    rng = np.random.default_rng(11)
    n_days = 50
    start = date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    rows = []
    # Three tickers in same sector with identical price paths → identical r_5d → median = each ticker's r_5d
    log_p_common = np.cumsum(rng.normal(0, 0.01, n_days))
    prices = 100.0 * np.exp(log_p_common)
    for ticker in ["AAA", "BBB", "CCC"]:
        for d, p in zip(dates, prices, strict=True):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "adj_close": float(p),
                    "sector": "Tech",
                }
            )
    panel = pl.DataFrame(rows).sort(["ticker", "date"])
    out = FEATURE_REGISTRY["sector_relative_return_5d"].compute(panel)
    valid = out.drop_nulls("sector_relative_return_5d")
    max_abs = float(valid["sector_relative_return_5d"].abs().max())
    assert max_abs < 1e-12, f"identical paths should give zero sector-relative, got {max_abs}"
