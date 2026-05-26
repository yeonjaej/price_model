"""Regression tests for the yfinance frame normalizer.

yfinance's return shape drifts across versions:
- Some versions return single-level columns: Open, High, Low, Close, Adj Close, Volume
- Newer versions return a MultiIndex with ('Field', 'Ticker') ordering
- Some versions return ('Ticker', 'Field') ordering instead (the one that broke us)
- 'Adj Close' may be absent when newer versions auto-adjust by default
- The date index may be named 'Date' or 'Datetime'

These tests construct synthetic pandas frames matching each shape we've seen in the wild
and verify the adapter produces a canonical polars panel without crashing.
"""

from __future__ import annotations

import pandas as pd

from price_model.data.sources.yfinance_source import _normalize_pandas_frame


def _make_single_level_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=3, freq="B"), name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Adj Close": [100.5, 101.5, 102.5],
            "Volume": [1_000_000, 1_100_000, 1_200_000],
        },
        index=idx,
    )


def test_normalize_single_level():
    """Original yfinance schema."""
    df = _normalize_pandas_frame(_make_single_level_frame(), "AAPL")
    assert set(df.columns) == {"date", "open", "high", "low", "close", "adj_close", "volume"}
    assert len(df) == 3


def test_normalize_field_ticker_multiindex():
    """MultiIndex with ('Field', 'Ticker') ordering."""
    base = _make_single_level_frame()
    base.columns = pd.MultiIndex.from_tuples([(c, "AAPL") for c in base.columns])
    df = _normalize_pandas_frame(base, "AAPL")
    assert "adj_close" in df.columns
    assert len(df) == 3


def test_normalize_ticker_field_multiindex():
    """MultiIndex with ('Ticker', 'Field') ordering — the layout that broke v0."""
    base = _make_single_level_frame()
    base.columns = pd.MultiIndex.from_tuples([("AAPL", c) for c in base.columns])
    df = _normalize_pandas_frame(base, "AAPL")
    assert set(df.columns) == {"date", "open", "high", "low", "close", "adj_close", "volume"}
    assert len(df) == 3
    # Spot-check a value to ensure we picked the right level, not the ticker name
    assert df["close"].iloc[0] == 100.5


def test_normalize_missing_adj_close_falls_back_to_close():
    """Newer yfinance versions sometimes drop Adj Close when auto-adjusting."""
    df = _make_single_level_frame().drop(columns=["Adj Close"])
    out = _normalize_pandas_frame(df, "AAPL")
    assert "adj_close" in out.columns
    assert (out["adj_close"] == out["close"]).all()


def test_normalize_datetime_index_name():
    """Intraday-shaped frame uses 'Datetime' as the index name."""
    df = _make_single_level_frame()
    df.index = df.index.rename("Datetime")
    out = _normalize_pandas_frame(df, "AAPL")
    assert "date" in out.columns
    assert len(out) == 3
