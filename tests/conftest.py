"""Shared test fixtures — synthetic panels so tests don't depend on yfinance."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def synthetic_panel() -> pl.DataFrame:
    """A small panel: 6 tickers x ~600 trading days of random-walk prices.

    Sectors are assigned deterministically so the sector-relative features have
    something non-trivial to do (need >1 ticker per sector for the median to differ
    from the value).
    """
    rng = np.random.default_rng(seed=42)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    sector_assignment = {
        "AAA": "Tech", "BBB": "Tech",
        "CCC": "Health", "DDD": "Health",
        "EEE": "Energy", "FFF": "Energy",
    }
    n_days = 600
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    rows = []
    for t in tickers:
        log_p = np.cumsum(rng.normal(0.0003, 0.015, size=n_days))
        prices = 100.0 * np.exp(log_p)
        vol = rng.integers(1_000_000, 10_000_000, size=n_days)
        for d, p, v in zip(dates, prices, vol):
            rows.append({
                "date": d, "ticker": t, "sector": sector_assignment[t],
                "open": float(p), "high": float(p) * 1.005, "low": float(p) * 0.995,
                "close": float(p), "adj_close": float(p), "volume": int(v),
            })
    return pl.DataFrame(rows).sort(["ticker", "date"])
