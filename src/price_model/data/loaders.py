"""Panel loader — the entry point everything else uses to get data.

Layered on top of the sources/ adapters. The rest of the codebase calls `load_panel(...)`
and never touches yfinance directly.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from price_model.data.sectors import attach_sector
from price_model.data.sources import yfinance_source
from price_model.data.universe import load_universe


def load_panel(
    universe: str = "sp500",
    start: str | date = "2015-01-01",
    end: str | date | None = None,
    raw_dir: Path | None = None,
    with_sector: bool = True,
) -> pl.DataFrame:
    """Load a price panel for a named universe.

    Returns long-format DataFrame: (date, ticker, open, high, low, close, adj_close,
    volume[, sector]). Sorted by (date, ticker).
    """
    tickers = load_universe(universe)
    panel = yfinance_source.fetch(tickers, start=start, end=end, raw_dir=raw_dir)
    if with_sector and panel.height > 0:
        panel = attach_sector(panel)
    return panel


def compute_returns(panel: pl.DataFrame, price_col: str = "adj_close") -> pl.DataFrame:
    """Add a log-return column. Sorted within ticker."""
    return (
        panel.sort(["ticker", "date"])
        .with_columns(
            (pl.col(price_col).log() - pl.col(price_col).log().shift(1).over("ticker"))
            .alias("log_return")
        )
    )
