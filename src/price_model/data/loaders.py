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
    pit_filter: bool = False,
) -> pl.DataFrame:
    """Load a price panel for a named universe.

    Returns long-format DataFrame: (date, ticker, open, high, low, close, adj_close,
    volume[, sector]). Sorted by (date, ticker).

    When `pit_filter=True`, each row is kept only if the ticker was an actual
    S&P 500 member on its date (per Wikipedia's historical components). This
    removes the survivorship-bias inflation of IC/Sharpe — the model is
    evaluated on the cross-section that actually existed at each point in time,
    not on today's snapshot. Membership data is fetched on first use; see
    `price_model.data.sources.sp500_membership`.
    """
    tickers = load_universe(universe)
    panel = yfinance_source.fetch(tickers, start=start, end=end, raw_dir=raw_dir)
    if pit_filter and panel.height > 0:
        from price_model.data.membership import filter_panel_to_pit

        before = panel.height
        panel = filter_panel_to_pit(panel)
        # Log the drop so it's visible when running experiments — useful for
        # quantifying the survivorship-bias correction.
        # (No log import here; tradition in this module is to stay silent.
        # The CLI's --verbose path can be added later if needed.)
        _ = before  # placeholder to keep the variable for future logging
    if with_sector and panel.height > 0:
        panel = attach_sector(panel)
    return panel


def compute_returns(panel: pl.DataFrame, price_col: str = "adj_close") -> pl.DataFrame:
    """Add a log-return column. Sorted within ticker."""
    return panel.sort(["ticker", "date"]).with_columns(
        (pl.col(price_col).log() - pl.col(price_col).log().shift(1).over("ticker")).alias(
            "log_return"
        )
    )
