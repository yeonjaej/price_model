"""Point-in-time S&P 500 membership lookup.

Three public functions, all backed by the same lazily-loaded membership table
from `sources.sp500_membership`:

- `members_on_date(d)` — set of tickers in the S&P 500 on date `d`.
- `is_member(ticker, d)` — convenience scalar wrapper.
- `filter_panel_to_pit(panel)` — filter a (date, ticker, ...) panel so each
  row survives only if the ticker was an index member on that row's date.

The membership table is loaded once per process via `lru_cache`. Tests that
want to swap in synthetic membership data should monkeypatch
`price_model.data.sources.sp500_membership.fetch` and call
`_load_membership_table.cache_clear()` (mirroring the pattern used for the
Ken French and factor-loading caches).

The PIT filter is the key building block for honest backtest reporting. Run
the same model with and without it; the delta in IC / Sharpe is your
survivorship-bias correction.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import polars as pl

from price_model.data.sources import sp500_membership


@lru_cache(maxsize=1)
def _load_membership_table() -> pl.DataFrame:
    """Lazy-load the membership table once per process.

    Schema: (ticker: Utf8, added: Date, removed: Date | None).
    """
    return sp500_membership.fetch()


def members_on_date(d: date) -> set[str]:
    """Tickers that were in the S&P 500 on date `d`.

    A ticker is "in" on `d` iff `added <= d` AND (`removed` IS NULL OR `removed > d`).
    The strict-greater on `removed` matches Wikipedia's convention that the
    removal date is the *effective* date (first day NOT in the index).
    """
    df = _load_membership_table()
    return set(
        df.filter(
            (pl.col("added") <= d)
            & (pl.col("removed").is_null() | (pl.col("removed") > d))
        )["ticker"].to_list()
    )


def is_member(ticker: str, d: date) -> bool:
    """Convenience scalar lookup. Returns False for unknown tickers."""
    return ticker in members_on_date(d)


def members_during_window(start: date, end: date) -> set[str]:
    """Tickers that were S&P 500 members at any point during [start, end].

    A ticker is "in" the window iff its membership interval `[added, removed)`
    overlaps `[start, end]`. Equivalently: `added <= end` AND
    (`removed` IS NULL OR `removed > start`).

    Used to build the universe file for PIT-correct backtests — you want
    yfinance data for every name that was ever a member during the training
    window, not just current survivors.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")
    df = _load_membership_table()
    return set(
        df.filter(
            (pl.col("added") <= end)
            & (pl.col("removed").is_null() | (pl.col("removed") > start))
        )["ticker"].to_list()
    )


def filter_panel_to_pit(panel: pl.DataFrame) -> pl.DataFrame:
    """Filter a long-form panel to point-in-time membership.

    Each row survives iff its ticker was an S&P 500 member on its date.
    Required input columns: `date`, `ticker`. All other columns are preserved.

    Implementation: anti-correlated join + boolean keep. We use a left join
    rather than a per-row Python loop because the panel can be millions of
    rows on the full universe.
    """
    if panel.height == 0:
        return panel
    required = {"date", "ticker"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"filter_panel_to_pit requires columns {required}; missing {missing}")

    membership = _load_membership_table()
    # Join membership ranges onto panel by ticker, then filter by date bounds.
    joined = panel.join(membership, on="ticker", how="left")
    kept = joined.filter(
        pl.col("added").is_not_null()
        & (pl.col("date") >= pl.col("added"))
        & (pl.col("removed").is_null() | (pl.col("date") < pl.col("removed")))
    )
    # Drop the membership cols we added for the filter, restore original schema
    return kept.drop(["added", "removed"]).select(panel.columns)


__all__ = [
    "filter_panel_to_pit",
    "is_member",
    "members_during_window",
    "members_on_date",
]
