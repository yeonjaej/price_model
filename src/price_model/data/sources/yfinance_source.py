"""yfinance adapter — fetch daily OHLCV, return a long-format polars panel.

This is the only module that touches yfinance directly. Everything downstream consumes
a panel with columns: date (Date), ticker (Utf8), open/high/low/close/adj_close (Float64),
volume (Int64). If we ever swap yfinance for Polygon/Sharadar, this is the only file
that changes — that's the point of isolating the source behind an adapter.

Caching: raw downloads land in data/raw/<ticker>.parquet keyed by ticker. Re-running
incrementally extends rather than re-fetching from scratch.

Robustness notes:
- yfinance's column layout has shifted over versions. Single-ticker downloads now return
  a MultiIndex columns frame where the levels may be ("Ticker", "Field") OR ("Field", "Ticker")
  depending on version. We detect which level holds the field names and pick that one.
- The date index may be named "Date" (daily) or "Datetime" (intraday); we normalize both.
- "Adj Close" may be absent when newer yfinance silently auto-adjusts; we fall back to
  "Close" with a warning.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import polars as pl

log = logging.getLogger(__name__)

# Schema we hand downstream. Pinned so changes are obvious.
PANEL_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Int64,
}

# The set of field names yfinance might use (case-sensitive as returned).
_FIELD_NAMES = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
# Map lowercased / normalized incoming names to our canonical schema column names.
_RENAME = {
    "date": "date",
    "datetime": "date",
    "index": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjusted close": "adj_close",
    "volume": "volume",
}


def _cache_path(raw_dir: Path, ticker: str) -> Path:
    return raw_dir / f"{ticker}.parquet"


def _flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce yfinance's MultiIndex columns to a single level of field names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    level0 = list(df.columns.get_level_values(0))
    level1 = list(df.columns.get_level_values(1))

    # Whichever level contains the known field names is the one we keep.
    if any(name in _FIELD_NAMES for name in level0):
        df.columns = pd.Index(level0)
    elif any(name in _FIELD_NAMES for name in level1):
        df.columns = pd.Index(level1)
    else:
        # Neither level has field names — concatenate as fallback
        df.columns = pd.Index(
            [f"{a}_{b}" if b else str(a) for a, b in zip(level0, level1, strict=True)]
        )
    return df


def _normalize_pandas_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Bring an arbitrary yfinance frame into our canonical pandas shape."""
    df = _flatten_multiindex(df)
    df = df.reset_index()

    # Drop accidental duplicates introduced by reset_index() + MultiIndex flattening
    df = df.loc[:, ~df.columns.duplicated()]

    # Build a rename map based on lowercased input names
    rename_map: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _RENAME:
            rename_map[col] = _RENAME[key]
    df = df.rename(columns=rename_map)

    # If adj_close is missing, fall back to close (newer yfinance auto-adjusts by default)
    if "adj_close" not in df.columns and "close" in df.columns:
        log.warning(
            "yfinance returned no 'Adj Close' for %s; using 'Close' as adj_close. "
            "Note: newer yfinance versions auto-adjust by default.",
            ticker,
        )
        df["adj_close"] = df["close"]

    required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"yfinance frame for {ticker} is missing required columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}"
        )

    return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]


def _yf_download_one(
    ticker: str,
    start: str | date,
    end: str | date | None,
    max_retries: int = 2,
    retry_sleep_s: float = 2.0,
) -> pl.DataFrame:
    """Hit yfinance for one ticker, normalize to our panel schema.

    yfinance occasionally returns no data for valid tickers due to transient errors
    (rate limit, DNS hiccup, "possibly delisted; no timezone found"). We retry up to
    `max_retries` times with a small sleep before giving up.
    """
    import time

    import yfinance as yf  # local import: yfinance pulls in heavy deps at import time

    df = None
    for attempt in range(max_retries + 1):
        df = yf.download(
            ticker,
            start=str(start),
            end=str(end) if end else None,
            progress=False,
            auto_adjust=False,
            actions=False,
        )
        if df is not None and not df.empty:
            break
        if attempt < max_retries:
            log.info("yfinance retry %d/%d for %s", attempt + 1, max_retries, ticker)
            time.sleep(retry_sleep_s)

    if df is None or df.empty:
        log.warning("yfinance returned no data for %s after %d attempts", ticker, max_retries + 1)
        return pl.DataFrame(schema=PANEL_SCHEMA)

    try:
        df = _normalize_pandas_frame(df, ticker)
    except ValueError as e:
        log.error("Failed to normalize yfinance frame for %s: %s", ticker, e)
        return pl.DataFrame(schema=PANEL_SCHEMA)

    out = pl.from_pandas(df).with_columns(
        pl.col("date").cast(pl.Date),
        pl.lit(ticker).alias("ticker"),
        pl.col("volume").cast(pl.Int64, strict=False),
        pl.col("open").cast(pl.Float64, strict=False),
        pl.col("high").cast(pl.Float64, strict=False),
        pl.col("low").cast(pl.Float64, strict=False),
        pl.col("close").cast(pl.Float64, strict=False),
        pl.col("adj_close").cast(pl.Float64, strict=False),
    )
    # Drop any rows with null prices (yfinance occasionally emits these for halts)
    out = out.drop_nulls(subset=["close"])
    return out.select(list(PANEL_SCHEMA.keys()))


def fetch(
    tickers: Iterable[str],
    start: str | date,
    end: str | date | None = None,
    raw_dir: Path | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    """Fetch a panel for the given tickers.

    Returns a long-format polars DataFrame with one row per (ticker, date).
    Cached per-ticker as parquet under raw_dir.
    """
    raw_dir = raw_dir or Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    end_date = (
        end
        if isinstance(end, date)
        else (
            datetime.fromisoformat(end).date() if isinstance(end, str) else datetime.today().date()
        )
    )

    frames: list[pl.DataFrame] = []
    for ticker in sorted(set(tickers)):
        cache = _cache_path(raw_dir, ticker)
        df: pl.DataFrame | None = None
        if use_cache and cache.exists():
            df = pl.read_parquet(cache)
            cached_max = df["date"].max() if df.height else None
            if cached_max is None or cached_max < end_date:
                fetch_start = str(cached_max) if cached_max else str(start)
                new_rows = _yf_download_one(ticker, fetch_start, end_date)
                if new_rows.height:
                    df = pl.concat([df, new_rows]).unique(subset=["date", "ticker"]).sort("date")
                    df.write_parquet(cache)
        else:
            df = _yf_download_one(ticker, start, end_date)
            if df.height:
                df.write_parquet(cache)
        if df is not None and df.height:
            df = df.filter(
                (pl.col("date") >= pl.lit(start).cast(pl.Date))
                & (pl.col("date") <= pl.lit(end_date).cast(pl.Date))
            )
            frames.append(df)
        else:
            log.warning("No data for %s", ticker)

    if not frames:
        return pl.DataFrame(schema=PANEL_SCHEMA)
    return pl.concat(frames).sort(["date", "ticker"])
