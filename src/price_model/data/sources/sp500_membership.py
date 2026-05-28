"""S&P 500 historical membership scraper — Wikipedia source.

Reconstructs a `(ticker, added, removed)` table for every name that has been
in the S&P 500 during the change-log coverage window. `removed` is null for
current members. This is the substrate for point-in-time (PIT) evaluation:
on any historical date `t`, a name is "in" iff `added <= t < removed` (or
`removed is null`).

Source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
The page has two relevant tables:

  Table 0 — "S&P 500 component stocks"
    Columns: Symbol, Security, GICS Sector, GICS Sub-Industry,
             Headquarters Location, Date added, CIK, Founded
    One row per *current* S&P 500 component (~503 names).

  Table 1 — "Selected changes to the S&P 500 components"
    MultiIndex header: ('', 'Date'), ('Added', 'Ticker'), ('Added', 'Security'),
                       ('Removed', 'Ticker'), ('Removed', 'Security'),
                       ('Reason', 'Reason')
    One row per index-membership event since ~2014. Either side
    (Added/Removed) may be blank for pure adds or pure removes.

Reconstruction algorithm:

  1. Start from Table 0 — every row contributes a (ticker, added) record
     with removed=None. The "Date added" column gives the canonical join date.

  2. Walk Table 1 from most-recent to oldest. For each row:
       - If "Added.Ticker" is present and not in our table → add it with
         added = row.Date, removed = None. (Edge case: Wikipedia's component
         table is current, so newly-added tickers are already there.)
       - If "Removed.Ticker" is present → set removed = row.Date for that
         ticker if it's in our table, or add it with added = "1970-01-01"
         (sentinel — predates yfinance coverage) and removed = row.Date.

  3. Apply ticker normalization (BRK.B -> BRK-B, BF.B -> BF-B) so the output
     matches yfinance's symbol convention.

Limitations of the Wikipedia source:
- The changes table is reliable back to ~2014. Pre-2014 membership requires
  a paid source. This is fine for a 2017-start training window.
- Wikipedia's "Date added" for current members is reliable; for delisted/
  removed members the only date we have is the removal date.
- Rename events (FB→META, FISV→FI) appear in the changes table as both a
  Remove and an Add. We DO NOT collapse them here — that's done downstream
  in tickers.py via the alias table, where the join logic merges histories.

Cache layout:
- data/raw/sp500_wiki.html        — the raw HTML, refreshable
- data/raw/sp500_membership.parquet — the parsed (ticker, added, removed) frame
"""

from __future__ import annotations

import io
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import polars as pl

log = logging.getLogger(__name__)

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HTML_CACHE_FILENAME = "sp500_wiki.html"
PARQUET_CACHE_FILENAME = "sp500_membership.parquet"

# Wikipedia uses BRK.B / BF.B; yfinance expects BRK-B / BF-B. Apply at parse time
# so every downstream consumer sees a single convention.
_WIKI_TO_YF_TICKER: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

# Sentinel for tickers whose "added" date predates the change log. yfinance
# coverage starts around 1970 for most names; we use this as a "very early"
# marker that won't accidentally exclude old rows from PIT joins.
_PREHISTORY = date(1970, 1, 1)


# ---------------------------------------------------------------------------
# HTTP fetch (network — wrapped so tests can monkeypatch)
# ---------------------------------------------------------------------------


def _download_html() -> str:
    """Fetch the Wikipedia HTML.

    Wikipedia rejects requests without a User-Agent header (returns 403), so
    we set a polite one identifying the project.
    """
    import urllib.request

    log.info("Downloading S&P 500 membership from %s", WIKIPEDIA_URL)
    # Wikipedia rejects requests without a User-Agent header; supply a generic
    # identifier rather than a personal one.
    req = urllib.request.Request(
        WIKIPEDIA_URL,
        headers={"User-Agent": "price-model/0.1 (research project)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


# ---------------------------------------------------------------------------
# Parsers (offline — tested against synthetic HTML fixtures)
# ---------------------------------------------------------------------------


def _normalize_ticker(t: str) -> str:
    """Apply Wikipedia→yfinance symbol convention."""
    t = (t or "").strip()
    return _WIKI_TO_YF_TICKER.get(t, t)


def _parse_date_string(s: str) -> date | None:
    """Parse Wikipedia's date format. Returns None on failure (we'll log).

    Wikipedia uses several formats: 'January 5, 2023', '2023-01-05',
    sometimes just '2023' (year-only). We try the common ones in order.
    """
    s = (s or "").strip()
    if not s or s.lower() in {"nan", "—", "-"}:
        return None
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%d %B %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # Year-only fallback (e.g., "1957" for early additions)
    try:
        return date(int(s), 1, 1)
    except ValueError:
        log.warning("Could not parse Wikipedia date %r — using prehistory sentinel", s)
        return None


def _parse_components_table(tables: list[pd.DataFrame]) -> pl.DataFrame:
    """Extract (ticker, added) from the first table (current components)."""
    if not tables:
        raise ValueError("read_html returned no tables")
    df = tables[0]
    # Wikipedia uses 'Symbol' (older) or 'Ticker' (newer) for the ticker column,
    # and 'Date added' (older) or 'Date first added' (newer) for the join date.
    col_map = {c.lower().strip(): c for c in df.columns.astype(str)}
    ticker_col = col_map.get("symbol") or col_map.get("ticker")
    date_col = col_map.get("date added") or col_map.get("date first added")
    if ticker_col is None or date_col is None:
        raise ValueError(f"Components table missing expected columns. Got: {list(df.columns)}")

    rows: list[dict] = []
    for _, row in df.iterrows():
        t = _normalize_ticker(str(row[ticker_col]))
        d = _parse_date_string(str(row[date_col])) or _PREHISTORY
        rows.append({"ticker": t, "added": d, "removed": None})
    return pl.DataFrame(
        rows,
        schema={"ticker": pl.Utf8, "added": pl.Date, "removed": pl.Date},
    )


def _parse_changes_table(tables: list[pd.DataFrame]) -> pl.DataFrame:
    """Extract (date, added_ticker, removed_ticker) from the changes table.

    The table has a MultiIndex header in current Wikipedia layout. pandas
    flattens it to strings like 'Added Ticker' / 'Removed Ticker'. We detect
    both layouts and select columns accordingly.
    """
    if len(tables) < 2:
        log.warning("read_html returned <2 tables — changes table missing, skipping")
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "added_ticker": pl.Utf8,
                "removed_ticker": pl.Utf8,
            }
        )

    df = tables[1].copy()
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join(str(c) for c in tup if str(c) != "nan").strip() for tup in df.columns
        ]

    # Find the relevant columns by lowercase match
    norm = {c.lower().strip(): c for c in df.columns.astype(str)}
    date_col = norm.get("date") or next((v for k, v in norm.items() if "date" in k), None)
    add_tk_col = next(
        (v for k, v in norm.items() if "added" in k and ("ticker" in k or "symbol" in k)),
        None,
    )
    rem_tk_col = next(
        (v for k, v in norm.items() if "removed" in k and ("ticker" in k or "symbol" in k)),
        None,
    )
    if not (date_col and (add_tk_col or rem_tk_col)):
        raise ValueError(f"Changes table missing required columns. Available: {list(df.columns)}")

    rows: list[dict] = []
    for _, row in df.iterrows():
        d = _parse_date_string(str(row[date_col]))
        if d is None:
            continue
        added = _normalize_ticker(str(row[add_tk_col])) if add_tk_col else ""
        removed = _normalize_ticker(str(row[rem_tk_col])) if rem_tk_col else ""
        # Strip nan/empty
        if added.lower() in {"", "nan"}:
            added = ""
        if removed.lower() in {"", "nan"}:
            removed = ""
        if not added and not removed:
            continue
        rows.append({"date": d, "added_ticker": added, "removed_ticker": removed})
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "added_ticker": pl.Utf8, "removed_ticker": pl.Utf8},
    )


def _build_membership_table(
    components: pl.DataFrame,
    changes: pl.DataFrame,
) -> pl.DataFrame:
    """Combine current components + change log into a (ticker, added, removed) frame.

    Strategy:
    - Components contributes the canonical (ticker, added, removed=None) rows.
    - Changes contributes removal dates for ex-members (and adds entries for
      tickers that were removed before they could ever appear in `components`).
    - Walk the changes table chronologically. For each remove event:
        * If ticker is currently in our map → set removed=event_date.
        * Else (the ticker was in pre-2014 and never current) → add with
          added=_PREHISTORY, removed=event_date.
    - For each add event:
        * If ticker is not in our map yet → it must be in `components`; ignore.
        * If it IS in our map but with removed != None → this is a re-add.
          Update added=event_date, removed=None. (Rare: GE was removed in
          2018 then re-added.)
    """
    # ticker -> {"added": date, "removed": date | None}
    table: dict[str, dict] = {}
    for row in components.iter_rows(named=True):
        table[row["ticker"]] = {"added": row["added"], "removed": None}

    # Process changes oldest-to-newest so re-adds work in order
    for row in changes.sort("date").iter_rows(named=True):
        d = row["date"]
        removed_tk = row["removed_ticker"]
        added_tk = row["added_ticker"]
        if removed_tk:
            existing = table.get(removed_tk)
            if existing is None:
                table[removed_tk] = {"added": _PREHISTORY, "removed": d}
            else:
                # Set removed only if we haven't already (most recent removal wins
                # only by virtue of being seen last; we process oldest-first so
                # later removals overwrite earlier ones — correct for re-add cycles).
                existing["removed"] = d
        if added_tk:
            existing = table.get(added_tk)
            if existing is None:
                # Ticker was added in the change log but isn't a current component
                # AND wasn't already in our table. This is unusual — probably a
                # short-lived membership. Add it.
                table[added_tk] = {"added": d, "removed": None}
            else:
                # Re-add: clear removed and update added.
                if existing.get("removed") is not None:
                    existing["added"] = d
                    existing["removed"] = None
                # If existing.removed is None and added matches, it's already
                # the canonical row from components — leave it.

    rows = [
        {"ticker": t, "added": v["added"], "removed": v["removed"]}
        for t, v in sorted(table.items())
    ]
    return pl.DataFrame(
        rows,
        schema={"ticker": pl.Utf8, "added": pl.Date, "removed": pl.Date},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _cache_paths(raw_dir: Path) -> tuple[Path, Path]:
    return raw_dir / HTML_CACHE_FILENAME, raw_dir / PARQUET_CACHE_FILENAME


def fetch(
    raw_dir: Path | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Return the historical S&P 500 membership table.

    Cached as both raw HTML and parsed parquet under `raw_dir`. The parquet is
    the fast path; the HTML is kept for reproducibility / re-parsing if we
    change the parser. Pass `force_refresh=True` to bypass both caches.

    Output schema: (ticker: Utf8, added: Date, removed: Date | None).
    """
    raw_dir = raw_dir or Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_cache, parquet_cache = _cache_paths(raw_dir)

    # Fast path: parquet exists, no refresh requested
    if use_cache and parquet_cache.exists() and not force_refresh:
        df = pl.read_parquet(parquet_cache)
        log.info(
            "Loaded SP500 membership from cache: %d tickers, %d currently active",
            df.height,
            df.filter(pl.col("removed").is_null()).height,
        )
        return df

    # Slow path: (re)download HTML and parse
    if use_cache and html_cache.exists() and not force_refresh:
        html = html_cache.read_text(encoding="utf-8")
    else:
        html = _download_html()
        html_cache.write_text(html, encoding="utf-8")

    tables = pd.read_html(io.StringIO(html))
    components = _parse_components_table(tables)
    changes = _parse_changes_table(tables)
    df = _build_membership_table(components, changes)
    df.write_parquet(parquet_cache)

    log.info(
        "Built SP500 membership: %d tickers total, %d currently active, %d historical",
        df.height,
        df.filter(pl.col("removed").is_null()).height,
        df.filter(pl.col("removed").is_not_null()).height,
    )
    return df


__all__ = [
    "WIKIPEDIA_URL",
    "fetch",
]
