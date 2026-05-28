"""Universe definitions.

A "universe" here is just a list of tickers. v0 ships two static lists:

- sp500: a curated 156-name large-cap subset of the actual S&P 500 (NOT the full
  ~503-name index). Roughly the top third by market cap. Used for training because
  cross-sectional learning needs breadth, but `breadth ≈ 156` leaves Sharpe on the
  table relative to the full index — see README "Known v0 limitations".
- top20_2026_01_01: deployment-time prediction universe (the user's focus).

Loading from text files (one ticker per line) keeps the lists trivially swappable —
later, a PIT-aware loader can return different memberships per date by walking
Wikipedia's S&P 500 history.

Two known shortcuts:

1. **Subset, not full index.** Expanding to all ~503 names roughly doubles the
   trading-relevant Sharpe at unchanged signal quality (Fundamental Law of Active
   Management: IR ~= IC * sqrt(breadth)). Cost: more manual sector mapping or a scrape,
   more yfinance failures to handle, more heterogeneous cross-section.

2. **Static membership = survivorship bias.** Today's list over-represents the names
   that survived to today; companies that fell out aren't present. All long-window
   backtests are upward-biased. Fix: PIT membership reconstruction (Wikipedia
   index-change history, or a paid PIT data provider).

Both are tracked TODOs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from price_model.data.tickers import resolve_ticker

UNIVERSE_DIR = Path(__file__).parent / "universes"


@lru_cache(maxsize=8)
def load_universe(name: str) -> tuple[str, ...]:
    """Load a universe by name. Returns a sorted tuple of tickers.

    Every ticker is routed through `tickers.resolve_ticker()` before being
    returned. That applies the rename map (FB → META, RTN → RTX, ...) and
    drops anything on `TICKER_DROP_LIST` (delisted / failed / went private,
    where yfinance has no usable history). The result is a clean universe
    that won't waste yfinance retries on known-dead symbols.
    """
    path = UNIVERSE_DIR / f"{name}.txt"
    if not path.exists():
        available = sorted(p.stem for p in UNIVERSE_DIR.glob("*.txt"))
        raise FileNotFoundError(f"Universe {name!r} not found. Available: {available}")
    raw = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    resolved = {r for r in (resolve_ticker(t) for t in raw) if r is not None}
    return tuple(sorted(resolved))


def list_universes() -> list[str]:
    return sorted(p.stem for p in UNIVERSE_DIR.glob("*.txt"))
