"""Universe definitions.

A "universe" here is just a list of tickers. Several universes ship with the
project:

- ``sp500``: a curated 156-name large-cap subset of the actual S&P 500 — the top
  third by market cap. Useful for quick experiments and as a survivorship-biased
  comparison baseline.
- ``sp500_pit``: 617-name expanded universe — every ticker that was an S&P 500
  member at any point during 2017-2026, reconstructed from Wikipedia's historical
  components log (see ``data.sources.sp500_membership``). Combined with
  ``load_panel(pit_filter=True)`` this is the survivorship-bias-corrected
  evaluation universe.
- ``top20_2026_01_01``: small deployment-time universe used for quick tests and
  the dashboard's default view.

Loading is from text files (one ticker per line). Tickers are routed through
``data.tickers.resolve_ticker`` so renames (FB→META, RTN→RTX) and delisted /
unavailable symbols are handled at load time.
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
