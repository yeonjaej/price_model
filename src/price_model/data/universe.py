"""Universe definitions.

A "universe" here is just a list of tickers. v0 ships two static lists:

- sp500: ~160 large US names, used for training (cross-sectional learning needs breadth)
- top20_2026_01_01: deployment-time prediction universe (the user's focus)

Loading from text files (one ticker per line) keeps the lists trivially swappable —
later, a PIT-aware loader can return different memberships per date by walking
Wikipedia's S&P 500 history.

Survivorship-bias caveat: the static lists are TODAY'S membership. Backtests on long
windows therefore overstate performance (we never see companies that fell out). This
is acknowledged and tracked as a v2 follow-up.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

UNIVERSE_DIR = Path(__file__).parent / "universes"


@lru_cache(maxsize=8)
def load_universe(name: str) -> tuple[str, ...]:
    """Load a universe by name. Returns a sorted tuple of tickers."""
    path = UNIVERSE_DIR / f"{name}.txt"
    if not path.exists():
        available = sorted(p.stem for p in UNIVERSE_DIR.glob("*.txt"))
        raise FileNotFoundError(f"Universe {name!r} not found. Available: {available}")
    tickers = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return tuple(sorted(set(tickers)))


def list_universes() -> list[str]:
    return sorted(p.stem for p in UNIVERSE_DIR.glob("*.txt"))
