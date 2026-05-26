"""Time-based splits for walk-forward evaluation.

The single most important property: for every (train, test) split, every training row's
target must resolve strictly before the test window begins. We enforce this with an
explicit `embargo_days` parameter that must be >= the forward target horizon.

Walk-forward generates a sequence of refit dates. At each refit, we train on everything
up to (refit_date - embargo) and predict for the next `refit_freq` window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator

import polars as pl


@dataclass(frozen=True)
class Split:
    """One walk-forward fold."""

    refit_date: date          # the cutoff for training data
    train_end: date           # last date used for training (refit_date - embargo)
    test_start: date          # first date predictions are made for
    test_end: date            # last date predictions are made for (inclusive)


def walk_forward_splits(
    start: date,
    end: date,
    refit_freq_days: int = 21,        # ~monthly
    embargo_days: int = 6,            # must be >= horizon
    min_train_days: int = 252 * 2,    # ~2y warmup before first refit
) -> Iterator[Split]:
    """Yield Split objects covering [start + min_train_days, end]."""
    if embargo_days < 1:
        raise ValueError("embargo_days must be >= 1")
    if refit_freq_days < 1:
        raise ValueError("refit_freq_days must be >= 1")

    first_refit = start + timedelta(days=min_train_days)
    refit = first_refit
    while refit <= end:
        train_end = refit - timedelta(days=embargo_days)
        if train_end <= start:
            refit += timedelta(days=refit_freq_days)
            continue
        test_start = refit
        test_end = min(refit + timedelta(days=refit_freq_days - 1), end)
        yield Split(
            refit_date=refit,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        refit += timedelta(days=refit_freq_days)


def slice_train(panel: pl.DataFrame, split: Split, date_col: str = "date") -> pl.DataFrame:
    return panel.filter(pl.col(date_col) <= pl.lit(split.train_end))


def slice_test(panel: pl.DataFrame, split: Split, date_col: str = "date") -> pl.DataFrame:
    return panel.filter(
        (pl.col(date_col) >= pl.lit(split.test_start))
        & (pl.col(date_col) <= pl.lit(split.test_end))
    )
