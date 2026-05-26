"""Walk-forward split correctness:

- every train_end + embargo strictly precedes test_start
- splits don't overlap each other
- embargo_days must be >= horizon
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pytest

from price_model.data.splits import walk_forward_splits


def test_no_train_test_overlap():
    splits = list(
        walk_forward_splits(
            start=date(2020, 1, 1),
            end=date(2023, 12, 31),
            refit_freq_days=21,
            embargo_days=6,
            min_train_days=252,
        )
    )
    assert splits, "Expected at least one split"
    for s in splits:
        assert s.train_end < s.test_start, f"Train/test overlap in {s}"
        gap = (s.test_start - s.train_end).days
        assert gap >= 6, f"Embargo too small in {s} (gap={gap})"


def test_splits_are_ordered_and_non_overlapping():
    splits = list(
        walk_forward_splits(
            start=date(2020, 1, 1),
            end=date(2023, 12, 31),
            refit_freq_days=21,
            embargo_days=6,
            min_train_days=252,
        )
    )
    for prev, curr in pairwise(splits):
        assert prev.test_end < curr.test_start, f"Overlap: {prev} -> {curr}"


def test_embargo_below_one_rejected():
    with pytest.raises(ValueError):
        list(walk_forward_splits(date(2020, 1, 1), date(2021, 1, 1), embargo_days=0))


def test_refit_freq_below_one_rejected():
    with pytest.raises(ValueError):
        list(walk_forward_splits(date(2020, 1, 1), date(2021, 1, 1), refit_freq_days=0))
