"""Tests for the purged forward-chain CV splitter used by the linear models.

These guard against the original bug: an integer `cv` on a ticker-sorted panel
silently became a non-temporal, unpurged KFold split. The splitter must be
temporal (train strictly before val), purged (embargo gap), and expanding.
"""

from __future__ import annotations

import numpy as np
import pytest

from price_model.models._cv import purged_forward_chain_folds


def _panel_dates(n_dates: int = 120, n_tickers: int = 5, shuffle: bool = True) -> np.ndarray:
    """Each date repeated n_tickers times, optionally shuffled (rows not date-sorted)."""
    dates = np.repeat(np.arange(n_dates), n_tickers)
    if shuffle:
        rng = np.random.default_rng(0)
        dates = dates[rng.permutation(dates.size)]
    return dates


def test_folds_are_temporal_and_purged():
    dates = _panel_dates(120, 5)
    embargo = 21
    folds = purged_forward_chain_folds(dates, n_splits=4, embargo=embargo)
    assert len(folds) >= 2
    for train_idx, val_idx in folds:
        train_dates, val_dates = dates[train_idx], dates[val_idx]
        # temporal: every training date strictly precedes every validation date
        assert train_dates.max() < val_dates.min()
        # purge: exactly `embargo` unique dates fall in the gap between them
        gap = np.unique(dates[(dates > train_dates.max()) & (dates < val_dates.min())])
        assert gap.size == embargo
        # disjoint index sets
        assert not (set(train_idx.tolist()) & set(val_idx.tolist()))


def test_window_is_expanding():
    folds = purged_forward_chain_folds(_panel_dates(120, 3), n_splits=4, embargo=10)
    train_sizes = [np.unique(_panel_dates(120, 3)[tr]).size for tr, _ in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]


def test_indices_recover_correct_dates():
    dates = _panel_dates(90, 4, shuffle=True)
    for train_idx, val_idx in purged_forward_chain_folds(dates, n_splits=3, embargo=15):
        # every index points at a row whose date is in the intended block
        assert np.all(dates[val_idx] >= dates[train_idx].max() + 15)


def test_unsorted_input_matches_sorted():
    base = _panel_dates(100, 3, shuffle=False)
    shuf = _panel_dates(100, 3, shuffle=True)
    fa = purged_forward_chain_folds(base, n_splits=3, embargo=12)
    fb = purged_forward_chain_folds(shuf, n_splits=3, embargo=12)
    # same set of (train_dates, val_dates) regardless of row order
    for (ta, va), (tb, vb) in zip(fa, fb, strict=True):
        assert set(base[ta].tolist()) == set(shuf[tb].tolist())
        assert set(base[va].tolist()) == set(shuf[vb].tolist())


def test_datetime64_dates_supported():
    days = np.repeat(np.arange(80), 3)
    dates = np.datetime64("2022-10-10") + days.astype("timedelta64[D]")
    folds = purged_forward_chain_folds(dates, n_splits=3, embargo=10)
    for train_idx, val_idx in folds:
        assert dates[train_idx].max() < dates[val_idx].min()


def test_too_few_dates_raises():
    with pytest.raises(ValueError, match=r"unique dates|too short"):
        purged_forward_chain_folds(np.array([1, 1, 2, 2]), n_splits=5, embargo=21)
