"""Tests for the turnover / net-of-cost analysis module."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from price_model.eval.turnover import (
    compare_models_costs,
    compute_turnover_and_costs,
)


def _make_eval_df(
    n_dates: int = 50,
    n_tickers: int = 50,
    seed: int = 42,
    static_rank: bool = False,
    daily_random_rank: bool = False,
) -> pl.DataFrame:
    """Synthetic (date, ticker, prediction, realized) panel.

    static_rank: every date uses the same ranking → zero turnover.
    daily_random_rank: every date is an independent random ranking → ~100% turnover.
    """
    rng = np.random.default_rng(seed)
    base_dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    rows: list[dict] = []
    # Static base prediction per ticker so realized has some signal
    base_pred = rng.normal(0.0, 1.0, n_tickers)
    for d in base_dates:
        if static_rank:
            pred = base_pred.copy()
        elif daily_random_rank:
            pred = rng.normal(0.0, 1.0, n_tickers)
        else:
            # Mostly stable with small daily noise
            pred = base_pred + rng.normal(0.0, 0.1, n_tickers)
        # Realized correlates loosely with prediction (positive IC)
        realized = pred * 0.01 + rng.normal(0.0, 0.02, n_tickers)
        for tk, p, r in zip(tickers, pred, realized, strict=True):
            rows.append({"date": d, "ticker": tk, "prediction": float(p), "realized": float(r)})
    return pl.DataFrame(rows)


def test_static_predictions_yield_near_zero_turnover_after_first_day():
    """If predictions never change, turnover should be ~0 except for day 1 setup."""
    df = _make_eval_df(static_rank=True, n_dates=20, n_tickers=40, seed=1)
    summary = compute_turnover_and_costs(df, top_frac=0.2, cost_bps=(5,))
    # First day's turnover is 1.0 (build the portfolio); subsequent days 0.
    # Average over 20 days ≈ 1/20 = 0.05.
    assert 0.03 < summary.daily_turnover_mean < 0.07
    # Annual turnover ≈ 252 * 0.05 = 12.6 — but synthetic so just check it's small
    assert summary.annual_turnover < 20.0


def test_random_predictions_yield_high_turnover():
    """Fully-randomized predictions → high turnover (book ~fully replaced each rebalance)."""
    df = _make_eval_df(daily_random_rank=True, n_dates=20, n_tickers=40, seed=2)
    summary = compute_turnover_and_costs(df, top_frac=0.2, cost_bps=(5,), horizon_days=5)
    # Turnover is measured per rebalance (= horizon_days) and annualized as
    # rebalances/year. Random reshuffling replaces ~the whole book each rebalance,
    # so the annualized turnover is large (a slow signal would be order ~1-15x).
    assert summary.annual_turnover > 40.0


def test_cost_adjustment_reduces_sharpe_at_high_cost():
    """Net Sharpe == gross at 0 bp, and a large cost reduces it below the gross value.

    Strict per-step monotonicity is NOT a mathematical invariant: net_t =
    ret_t - 2*turnover_t*bp, and when per-rebalance turnover correlates with
    per-rebalance returns, a small cost can shrink the return variance faster
    than the mean and briefly *raise* the Sharpe. The robust invariant is that a
    large enough cost pushes net Sharpe below the zero-cost (gross) value.
    """
    df = _make_eval_df(daily_random_rank=False, n_dates=40, n_tickers=40, seed=3)
    summary = compute_turnover_and_costs(df, top_frac=0.2, cost_bps=(0, 200), horizon_days=5)
    sharpes = summary.after_cost_sharpe_by_bp
    # At 0 bp the after-cost Sharpe equals the gross Sharpe.
    if not math.isnan(summary.gross_long_short_sharpe) and not math.isnan(sharpes[0]):
        assert abs(sharpes[0] - summary.gross_long_short_sharpe) < 1e-9
    # A large cost must reduce it.
    assert sharpes[200] < sharpes[0]


def test_zero_height_df_returns_nan():
    """Empty input → all-nan summary, no crash."""
    df = pl.DataFrame(
        schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "prediction": pl.Float64,
            "realized": pl.Float64,
        }
    )
    summary = compute_turnover_and_costs(df, cost_bps=(5,))
    assert summary.n_observations == 0
    assert summary.n_dates == 0
    assert math.isnan(summary.gross_ic)
    assert math.isnan(summary.daily_turnover_mean)


def test_compare_models_costs_produces_expected_columns():
    """Multi-model dispatch returns a long-form table with all summary fields."""
    df = _make_eval_df(n_dates=15, n_tickers=30, seed=4).with_columns(
        pl.lit("model_a").alias("model_id")
    )
    df2 = _make_eval_df(n_dates=15, n_tickers=30, seed=5).with_columns(
        pl.lit("model_b").alias("model_id")
    )
    joined = pl.concat([df, df2])
    out = compare_models_costs(joined, cost_bps=(3, 10))
    assert out.height == 2
    assert "model_id" in out.columns
    assert "gross_ic" in out.columns
    assert "annual_turnover" in out.columns
    assert "after_cost_sharpe_3bp" in out.columns
    assert "after_cost_sharpe_10bp" in out.columns


def test_compare_models_costs_requires_model_id():
    """Calling without a model_id column should raise."""
    df = _make_eval_df(n_dates=10, n_tickers=20, seed=6)
    with pytest.raises(ValueError, match="model_id"):
        compare_models_costs(df)


def test_summary_dict_round_trip():
    """as_dict() must contain all the cost-level fields, named correctly."""
    df = _make_eval_df(n_dates=15, n_tickers=20, seed=7)
    summary = compute_turnover_and_costs(df, cost_bps=(3, 10, 20))
    d = summary.as_dict()
    for bp in (3, 10, 20):
        assert f"after_cost_sharpe_{bp}bp" in d
    assert d["n_dates"] >= 1
    assert isinstance(d["gross_ic"], float)
