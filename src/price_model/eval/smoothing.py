"""Per-ticker signal smoothing for cross-sectional prediction signals.

Motivation
----------
Cross-sectional ML signals on daily-rebalanced large-cap universes
routinely suffer from prediction volatility that drives turnover above what
trading costs can support. A standard remedy in the literature is to apply
a low-pass filter to the raw signal — typically an exponentially weighted
moving average (EWM) — *per ticker, across consecutive dates*. The
smoothed signal preserves most of the underlying ranking information at the
cost of a small lag while substantially reducing the day-to-day reshuffling
of the long-short composition.

This is the same engineering trick Asness-Frazzini-Pedersen-style factor
construction uses: published quality, value, and momentum signals are
smoothed before being traded, not used raw.

Mechanics
---------
For each ticker `i`, sort by date and apply an exponentially weighted mean
with the requested halflife (measured in panel-row positions, which on a
daily walk-forward = trading days). The smoothed prediction at row `t` is:

    smoothed_i,t = (1 - alpha) * smoothed_i,t-1 + alpha * raw_i,t
    where alpha = 1 - 0.5^(1/halflife_days)

The result is causal — at date `t` the smoothed value uses only data from
`t` and earlier on the same ticker. No future leakage.

Two consequences worth flagging:

- The smoothing introduces a phase lag of roughly `halflife_days /
  ln(2) ≈ 1.44 * halflife_days` trading days. With a 21-day halflife,
  the smoothed signal lags raw by ~30 trading days. On signals whose
  predictive horizon is much longer than that lag (e.g., quarterly
  momentum), the IC degradation is small. On a 5-day-forward target, the
  IC degradation is meaningful but typically far smaller than the
  turnover reduction it buys.
- Cross-sectional dispersion of smoothed predictions is tighter than the
  raw cross-section. The rank ordering still works, but ties become more
  common; the long-short quintile assignment is more stable.

Usage
-----
Smoothing is applied to the prediction column only; realized targets and
all other columns pass through unchanged. The output is shape-equivalent
to the input. Downstream functions (`compute_turnover_and_costs`,
`summarize`, `time_split_evaluate`) consume the smoothed DataFrame
without any code changes.
"""

from __future__ import annotations

import polars as pl


def smooth_predictions(
    df: pl.DataFrame,
    halflife_days: int,
    *,
    pred_col: str = "prediction",
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pl.DataFrame:
    """Apply per-ticker exponentially weighted smoothing to the prediction column.

    Args:
        df: long-format DataFrame containing at minimum `(date_col, ticker_col,
            pred_col)`. Additional columns (e.g., `realized`, `model_id`) pass
            through unchanged.
        halflife_days: positive integer EWM halflife in trading days.
            `halflife_days=21` → 50% weight reaches back roughly one month.
        pred_col: name of the column to smooth. Defaults to `"prediction"`.
        ticker_col: per-ticker grouping column. Defaults to `"ticker"`.
        date_col: temporal ordering column. Defaults to `"date"`.

    Returns:
        A DataFrame with the same schema as `df` but with `pred_col` replaced
        by its smoothed values. Row order is normalized to
        `(ticker_col, date_col)` ascending. Empty input returns empty output.
    """
    if halflife_days <= 0:
        raise ValueError(f"halflife_days must be positive; got {halflife_days}")
    if df.height == 0:
        return df

    required = {date_col, ticker_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"smooth_predictions requires columns {required}; missing {missing}")

    # ewm_mean with half_life applies the standard formula
    #   alpha = 1 - 0.5 ** (1 / half_life)
    # which is the trading-day-equivalent of the EWM halflife the docstring describes.
    # .over(ticker_col) keeps the recursion isolated per ticker. adjust=False is the
    # causal recursive form (the default `adjust=True` divides by the sum of weights
    # in the trailing window, which is also causal but produces a slightly different
    # transient at the start of each ticker's history; recursive form matches the
    # finance-literature convention).
    return df.sort([ticker_col, date_col]).with_columns(
        pl.col(pred_col)
        .ewm_mean(half_life=halflife_days, adjust=False)
        .over(ticker_col)
        .alias(pred_col)
    )
