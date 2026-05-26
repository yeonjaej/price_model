"""Target construction.

The target is the cross-sectional excess return over a forward horizon:

    excess_return_h = log_return_{t -> t+h} - mean_universe(log_return_{t -> t+h})

This removes the market-wide move and forces the model to predict *relative*
performance, which is what cross-sectional models can actually learn.

Important leakage note: the target uses FUTURE returns, so any row whose target
window extends past the training cutoff MUST be excluded from training. The
walk-forward harness does this via the embargo.
"""

from __future__ import annotations

import polars as pl


def add_forward_excess_return(
    panel: pl.DataFrame,
    horizon_days: int = 5,
    price_col: str = "adj_close",
    date_col: str = "date",
    target_col: str = "y",
) -> pl.DataFrame:
    """Add a forward excess-return column.

    `y_t = log(P_{t+h} / P_t) - mean_{universe at t}(log(P_{t+h} / P_t))`

    Rows where the forward price isn't yet observed get null targets and should
    be dropped before training (but kept for inference).
    """
    c = pl.col(price_col)
    raw_fwd = (c.shift(-horizon_days).log() - c.log()).over("ticker")
    panel = panel.with_columns(raw_fwd.alias(f"_raw_fwd_{horizon_days}"))
    panel = panel.with_columns(
        (
            pl.col(f"_raw_fwd_{horizon_days}")
            - pl.col(f"_raw_fwd_{horizon_days}").mean().over(date_col)
        ).alias(target_col)
    )
    return panel.drop(f"_raw_fwd_{horizon_days}")
