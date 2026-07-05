"""Turnover and net-of-cost analysis for cross-sectional long-short signals.

Gross IC and gross Sharpe describe how well a model *ranks* the cross-section.
Net of transaction costs is what matters for portfolio realism, and on a
liquid universe like the S&P 500 daily, turnover differences across models
can flip the net ranking — especially when comparing a low-turnover momentum
proxy (e.g., annual-refit ARIMA whose predictions are static within each
refit window) against a high-turnover engineered LightGBM (whose predictions
update daily).

Method
------
For each evaluation date `t`:

1. Sort tickers by predicted return. Take the top quintile (top 20%) as the
   long leg, the bottom quintile as the short leg, equal-weighted within
   each leg. This matches `_long_short_returns` in `eval/metrics.py`.
2. Define a weight vector `w_t` over the union of tickers: `+1/k` if in
   long, `-1/k` if in short, `0` otherwise, where `k` is the leg size on
   date `t`.
3. One-sided turnover over one HOLDING PERIOD (= `horizon_days`, the rebalance
   interval matched to the H-day return):
       turnover_t = 0.5 * sum_i | w_t[i] - w_{t-H}[i] |
   A turnover of 1.0 corresponds to fully replacing every position in one
   rebalance. (Using a 1-day delta here while crediting an H-day return — the
   prior bug — pairs a daily cost with a monthly return and under-charges by ~H.)
4. Long-short gross return: `mean(top.realized) - mean(bot.realized)` (the
   same definition used by the existing metrics module).
5. After-cost return per holding period at cost `b` basis points per side,
   charged round-trip (buy one-sided turnover + sell one-sided turnover):
       net_ret_t = gross_ret_t - 2 * turnover_t * b / 10000
6. Annualize both gross and net Sharpe via `sqrt(252 / horizon_days)`,
   matching the existing `summarize` convention.

Annual turnover = per-rebalance turnover * (252 / horizon_days) = rebalances
per year * per-rebalance turnover. An annual turnover of 1.0 means the book
replaces itself once per year; low-turnover momentum books run ~1-15x,
a daily-signal book rebalanced at its 21-day horizon runs ~5-15x. (Rebalancing
the SAME signal *daily* would be ~21x higher, ~100-300x, but you would then be
earning 1-day returns, not the 21-day return the Sharpe is built on — so the
honest figure is the horizon-matched one computed here.)

Cost ranges in practice (one-sided, all-in: spread + commission + impact):

  - 1-3 bp: large institutional with internal execution on liquid US large-caps
  - 5-10 bp: smaller institutional / hedge fund retail brokerage
  - 15-30 bp: retail investor with discount broker on top names
  - 50+ bp: smaller / less liquid universes

CAVEAT (unchanged by this module): `returns_seq` are OVERLAPPING H-day returns
(one per date), so the Sharpe's standard deviation is autocorrelation-smoothed
and the Sharpe is somewhat inflated — gross AND net. This module fixes the COST
(clock + round-trip); for an overlap-free Sharpe use a non-overlapping
H-day-rebalance evaluation (`scripts/rebalance_cadence_net_cost.py`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from price_model.eval.metrics import _per_date_ic


@dataclass
class CostAdjustedSummary:
    """Bundle of gross and net-of-cost metrics for one model on one window."""

    n_observations: int
    n_dates: int

    # Gross metrics (same definitions as eval/metrics.py)
    gross_ic: float
    gross_ic_t_stat: float
    gross_long_short_sharpe: float

    # Turnover (one-sided), measured per rebalance (= horizon-day holding) and
    # annualized as rebalances/year. `daily_turnover_mean` is the per-day
    # equivalent (annual / 252), kept for backward-compatible reporting.
    daily_turnover_mean: float
    annual_turnover: float  # per_rebalance_turnover * (252 / horizon_days)

    # After-cost Sharpe at each requested cost level in basis points
    after_cost_sharpe_by_bp: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "n_observations": self.n_observations,
            "n_dates": self.n_dates,
            "gross_ic": self.gross_ic,
            "gross_ic_t_stat": self.gross_ic_t_stat,
            "gross_long_short_sharpe": self.gross_long_short_sharpe,
            "daily_turnover_mean": self.daily_turnover_mean,
            "annual_turnover": self.annual_turnover,
        }
        for bp, sharpe in sorted(self.after_cost_sharpe_by_bp.items()):
            out[f"after_cost_sharpe_{bp}bp"] = sharpe
        return out


def _long_short_weights_and_returns(
    df: pl.DataFrame, top_frac: float = 0.2
) -> tuple[list[dict[str, float]], list[float], list[object]]:
    """Build per-date (weight_dict, ls_return, date) lists.

    Returns three parallel lists in chronological date order so the caller
    can iterate to compute turnover (which needs the previous date's
    weights).
    """
    weights_seq: list[dict[str, float]] = []
    returns_seq: list[float] = []
    dates_seq: list[object] = []

    # Iterate dates in chronological order
    by_date = sorted(df["date"].unique().to_list())
    for d in by_date:
        sub = df.filter(pl.col("date") == d).drop_nulls(subset=["prediction", "realized"])
        n = sub.height
        if n < 10:
            continue
        k = max(1, round(n * top_frac))
        sorted_sub = sub.sort("prediction")
        # Top quintile = largest predictions; bottom = smallest.
        top = sorted_sub.tail(k)
        bot = sorted_sub.head(k)
        w_long = 1.0 / k
        w_short = -1.0 / k
        weights: dict[str, float] = {}
        for t in top["ticker"].to_list():
            weights[t] = w_long
        for t in bot["ticker"].to_list():
            weights[t] = w_short
        ls_ret = float(top["realized"].mean() - bot["realized"].mean())
        weights_seq.append(weights)
        returns_seq.append(ls_ret)
        dates_seq.append(d)
    return weights_seq, returns_seq, dates_seq


def _one_sided_turnover_series(weights_seq: list[dict[str, float]], lag: int = 1) -> list[float]:
    """One-sided turnover over a `lag`-observation holding period:

        turnover_t = 0.5 * sum_i | w_t[i] - w_{t-lag}[i] |.

    `lag` is the rebalance interval in observations. `lag=1` is daily
    rebalancing; `lag=H` matches a strategy that rebalances every H days and
    holds for H days — the realistic cadence for an H-day-horizon signal, and
    the one whose cost belongs next to an H-day return. Each of the first `lag`
    observations is measured against an empty book (the one-time initial build).
    A value of 1.0 corresponds to fully replacing every position in one rebalance.
    """
    turnovers: list[float] = []
    for i, weights in enumerate(weights_seq):
        prev = weights_seq[i - lag] if i >= lag else {}
        tickers = set(weights.keys()) | set(prev.keys())
        change = sum(abs(weights.get(t, 0.0) - prev.get(t, 0.0)) for t in tickers)
        turnovers.append(0.5 * change)
    return turnovers


def compute_turnover_and_costs(
    df: pl.DataFrame,
    *,
    top_frac: float = 0.2,
    cost_bps: Iterable[int] = (3, 10, 20),
    horizon_days: int = 5,
) -> CostAdjustedSummary:
    """Compute gross metrics, daily / annual turnover, and after-cost Sharpe.

    `df` must have columns `(date, ticker, prediction, realized)`. Predictions
    and realizations are joined by `(date, ticker)` upstream; this function
    expects them already aligned. See `eval/metrics.py::summarize` for the
    metric conventions reused here.
    """
    valid = df.drop_nulls(subset=["prediction", "realized"])
    if valid.height == 0:
        nan_costs = {int(bp): float("nan") for bp in cost_bps}
        return CostAdjustedSummary(
            n_observations=0,
            n_dates=0,
            gross_ic=float("nan"),
            gross_ic_t_stat=float("nan"),
            gross_long_short_sharpe=float("nan"),
            daily_turnover_mean=float("nan"),
            annual_turnover=float("nan"),
            after_cost_sharpe_by_bp=nan_costs,
        )

    weights_seq, returns_seq, _dates_seq = _long_short_weights_and_returns(valid, top_frac=top_frac)
    # Cost clock must match the return clock. `returns_seq` are horizon-day
    # forward returns, so the book is rebalanced every `horizon_days` and held
    # `horizon_days` — turnover is the change since the previous rebalance
    # (horizon-day lag), NOT a daily delta. Using a daily delta here (lag=1) is
    # the bug that paired a 1-day turnover with a horizon-day return.
    rebal = max(int(horizon_days), 1)
    turnover_seq = _one_sided_turnover_series(weights_seq, lag=rebal)

    n_dates = len(returns_seq)

    # Gross IC and t-stat
    ic_df = _per_date_ic(valid)
    if ic_df.height >= 2:
        ic_mean = float(ic_df["ic"].mean())
        ic_std = float(ic_df["ic"].std())
        gross_ic_t = (ic_mean * math.sqrt(ic_df.height)) / ic_std if ic_std > 0 else float("nan")
    else:
        ic_mean = float("nan")
        gross_ic_t = float("nan")

    # Gross Sharpe of the long-short portfolio (matches existing convention)
    ret_arr = np.array(returns_seq, dtype=float)
    turnover_arr = np.array(turnover_seq, dtype=float)
    ann_factor = math.sqrt(252 / max(horizon_days, 1))
    if ret_arr.size >= 2 and ret_arr.std(ddof=1) > 0:
        gross_sharpe = float(ret_arr.mean() / ret_arr.std(ddof=1) * ann_factor)
    else:
        gross_sharpe = float("nan")

    # `turnover_arr` is now per-rebalance (one value per horizon-day holding),
    # so the annualization is rebalances-per-year = 252 / horizon_days, not 252.
    period_to = float(turnover_arr.mean()) if turnover_arr.size > 0 else float("nan")
    rebals_per_year = 252.0 / rebal
    annual_to = period_to * rebals_per_year if not math.isnan(period_to) else float("nan")
    daily_to = annual_to / 252.0 if not math.isnan(annual_to) else float("nan")

    # After-cost Sharpe per cost level. Round-trip: a rebalance buys one-sided
    # turnover AND sells one-sided turnover, so cost = 2 * turnover * bp/side
    # (the factor the module docstring claimed but the code was missing).
    after_cost: dict[int, float] = {}
    for bp in cost_bps:
        bp_int = int(bp)
        cost_decimal = bp_int / 10000.0
        net = ret_arr - turnover_arr * cost_decimal * 2.0
        if net.size >= 2 and net.std(ddof=1) > 0:
            after_cost[bp_int] = float(net.mean() / net.std(ddof=1) * ann_factor)
        else:
            after_cost[bp_int] = float("nan")

    return CostAdjustedSummary(
        n_observations=int(valid.height),
        n_dates=int(n_dates),
        gross_ic=ic_mean,
        gross_ic_t_stat=gross_ic_t,
        gross_long_short_sharpe=gross_sharpe,
        daily_turnover_mean=daily_to,
        annual_turnover=annual_to,
        after_cost_sharpe_by_bp=after_cost,
    )


def compare_models_costs(
    df: pl.DataFrame,
    *,
    model_ids: Iterable[str] | None = None,
    top_frac: float = 0.2,
    cost_bps: Iterable[int] = (3, 10, 20),
    horizon_days: int = 5,
) -> pl.DataFrame:
    """Run `compute_turnover_and_costs` once per model_id; return a comparison table."""
    if "model_id" not in df.columns:
        raise ValueError("DataFrame must have a model_id column for comparison")
    ids = list(model_ids) if model_ids else sorted(df["model_id"].unique().to_list())
    rows: list[dict[str, object]] = []
    for mid in ids:
        sub = df.filter(pl.col("model_id") == mid).select(
            "date", "ticker", "prediction", "realized"
        )
        summary = compute_turnover_and_costs(
            sub, top_frac=top_frac, cost_bps=cost_bps, horizon_days=horizon_days
        ).as_dict()
        summary["model_id"] = mid
        rows.append(summary)
    if not rows:
        return pl.DataFrame()
    # Sort by gross IC descending for a stable display
    out = pl.DataFrame(rows)
    if "gross_ic" in out.columns:
        out = out.sort("gross_ic", descending=True, nulls_last=True)
    # Put model_id first
    cols = ["model_id"] + [c for c in out.columns if c != "model_id"]
    return out.select(cols)
