"""Honest net-of-cost: overlapping vs non-overlapping return series, both 21-day rebalance.

After the cost fix, `compute_turnover_and_costs` measures turnover over the 21-day
HOLDING period (lag = horizon_days) and annualizes as rebalances/year (252/21 ~= 12),
so it now reports the deployable ~9x turnover -- NOT a daily-churn figure. The only
thing "overlapping" about it is the RETURN series: it forms a book on every date and
credits the 21-day-forward return (348 overlapping observations). This compares it
against the same 21-day-rebalance strategy evaluated on NON-overlapping books:

  overlap      : compute_turnover_and_costs -- book formed each date, 21-day return,
                 348 overlapping obs. Smoother Sharpe estimate (overlap-correlated).
  non-overlap  : rebalance every 21 trading days, each book held 21 days, ~17 obs.
                 Per-book return matched to per-book cost. Same strategy, honest std.

Both rows are a 21-day-rebalance strategy, so both show ~9x turnover and the same IC
quality; they differ only in how the Sharpe's denominator treats serial overlap.

Usage: PYTHONPATH=src .venv/bin/python scripts/rebalance_cadence_net_cost.py
"""
from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit
from price_model.eval.turnover import compute_turnover_and_costs
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.pipeline.walk_forward import join_with_realized, run_walk_forward

FEATS = ["momentum_12_1", "momentum_756", "return_1d", "vol_ewm_20", "distance_52w_high", "log_dollar_volume"]
TRAIN_START, FIRST_REFIT, OOS = date(2022, 10, 10), date(2025, 1, 2), date(2025, 1, 1)
COSTS = (3, 10, 20)


def eval_21d(df: pl.DataFrame, step: int = 21) -> dict:
    """Non-overlapping rebalance every `step` trading days; matched per-book cost."""
    dates = sorted(df["date"].unique().to_list())
    rebal = dates[::step]
    prev: dict[str, float] = {}
    rets, turns, ics = [], [], []
    for d in rebal:
        sub = df.filter(pl.col("date") == d).drop_nulls(["prediction", "realized"])
        n = sub.height
        if n < 10:
            continue
        ic, _ = spearmanr(sub["prediction"].to_numpy(), sub["realized"].to_numpy())
        ics.append(float(ic))
        k = max(1, round(n * 0.2))
        s = sub.sort("prediction")
        top, bot = s.tail(k), s.head(k)
        rets.append(float(top["realized"].mean() - bot["realized"].mean()))
        w = {t: 1.0 / k for t in top["ticker"].to_list()} | {t: -1.0 / k for t in bot["ticker"].to_list()}
        turns.append(0.5 * sum(abs(w.get(t, 0.0) - prev.get(t, 0.0)) for t in set(w) | set(prev)))
        prev = w
    r, to = np.array(rets), np.array(turns)
    ann = np.sqrt(252 / 21)
    gross = r.mean() / r.std(ddof=1) * ann
    annual_turn = float(to.mean() * (252 / 21))  # ~12 rebalances/year
    net = {}
    for bp in COSTS:
        nr = r - to * (bp / 10000) * 2  # round-trip
        net[bp] = float(nr.mean() / nr.std(ddof=1) * ann)
    return {"n": len(rets), "ic": float(np.nanmean(ics)), "gross": float(gross), "turn": annual_turn, "net": net}


def main() -> None:
    warnings.filterwarnings("ignore")
    con = Console()
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    m = build_feature_matrix(raw, FEATS, "rank", 21).pipe(drop_warmup_rows, FEATS).sort(["ticker", "date"])
    target = m.select("date", "ticker", "y")

    def preds_for(cls, params):
        model = build_model(cls, ModelConfig("x", tuple(FEATS), "y", params))
        p = run_walk_forward(m, model=model, feature_cols=FEATS, target_col="y", experiment_id="x",
                             horizon_days=21, refit_freq_days=9999, embargo_days=33, min_train_days=504,
                             train_start=TRAIN_START, first_refit=FIRST_REFIT)
        return join_with_realized(p, target).filter(pl.col("date") >= OOS)

    con.print("Lasso-6 ..."); lasso = preds_for("LassoCrossSectional", {"cv": 3})
    con.print("Ridge-6 ..."); ridge = preds_for("RidgeCrossSectional", {"cv": 3})
    con.print("momentum_756 ...")
    mom = (m.filter(pl.col("date") >= OOS).select("date", "ticker", pl.col("momentum_756").alias("prediction"))
           .join(target.rename({"y": "realized"}), on=["date", "ticker"], how="left"))

    series = {"Lasso-6": lasso, "Ridge-6": ridge, "mom_756": mom}

    t = Table(title="Net-of-cost: both 21-day rebalance; overlapping vs non-overlapping return series")
    for c in ("model", "return series", "gross IC", "gross Sh", "ann.turn", "net@3", "net@10", "net@20"):
        t.add_column(c, justify=("left" if c in ("model", "return series") else "right"))
    for name, df in series.items():
        d = df.select("date", "ticker", "prediction", "realized")
        sd = compute_turnover_and_costs(d, cost_bps=COSTS, horizon_days=21)
        t.add_row(name, "overlap (all dates)", f"{sd.gross_ic:+.4f}", f"{sd.gross_long_short_sharpe:+.2f}",
                  f"{sd.annual_turnover:.0f}x", *[f"{sd.after_cost_sharpe_by_bp[b]:+.2f}" for b in COSTS])
        r = eval_21d(d)
        t.add_row("", f"non-overlap (n={r['n']})", f"{r['ic']:+.4f}", f"{r['gross']:+.2f}", f"{r['turn']:.0f}x",
                  *[f"{r['net'][b]:+.2f}" for b in COSTS])
        t.add_section()
    con.print(t)


if __name__ == "__main__":
    main()
