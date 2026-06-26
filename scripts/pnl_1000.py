"""Dollar PnL of the regime-confined Lasso-6 long-short book on a $1000 portfolio.

Non-overlapping 21-day rebalance on the 2025+ test slice. Per rebalance the
long-short SPREAD return is r = mean(top-quintile 21d fwd) - mean(bottom-quintile).
Two sizings of "$1000 long-short":
  (A) $1000 capital, dollar-neutral -> $500 long + $500 short -> period return = r/2
  (B) $1000 per leg ($1000 long + $1000 short, $2000 gross) -> period PnL = $1000 * r
Reports gross and net @10bp (round-trip), summed and compounded.

Usage: PYTHONPATH=src .venv/bin/python scripts/pnl_1000.py
"""
from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.pipeline.walk_forward import join_with_realized, run_walk_forward

FEATS = ["momentum_12_1", "momentum_756", "return_1d", "vol_ewm_20", "distance_52w_high", "log_dollar_volume"]
TRAIN_START, FIRST_REFIT, OOS = date(2022, 10, 10), date(2025, 1, 2), date(2025, 1, 1)
BP = 10  # cost per side, basis points
CAP = 1000.0


def main() -> None:
    warnings.filterwarnings("ignore")
    con = Console()
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    m = build_feature_matrix(raw, FEATS, "rank", 21).pipe(drop_warmup_rows, FEATS).sort(["ticker", "date"])
    target = m.select("date", "ticker", "y")
    model = build_model("LassoCrossSectional", ModelConfig("x", tuple(FEATS), "y", {"cv": 5}))
    p = run_walk_forward(m, model=model, feature_cols=FEATS, target_col="y", experiment_id="x",
                         horizon_days=21, refit_freq_days=9999, embargo_days=33, min_train_days=504,
                         train_start=TRAIN_START, first_refit=FIRST_REFIT)
    df = join_with_realized(p, target).filter(pl.col("date") >= OOS).select(
        "date", "ticker", "prediction", "realized")

    dates = sorted(df["date"].unique().to_list())
    rebal = dates[::21]
    prev: dict[str, float] = {}
    rows = []
    for d in rebal:
        sub = df.filter(pl.col("date") == d).drop_nulls(["prediction", "realized"])
        n = sub.height
        if n < 10:
            continue
        k = max(1, round(n * 0.2))
        s = sub.sort("prediction")
        top, bot = s.tail(k), s.head(k)
        r = float(top["realized"].mean() - bot["realized"].mean())
        w = {t: 1.0 / k for t in top["ticker"].to_list()} | {t: -1.0 / k for t in bot["ticker"].to_list()}
        to = 0.5 * sum(abs(w.get(t, 0.0) - prev.get(t, 0.0)) for t in set(w) | set(prev))
        cost = to * (BP / 10000) * 2  # round-trip, as a fraction of gross-exposure-per-leg
        rows.append({"date": d, "r": r, "turn": to, "net": r - cost})
        prev = w

    rr = np.array([x["r"] for x in rows])
    nn = np.array([x["net"] for x in rows])
    n = len(rows)

    def summarize(series: np.ndarray, label: str) -> list[str]:
        # (A) $1000 capital, $500/leg -> period return on capital = r/2
        a_sum = CAP * series.sum() / 2
        a_cmp = CAP * (np.prod(1 + series / 2) - 1)
        # (B) $1000 per leg -> period PnL = $1000 * r
        b_sum = CAP * series.sum()
        return [label, f"{series.mean()*100:+.2f}%", f"{series.sum()*100:+.1f}%",
                f"${a_sum:+,.0f}", f"${a_cmp:+,.0f}", f"${b_sum:+,.0f}"]

    t = Table(title=f"Lasso-6 long-short PnL on $1000 — {n} rebalances, 2025+ ({rebal[0]} to {rebal[-1] if rebal else '?'})")
    for c in ("", "avg/rebal", "sum of spreads", "(A) $500/leg sum", "(A) $500/leg compounded", "(B) $1000/leg sum"):
        t.add_column(c, justify=("left" if c == "" else "right"))
    t.add_row(*summarize(rr, "gross"))
    t.add_row(*summarize(nn, f"net @{BP}bp"))
    con.print(t)
    con.print(f"\nAvg per-rebalance one-sided turnover: {np.mean([x['turn'] for x in rows]):.2f}  "
              f"(annualized ~{np.mean([x['turn'] for x in rows])*252/21:.0f}x)")
    con.print("(A) = $1000 total capital, dollar-neutral ($500 long + $500 short).")
    con.print("(B) = $1000 on each leg ($1000 long + $1000 short, $2000 gross exposure).")


if __name__ == "__main__":
    main()
