"""Long-only top-20% book, 21-day rebalance, on a Robinhood-style retail account.

The project target is CROSS-SECTIONAL EXCESS return (raw - equal-weight universe
mean), so a long-only book is NOT market-neutral: your account return is RAW =
market(beta) + alpha. This reconstructs raw 21-day forward returns from adj_close
and decomposes the long-only top-quintile book into:

  book      : equal-weight raw return of the top-20%-by-prediction names (what you earn)
  benchmark : equal-weight raw return of the FULL universe (~equal-weight S&P / VOO-ish)
  alpha     : book - benchmark  (the only part the model actually adds)
  turnover  : one-sided fraction of the long book replaced per 21-day rebalance

Net applies a per-side retail cost (spread + PFOF), round-trip, to the traded fraction.
PnL shown on a $1000 long-only account (fully invested, fractional shares).

Usage: PYTHONPATH=src .venv/bin/python scripts/long_only_robinhood.py
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
CAP = 1000.0
COST_SIDES_BPS = {"base 5bp/side": 5.0, "conservative 12bp/side": 12.0}  # per side; round-trip = x2


def main() -> None:
    warnings.filterwarnings("ignore")
    con = Console()
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    # raw 21-day forward SIMPLE return per ticker (what an account actually earns)
    fwd = (raw.sort(["ticker", "date"])
           .with_columns((pl.col("adj_close").shift(-21).over("ticker") / pl.col("adj_close") - 1.0)
                         .alias("raw_fwd21"))
           .select("date", "ticker", "raw_fwd21"))

    m = build_feature_matrix(raw, FEATS, "rank", 21).pipe(drop_warmup_rows, FEATS).sort(["ticker", "date"])
    target = m.select("date", "ticker", "y")
    model = build_model("LassoCrossSectional", ModelConfig("x", tuple(FEATS), "y", {"cv": 5}))
    p = run_walk_forward(m, model=model, feature_cols=FEATS, target_col="y", experiment_id="x",
                         horizon_days=21, refit_freq_days=9999, embargo_days=33, min_train_days=504,
                         train_start=TRAIN_START, first_refit=FIRST_REFIT)
    df = (join_with_realized(p, target).filter(pl.col("date") >= OOS)
          .join(fwd, on=["date", "ticker"], how="left")
          .select("date", "ticker", "prediction", "raw_fwd21").drop_nulls("raw_fwd21"))

    dates = sorted(df["date"].unique().to_list())
    rebal = dates[::21]
    prev: set[str] = set()
    rows = []
    for d in rebal:
        sub = df.filter(pl.col("date") == d)
        n = sub.height
        if n < 10:
            continue
        k = max(1, round(n * 0.2))
        top = sub.sort("prediction").tail(k)
        names = set(top["ticker"].to_list())
        book = float(top["raw_fwd21"].mean())          # account return (raw)
        bench = float(sub["raw_fwd21"].mean())          # equal-weight universe (beta)
        turn = len(names - prev) / k if prev else 1.0   # one-sided: fraction newly bought
        rows.append({"book": book, "bench": bench, "turn": turn})
        prev = names

    bk = np.array([r["book"] for r in rows])
    bn = np.array([r["bench"] for r in rows])
    tn = np.array([r["turn"] for r in rows])
    n = len(rows)
    ann_turn = tn.mean() * 252 / 21

    def comp(series):  # compounded $ PnL on CAP
        return CAP * (np.prod(1 + series) - 1)

    con.print(f"[bold]Long-only top-20%, 21-day rebalance, {n} rebalances 2025+[/bold]")
    con.print(f"  avg per-rebalance one-sided turnover {tn.mean():.2f}  (~{ann_turn:.1f}x/yr)\n")

    t = Table(title=f"$1000 long-only — compounded PnL over {n} rebalances")
    for c in ("series", "avg/rebal", "cum return", "$ PnL on $1000"):
        t.add_column(c, justify=("left" if c == "series" else "right"))
    t.add_row("book (top-20%, gross)", f"{bk.mean()*100:+.2f}%", f"{(np.prod(1+bk)-1)*100:+.1f}%", f"${comp(bk):+,.0f}")
    t.add_row("benchmark (EW universe)", f"{bn.mean()*100:+.2f}%", f"{(np.prod(1+bn)-1)*100:+.1f}%", f"${comp(bn):+,.0f}")
    t.add_row("alpha (book - bench, gross)", f"{(bk-bn).mean()*100:+.2f}%", "", f"${comp(bk)-comp(bn):+,.0f}")
    t.add_section()
    for label, bps in COST_SIDES_BPS.items():
        net = bk - tn * (bps / 10000) * 2  # round-trip on traded fraction
        drag = comp(bk) - comp(net)
        t.add_row(f"book NET, {label}", f"{net.mean()*100:+.2f}%", f"{(np.prod(1+net)-1)*100:+.1f}%", f"${comp(net):+,.0f}")
        t.add_row(f"   alpha net of cost vs bench", "", "", f"${comp(net)-comp(bn):+,.0f}  (cost drag ${drag:,.0f})")
    con.print(t)
    con.print("\nbeta share of gross book PnL: "
              f"{comp(bn)/comp(bk)*100:.0f}%  — i.e. most of the dollars are just 'long the market'.")


if __name__ == "__main__":
    main()
