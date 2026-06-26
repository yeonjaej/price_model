"""Survivorship-bias ablation on the headline models: Lasso-6, Ridge-6, tuned LightGBM.

All arms draw from the SAME price source (sp500_pit, pit_filter=False) and use each
model's headline recipe, regime-confined (train 2022-10-10 -> 2024-11-30 via
train_start + first_refit, embargo 33, single refit), tested on 2025+. The three
arms differ ONLY in which cross-section is used on each date:

  1. PIT-on (honest = headline)  — per-date S&P 500 membership (filter_panel_to_pit).
  2. Survivor-snapshot (biased)  — the CURRENT roster (members_on_date at the last
                                   date), used across all history.
  3. All-available               — every name whenever it has data (no filter).

Models (each on its own headline panel; HPs held fixed across arms so the only
thing varying is the universe — re-tuning per arm would conflate HP-selection
with survivorship):
  - Lasso-6 / Ridge-6  : 6-feature rank panel, CV-selected regularization.
  - LightGBM (rank-9)  : 9-feature rank panel, HPs from the held-out Optuna config
                         (lightgbm_rank9_h21_hp_pre20241231.yaml).

Reports gross IC / t / Sharpe + 20 bp net on the 2025+ test slice; survivorship
inflation = (survivor OOS) − (PIT-on OOS), per model, for IC and Sharpe.

Usage: PYTHONPATH=src .venv/bin/python scripts/survivorship_ablation_headline.py
"""

from __future__ import annotations

import warnings
from datetime import date

import polars as pl
import yaml
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit, members_on_date
from price_model.eval.turnover import compute_turnover_and_costs
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.pipeline.walk_forward import join_with_realized, run_walk_forward

FEATS6 = ["momentum_12_1", "momentum_756", "return_1d",
          "vol_ewm_20", "distance_52w_high", "log_dollar_volume"]
TRAIN_START, FIRST_REFIT, OOS_START = date(2022, 10, 10), date(2025, 1, 2), date(2025, 1, 1)


def lgbm_spec() -> tuple:
    cfg = yaml.safe_load(open("config/experiments/lightgbm_rank9_h21_hp_pre20241231.yaml"))
    params = next(m["params"] for m in cfg["models"] if m["class"] == "LightGBMModel")
    return ("LightGBM rank-9 (tuned)", list(cfg["features"]), cfg.get("normalize_kind", "rank"),
            "LightGBMModel", params)


def run_cell(spec: tuple, raw: pl.DataFrame) -> dict:
    name, feats, norm, cls, params = spec
    matrix = build_feature_matrix(raw, feats, norm, 21).pipe(drop_warmup_rows, feats).sort(["ticker", "date"])
    target = matrix.select("date", "ticker", "y")
    model = build_model(cls, ModelConfig(model_id="surv", feature_cols=tuple(feats), params=params))
    preds = run_walk_forward(
        matrix, model=model, feature_cols=feats, target_col="y",
        experiment_id="surv_ablation", horizon_days=21,
        refit_freq_days=9999, embargo_days=33, min_train_days=504,
        train_start=TRAIN_START, first_refit=FIRST_REFIT,
    )
    joined = join_with_realized(preds, target).filter(pl.col("date") >= OOS_START)
    s = compute_turnover_and_costs(
        joined.select("date", "ticker", "prediction", "realized"),
        cost_bps=(3, 10, 20), horizon_days=21,
    )
    return {"tickers": joined["ticker"].n_unique(), "ic": s.gross_ic, "t": s.gross_ic_t_stat,
            "sharpe": s.gross_long_short_sharpe, "turn": s.annual_turnover,
            "net20": s.after_cost_sharpe_by_bp[20]}


def main() -> None:
    warnings.filterwarnings("ignore")
    con = Console()
    con.print("Loading sp500_pit price panel (no membership filter)...")
    raw = load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False)
    last = raw["date"].max()
    current = members_on_date(last) & set(raw["ticker"].unique().to_list())
    arms = {
        "PIT-on (honest)": filter_panel_to_pit(raw),
        "survivor-snapshot": raw.filter(pl.col("ticker").is_in(list(current))),
        "all-available": raw,
    }
    models = [
        ("Lasso-6", FEATS6, "rank", "LassoCrossSectional", {"cv": 5}),
        ("Ridge-6", FEATS6, "rank", "RidgeCrossSectional", {"cv": 5}),
        lgbm_spec(),
    ]

    rows = {}
    for spec in models:
        mname = spec[0]
        for aname, araw in arms.items():
            con.print(f"  {mname}  ×  {aname} ...")
            rows[(mname, aname)] = run_cell(spec, araw)

    t = Table(title="Survivorship ablation across models (regime-confined, test 2025+)")
    for c in ("model", "arm", "tickers", "OOS IC", "t", "gross Sh", "ann.turn", "net@20"):
        t.add_column(c, justify=("left" if c in ("model", "arm") else "right"))
    for spec in models:
        mname = spec[0]
        for aname in arms:
            r = rows[(mname, aname)]
            t.add_row(mname, aname, str(r["tickers"]), f"{r['ic']:+.4f}", f"{r['t']:+.2f}",
                      f"{r['sharpe']:+.2f}", f"{r['turn']:.0f}x", f"{r['net20']:+.2f}")
        t.add_section()
    con.print(t)

    con.print("\n[bold]Survivorship inflation (survivor − PIT-on):[/bold]")
    for spec in models:
        mname = spec[0]
        h, s = rows[(mname, "PIT-on (honest)")], rows[(mname, "survivor-snapshot")]
        con.print(f"  {mname:24s}  IC {s['ic'] - h['ic']:+.4f}  "
                  f"({(s['ic'] - h['ic']) / h['ic'] * 100:+.0f}%)   "
                  f"gross Sharpe {s['sharpe'] - h['sharpe']:+.2f}")


if __name__ == "__main__":
    main()
