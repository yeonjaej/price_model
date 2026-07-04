"""Net-of-cost (3/10/20 bp) for the regime-confined headline models.

The shared `compare_net_of_cost.py` intersects ~37 accumulated store models and
dropped most of the new regime-confined runs. This pulls each headline model
directly from the store (deduped to its latest generated_at), joins to the
realized 21-day excess return, and computes gross + 3/10/20 bp net via
eval/turnover.compute_turnover_and_costs.

Usage: PYTHONPATH=src .venv/bin/python scripts/net_cost_regime.py
"""
from __future__ import annotations

import warnings
from datetime import date

import polars as pl
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit
from price_model.eval.turnover import compute_turnover_and_costs
from price_model.features.targets import add_forward_excess_return
from price_model.serving.store import PredictionStore

warnings.filterwarnings("ignore")

MODELS = [
    ("Lasso-6 (rank)",   "lasso_curated6_pit_h21"),
    ("Ridge-6 (rank)",   "ridge_curated6_pit_h21"),
    ("LightGBM (rank9)", "lightgbm_rank9_h21_hp_pre20241231"),
    ("CatBoost (rank9)", "catboost_rank9_h21_hp_pre20241231"),
    ("XGBoost (rank9)",  "xgboost_rank9_h21_hp_pre20241231"),
    ("LightGBM (eng14)", "lightgbm_kaggle_v3_curated_h21_hp_pre20241231"),
    ("CatBoost (eng14)", "catboost_v3_curated_h21_hp_pre20241231"),
    ("XGBoost (eng14)",  "xgboost_v3_curated_h21_hp_pre20241231"),
    ("mom_756",          "mom_756_factor_h21"),
]
OOS_START = date(2025, 1, 1)


def main() -> None:
    con = Console()
    con.print("Building realized 21-day excess-return panel...")
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    tgt = (
        add_forward_excess_return(raw.sort(["ticker", "date"]), horizon_days=21, target_col="y")
        .select("date", "ticker", pl.col("y").alias("realized"))
    )

    store = PredictionStore(read_only=True)
    rows = []
    for label, mid in MODELS:
        df = store.query(
            f"""
            WITH d AS (
                SELECT prediction_date AS date, ticker, prediction,
                       ROW_NUMBER() OVER (PARTITION BY prediction_date, ticker
                                          ORDER BY generated_at DESC) AS rn
                FROM predictions WHERE model_id = '{mid}'
            )
            SELECT date, ticker, prediction FROM d WHERE rn = 1
            """
        )
        if df.height == 0:
            con.print(f"[red]no predictions in store for {mid}")
            continue
        j = df.join(tgt, on=["date", "ticker"], how="left").filter(pl.col("date") >= OOS_START)
        s = compute_turnover_and_costs(
            j.select("date", "ticker", "prediction", "realized"),
            cost_bps=(3, 10, 20), horizon_days=21,
        )
        rows.append((label, s))

    t = Table(title="Regime-confined net-of-cost (test 2025+, 21-day, top_frac=0.2)")
    for c in ("model", "n_dates", "gross IC", "t", "gross Sh", "ann.turn", "net@3", "net@10", "net@20"):
        t.add_column(c, justify=("left" if c == "model" else "right"))
    for label, s in rows:
        t.add_row(
            label, str(s.n_dates), f"{s.gross_ic:+.4f}", f"{s.gross_ic_t_stat:+.2f}",
            f"{s.gross_long_short_sharpe:+.2f}", f"{s.annual_turnover:.0f}x",
            f"{s.after_cost_sharpe_by_bp[3]:+.2f}", f"{s.after_cost_sharpe_by_bp[10]:+.2f}",
            f"{s.after_cost_sharpe_by_bp[20]:+.2f}",
        )
    con.print(t)


if __name__ == "__main__":
    main()
