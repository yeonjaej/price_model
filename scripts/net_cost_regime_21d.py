"""Honest 21-day headline for the regime-confined models (framing B).

The linear family (OLS-6 / Ridge-6 / Lasso-6) is fit FRESH on the curated-6 panel
with the temporal, purged, IC-scored forward-chain CV (cv=3) — the store holds
stale pre-IC-grader predictions and no OLS. On this panel Ridge/Lasso reduce to OLS
(CV drives α→0), so all three are shown together. Trees + momentum are read from the
store (their grader is unchanged).

IC POINT ESTIMATE keeps all daily dates (best use of data), but its t-stat is
reported BOTH naively and Newey-West HAC-corrected (lag 21 = the overlap length)
so the standard error accounts for the autocorrelation that the 21-day overlapping
windows induce -- the naive daily t (~10) badly overstates significance.

PORTFOLIO metrics (gross Sharpe, turnover, net) are on the deployable
NON-OVERLAPPING 21-day schedule (dates[::21], ~17 books each held to expiry):

  IC (all)     : mean per-DATE Spearman(pred, realized) over all ~348 dates.
  t naive      : mean / (std/sqrt(n_dates))  -- overlap-inflated, for reference.
  t HAC(21)    : Newey-West HAC t with Bartlett lag 21 -- the honest headline t.
  gross Sharpe : mean/std of the ~17 non-overlapping long-short spreads * sqrt(252/21).
  turnover     : one-sided book replacement per rebalance, annualized * (252/21).
  net @b bp    : spread - turnover * b/1e4 * 2 (round-trip), same annualization.

Usage: PYTHONPATH=src .venv/bin/python scripts/net_cost_regime_21d.py
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
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.features.targets import add_forward_excess_return
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.pipeline.walk_forward import join_with_realized, run_walk_forward
from price_model.serving.store import PredictionStore

warnings.filterwarnings("ignore")

# Linear family is fit FRESH (the store holds stale, pre-IC-grader predictions and
# no OLS at all). All three use the curated-6 panel + the temporal, purged,
# IC-scored forward-chain CV (cv=3) — OLS has no penalty, so Ridge/Lasso reduce to
# it in-regime; they're shown together to make that explicit.
FEATS6 = ["momentum_12_1", "momentum_756", "return_1d", "vol_ewm_20", "distance_52w_high", "log_dollar_volume"]
LINEAR_FRESH = [
    ("OLS-6 (rank)",   "OLSCrossSectional"),
    ("Ridge-6 (rank)", "RidgeCrossSectional"),
    ("Lasso-6 (rank)", "LassoCrossSectional"),
]
# Trees + momentum come from the store (their grader is unchanged, predictions valid).
STORE_MODELS = [
    ("LightGBM (rank9)", "lightgbm_rank9_h21_hp_pre20241231"),
    ("CatBoost (rank9)", "catboost_rank9_h21_hp_pre20241231"),
    ("XGBoost (rank9)",  "xgboost_rank9_h21_hp_pre20241231"),
    ("LightGBM (eng14)", "lightgbm_kaggle_v3_curated_h21_hp_pre20241231"),
    ("CatBoost (eng14)", "catboost_v3_curated_h21_hp_pre20241231"),
    ("XGBoost (eng14)",  "xgboost_v3_curated_h21_hp_pre20241231"),
    ("mom_756",          "mom_756_factor_h21"),
]
TRAIN_START, FIRST_REFIT, OOS_START = date(2022, 10, 10), date(2025, 1, 2), date(2025, 1, 1)
COSTS = (3, 10, 20)
ANN = np.sqrt(252 / 21)


def hac_t(x: np.ndarray, lag: int) -> float:
    """Newey-West HAC t-stat for the mean of x, Bartlett kernel, given lag."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    d = x - x.mean()
    var = (d @ d) / n  # gamma_0
    for k in range(1, min(lag, n - 1) + 1):
        gk = (d[k:] @ d[:-k]) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    se = np.sqrt(var / n) if var > 0 else float("nan")
    return float(x.mean() / se) if se and se > 0 else float("nan")


def stats_21d(j: pl.DataFrame) -> dict:
    """j: (date, ticker, prediction, realized) on the OOS slice."""
    j = j.drop_nulls(["prediction", "realized"])
    dates = sorted(j["date"].unique().to_list())

    # --- IC point estimate + honest t: ALL daily dates ---
    daily_ic = []
    for d in dates:
        sub = j.filter(pl.col("date") == d)
        if sub.height < 10:
            continue
        ic, _ = spearmanr(sub["prediction"].to_numpy(), sub["realized"].to_numpy())
        daily_ic.append(float(ic))
    ic_arr = np.array(daily_ic)
    ic_mean = float(np.nanmean(ic_arr))
    t_naive = ic_mean * np.sqrt(len(ic_arr)) / ic_arr.std(ddof=1)
    t_hac = hac_t(ic_arr, lag=21)

    # --- portfolio: NON-OVERLAPPING 21-day rebalance ---
    rebal = dates[::21]
    prev: dict[str, float] = {}
    rets, turns = [], []
    for d in rebal:
        sub = j.filter(pl.col("date") == d)
        n = sub.height
        if n < 10:
            continue
        k = max(1, round(n * 0.2))
        s = sub.sort("prediction")
        top, bot = s.tail(k), s.head(k)
        rets.append(float(top["realized"].mean() - bot["realized"].mean()))
        w = {t: 1.0 / k for t in top["ticker"].to_list()} | {t: -1.0 / k for t in bot["ticker"].to_list()}
        turns.append(0.5 * sum(abs(w.get(t, 0.0) - prev.get(t, 0.0)) for t in set(w) | set(prev)))
        prev = w
    r, to = np.array(rets), np.array(turns)
    gross = r.mean() / r.std(ddof=1) * ANN
    net = {b: float((r - to * (b / 10000) * 2).mean() / (r - to * (b / 10000) * 2).std(ddof=1) * ANN)
           for b in COSTS}
    return {"n_dates": len(ic_arr), "ic": ic_mean, "t_naive": float(t_naive), "t_hac": t_hac,
            "n_rebal": len(rets), "gross": float(gross), "turn": float(to.mean() * 252 / 21), "net": net}


def _add_row(t: Table, label: str, j: pl.DataFrame) -> None:
    s = stats_21d(j.select("date", "ticker", "prediction", "realized"))
    t.add_row(label, f"{s['ic']:+.4f}", f"{s['t_naive']:+.2f}", f"{s['t_hac']:+.2f}", str(s["n_rebal"]),
              f"{s['gross']:+.2f}", f"{s['turn']:.0f}x", *[f"{s['net'][b]:+.2f}" for b in COSTS])


def main() -> None:
    con = Console()
    con.print("Building realized 21-day excess-return panel...")
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    tgt = (add_forward_excess_return(raw.sort(["ticker", "date"]), horizon_days=21, target_col="y")
           .select("date", "ticker", pl.col("y").alias("realized")))

    # Curated-6 matrix for the fresh linear fits.
    m = build_feature_matrix(raw, FEATS6, "rank", 21).pipe(drop_warmup_rows, FEATS6).sort(["ticker", "date"])
    target = m.select("date", "ticker", "y")

    t = Table(title="Regime-confined honest 21-day headline (test 2025+, top_frac=0.2): IC all-dates + HAC t; portfolio non-overlap")
    cols = ("model", "IC (all)", "t naive", "t HAC", "n_reb", "gross Sh", "ann.turn", "net@3", "net@10", "net@20")
    for c in cols:
        t.add_column(c, justify=("left" if c == "model" else "right"))

    # --- linear family, fit fresh (OLS / Ridge / Lasso on curated-6) ---
    for label, cls in LINEAR_FRESH:
        con.print(f"fitting {label} (fresh, temporal IC CV) ...")
        model = build_model(cls, ModelConfig(label, tuple(FEATS6), "y", {"cv": 3}))
        p = run_walk_forward(m, model=model, feature_cols=FEATS6, target_col="y", experiment_id="x",
                             horizon_days=21, refit_freq_days=9999, embargo_days=33, min_train_days=504,
                             train_start=TRAIN_START, first_refit=FIRST_REFIT)
        j = join_with_realized(p, target).filter(pl.col("date") >= OOS_START)
        _add_row(t, label, j)
    t.add_section()

    # --- trees + momentum, from the store (grader unchanged) ---
    store = PredictionStore(read_only=True)
    for label, mid in STORE_MODELS:
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
        _add_row(t, label, j)

    con.print(t)
    con.print("\n[dim]Linear family fit fresh (curated-6, temporal IC CV cv=3); trees + momentum from store. "
              "IC & t use all ~348 daily dates; t HAC = Newey-West Bartlett lag 21. "
              "gross Sh / turn / net use the ~17 non-overlapping 21-day rebalances.[/dim]")


if __name__ == "__main__":
    main()
