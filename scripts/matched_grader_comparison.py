"""Matched-grader model comparison, confined to the bullish regime.

Motivation
----------
The README headline ("L1 beats every ML variant") is confounded by an
HP-selection asymmetry: the Lasso's alpha is chosen by a non-temporal,
MSE-scored KFold (lenient), while the tree models' HPs are chosen by a purged,
embargoed, IC-scored walk-forward (strict). This script removes the confound by
holding the GRADER fixed across model classes and confining everything to one
regime (2022-10-10 -> present), so the comparison isolates model class.

Split (all in-regime; warmup satisfied by pre-2022 prices):
  TRAIN   2022-10-10 -> 2024-11-29   (539 trading dates)
  EMBARGO 2024-12-02 -> 2024-12-31   (21 dates, purged)
  TEST    2025-01-02 -> 2026-05-22   (348 dates, held out; == headline OOS slice)

Inner selection (on TRAIN only): purged forward-chain folds (de Prado), scored
by mean per-date Spearman IC -- the same _purged_walk_forward_folds the trees
use via optuna_sweep.py.

Arms:
  A. Lasso, MSE-alpha via LassoCV(cv=5)        -- current lenient grader (reference)
  B. Lasso, IC-alpha via purged forward-chain  -- matched grader
  C. LightGBM, default HPs (no inner CV)        -- clean tree baseline
  D. LightGBM, IC-HPs via purged forward-chain  -- matched grader

Usage:
    PYTHONPATH=src .venv/bin/python scripts/matched_grader_comparison.py
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import Lasso

from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit
from price_model.eval.metrics import _per_date_ic, summarize
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.models.boosting import DEFAULT_PARAMS

FEATS = [
    "momentum_12_1", "momentum_756", "return_1d",
    "vol_ewm_20", "idio_vol_20", "max_return_21d",
    "distance_52w_high", "log_dollar_volume", "beta_60",
]
REGIME_START = date(2022, 10, 10)
TRAIN_END = date(2024, 11, 29)
TEST_START = date(2025, 1, 2)
EMBARGO = 21  # trading days, purge between inner folds


def purged_folds(dates: list, n_folds: int = 4, embargo: int = EMBARGO):
    n = len(dates); chunk = n // n_folds; out = []
    for i in range(1, n_folds):
        vs = i * chunk; ve = (i + 1) * chunk if i < n_folds - 1 else n
        val = dates[vs:ve]; tr = dates[: max(0, vs - embargo)]
        if len(tr) >= embargo and len(val) >= 5:
            out.append((tr, val))
    return out


def mean_fold_ic(pred, y, dts) -> float:
    df = pl.DataFrame({"date": dts, "prediction": pred, "realized": y})
    icdf = _per_date_ic(df)
    return float(icdf["ic"].mean()) if icdf.height else float("nan")


def test_metrics(pred_df: pl.DataFrame, target: pl.DataFrame) -> dict:
    j = pred_df.join(target.rename({"y": "realized"}), on=["date", "ticker"], how="left")
    s = summarize(j.select("date", "ticker", "prediction", "realized"), horizon_days=21)
    return {"ic": s.information_coefficient, "t": s.ic_t_stat, "sharpe": s.long_short_sharpe}


def main() -> None:
    console = Console()
    console.print("Building rank-normalized 9-feature matrix (h=21, sp500_pit + PIT)...")
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    m = build_feature_matrix(raw, FEATS, "rank", 21).pipe(drop_warmup_rows, FEATS).sort(["ticker", "date"])

    train = m.filter((pl.col("date") >= REGIME_START) & (pl.col("date") <= TRAIN_END)).drop_nulls(["y", *FEATS])
    test = m.filter(pl.col("date") >= TEST_START).drop_nulls(FEATS)
    target = m.select("date", "ticker", "y")
    tr_dates = sorted(train["date"].unique().to_list())
    folds = purged_folds(tr_dates)
    console.print(
        f"train {tr_dates[0]}..{tr_dates[-1]} ({len(tr_dates)} dates, {train.height:,} rows) | "
        f"test {test['date'].min()}..{test['date'].max()} ({test['date'].n_unique()} dates) | "
        f"{len(folds)} purged forward-chain folds\n"
    )

    Xtr = train.select(FEATS).to_numpy(); ytr = train["y"].to_numpy()
    Xte = test.select(FEATS).fill_null(0.0).to_numpy()
    results = []

    # --- A. Lasso, MSE-alpha via LassoCV(cv=5) ---
    console.print("A. Lasso MSE-alpha (LassoCV cv=5)...")
    a = build_model("LassoCrossSectional", ModelConfig("A", tuple(FEATS), "y", {"cv": 5}))
    a.fit(train)
    results.append(("Lasso", "MSE-alpha / KFold cv=5 (lenient)", a.selected_alpha(), test_metrics(a.predict(test), target)))

    # --- B. Lasso, IC-alpha via purged forward-chain ---
    console.print("B. Lasso IC-alpha (purged forward-chain)...")
    alpha_grid = np.logspace(-5, -2.3, 25)
    fold_arrays = [
        (train.filter(pl.col("date").is_in(trd)), train.filter(pl.col("date").is_in(vad)))
        for trd, vad in folds
    ]
    best_a, best_ic = None, -np.inf
    for al in alpha_grid:
        ics = []
        for ftr, fva in fold_arrays:
            lm = Lasso(alpha=al, max_iter=5000, tol=1e-4).fit(ftr.select(FEATS).to_numpy(), ftr["y"].to_numpy())
            p = fva.select(FEATS).fill_null(0.0).to_numpy() @ lm.coef_ + lm.intercept_
            ics.append(mean_fold_ic(p, fva["y"].to_numpy(), fva["date"].to_numpy()))
        mic = float(np.nanmean(ics))
        if mic > best_ic:
            best_ic, best_a = mic, al
    lm = Lasso(alpha=best_a, max_iter=5000, tol=1e-4).fit(Xtr, ytr)
    predB = test.select("date", "ticker").with_columns(pl.Series("prediction", Xte @ lm.coef_ + lm.intercept_))
    console.print(f"   chosen alpha={best_a:.2e} (CV fold-IC={best_ic:+.4f}); nonzero coefs={int(np.count_nonzero(lm.coef_))}/9")
    results.append(("Lasso", "IC-alpha / purged fwd-chain (matched)", best_a, test_metrics(predB, target)))

    # --- C. LightGBM, default HPs ---
    console.print("C. LightGBM default HPs...")
    c = build_model("LightGBMModel", ModelConfig("C", tuple(FEATS), "y", dict(DEFAULT_PARAMS)))
    c.fit(train)
    results.append(("LightGBM", "default HPs (no inner CV)", None, test_metrics(c.predict(test), target)))

    # --- D. LightGBM, IC-HPs via purged forward-chain (small grid) ---
    console.print("D. LightGBM IC-HPs (purged forward-chain, small grid)...")
    grid = [
        {"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": 0.05, "n_estimators": 400}
        for nl in (15, 31, 63) for md in (100, 300)
    ]
    best_hp, best_ic_d = None, -np.inf
    for hp in grid:
        params = {**DEFAULT_PARAMS, **hp}
        ics = []
        for ftr, fva in fold_arrays:
            mdl = build_model("LightGBMModel", ModelConfig("d", tuple(FEATS), "y", params))
            mdl.fit(ftr)
            p = mdl.predict(fva)["prediction"].to_numpy()
            ics.append(mean_fold_ic(p, fva["y"].to_numpy(), fva["date"].to_numpy()))
        mic = float(np.nanmean(ics))
        if mic > best_ic_d:
            best_ic_d, best_hp = mic, hp
    d = build_model("LightGBMModel", ModelConfig("D", tuple(FEATS), "y", {**DEFAULT_PARAMS, **best_hp, "n_estimators": 500}))
    d.fit(train)
    console.print(f"   chosen HPs={best_hp} (CV fold-IC={best_ic_d:+.4f})")
    results.append(("LightGBM", "IC-HPs / purged fwd-chain (matched)", None, test_metrics(d.predict(test), target)))

    # --- Report ---
    table = Table(title="Matched-grader comparison — bullish regime, TEST 2025-01-02..2026-05-22")
    for col in ("model", "HP selection", "alpha", "test IC", "t", "Sharpe"):
        table.add_column(col, justify="right" if col in ("test IC", "t", "Sharpe") else "left")
    for model, sel, al, mt in results:
        table.add_row(model, sel, f"{al:.2e}" if al else "—",
                      f"{mt['ic']:+.4f}" if mt['ic'] == mt['ic'] else "null",
                      f"{mt['t']:+.2f}" if mt['t'] == mt['t'] else "—",
                      f"{mt['sharpe']:+.2f}")
    console.print(table)


if __name__ == "__main__":
    main()
