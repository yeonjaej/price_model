"""Model x Panel spectrum comparison, matched grader, bullish regime.

Spectrum: Lasso (aggressive sparsity, needs clean panel) -> Ridge (shrinks but
keeps all, tolerates collinearity) -> LightGBM (handles complexity/interactions).

Grid (all under the SAME purged forward-chain, IC-scored selection; same
bullish-regime split train 2022-10-10..2024-11-29 / test 2025-01-02..2026-05-22):

  Panel A: 9-feature  RANK   (curated-ish anomaly panel; the headline Lasso panel)
  Panel B: 14-feature ZSCORE (v3 engineered/regime/sector panel; the tree panel)

For the linear cells we also report the volatility-cluster coefficients to show
L1 cancellation vs L2 weight-sharing on the 0.79-correlated vol_ewm/idio_vol pair.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/spectrum_comparison.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from datetime import date
from sklearn.linear_model import Lasso, Ridge

from price_model.data.loaders import load_panel
from price_model.data.membership import filter_panel_to_pit
from price_model.eval.metrics import _per_date_ic, summarize
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.models.boosting import DEFAULT_PARAMS

FEATS_A = ["momentum_12_1", "momentum_756", "return_1d", "vol_ewm_20", "idio_vol_20",
           "max_return_21d", "distance_52w_high", "log_dollar_volume", "beta_60"]
FEATS_B = ["abnormal_volume", "distance_52w_high", "mom60_minus_dist_ma200", "return_5d",
           "max_return_21d", "momentum_12_1", "rsi_14", "log_dollar_volume", "vol_ewm_20",
           "momentum_60_rank", "cs_return_dispersion_20", "sector_relative_return_5d",
           "momentum_504", "beta_60"]
VOL_CLUSTER = ["vol_ewm_20", "idio_vol_20", "max_return_21d", "beta_60"]
TRAIN_LO, TRAIN_HI, TEST_LO = date(2022, 10, 10), date(2024, 11, 29), date(2025, 1, 2)


def purged_folds(dates, k=4, embargo=21):
    n = len(dates); chunk = n // k
    return [(dates[: max(0, i * chunk - embargo)], dates[i * chunk:((i + 1) * chunk if i < k - 1 else n)])
            for i in range(1, k)]


def fold_ic(pred, y, d):
    icdf = _per_date_ic(pl.DataFrame({"date": d, "prediction": pred, "realized": y}))
    return float(icdf["ic"].mean()) if icdf.height else float("nan")


def test_ic(pred_df, target):
    j = pred_df.join(target.rename({"y": "realized"}), on=["date", "ticker"], how="left")
    s = summarize(j.select("date", "ticker", "prediction", "realized"), horizon_days=21)
    return s.information_coefficient, s.ic_t_stat, s.long_short_sharpe


def linear_search(Maker, grid, feats, fa, train, test):
    best_a, best_ic = None, -np.inf
    for al in grid:
        ics = []
        for ftr, fva in fa:
            lm = Maker(al).fit(ftr.select(feats).to_numpy(), ftr["y"].to_numpy())
            p = fva.select(feats).fill_null(0.0).to_numpy() @ lm.coef_ + lm.intercept_
            ics.append(fold_ic(p, fva["y"].to_numpy(), fva["date"].to_numpy()))
        mic = float(np.nanmean(ics))
        if mic > best_ic:
            best_ic, best_a = mic, al
    lm = Maker(best_a).fit(train.select(feats).to_numpy(), train["y"].to_numpy())
    Xte = test.select(feats).fill_null(0.0).to_numpy()
    pred = test.select("date", "ticker").with_columns(pl.Series("prediction", Xte @ lm.coef_ + lm.intercept_))
    coefs = dict(zip(feats, lm.coef_))
    return best_a, best_ic, pred, coefs


def lgbm_search(feats, fa, train, test):
    grid = [{"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": 0.05, "n_estimators": 400}
            for nl in (15, 31, 63) for md in (100, 300)]
    best_hp, best_ic = None, -np.inf
    for hp in grid:
        params = {**DEFAULT_PARAMS, **hp}; ics = []
        for ftr, fva in fa:
            mdl = build_model("LightGBMModel", ModelConfig("d", tuple(feats), "y", params)); mdl.fit(ftr)
            p = mdl.predict(fva)["prediction"].to_numpy()
            ics.append(fold_ic(p, fva["y"].to_numpy(), fva["date"].to_numpy()))
        mic = float(np.nanmean(ics))
        if mic > best_ic:
            best_ic, best_hp = mic, hp
    mdl = build_model("LightGBMModel", ModelConfig("D", tuple(feats), "y", {**DEFAULT_PARAMS, **best_hp, "n_estimators": 500}))
    mdl.fit(train)
    return best_hp, best_ic, mdl.predict(test)


def cancel_ratio(coefs):
    v = np.array([coefs.get(f, 0.0) for f in VOL_CLUSTER])
    s = np.abs(v).sum()
    return abs(v.sum()) / s if s > 1e-12 else 1.0


def main():
    raw = filter_panel_to_pit(load_panel(universe="sp500_pit", start="2017-01-01", pit_filter=False))
    rows = []
    coef_report = []
    for pname, feats, norm in [("A_9feat_rank", FEATS_A, "rank"), ("B_14feat_zscore", FEATS_B, "zscore")]:
        print(f"Building panel {pname} ({len(feats)} feats, {norm})...", flush=True)
        m = build_feature_matrix(raw, feats, norm, 21).pipe(drop_warmup_rows, feats).sort(["ticker", "date"])
        train = m.filter((pl.col("date") >= TRAIN_LO) & (pl.col("date") <= TRAIN_HI)).drop_nulls(["y", *feats])
        test = m.filter(pl.col("date") >= TEST_LO).drop_nulls(feats)
        target = m.select("date", "ticker", "y")
        trd = sorted(train["date"].unique().to_list())
        folds = purged_folds(trd)
        fa = [(train.filter(pl.col("date").is_in(t)), train.filter(pl.col("date").is_in(v))) for t, v in folds]

        print(f"  Lasso on {pname}...", flush=True)
        a, cvic, pred, coefs = linear_search(lambda al: Lasso(al, max_iter=5000, tol=1e-4),
                                             np.logspace(-5, -2.3, 25), feats, fa, train, test)
        ic, t, sh = test_ic(pred, target)
        rows.append((pname, "Lasso", f"a={a:.1e}", ic, t, sh))
        coef_report.append((pname, "Lasso", coefs, cancel_ratio(coefs)))

        print(f"  Ridge on {pname}...", flush=True)
        a, cvic, pred, coefs = linear_search(lambda al: Ridge(al), np.logspace(-3, 4, 25), feats, fa, train, test)
        ic, t, sh = test_ic(pred, target)
        rows.append((pname, "Ridge", f"a={a:.1e}", ic, t, sh))
        coef_report.append((pname, "Ridge", coefs, cancel_ratio(coefs)))

        print(f"  LightGBM on {pname}...", flush=True)
        hp, cvic, pred = lgbm_search(feats, fa, train, test)
        ic, t, sh = test_ic(pred, target)
        rows.append((pname, "LightGBM", f"leaves={hp['num_leaves']}", ic, t, sh))

    print("\n==== SPECTRUM GRID (TEST 2025-01-02..2026-05-22) ====")
    print(f"{'panel':16s} {'model':9s} {'selected':14s} {'test_IC':>8s} {'t':>7s} {'Sharpe':>7s}")
    for p, mdl, sel, ic, t, sh in rows:
        print(f"{p:16s} {mdl:9s} {sel:14s} {ic:+8.4f} {t:+7.2f} {sh:+7.2f}")

    print("\n==== VOL-CLUSTER COEFFICIENTS (linear cells) ====")
    print("cancel ratio: 1.0 = no cancellation, ~0 = full cancellation")
    for p, mdl, coefs, cr in coef_report:
        ve = coefs.get("vol_ewm_20", float("nan")); iv = coefs.get("idio_vol_20", float("nan"))
        print(f"  {p:16s} {mdl:9s}  vol_ewm_20={ve:+.4f}  idio_vol_20={iv:+.4f}  cluster_cancel_ratio={cr:.2f}")


if __name__ == "__main__":
    main()
