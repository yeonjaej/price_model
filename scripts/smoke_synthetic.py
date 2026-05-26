"""End-to-end smoke test using a synthetic panel.

Doesn't touch yfinance. Verifies:
  - feature matrix construction
  - walk-forward harness across multiple splits
  - LightGBM fit/predict
  - prediction store round-trip
  - metrics comparison

Usage: PYTHONPATH=src python scripts/smoke_synthetic.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# Ensure src/ is importable when run from the repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from price_model.eval.metrics import compare_models
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.pipeline.walk_forward import join_with_realized, run_walk_forward
from price_model.serving.store import PredictionStore


def synthetic_panel(n_tickers: int = 20, n_days: int = 1500, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed=seed)
    tickers = [f"SYN{i:02d}" for i in range(n_tickers)]
    sectors = ["Tech", "Health", "Energy", "Financials"]
    sector_for = {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}
    start = date(2018, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    rows: list[dict] = []
    market = rng.normal(0.0003, 0.010, size=n_days)
    for t in tickers:
        beta = rng.uniform(0.6, 1.4)
        idio = rng.normal(0.0002, 0.012, size=n_days)
        log_ret = beta * market + idio
        prices = 100.0 * np.exp(np.cumsum(log_ret))
        for d, p in zip(dates, prices):
            rows.append({
                "date": d, "ticker": t, "sector": sector_for[t],
                "open": float(p), "high": float(p) * 1.01, "low": float(p) * 0.99,
                "close": float(p), "adj_close": float(p),
                "volume": int(rng.integers(1_000_000, 10_000_000)),
            })
    return pl.DataFrame(rows).sort(["ticker", "date"])


def main() -> int:
    print("→ generating synthetic panel...")
    panel = synthetic_panel()
    print(f"  panel: {panel.height:,} rows, {panel['ticker'].n_unique()} tickers, "
          f"{panel['date'].min()} → {panel['date'].max()}")

    feats = [
        "return_5d", "momentum_60", "vol_20", "rsi_14", "distance_ma_200",
        "momentum_60_sector_rel", "idio_vol_20",
        "momentum_60_rank", "vol_20_rank", "distance_ma_200_rank",
    ]
    print("→ building feature matrix...")
    matrix = build_feature_matrix(panel, feature_names=feats, target_horizon=5)
    matrix = drop_warmup_rows(matrix, feats)
    print(f"  matrix: {matrix.height:,} rows after warmup drop")

    with tempfile.TemporaryDirectory() as tmp:
        store = PredictionStore(Path(tmp) / "preds.duckdb")
        all_preds = []
        for spec in [
            ("zero", "ZeroPredictor", {}),
            ("last_ret", "LastReturnPredictor", {}),
            ("lgbm", "LightGBMModel", {
                "n_estimators": 80, "num_leaves": 15, "min_data_in_leaf": 50,
                "learning_rate": 0.1,
            }),
        ]:
            mid, cls, params = spec
            print(f"→ walk-forward for {mid}...")
            cfg = ModelConfig(model_id=mid, feature_cols=tuple(feats), params=params)
            model = build_model(cls, cfg)
            preds = run_walk_forward(
                matrix,
                model=model,
                feature_cols=feats,
                target_col="y",
                experiment_id="smoke",
                horizon_days=5,
                refit_freq_days=63,
                embargo_days=6,
                min_train_days=300,
                store=store,
            )
            preds = preds.with_columns(pl.lit(mid).alias("model_id"))
            all_preds.append(preds)
            print(f"  {mid}: {preds.height:,} predictions")

        joined = pl.concat(all_preds)
        eval_df = join_with_realized(joined, matrix)
        summary = compare_models(eval_df, horizon_days=5)

        print("\n=== comparison ===")
        with pl.Config(tbl_width_chars=200, tbl_cols=20):
            print(summary)

        store_rows = store.query("SELECT COUNT(*) AS n FROM predictions")["n"][0]
        print(f"\nstore rows: {store_rows:,}")
        store.close()

    print("\n✓ smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
