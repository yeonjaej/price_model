"""LightGBM wrapper.

The first "real" model. Trains on (feature_cols) -> target on the pooled cross-section.
No stock-identity feature — so it generalizes to any ticker at inference time.

Two features beyond a plain LightGBM call, both inherited from the Kaggle hedge-fund
notebooks we surveyed:

1. **Internal time-based validation + early stopping.** When `early_stopping_rounds`
   is provided in `config.params`, we carve off the last `val_fraction` *dates* of
   the training panel as an internal validation set, hand it to LightGBM as
   `valid_sets`, and stop boosting when validation error plateaus. The split is
   date-based, not row-based, so it's leakage-free.

2. **Multi-seed averaging.** When `seeds: [s1, s2, ...]` is in `config.params`, we
   train one booster per seed and average predictions at inference. Cheap variance
   reduction, no overfitting risk. With a single seed (default), behavior matches
   the original single-booster implementation.

Both are opt-in via config; defaults preserve the original behavior so existing
experiments don't change underfoot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from price_model.models.base import Model, ModelConfig, load_config, save_config

# Keys that are OUR knobs, not lgb.train params — popped off before passing to LightGBM.
_INTERNAL_KEYS = {"n_estimators", "seeds", "early_stopping_rounds", "val_fraction"}

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "n_estimators": 500,
    "verbosity": -1,
    # Internal knobs (defaults preserve original single-seed, no-early-stop behavior)
    "seeds": [42],
    "early_stopping_rounds": 0,  # 0 disables early stopping
    "val_fraction": 0.0,  # fraction of training DATES carved off as val
}


def _split_by_date(
    panel: pl.DataFrame, val_fraction: float
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Take the last `val_fraction` unique dates of `panel` as validation.

    Returns (train, val) where val may be None if val_fraction <= 0 or there
    aren't enough dates for a meaningful split.
    """
    if val_fraction <= 0:
        return panel, None
    all_dates = sorted(panel["date"].unique().to_list())
    n_val_dates = round(len(all_dates) * val_fraction)
    if n_val_dates < 1 or n_val_dates >= len(all_dates):
        return panel, None
    cutoff = all_dates[-n_val_dates]
    tr = panel.filter(pl.col("date") < pl.lit(cutoff))
    vl = panel.filter(pl.col("date") >= pl.lit(cutoff))
    if tr.height == 0 or vl.height == 0:
        return panel, None
    return tr, vl


class LightGBMModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._boosters: list = []  # one Booster per seed

    def _split_params(self) -> tuple[dict, dict]:
        """Return (lgb_params, internal) — separates lgb.train kwargs from our knobs."""
        merged = {**DEFAULT_PARAMS, **self.config.params}
        internal = {k: merged.pop(k) for k in list(merged) if k in _INTERNAL_KEYS}
        return merged, internal

    def fit(self, panel: pl.DataFrame) -> None:
        import lightgbm as lgb

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats]).sort("date")
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        lgb_params, internal = self._split_params()
        n_estimators = int(internal.get("n_estimators", 500))
        seeds = list(internal.get("seeds", [42]))
        early_stop = int(internal.get("early_stopping_rounds", 0))
        val_fraction = float(internal.get("val_fraction", 0.0))

        # Optional internal validation split (date-based to avoid leakage)
        tr_panel, val_panel = _split_by_date(train, val_fraction if early_stop > 0 else 0.0)

        X_tr = tr_panel.select(feats).to_numpy()
        y_tr = tr_panel[target].to_numpy()

        valid_sets: list = []
        valid_names: list[str] = []
        callbacks: list = []
        if val_panel is not None:
            X_vl = val_panel.select(feats).to_numpy()
            y_vl = val_panel[target].to_numpy()
            # reference will be re-bound per seed below
            valid_sets = [(X_vl, y_vl)]
            valid_names = ["valid"]
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stop, verbose=False))

        self._boosters = []
        for seed in seeds:
            params = {**lgb_params, "seed": seed}
            train_ds = lgb.Dataset(X_tr, label=y_tr, feature_name=feats)
            if valid_sets:
                val_lgb = lgb.Dataset(valid_sets[0][0], label=valid_sets[0][1], reference=train_ds)
                booster = lgb.train(
                    params,
                    train_ds,
                    num_boost_round=n_estimators,
                    valid_sets=[val_lgb],
                    valid_names=valid_names,
                    callbacks=callbacks,
                )
            else:
                booster = lgb.train(
                    params,
                    train_ds,
                    num_boost_round=n_estimators,
                )
            self._boosters.append(booster)
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        if not self._boosters:
            raise RuntimeError("LightGBMModel has no fitted boosters")
        feats = list(self.config.feature_cols)
        # Cross-sectionally normalized features have mean ~0 by construction;
        # fill nulls with 0 to make predictions for partial-warmup rows tolerable.
        X = panel.select(feats).fill_null(0.0).to_numpy()
        per_seed_preds = np.stack([b.predict(X) for b in self._boosters], axis=0)
        avg_preds = per_seed_preds.mean(axis=0)
        return self._format_predictions(panel, np.asarray(avg_preds))

    def feature_importance(self) -> dict[str, float]:
        """Gain-based importance, averaged across all seeded boosters."""
        self._check_fitted()
        if not self._boosters:
            raise RuntimeError("LightGBMModel has no fitted boosters")
        feats = list(self.config.feature_cols)
        per_seed = np.stack(
            [b.feature_importance(importance_type="gain") for b in self._boosters],
            axis=0,
        )
        avg = per_seed.mean(axis=0)
        return dict(zip(feats, [float(x) for x in avg], strict=True))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        for i, booster in enumerate(self._boosters):
            booster.save_model(str(path / f"booster_{i:02d}.txt"))

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        import lightgbm as lgb

        config = load_config(path / "config.json")
        m = cls(config)
        # Load any boosters (new naming `booster_XX.txt`, plus legacy `booster.txt`)
        booster_files = sorted(path.glob("booster_*.txt"))
        if not booster_files:
            legacy = path / "booster.txt"
            if legacy.exists():
                booster_files = [legacy]
        for bf in booster_files:
            m._boosters.append(lgb.Booster(model_file=str(bf)))
        if m._boosters:
            m._fitted = True
        return m
