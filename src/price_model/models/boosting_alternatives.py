"""XGBoost and CatBoost wrappers — alternative tree-boosting model classes.

The project's primary tree model is LightGBM (`boosting.py`). XGBoost and
CatBoost are gradient-boosted-tree libraries with subtly different default
behaviors:

  - **XGBoost** uses level-wise tree growth and exact split-finding (or
    approximate via the `approx` algorithm). Defaults to a histogram-based
    optimization since version 1.5. Strong on tabular data with mixed
    feature distributions; widely used in financial Kaggle competitions
    (Two Sigma, Numerai, JPX).

  - **CatBoost** uses oblivious (symmetric) tree growth and Ordered Boosting,
    which provides a built-in regularization mechanism that reduces
    overfitting on small-to-medium datasets. Strong on categorical features
    and on problems where over-fitting is a known concern; the default
    hyperparameters are generally more conservative than LightGBM's.

Both share the same project Model ABC interface as LightGBMModel: pooled
cross-sectional fit on `(feature_cols) -> target`, multi-seed averaging
optional, save/load via library-native serialization, gain-based
feature importance.

The point of including all three is ensemble diversification. The same
feature panel produces three different prediction streams (LightGBM,
XGBoost, CatBoost), each with different inductive biases. A stacking layer
that combines them can extract more signal than any single library alone —
this is the standard Kaggle-style winning approach for cross-sectional
return prediction.

Install:
    pip install xgboost catboost
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from price_model.models.base import Model, ModelConfig, load_config, save_config

# Keys we own (popped before passing to xgb/cb)
_INTERNAL_KEYS = {"n_estimators", "seeds", "early_stopping_rounds", "val_fraction"}

XGB_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "n_estimators": 500,
    "verbosity": 0,
    # Internal knobs
    "seeds": [42],
    "early_stopping_rounds": 0,
    "val_fraction": 0.0,
}

CB_DEFAULT_PARAMS: dict[str, Any] = {
    "loss_function": "RMSE",
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "rsm": 0.8,  # random subspace method = colsample
    "subsample": 0.8,
    "bootstrap_type": "Bernoulli",
    "n_estimators": 500,
    "verbose": False,
    "allow_writing_files": False,
    # Internal knobs
    "seeds": [42],
    "early_stopping_rounds": 0,
    "val_fraction": 0.0,
}


def _split_by_date(
    panel: pl.DataFrame, val_fraction: float
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Date-based train/val split — same convention as boosting.py."""
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


class XGBoostModel(Model):
    """XGBoost gradient-boosted regression on the pooled cross-section.

    Drop-in alternative to LightGBMModel — same feature panel, same target,
    same save/load contract, but uses xgboost.XGBRegressor under the hood.
    The library's `hist` tree method makes it competitive with LightGBM on
    speed for our dataset size.

    Hyperparameters via `config.params`:
      - learning_rate, max_depth, min_child_weight, subsample,
        colsample_bytree, reg_alpha, reg_lambda, n_estimators
      - seeds: list of integer seeds for multi-seed averaging
      - early_stopping_rounds + val_fraction for date-based early stopping

    All params have defaults set to "Kaggle-reasonable" starting points.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._boosters: list = []

    def _split_params(self) -> tuple[dict, dict]:
        merged = {**XGB_DEFAULT_PARAMS, **self.config.params}
        internal = {k: merged.pop(k) for k in list(merged) if k in _INTERNAL_KEYS}
        return merged, internal

    def fit(self, panel: pl.DataFrame) -> None:
        import xgboost as xgb

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats]).sort("date")
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        xgb_params, internal = self._split_params()
        n_estimators = int(internal.get("n_estimators", 500))
        seeds = list(internal.get("seeds", [42]))
        early_stop = int(internal.get("early_stopping_rounds", 0))
        val_fraction = float(internal.get("val_fraction", 0.0))

        tr_panel, val_panel = _split_by_date(train, val_fraction if early_stop > 0 else 0.0)

        X_tr = tr_panel.select(feats).to_numpy()
        y_tr = tr_panel[target].to_numpy()
        eval_set: list = []
        if val_panel is not None:
            X_vl = val_panel.select(feats).to_numpy()
            y_vl = val_panel[target].to_numpy()
            eval_set = [(X_vl, y_vl)]

        self._boosters = []
        for seed in seeds:
            params = {**xgb_params, "random_state": seed, "n_estimators": n_estimators}
            if early_stop > 0:
                params["early_stopping_rounds"] = early_stop
            booster = xgb.XGBRegressor(**params)
            if eval_set:
                booster.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
            else:
                booster.fit(X_tr, y_tr, verbose=False)
            self._boosters.append(booster)
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        if not self._boosters:
            raise RuntimeError("XGBoostModel has no fitted boosters")
        feats = list(self.config.feature_cols)
        X = panel.select(feats).fill_null(0.0).to_numpy()
        per_seed_preds = np.stack([b.predict(X) for b in self._boosters], axis=0)
        avg_preds = per_seed_preds.mean(axis=0)
        return self._format_predictions(panel, np.asarray(avg_preds))

    def feature_importance(self) -> dict[str, float]:
        """Gain-based importance, averaged across all seeded boosters."""
        self._check_fitted()
        if not self._boosters:
            raise RuntimeError("XGBoostModel has no fitted boosters")
        feats = list(self.config.feature_cols)
        per_seed = np.stack(
            [
                np.asarray(
                    [
                        b.get_booster()
                        .get_score(importance_type="gain")
                        .get(f"f{i}", 0.0)
                        for i in range(len(feats))
                    ]
                )
                for b in self._boosters
            ],
            axis=0,
        )
        avg = per_seed.mean(axis=0)
        return dict(zip(feats, [float(x) for x in avg], strict=True))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        for i, booster in enumerate(self._boosters):
            booster.save_model(str(path / f"booster_{i:02d}.ubj"))

    @classmethod
    def load(cls, path: Path) -> XGBoostModel:
        import xgboost as xgb

        config = load_config(path / "config.json")
        m = cls(config)
        booster_files = sorted(path.glob("booster_*.ubj"))
        for bf in booster_files:
            booster = xgb.XGBRegressor()
            booster.load_model(str(bf))
            m._boosters.append(booster)
        if m._boosters:
            m._fitted = True
        return m


class CatBoostModel(Model):
    """CatBoost gradient-boosted regression on the pooled cross-section.

    Uses oblivious (symmetric) tree growth and Ordered Boosting — the
    library's default regularization mechanism that reduces overfitting on
    small-to-medium datasets. Conservative defaults relative to LightGBM /
    XGBoost; often produces decorrelated predictions vs the other two
    libraries, making it valuable as an ensemble diversifier.

    Hyperparameters via `config.params`:
      - learning_rate, depth, l2_leaf_reg, rsm, subsample, n_estimators
      - seeds: list of integer seeds
      - early_stopping_rounds + val_fraction for date-based early stopping

    The verbose=False + allow_writing_files=False suppress CatBoost's
    default tendency to log progress and write checkpoint files to disk.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._boosters: list = []

    def _split_params(self) -> tuple[dict, dict]:
        merged = {**CB_DEFAULT_PARAMS, **self.config.params}
        internal = {k: merged.pop(k) for k in list(merged) if k in _INTERNAL_KEYS}
        return merged, internal

    def fit(self, panel: pl.DataFrame) -> None:
        from catboost import CatBoostRegressor

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats]).sort("date")
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        cb_params, internal = self._split_params()
        n_estimators = int(internal.get("n_estimators", 500))
        seeds = list(internal.get("seeds", [42]))
        early_stop = int(internal.get("early_stopping_rounds", 0))
        val_fraction = float(internal.get("val_fraction", 0.0))

        tr_panel, val_panel = _split_by_date(train, val_fraction if early_stop > 0 else 0.0)

        X_tr = tr_panel.select(feats).to_numpy()
        y_tr = tr_panel[target].to_numpy()
        eval_set = None
        if val_panel is not None:
            X_vl = val_panel.select(feats).to_numpy()
            y_vl = val_panel[target].to_numpy()
            eval_set = (X_vl, y_vl)

        self._boosters = []
        for seed in seeds:
            params = {**cb_params, "random_seed": seed, "iterations": n_estimators}
            if early_stop > 0:
                params["early_stopping_rounds"] = early_stop
            params.pop("n_estimators", None)
            booster = CatBoostRegressor(**params)
            booster.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
            self._boosters.append(booster)
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        if not self._boosters:
            raise RuntimeError("CatBoostModel has no fitted boosters")
        feats = list(self.config.feature_cols)
        X = panel.select(feats).fill_null(0.0).to_numpy()
        per_seed_preds = np.stack([b.predict(X) for b in self._boosters], axis=0)
        avg_preds = per_seed_preds.mean(axis=0)
        return self._format_predictions(panel, np.asarray(avg_preds))

    def feature_importance(self) -> dict[str, float]:
        """LossFunctionChange importance averaged across seeded boosters."""
        self._check_fitted()
        if not self._boosters:
            raise RuntimeError("CatBoostModel has no fitted boosters")
        feats = list(self.config.feature_cols)
        per_seed = np.stack(
            [np.asarray(b.feature_importances_) for b in self._boosters],
            axis=0,
        )
        avg = per_seed.mean(axis=0)
        return dict(zip(feats, [float(x) for x in avg], strict=True))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        for i, booster in enumerate(self._boosters):
            booster.save_model(str(path / f"booster_{i:02d}.cbm"))

    @classmethod
    def load(cls, path: Path) -> CatBoostModel:
        from catboost import CatBoostRegressor

        config = load_config(path / "config.json")
        m = cls(config)
        booster_files = sorted(path.glob("booster_*.cbm"))
        for bf in booster_files:
            booster = CatBoostRegressor()
            booster.load_model(str(bf))
            m._boosters.append(booster)
        if m._boosters:
            m._fitted = True
        return m
