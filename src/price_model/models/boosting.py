"""LightGBM wrapper.

The first "real" model. Trains on (feature_cols) -> target on the pooled cross-section.
No stock-identity feature — so it generalizes to any ticker at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from price_model.models.base import Model, ModelConfig, save_config, load_config

DEFAULT_PARAMS = {
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
}


class LightGBMModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._booster = None  # lightgbm.Booster after fit

    def _params(self) -> dict:
        merged = {**DEFAULT_PARAMS, **self.config.params}
        return merged

    def fit(self, panel: pl.DataFrame) -> None:
        import lightgbm as lgb

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        # Drop rows with null target (forward window not yet observed)
        train = panel.drop_nulls(subset=[target, *feats])
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        X = train.select(feats).to_numpy()
        y = train[target].to_numpy()

        params = self._params()
        n_estimators = int(params.pop("n_estimators", 500))
        train_ds = lgb.Dataset(X, label=y, feature_name=feats)
        self._booster = lgb.train(
            params,
            train_ds,
            num_boost_round=n_estimators,
        )
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        assert self._booster is not None
        feats = list(self.config.feature_cols)
        # Predict even if features are partially null — fill with 0 (cross-sectionally
        # normalized features have mean ~0 by construction). Track these as low-confidence.
        X = panel.select(feats).fill_null(0.0).to_numpy()
        preds = self._booster.predict(X)
        return self._format_predictions(panel, np.asarray(preds))

    def feature_importance(self) -> dict[str, float]:
        self._check_fitted()
        assert self._booster is not None
        importance = self._booster.feature_importance(importance_type="gain")
        return dict(zip(list(self.config.feature_cols), [float(x) for x in importance]))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        if self._booster is not None:
            self._booster.save_model(str(path / "booster.txt"))

    @classmethod
    def load(cls, path: Path) -> "LightGBMModel":
        import lightgbm as lgb

        config = load_config(path / "config.json")
        m = cls(config)
        booster_path = path / "booster.txt"
        if booster_path.exists():
            m._booster = lgb.Booster(model_file=str(booster_path))
            m._fitted = True
        return m
