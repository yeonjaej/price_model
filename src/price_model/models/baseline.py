"""Baseline models — the floor every real model must clear.

If your fancy transformer can't beat ZeroPredictor's IC (which by definition is zero on
random samples but ~0 in expectation everywhere), you have a bug. If it can't beat
LastReturnPredictor, your fancy model isn't adding much over short-term momentum/reversal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from price_model.models.base import Model, load_config, save_config


class ZeroPredictor(Model):
    """Always predicts 0. Useful sanity check for the harness."""

    def fit(self, panel: pl.DataFrame) -> None:
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        return self._format_predictions(panel, np.zeros(panel.height))

    def save(self, path: Path) -> None:
        save_config(self.config, path / "config.json")

    @classmethod
    def load(cls, path: Path) -> ZeroPredictor:
        m = cls(load_config(path / "config.json"))
        m._fitted = True
        return m


class LastReturnPredictor(Model):
    """Predicts the most recent normalized `return_5d` feature.

    Tests whether short-horizon momentum (or reversal, depending on sign) is meaningful.
    Expects `return_5d` to be in feature_cols.
    """

    SOURCE_FEATURE = "return_5d"

    def fit(self, panel: pl.DataFrame) -> None:
        if self.SOURCE_FEATURE not in self.config.feature_cols:
            raise ValueError(f"LastReturnPredictor needs {self.SOURCE_FEATURE!r} in feature_cols")
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        preds = panel[self.SOURCE_FEATURE].to_numpy()
        return self._format_predictions(panel, preds)

    def save(self, path: Path) -> None:
        save_config(self.config, path / "config.json")

    @classmethod
    def load(cls, path: Path) -> LastReturnPredictor:
        m = cls(load_config(path / "config.json"))
        m._fitted = True
        return m


class MomentumFactor(Model):
    """Cross-sectional documented-factor baseline: rank tickers by a single feature.

    This is the literature-standard "naive momentum factor" baseline for
    cross-sectional ML evaluation on US equities. The prediction on date `t`
    for ticker `i` is just the value of the configured feature (typically
    `momentum_504`, `momentum_378`, or `momentum_12_1`) for that
    `(t, i)`. No training, no hyperparameters — the model is the feature.

    Configure via `config.params["feature_name"]`. The named feature must
    appear in `config.feature_cols` so the harness routes it through the
    feature pipeline. Example YAML:

        - id: momentum_504_factor
          class: MomentumFactor
          features: [momentum_504]
          params:
            feature_name: momentum_504

    Why this exists
    ---------------
    Prior comparisons in the project used `ArimaPerTicker` as the "classical"
    baseline, but at annual-refit cadence ARIMA's forecast is dominated by
    its constant trend term, which is `≈ mean(log_return)` over the training
    window — operationally identical to ranking by trailing momentum. So
    "LightGBM vs ARIMA" has been "LightGBM vs naive momentum factor"
    indirectly. This class makes the comparison direct, matching the
    convention in Gu-Kelly-Xiu (2020), Avramov-Cheng-Metzker (2023), and
    Han-He-Rapach-Zhou (2024), where the cross-sectional ML model is judged
    against an explicit documented factor — not a time-series wrapper.

    No fit step is required: the predict step simply returns the feature's
    value, which the walk-forward harness will compare to forward excess
    returns. The cross-sectional ranking it induces is the entire signal.
    Per-date cross-sectional demeaning is applied at the eval layer
    (the target `y` is already cross-sectionally demeaned), so this
    baseline's IC is the univariate cross-sectional IC of the named
    feature.
    """

    def fit(self, panel: pl.DataFrame) -> None:
        feature_name = self.config.params.get("feature_name")
        if feature_name is None:
            raise ValueError("MomentumFactor requires params['feature_name'] (e.g. 'momentum_504')")
        if feature_name not in self.config.feature_cols:
            raise ValueError(
                f"MomentumFactor: feature_name {feature_name!r} not in feature_cols "
                f"{self.config.feature_cols!r}"
            )
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        feature_name = self.config.params["feature_name"]
        preds = panel[feature_name].to_numpy()
        return self._format_predictions(panel, preds)

    def save(self, path: Path) -> None:
        save_config(self.config, path / "config.json")

    @classmethod
    def load(cls, path: Path) -> MomentumFactor:
        m = cls(load_config(path / "config.json"))
        m._fitted = True
        return m
