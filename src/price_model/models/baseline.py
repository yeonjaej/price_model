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

    This is the literature-standard "naive factor" baseline for cross-sectional
    ML evaluation on US equities. The prediction on date `t` for ticker `i` is
    `sign * feature_value(t, i)` for the configured feature. No training, no
    hyperparameters — the model is the feature.

    Configure via `config.params`:
      - `feature_name` (required): the feature to rank on
      - `sign` (default +1): +1 for momentum-direction factors; -1 for
        reversal-direction factors (Lehmann 1990 / Jegadeesh 1990 short-term
        reversal). With sign=-1 the IC will be POSITIVE when high recent
        returns predict LOW future returns — the canonical daily-horizon
        anomaly direction. This keeps all model_id comparisons in the table
        like-for-like: a positive IC is "the factor works as predicted."

    Example YAML:

        # Momentum direction (JT canonical)
        - id: mom_504_factor
          class: MomentumFactor
          features: [momentum_504]
          params:
            feature_name: momentum_504
            sign: 1

        # Reversal direction (Lehmann canonical, daily horizon)
        - id: reversal_1d_factor
          class: MomentumFactor
          features: [return_1d]
          params:
            feature_name: return_1d
            sign: -1

    Why this exists
    ---------------
    The project's cross-sectional comparisons need a literature-canonical
    single-factor baseline at the Tier-1 level: "do you have signal beyond
    the most-documented anomaly for this horizon?" At daily / 5-day-forward
    horizons, the canonical anomaly is short-term reversal, not Jegadeesh-
    Titman 12-1 month momentum (which is the *monthly* canonical). Both
    directions are useful comparison rows depending on the regime under
    study. The `sign` flag makes this clean without proliferating model
    classes.

    Per-date cross-sectional demeaning is applied at the eval layer (the
    target `y` is already cross-sectionally demeaned), so this baseline's
    IC is the univariate cross-sectional IC of `sign * feature`.
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
        # Coerce sign to ±1; default to +1 (momentum direction). Anything else
        # is a config error.
        sign = self.config.params.get("sign", 1)
        if sign not in (1, -1, 1.0, -1.0):
            raise ValueError(f"MomentumFactor: sign must be +1 or -1, got {sign!r}")
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        feature_name = self.config.params["feature_name"]
        sign = float(self.config.params.get("sign", 1))
        preds = sign * panel[feature_name].to_numpy()
        return self._format_predictions(panel, preds)

    def save(self, path: Path) -> None:
        save_config(self.config, path / "config.json")

    @classmethod
    def load(cls, path: Path) -> MomentumFactor:
        m = cls(load_config(path / "config.json"))
        m._fitted = True
        return m
