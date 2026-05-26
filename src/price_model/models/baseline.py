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
