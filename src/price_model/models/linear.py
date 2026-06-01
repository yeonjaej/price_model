"""Regularized linear cross-sectional models — the modern academic baseline.

Han, He, Rapach, and Zhou (2024, Review of Finance) document that
regularized linear cross-sectional regressions ("E-LASSO" — Lasso applied
to Fama-MacBeth-style cross-sectional regressions) match or beat
heavier ML methods on liquid US equity universes net of transaction
costs. The key mechanism is L1 penalty driving automatic feature
selection on a panel where the "factor zoo" produces many candidate
predictors but only 3-8 actually carry signal.

This module wires `sklearn.linear_model.LassoCV` into the project's
`Model` ABC. The fit step pools all training rows from the walk-forward
window, runs cross-validated Lasso to pick α, and stores the resulting
coefficient vector. The predict step is a single linear function
evaluation, making predictions:

  - Cross-sectionally smooth (no tree bin-noise)
  - Slow-changing when inputs are slow-changing (no per-bin shrinkage)
  - Sparse (most coefficients exactly zero from L1)
  - Auditable (every coefficient is named and signed)

Compared to LightGBM on the same feature panel, Lasso loses the ability
to capture non-linear feature interactions but gains stability, sparsity,
and a much smaller cross-sectional dispersion compression problem.
On heavily-arbitraged liquid US universes, the empirical record is that
this trade-off favors the linear model.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from price_model.models.base import Model, load_config, save_config


class LassoCrossSectional(Model):
    """Cross-sectional Lasso regression with cross-validated regularization.

    Fits `sklearn.linear_model.LassoCV` on the pooled `(date, ticker)` feature
    panel against `y` (forward excess return). The L1 penalty forces
    sparsity; with cross-validated `α`, the model selects its own
    regularization strength on each refit.

    Parameters (via `config.params`):

    - `alphas` (list of float, optional): grid of α values to search. If
      None, sklearn's default geometric grid is used.
    - `cv` (int, default 5): K-fold cross-validation folds for picking α.
    - `max_iter` (int, default 5000): coordinate-descent iterations cap.
    - `tol` (float, default 1e-4): convergence tolerance.
    - `selection` (str, default "cyclic"): "cyclic" or "random" coordinate
      descent order.
    - `fit_intercept` (bool, default True): include an intercept (matches
      cross-sectional excess return target, which is zero-mean per date).

    Save/load uses pickle for the coefficient vector and intercept;
    config.json carries the feature_cols and hyperparameters.
    """

    def __init__(self, config):
        super().__init__(config)
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None
        self._alpha: float | None = None
        self._feat_names: tuple[str, ...] = tuple(config.feature_cols)

    def _params(self) -> dict[str, Any]:
        # sklearn 1.7+ deprecates `alphas=None` (the default geometric grid
        # signal). Replace None with the explicit int 100 so we keep the same
        # default behavior across versions without warnings.
        alphas_cfg = self.config.params.get("alphas")
        alphas_resolved = 100 if alphas_cfg is None else alphas_cfg
        return {
            "alphas": alphas_resolved,
            "cv": int(self.config.params.get("cv", 5)),
            "max_iter": int(self.config.params.get("max_iter", 5000)),
            "tol": float(self.config.params.get("tol", 1e-4)),
            "selection": str(self.config.params.get("selection", "cyclic")),
            "fit_intercept": bool(self.config.params.get("fit_intercept", True)),
        }

    def fit(self, panel: pl.DataFrame) -> None:
        from sklearn.linear_model import LassoCV

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats])
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        X = train.select(feats).to_numpy()
        y = train[target].to_numpy()
        params = self._params()

        model = LassoCV(
            alphas=params["alphas"],
            cv=params["cv"],
            max_iter=params["max_iter"],
            tol=params["tol"],
            selection=params["selection"],
            fit_intercept=params["fit_intercept"],
            n_jobs=-1,
        )
        model.fit(X, y)
        self._coef = np.asarray(model.coef_)
        self._intercept = float(model.intercept_) if params["fit_intercept"] else 0.0
        self._alpha = float(model.alpha_)
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        if self._coef is None:
            raise RuntimeError("LassoCrossSectional has no fitted coefficients")
        feats = list(self.config.feature_cols)
        # Fill nulls with 0 — for z-scored features this is the cross-sectional
        # mean; for rank features it's near-middle. Predictions for warmup
        # rows will be dominated by the intercept and treated as low-conviction.
        X = panel.select(feats).fill_null(0.0).to_numpy()
        preds = X @ self._coef + (self._intercept or 0.0)
        return self._format_predictions(panel, preds)

    def feature_importance(self) -> dict[str, float]:
        """Coefficient magnitude per feature. Sign-preserved; non-zero = selected."""
        self._check_fitted()
        if self._coef is None:
            raise RuntimeError("LassoCrossSectional has no fitted coefficients")
        return dict(zip(self._feat_names, [float(c) for c in self._coef], strict=True))

    def selected_alpha(self) -> float | None:
        """The α value LassoCV picked via cross-validation. Useful for diagnostics."""
        return self._alpha

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        with (path / "state.pkl").open("wb") as f:
            pickle.dump(
                {
                    "coef": self._coef,
                    "intercept": self._intercept,
                    "alpha": self._alpha,
                    "feat_names": self._feat_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> LassoCrossSectional:
        m = cls(load_config(path / "config.json"))
        with (path / "state.pkl").open("rb") as f:
            state = pickle.load(f)
        m._coef = state["coef"]
        m._intercept = state["intercept"]
        m._alpha = state["alpha"]
        m._feat_names = state["feat_names"]
        m._fitted = True
        return m
