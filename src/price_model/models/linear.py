"""Regularized linear cross-sectional models — the modern academic baselines.

Three model classes, paired with different penalty structures:

  LassoCrossSectional — L1 penalty (sparsity / feature selection).
    Han, He, Rapach, Zhou (2024, Review of Finance) document that
    L1-regularized cross-sectional regressions ("E-LASSO") match or beat
    heavier ML methods on liquid US equity universes net of transaction
    costs WHEN the feature panel is pre-cleaned to one representative
    per economically distinct factor. L1's failure mode on collinear
    panels — splitting correlated columns into cancelling signs — is
    documented across the project's own diagnostics.

  RidgeCrossSectional — L2 penalty (smooth shrinkage, no feature
    elimination). The standard regularized-linear baseline in the
    asset-pricing literature for cross-sectional studies where the
    feature panel has known collinearity. Gu-Kelly-Xiu (2020), Avramov-
    Cheng-Metzker (2023), and the broader factor-zoo literature use
    Ridge (or OLS) as the reference linear baseline that nonlinear ML
    models must beat to justify their complexity. Unlike L1, L2 handles
    correlated regressors smoothly: weight is shared across collinear
    columns proportional to the joint signal, with no cancellation
    failure mode.

  ElasticNetCrossSectional — convex combination of L1 + L2 (Zou-Hastie 2005).
    The hybrid penalty handles the regime where features are correlated
    BUT one or two are noise: L2 keeps correlated features stable while
    L1 still zeros out the genuinely irrelevant ones. Mathematically it
    interpolates between Lasso (l1_ratio=1) and Ridge (l1_ratio=0). The
    optimal l1_ratio is selected per refit via inner CV. Particularly
    useful for the momentum-family experiments where mom_378, mom_504,
    mom_756 are highly collinear (L2 stabilizes them) but mom_12_1 may
    or may not carry additional signal (L1 can drop it if not).

All three classes share the same fit/predict/save/load interface and the
same `feature_importance()` / `selected_alpha()` diagnostic surface.
They're intended to be drop-in alternatives on the same experiment YAML
so the L1-vs-L2-vs-blend comparison is one model_id change apart.

The predict step is a single linear function evaluation, making predictions:

  - Cross-sectionally smooth (no tree bin-noise)
  - Slow-changing when inputs are slow-changing (no per-bin shrinkage)
  - Auditable (every coefficient is named and signed)
  - Sparse (Lasso only) or smoothly shrunk (Ridge)

Compared to LightGBM on the same feature panel, both linear models lose
the ability to capture non-linear feature interactions but gain stability,
interpretability, and a much smaller cross-sectional dispersion
compression problem. On heavily-arbitraged liquid US universes, the
empirical record is mixed: the literature suggests linear models match
ML when the feature panel is curated; the project's own data on this
PIT universe at 5-day horizon shows trees retain an edge that the linear
models do not match.
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


class RidgeCrossSectional(Model):
    """Cross-sectional Ridge regression with cross-validated regularization.

    Fits `sklearn.linear_model.RidgeCV` on the pooled `(date, ticker)` feature
    panel against `y` (forward excess return). Unlike Lasso, the L2 penalty
    keeps every coefficient non-zero and shrinks them smoothly toward zero.
    Critical property on collinear panels: when two features are highly
    correlated, Ridge splits their weight proportionally rather than
    arbitrarily assigning all weight to one or, worse (as L1 does on this
    project's data), cancelling them into opposing signs.

    Parameters (via `config.params`):

    - `alphas` (list of float, optional): grid of α values to search. If
      None, sklearn's default `[0.1, 1.0, 10.0]` is used; we override to
      a log-spaced 20-point grid spanning 1e-4 to 1e4 for finer resolution.
    - `cv` (int, default 5): K-fold cross-validation folds for picking α.
    - `fit_intercept` (bool, default True): include an intercept (matches
      cross-sectional excess return target, which is zero-mean per date).

    Save/load uses pickle for the coefficient vector and intercept;
    config.json carries the feature_cols and hyperparameters.

    Why Ridge alongside Lasso
    -------------------------
    On the project's feature panels, L1 produces unstable solutions when
    correlated features are present (documented in the lasso_pit_v1 / v2
    diagnostics — momentum siblings get cancelling signs of equal
    magnitude). Ridge sidesteps this entirely. The L1-vs-L2 IC comparison
    on the same panel quantifies how much of the Lasso failure is L1-
    specific versus structural to pooled-coefficient linear modeling
    on this universe.
    """

    def __init__(self, config):
        super().__init__(config)
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None
        self._alpha: float | None = None
        self._feat_names: tuple[str, ...] = tuple(config.feature_cols)

    def _params(self) -> dict[str, Any]:
        # sklearn's RidgeCV default alphas=[0.1, 1.0, 10.0] is too coarse
        # for the magnitudes the project sees on z-scored features; use a
        # finer log-spaced grid unless caller overrides.
        alphas_cfg = self.config.params.get("alphas")
        alphas_resolved = np.logspace(-4, 4, 20).tolist() if alphas_cfg is None else alphas_cfg
        return {
            "alphas": alphas_resolved,
            "cv": int(self.config.params.get("cv", 5)),
            "fit_intercept": bool(self.config.params.get("fit_intercept", True)),
        }

    def fit(self, panel: pl.DataFrame) -> None:
        from sklearn.linear_model import RidgeCV

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats])
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        X = train.select(feats).to_numpy()
        y = train[target].to_numpy()
        params = self._params()

        # RidgeCV uses leave-one-out by default when cv is None; we pin
        # cv=5 to match Lasso's setup so the comparison isolates the
        # penalty type, not the cross-validation scheme.
        model = RidgeCV(
            alphas=params["alphas"],
            cv=params["cv"],
            fit_intercept=params["fit_intercept"],
        )
        model.fit(X, y)
        self._coef = np.asarray(model.coef_)
        self._intercept = float(model.intercept_) if params["fit_intercept"] else 0.0
        self._alpha = float(model.alpha_)
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        if self._coef is None:
            raise RuntimeError("RidgeCrossSectional has no fitted coefficients")
        feats = list(self.config.feature_cols)
        X = panel.select(feats).fill_null(0.0).to_numpy()
        preds = X @ self._coef + (self._intercept or 0.0)
        return self._format_predictions(panel, preds)

    def feature_importance(self) -> dict[str, float]:
        """Coefficient magnitude per feature. Sign-preserved; never exactly zero (L2)."""
        self._check_fitted()
        if self._coef is None:
            raise RuntimeError("RidgeCrossSectional has no fitted coefficients")
        return dict(zip(self._feat_names, [float(c) for c in self._coef], strict=True))

    def selected_alpha(self) -> float | None:
        """The α value RidgeCV picked via cross-validation. Useful for diagnostics."""
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
    def load(cls, path: Path) -> RidgeCrossSectional:
        m = cls(load_config(path / "config.json"))
        with (path / "state.pkl").open("rb") as f:
            state = pickle.load(f)
        m._coef = state["coef"]
        m._intercept = state["intercept"]
        m._alpha = state["alpha"]
        m._feat_names = state["feat_names"]
        m._fitted = True
        return m


class ElasticNetCrossSectional(Model):
    """Cross-sectional ElasticNet regression with cross-validated regularization.

    Fits `sklearn.linear_model.ElasticNetCV` on the pooled `(date, ticker)`
    feature panel against `y` (forward excess return). The hybrid L1 + L2
    penalty interpolates between Lasso (l1_ratio=1) and Ridge (l1_ratio=0).
    The CV jointly selects α (overall regularization strength) and
    l1_ratio (mix between L1 and L2) per refit.

    Use case versus pure Lasso / Ridge
    ----------------------------------
    Lasso's failure mode on correlated panels (documented in this
    project's lasso_pit_v1 diagnostics): when two features are highly
    collinear, L1 picks one arbitrarily and zeros the other — or, worse,
    splits weight into cancelling signs. Ridge sidesteps this by sharing
    weight smoothly but never produces sparsity. ElasticNet's L2
    component stabilizes the correlated subspace while the L1 component
    can still drop genuinely irrelevant features. This is the recipe Zou
    & Hastie (2005) introduced for "Lasso's instability on grouped
    variables" — exactly the failure mode this project encountered.

    Parameters (via `config.params`):

    - `l1_ratios` (list of float, optional): grid of L1 mix values to
      search. Default `[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]` — sklearn
      recommends weighting toward higher l1_ratios since smaller values
      collapse to Ridge.
    - `alphas` (list of float, optional): grid of α values to search.
      Default: sklearn's data-driven geometric grid (n_alphas=100).
    - `cv` (int, default 5): K-fold cross-validation folds.
    - `max_iter` (int, default 5000): coordinate-descent iterations cap.
    - `tol` (float, default 1e-4): convergence tolerance.
    - `selection` (str, default "cyclic"): "cyclic" or "random" coordinate
      descent order.
    - `fit_intercept` (bool, default True): include an intercept.

    Save/load uses pickle for the coefficient vector + intercept +
    selected (α, l1_ratio); config.json carries the feature_cols and HPs.
    """

    def __init__(self, config):
        super().__init__(config)
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None
        self._alpha: float | None = None
        self._l1_ratio: float | None = None
        self._feat_names: tuple[str, ...] = tuple(config.feature_cols)

    def _params(self) -> dict[str, Any]:
        # sklearn 1.7+ deprecates `alphas=None`. Replace with explicit int 100
        # to preserve the default geometric grid without warnings.
        alphas_cfg = self.config.params.get("alphas")
        alphas_resolved = 100 if alphas_cfg is None else alphas_cfg
        l1_ratios_cfg = self.config.params.get("l1_ratios")
        l1_ratios_resolved = (
            [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
            if l1_ratios_cfg is None
            else l1_ratios_cfg
        )
        return {
            "alphas": alphas_resolved,
            "l1_ratios": l1_ratios_resolved,
            "cv": int(self.config.params.get("cv", 5)),
            "max_iter": int(self.config.params.get("max_iter", 5000)),
            "tol": float(self.config.params.get("tol", 1e-4)),
            "selection": str(self.config.params.get("selection", "cyclic")),
            "fit_intercept": bool(self.config.params.get("fit_intercept", True)),
        }

    def fit(self, panel: pl.DataFrame) -> None:
        from sklearn.linear_model import ElasticNetCV

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats])
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        X = train.select(feats).to_numpy()
        y = train[target].to_numpy()
        params = self._params()

        model = ElasticNetCV(
            l1_ratio=params["l1_ratios"],
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
        self._l1_ratio = float(model.l1_ratio_)
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        if self._coef is None:
            raise RuntimeError("ElasticNetCrossSectional has no fitted coefficients")
        feats = list(self.config.feature_cols)
        X = panel.select(feats).fill_null(0.0).to_numpy()
        preds = X @ self._coef + (self._intercept or 0.0)
        return self._format_predictions(panel, preds)

    def feature_importance(self) -> dict[str, float]:
        """Coefficient magnitude per feature. Sign-preserved; zeros possible (L1 component)."""
        self._check_fitted()
        if self._coef is None:
            raise RuntimeError("ElasticNetCrossSectional has no fitted coefficients")
        return dict(zip(self._feat_names, [float(c) for c in self._coef], strict=True))

    def selected_alpha(self) -> float | None:
        """The α value ElasticNetCV picked. Useful for diagnostics."""
        return self._alpha

    def selected_l1_ratio(self) -> float | None:
        """The l1_ratio ElasticNetCV picked. 1.0 = Lasso, 0.0 = Ridge.

        Diagnostic: if CV consistently picks l1_ratio close to 1.0 across
        refits, the L2 stabilization isn't earning its keep on this panel
        and a pure Lasso would be effectively equivalent. If CV picks
        l1_ratio close to 0.0, pure Ridge would be effectively equivalent.
        Values between 0.5 and 0.9 indicate the hybrid is actually adding
        value over either extreme.
        """
        return self._l1_ratio

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        with (path / "state.pkl").open("wb") as f:
            pickle.dump(
                {
                    "coef": self._coef,
                    "intercept": self._intercept,
                    "alpha": self._alpha,
                    "l1_ratio": self._l1_ratio,
                    "feat_names": self._feat_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> ElasticNetCrossSectional:
        m = cls(load_config(path / "config.json"))
        with (path / "state.pkl").open("rb") as f:
            state = pickle.load(f)
        m._coef = state["coef"]
        m._intercept = state["intercept"]
        m._alpha = state["alpha"]
        m._l1_ratio = state["l1_ratio"]
        m._feat_names = state["feat_names"]
        m._fitted = True
        return m
