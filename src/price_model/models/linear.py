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

    Fits a plain `sklearn.linear_model.Lasso` on the pooled `(date, ticker)`
    feature panel against `y` (forward excess return). The L1 penalty forces
    sparsity; `α` is chosen per refit by a **temporal, purged forward-chain CV
    scored on per-date IC** — the same grader the tree sweeps use — then the
    model is refit once on the full training block at the selected `α`.

    Parameters (via `config.params`):

    - `alphas` (list of float, optional): explicit grid of α values to search.
      If None, a data-driven log grid (sklearn's `alpha_max·eps … alpha_max`
      convention) is generated; see `models/_cv.l1_alpha_grid`.
    - `cv` (int, default 3): number of **purged, expanding-window, forward-chain**
      CV folds for picking α, scored on **per-date IC** (temporal — see
      `models/_cv.py`). NOT K-fold, NOT MSE.
    - `cv_embargo` (int, default 21): unique trailing dates purged before each
      validation fold; set ≥ the forward-return horizon so the label can't leak.
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
        # When `alphas` is unset, fit() generates a data-driven log grid with this
        # many points (see models/_cv.l1_alpha_grid); IC-based selection does not
        # need a fine grid, so 25 keeps the per-refit fold search fast.
        alphas_cfg = self.config.params.get("alphas")
        alphas_resolved = 25 if alphas_cfg is None else alphas_cfg
        return {
            "alphas": alphas_resolved,
            "n_splits": int(self.config.params.get("cv", 3)),
            "cv_embargo": int(self.config.params.get("cv_embargo", 21)),
            "max_iter": int(self.config.params.get("max_iter", 5000)),
            "tol": float(self.config.params.get("tol", 1e-4)),
            "selection": str(self.config.params.get("selection", "cyclic")),
            "fit_intercept": bool(self.config.params.get("fit_intercept", True)),
        }

    def fit(self, panel: pl.DataFrame) -> None:
        from sklearn.linear_model import Lasso

        from ._cv import l1_alpha_grid, purged_forward_chain_folds, select_alpha_by_ic

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats])
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        X = train.select(feats).to_numpy()
        y = train[target].to_numpy()
        dates = train["date"].to_numpy()
        params = self._params()

        # α selected by a temporal, purged forward-chain CV scored on per-date IC —
        # the SAME grader the tree sweeps use (not sklearn's non-temporal, MSE-path
        # LassoCV). See models/_cv.py.
        folds = purged_forward_chain_folds(
            dates, n_splits=params["n_splits"], embargo=params["cv_embargo"]
        )
        alphas = (
            np.asarray(params["alphas"], dtype=float)
            if isinstance(params["alphas"], (list, tuple, np.ndarray))
            else l1_alpha_grid(X, y, n_alphas=int(params["alphas"]), l1_ratio=1.0)
        )
        common = {
            "max_iter": params["max_iter"],
            "tol": params["tol"],
            "selection": params["selection"],
            "fit_intercept": params["fit_intercept"],
        }
        best, _, _ = select_alpha_by_ic(
            [{"alpha": float(a)} for a in alphas],
            lambda p: Lasso(alpha=p["alpha"], **common),
            X, y, dates, folds,
            verbose=bool(self.config.params.get("verbose", False)),
            label=self.config.model_id,
        )
        final = Lasso(alpha=best["alpha"], **common).fit(X, y)
        self._coef = np.asarray(final.coef_)
        self._intercept = float(final.intercept_) if params["fit_intercept"] else 0.0
        self._alpha = float(best["alpha"])
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

    Fits a plain `sklearn.linear_model.Ridge` on the pooled `(date, ticker)`
    feature panel against `y` (forward excess return); `α` is chosen per refit by
    a **temporal, purged forward-chain CV scored on per-date IC** (same grader as
    Lasso and the tree sweeps), then refit once at the selected `α`. Unlike Lasso, the L2 penalty
    keeps every coefficient non-zero and shrinks them smoothly toward zero.
    Critical property on collinear panels: when two features are highly
    correlated, Ridge splits their weight proportionally rather than
    arbitrarily assigning all weight to one or, worse (as L1 does on this
    project's data), cancelling them into opposing signs.

    Parameters (via `config.params`):

    - `alphas` (list of float, optional): explicit grid of α to search. If
      None, fit() builds a **data-scaled** grid (`models/_cv.ridge_alpha_grid`):
      ±4 decades around the mean Gram-matrix eigenvalue `trace(XᵀX)/p`, so the
      grid spans effectively-OLS → heavily-shrunk regardless of feature scaling
      or sample size. An int here sets the number of grid points (default 20).
      (A fixed absolute grid like 1e-4…1e4 is mostly below the shrinkage scale
      for a pooled fit and leaves nearly every point at OLS.)
    - `cv` (int, default 3): number of **purged, expanding-window, forward-chain**
      CV folds for picking α, scored on **per-date IC** (temporal — see
      `models/_cv.py`). NOT K-fold, NOT MSE.
    - `cv_embargo` (int, default 21): unique trailing dates purged before each
      validation fold; set ≥ the forward-return horizon so the label can't leak.
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
        # L2 has no finite alpha_max, so when `alphas` is unset fit() builds a
        # DATA-SCALED grid (models/_cv.ridge_alpha_grid) anchored to the Gram-matrix
        # eigenvalue scale; an int here is the number of grid points. A fixed
        # absolute grid (e.g. 1e-4..1e4) is mostly below the shrinkage-relevant
        # scale for a pooled ~50k-row fit, leaving nearly every point at OLS.
        alphas_cfg = self.config.params.get("alphas")
        alphas_resolved = 20 if alphas_cfg is None else alphas_cfg
        return {
            "alphas": alphas_resolved,
            "n_splits": int(self.config.params.get("cv", 3)),
            "cv_embargo": int(self.config.params.get("cv_embargo", 21)),
            "fit_intercept": bool(self.config.params.get("fit_intercept", True)),
        }

    def fit(self, panel: pl.DataFrame) -> None:
        from sklearn.linear_model import Ridge

        from ._cv import purged_forward_chain_folds, ridge_alpha_grid, select_alpha_by_ic

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats])
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        X = train.select(feats).to_numpy()
        y = train[target].to_numpy()
        dates = train["date"].to_numpy()
        params = self._params()

        # α selected by a temporal, purged forward-chain CV scored on per-date IC —
        # the SAME grader as Lasso and the tree sweeps (not sklearn RidgeCV's
        # non-temporal, MSE/R²-scored path). See models/_cv.py.
        folds = purged_forward_chain_folds(
            dates, n_splits=params["n_splits"], embargo=params["cv_embargo"]
        )
        alphas = (
            np.asarray(params["alphas"], dtype=float)
            if isinstance(params["alphas"], (list, tuple, np.ndarray))
            else ridge_alpha_grid(X, n_alphas=int(params["alphas"]))
        )
        best, _, _ = select_alpha_by_ic(
            [{"alpha": float(a)} for a in alphas],
            lambda p: Ridge(alpha=p["alpha"], fit_intercept=params["fit_intercept"]),
            X, y, dates, folds,
            verbose=bool(self.config.params.get("verbose", False)),
            label=self.config.model_id,
        )
        final = Ridge(alpha=best["alpha"], fit_intercept=params["fit_intercept"]).fit(X, y)
        self._coef = np.asarray(final.coef_)
        self._intercept = float(final.intercept_) if params["fit_intercept"] else 0.0
        self._alpha = float(best["alpha"])
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

    Fits a plain `sklearn.linear_model.ElasticNet` on the pooled `(date, ticker)`
    feature panel against `y` (forward excess return). The hybrid L1 + L2
    penalty interpolates between Lasso (l1_ratio=1) and Ridge (l1_ratio=0).
    `(α, l1_ratio)` are jointly selected per refit by a **temporal, purged
    forward-chain CV scored on per-date IC** (same grader as Lasso/Ridge and the
    tree sweeps), then the model is refit once at the selected pair.

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
    - `cv` (int, default 3): number of **purged, expanding-window, forward-chain**
      CV folds for jointly picking α and l1_ratio, scored on **per-date IC**
      (temporal — see `models/_cv.py`). NOT K-fold, NOT MSE.
    - `cv_embargo` (int, default 21): unique trailing dates purged before each
      validation fold; set ≥ the forward-return horizon so the label can't leak.
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
        # When `alphas` is unset, fit() generates a per-l1_ratio data-driven log
        # grid with this many points (see models/_cv.l1_alpha_grid); 25 keeps the
        # joint (alpha, l1_ratio) IC search tractable.
        alphas_cfg = self.config.params.get("alphas")
        alphas_resolved = 25 if alphas_cfg is None else alphas_cfg
        l1_ratios_cfg = self.config.params.get("l1_ratios")
        l1_ratios_resolved = (
            [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0] if l1_ratios_cfg is None else l1_ratios_cfg
        )
        return {
            "alphas": alphas_resolved,
            "l1_ratios": l1_ratios_resolved,
            "n_splits": int(self.config.params.get("cv", 3)),
            "cv_embargo": int(self.config.params.get("cv_embargo", 21)),
            "max_iter": int(self.config.params.get("max_iter", 5000)),
            "tol": float(self.config.params.get("tol", 1e-4)),
            "selection": str(self.config.params.get("selection", "cyclic")),
            "fit_intercept": bool(self.config.params.get("fit_intercept", True)),
        }

    def fit(self, panel: pl.DataFrame) -> None:
        from sklearn.linear_model import ElasticNet

        from ._cv import l1_alpha_grid, purged_forward_chain_folds, select_alpha_by_ic

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats])
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        X = train.select(feats).to_numpy()
        y = train[target].to_numpy()
        dates = train["date"].to_numpy()
        params = self._params()

        # (α, l1_ratio) jointly selected by a temporal, purged forward-chain CV
        # scored on per-date IC — the SAME grader as Lasso/Ridge and the tree
        # sweeps (not sklearn ElasticNetCV's non-temporal, MSE path). See _cv.py.
        folds = purged_forward_chain_folds(
            dates, n_splits=params["n_splits"], embargo=params["cv_embargo"]
        )
        explicit_alphas = isinstance(params["alphas"], (list, tuple, np.ndarray))
        candidates: list[dict] = []
        for l1 in params["l1_ratios"]:
            alphas = (
                np.asarray(params["alphas"], dtype=float)
                if explicit_alphas
                else l1_alpha_grid(X, y, n_alphas=int(params["alphas"]), l1_ratio=float(l1))
            )
            candidates.extend({"alpha": float(a), "l1_ratio": float(l1)} for a in alphas)
        common = {
            "max_iter": params["max_iter"],
            "tol": params["tol"],
            "selection": params["selection"],
            "fit_intercept": params["fit_intercept"],
        }
        best, _, _ = select_alpha_by_ic(
            candidates,
            lambda p: ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], **common),
            X, y, dates, folds,
            verbose=bool(self.config.params.get("verbose", False)),
            label=self.config.model_id,
        )
        final = ElasticNet(alpha=best["alpha"], l1_ratio=best["l1_ratio"], **common).fit(X, y)
        self._coef = np.asarray(final.coef_)
        self._intercept = float(final.intercept_) if params["fit_intercept"] else 0.0
        self._alpha = float(best["alpha"])
        self._l1_ratio = float(best["l1_ratio"])
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


class OLSCrossSectional(Model):
    """Cross-sectional ordinary least squares — the unregularized linear baseline.

    Plain `sklearn.linear_model.LinearRegression` on the pooled `(date, ticker)`
    feature panel against `y` (forward excess return). No penalty, no
    cross-validation, no hyperparameter.

    Why it exists: on a small curated panel with a large pooled sample the Gram
    matrix `XᵀX` is extremely well-conditioned (this project's headline: ~238k
    rows × 6 features), so regularization is unnecessary — the IC-scored forward-
    chain CV drives both Lasso and Ridge to α→0, i.e. they *reduce to this model*.
    OLS makes that explicit and removes the (inactive) penalty. On this panel
    `OLSCrossSectional`, `LassoCrossSectional`, and `RidgeCrossSectional` produce
    the same predictions to within selection noise; OLS is the honest headline.

    Parameters (via `config.params`):

    - `fit_intercept` (bool, default True): include an intercept (the
      cross-sectional excess-return target is ~zero-mean per date).
    """

    def __init__(self, config):
        super().__init__(config)
        self._coef: np.ndarray | None = None
        self._intercept: float | None = None
        self._feat_names: tuple[str, ...] = tuple(config.feature_cols)

    def fit(self, panel: pl.DataFrame) -> None:
        from sklearn.linear_model import LinearRegression

        feats = list(self.config.feature_cols)
        target = self.config.target_col
        train = panel.drop_nulls(subset=[target, *feats])
        if train.height == 0:
            raise ValueError("No training rows after dropping nulls — check warmup/embargo.")

        X = train.select(feats).to_numpy()
        y = train[target].to_numpy()
        fit_intercept = bool(self.config.params.get("fit_intercept", True))
        model = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
        self._coef = np.asarray(model.coef_)
        self._intercept = float(model.intercept_) if fit_intercept else 0.0
        self._fitted = True

    def predict(self, panel: pl.DataFrame) -> pl.DataFrame:
        self._check_fitted()
        if self._coef is None:
            raise RuntimeError("OLSCrossSectional has no fitted coefficients")
        feats = list(self.config.feature_cols)
        X = panel.select(feats).fill_null(0.0).to_numpy()
        preds = X @ self._coef + (self._intercept or 0.0)
        return self._format_predictions(panel, preds)

    def feature_importance(self) -> dict[str, float]:
        """Coefficient magnitude per feature. Sign-preserved (dense — no shrinkage)."""
        self._check_fitted()
        if self._coef is None:
            raise RuntimeError("OLSCrossSectional has no fitted coefficients")
        return dict(zip(self._feat_names, [float(c) for c in self._coef], strict=True))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_config(self.config, path / "config.json")
        with (path / "state.pkl").open("wb") as f:
            pickle.dump(
                {"coef": self._coef, "intercept": self._intercept, "feat_names": self._feat_names},
                f,
            )

    @classmethod
    def load(cls, path: Path) -> OLSCrossSectional:
        m = cls(load_config(path / "config.json"))
        with (path / "state.pkl").open("rb") as f:
            state = pickle.load(f)
        m._coef = state["coef"]
        m._intercept = state["intercept"]
        m._feat_names = state["feat_names"]
        m._fitted = True
        return m
