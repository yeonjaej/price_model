"""Purged, embargoed forward-chain cross-validation folds for temporal panels.

The cross-sectional linear models tune their regularization with sklearn's `*CV`
estimators (`LassoCV` / `RidgeCV` / `ElasticNetCV`). Passing an **integer** `cv`
makes sklearn use `KFold(shuffle=False)` — and because the training panel is sorted
`["ticker", "date"]` (required for correct rolling-feature computation), those
contiguous folds fall on **ticker** boundaries. The result is a *non-temporal*,
by-ticker split with **no purge**: validation rows share their dates with training
rows, and on an `H`-day forward-return target adjacent rows leak across the
train/val boundary. That silently selects the regularization strength with future
information — the exact mistake a temporal task must avoid.

`purged_forward_chain_folds` builds the same expanding-window, embargoed,
forward-chain folds the tree hyperparameter sweeps use (`scripts/optuna_sweep.py`),
keyed on the **date** axis, so every model in the project selects hyperparameters
the same temporally-honest way. The return value is an explicit list of
`(train_idx, val_idx)` index arrays, which is a valid `cv` argument to any sklearn
`*CV` estimator or `cross_val_*` helper.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


def purged_forward_chain_folds(
    dates: NDArray,
    n_splits: int = 5,
    embargo: int = 21,
) -> list[tuple[NDArray, NDArray]]:
    """Expanding-window, purged, forward-chain CV folds for a temporal panel.

    The unique dates are sorted ascending and cut into ``n_splits + 1`` contiguous
    chunks. Chunk 0 is the initial training seed (never validated); chunk ``i``
    (1..n_splits) is a validation block whose training set is *all earlier dates
    minus the last ``embargo`` of them* — the purge that stops an ``H``-day forward
    target from straddling the boundary. Every training date is therefore strictly
    earlier than (and ``embargo`` dates removed from) its validation block.

    Args:
        dates: per-row date values (array-like, any dtype, **any order** — rows
            need not be pre-sorted; folds are built on the unique-date axis).
        n_splits: number of (train, val) folds to produce. Clamped down if there
            are too few unique dates.
        embargo: number of unique trailing dates dropped from each fold's training
            set immediately before its validation block. Set ``>=`` the forward
            return horizon (in trading days) so the label cannot leak.

    Returns:
        List of ``(train_idx, val_idx)`` int arrays indexing into ``dates``. Usable
        directly as the ``cv`` argument of sklearn ``*CV`` estimators.

    Raises:
        ValueError: if fewer than two temporally-valid folds can be built (the
            training window is too short for the requested embargo / splits).
    """
    dates = np.asarray(dates)
    uniq = np.unique(dates)  # sorted ascending
    n = len(uniq)
    n_splits = min(n_splits, n - 1)
    if n_splits < 2:
        raise ValueError(
            f"Cannot build temporal CV folds: only {n} unique dates "
            f"(need >= 3 for >= 2 folds). Check the training window / embargo."
        )

    chunk = n // (n_splits + 1)
    # On short windows the requested embargo can starve every fold's training set;
    # clamp it so the first fold keeps >= 1 training date. Temporal ordering (train
    # strictly before val) is preserved regardless — only the purge width shrinks.
    # Production windows (hundreds of dates) are long enough that this never fires.
    embargo = max(0, min(embargo, chunk - 1))
    folds: list[tuple[NDArray, NDArray]] = []
    for i in range(1, n_splits + 1):
        val_start = i * chunk
        val_end = (i + 1) * chunk if i < n_splits else n
        val_dates = uniq[val_start:val_end]
        train_end = max(0, val_start - embargo)
        train_dates = uniq[:train_end]
        if train_dates.size == 0 or val_dates.size == 0:
            # Early fold has no admissible history once the embargo is applied —
            # skip it rather than train on a leaking or empty window.
            continue
        train_idx = np.flatnonzero(np.isin(dates, train_dates))
        val_idx = np.flatnonzero(np.isin(dates, val_dates))
        if train_idx.size and val_idx.size:
            folds.append((train_idx, val_idx))

    if len(folds) < 2:
        raise ValueError(
            "Could not build >= 2 temporally-valid CV folds with embargo="
            f"{embargo} over {n} unique dates — training window too short."
        )
    return folds


def l1_alpha_grid(
    X: NDArray, y: NDArray, n_alphas: int = 100, eps: float = 1e-3, l1_ratio: float = 1.0
) -> NDArray:
    """Data-driven α grid for L1/ElasticNet, matching sklearn's `*CV` convention.

    `alpha_max = max_j |x_j·y| / (n · l1_ratio)` is the smallest α that zeros every
    coefficient; the grid is `n_alphas` log-spaced values from `alpha_max·eps` up to
    `alpha_max`. Features/target are centered to match `fit_intercept=True`.
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    alpha_max = float(np.max(np.abs(Xc.T @ yc))) / (n * max(l1_ratio, 1e-3))
    if not np.isfinite(alpha_max) or alpha_max <= 0:
        alpha_max = 1.0
    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), int(n_alphas))


def ridge_alpha_grid(
    X: NDArray, n_alphas: int = 20, lo_decade: float = -4.0, hi_decade: float = 4.0
) -> NDArray:
    """Data-scaled α grid for Ridge (L2 has no finite `alpha_max`).

    Ridge shrinks the k-th principal direction by `d_k / (d_k + α)`, where `d_k`
    are the eigenvalues of the centered Gram matrix `XᵀX`, so α only matters
    *relative to that spectrum*. A fixed absolute grid (e.g. 1e-4 … 1e4) is mostly
    below the spectrum for a pooled fit (eigenvalues ∝ n) and leaves nearly every
    point at effectively-OLS. Instead anchor to the **mean eigenvalue**
    `trace(XᵀX) / p` and span `10**lo_decade … 10**hi_decade` around it (default
    ±4 decades) — running from effectively-OLS to heavily-shrunk regardless of
    feature scaling or sample size.
    """
    Xc = X - X.mean(axis=0)
    scale = float((Xc * Xc).sum()) / X.shape[1]  # trace(XᵀX_centered) / n_features
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.logspace(lo_decade, hi_decade, int(n_alphas)) * scale


def select_alpha_by_ic(
    candidates: list[dict[str, Any]],
    make_estimator: Callable[[dict[str, Any]], Any],
    X: NDArray,
    y: NDArray,
    dates: NDArray,
    folds: list[tuple[NDArray, NDArray]],
    verbose: bool = False,
    label: str = "",
) -> tuple[dict[str, Any], float, list[tuple[dict[str, Any], float]]]:
    """Pick the hyperparameter dict that maximizes mean per-fold, per-date IC.

    For each candidate, fit a fresh estimator on each fold's training rows, predict
    its validation rows, and score by mean per-date IC; the candidate's selection
    score is the mean of its fold ICs. This is the **same temporally-honest,
    IC-scored grader the tree sweeps use** — replacing sklearn `*CV`'s internal
    MSE path so linear and tree hyperparameters are chosen on the same criterion.

    Per-date IC is Spearman = Pearson-of-ranks. To stay fast across many candidates,
    each validation fold's date groups and the centered rank of the target within
    each group are precomputed once and reused for every candidate (no per-call
    DataFrame construction). Dates with < 5 valid rows are skipped, matching the
    project's IC definition.

    Returns `(best_params, best_score, [(params, score), ...])`. If every candidate
    degenerates to NaN (e.g. all α zero the model), falls back to the
    least-regularized candidate so a usable model is still returned.
    """
    from scipy.stats import rankdata

    # Precompute per fold: (train_idx, val_idx, [(local_rows, centered_y_rank, ss), ...]).
    fold_groups: list[tuple[NDArray, NDArray, list]] = []
    for tr, va in folds:
        dv, yv = dates[va], y[va]
        groups = []
        for ud in np.unique(dv):
            loc = np.flatnonzero(dv == ud)
            if loc.size < 5:
                continue
            yr = rankdata(yv[loc])
            yr_c = yr - yr.mean()
            ss = float(yr_c @ yr_c)
            if ss > 0:  # target not constant on this date
                groups.append((loc, yr_c, ss))
        fold_groups.append((tr, va, groups))

    if verbose:
        tag = f"[{label}] " if label else ""
        n_dates = np.unique(dates).size
        print(
            f"\n{tag}training matrix: {X.shape[0]} rows × {X.shape[1]} features "
            f"over {n_dates} dates (~{X.shape[0] / max(n_dates, 1):.0f} names/date)"
        )
        print(f"{tag}purged forward-chain CV — {len(fold_groups)} folds (embargo applied):")
        for i, (tr, va, groups) in enumerate(fold_groups, 1):
            td, vd = np.unique(dates[tr]), np.unique(dates[va])
            print(
                f"  fold {i}: train {td.min()}..{td.max()} ({td.size}d) | "
                f"val {vd.min()}..{vd.max()} ({vd.size}d, {len(groups)} scored dates)"
            )
        agrid = sorted({float(c["alpha"]) for c in candidates})
        print(f"  alpha grid: {len(agrid)} pts, {agrid[0]:.4e} .. {agrid[-1]:.4e}")
        if any("l1_ratio" in c for c in candidates):
            print(f"  l1_ratios: {sorted({float(c['l1_ratio']) for c in candidates})}")

    def _fold_ic(pred_va: NDArray, groups: list) -> float:
        rhos = []
        for loc, yr_c, yr_ss in groups:
            pr = rankdata(pred_va[loc])
            pr_c = pr - pr.mean()
            denom = (pr_c @ pr_c) * yr_ss
            if denom > 0:  # prediction not constant on this date
                rhos.append(float((pr_c @ yr_c) / np.sqrt(denom)))
        return float(np.mean(rhos)) if rhos else float("nan")

    best_params: dict[str, Any] | None = None
    best_score = -np.inf
    table: list[tuple[dict[str, Any], float]] = []
    for params in candidates:
        fold_ics = [
            _fold_ic(make_estimator(params).fit(X[tr], y[tr]).predict(X[va]), groups)
            for tr, va, groups in fold_groups
        ]
        valid = [v for v in fold_ics if np.isfinite(v)]
        score = float(np.mean(valid)) if valid else float("nan")
        table.append((params, score))
        if np.isfinite(score) and score > best_score:
            best_params, best_score = params, score
    if best_params is None:
        best_params = min(candidates, key=lambda p: p.get("alpha", 0.0))
        best_score = float("nan")

    if verbose:
        tag = f"[{label}] " if label else ""
        ranked = sorted(table, key=lambda t: (t[1] if np.isfinite(t[1]) else -np.inf), reverse=True)
        print(f"{tag}candidate scores (mean CV IC), top 10 of {len(table)}:")
        for params, score in ranked[:10]:
            extra = f" l1={params['l1_ratio']:.2f}" if "l1_ratio" in params else ""
            mark = "  <== SELECTED" if params is best_params else ""
            print(f"   alpha={params['alpha']:.4e}{extra}   IC={score:+.4f}{mark}")
        be = f" l1_ratio={best_params['l1_ratio']:.2f}" if "l1_ratio" in best_params else ""
        print(f"{tag}SELECTED alpha={best_params['alpha']:.4e}{be}  (CV IC={best_score:+.4f})\n")

    return best_params, best_score, table
