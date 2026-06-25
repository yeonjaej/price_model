"""Optuna hyperparameter sweep for tree-boosting models.

Optimizes the hyperparameter space of a boosting model on a given feature
panel against walk-forward cross-validation IC. Supports LightGBM, XGBoost,
and CatBoost — the sweep grid is library-aware.

Lookahead-safety: the CV splits are PURGED walk-forward (de Prado Ch. 7)
with a configurable embargo. Each Optuna trial fits the model on k-1 folds
and evaluates IC on the held-out fold; the embargo gap between train and
val ensures no overlapping forward returns leak into evaluation.

The sweep optimizes mean-CV-fold IC. After Optuna picks the best HPs, the
script optionally writes a tuned YAML config that can be used by the CLI
runner directly.

Usage:
    PYTHONPATH=src python scripts/optuna_sweep.py --experiment extended_kaggle_v3_curated
    PYTHONPATH=src python scripts/optuna_sweep.py --experiment extended_kaggle_v3_xgboost --n-trials 100
    PYTHONPATH=src python scripts/optuna_sweep.py --experiment extended_kaggle_v3_catboost --write-tuned

The output YAML (with --write-tuned) is written to
config/experiments/<experiment>_tuned.yaml — identical except the LightGBM/
XGBoost/CatBoost params block is replaced with the Optuna-selected HPs.
"""

from __future__ import annotations

import argparse
import copy
import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from rich.console import Console
from rich.table import Table
from scipy.stats import spearmanr

from price_model.data.loaders import load_panel
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig


def _purged_walk_forward_folds(
    dates: list, n_folds: int, embargo_days: int
) -> list[tuple[list, list]]:
    """Build n_folds (train_dates, val_dates) splits with embargo.

    Date list is partitioned into n_folds contiguous chunks; chunk i is the
    val fold, chunks 0..i-1 are training. The last `embargo_days` dates of
    the training portion immediately preceding the val fold are dropped to
    prevent overlap-leakage of forward returns.
    """
    n = len(dates)
    chunk = n // n_folds
    folds: list[tuple[list, list]] = []
    for i in range(1, n_folds):  # expanding-window: skip fold 0 as val
        val_start = i * chunk
        val_end = (i + 1) * chunk if i < n_folds - 1 else n
        val_dates = dates[val_start:val_end]
        train_end = max(0, val_start - embargo_days)
        train_dates = dates[:train_end]
        if len(train_dates) < embargo_days or len(val_dates) < 5:
            continue
        folds.append((train_dates, val_dates))
    return folds


def _per_date_ic(predictions: np.ndarray, targets: np.ndarray, dates: np.ndarray) -> float:
    """Mean per-date Spearman IC."""
    rows = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 5:
            continue
        rho, _ = spearmanr(predictions[mask], targets[mask])
        if rho is not None and not math.isnan(rho):
            rows.append(float(rho))
    return float(np.mean(rows)) if rows else 0.0


def _objective_factory(
    cfg: dict,
    matrix: pl.DataFrame,
    folds: list[tuple[list, list]],
    model_class: str,
    console: Console,
):
    feats = cfg["features"]
    target = "y"

    def objective(trial) -> float:
        import optuna  # noqa: F401  (trial type comes from optuna)

        # Library-specific HP grid
        if model_class == "LightGBMModel":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 200),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.95),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.95),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 50, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 50, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "bagging_freq": 5,
                "verbosity": -1,
                "seeds": [42],
            }
        elif model_class == "XGBoostModel":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "tree_method": "hist",
                "verbosity": 0,
                "seeds": [42],
            }
        elif model_class == "CatBoostModel":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 50, log=True),
                "rsm": trial.suggest_float("rsm", 0.5, 0.95),
                "subsample": trial.suggest_float("subsample", 0.5, 0.95),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "bootstrap_type": "Bernoulli",
                "verbose": False,
                "allow_writing_files": False,
                "seeds": [42],
            }
        else:
            raise ValueError(f"HP sweep not configured for {model_class}")

        fold_ics: list[float] = []
        for train_dates, val_dates in folds:
            train = matrix.filter(pl.col("date").is_in(train_dates))
            val = matrix.filter(pl.col("date").is_in(val_dates))
            train = train.drop_nulls(subset=[target, *feats])
            val = val.drop_nulls(subset=[target, *feats])
            if train.height == 0 or val.height == 0:
                continue

            model_cfg = ModelConfig(
                model_id="optuna_trial",
                feature_cols=tuple(feats),
                target_col=target,
                params=params,
            )
            model = build_model(model_class, model_cfg)
            try:
                model.fit(train)
                preds = model.predict(val)["prediction"].to_numpy()
            except Exception as e:
                # Trial failed — return very bad score
                console.print(f"  [yellow]Trial failed:[/yellow] {e}")
                return -1.0

            y = val[target].to_numpy()
            d = val["date"].to_numpy()
            ic = _per_date_ic(preds, y, d)
            fold_ics.append(ic)

        if not fold_ics:
            return -1.0
        return float(np.mean(fold_ics))

    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiment", required=True,
        help="Experiment YAML name (without .yaml suffix) to sweep.",
    )
    ap.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials.",
    )
    ap.add_argument(
        "--n-folds", type=int, default=4,
        help="Number of walk-forward CV folds.",
    )
    ap.add_argument(
        "--embargo-days", type=int, default=6,
        help="Embargo gap between train and val in CV (matches harness).",
    )
    ap.add_argument(
        "--write-tuned", action="store_true",
        help="Write <experiment>_tuned.yaml with the best HPs.",
    )
    ap.add_argument(
        "--max-date", default=None,
        help=(
            "If set (YYYY-MM-DD), restrict the Optuna CV to dates <= this. "
            "Use this to hold out a post-cutoff period for HP-free "
            "out-of-sample evaluation. The output YAML name encodes the "
            "cutoff so it does not collide with a non-restricted sweep."
        ),
    )
    args = ap.parse_args()
    max_date: date | None = (
        date.fromisoformat(args.max_date) if args.max_date else None
    )

    console = Console()
    try:
        import optuna
    except ImportError:
        console.print("[red]optuna is not installed. Install via: pip install optuna")
        return

    # Suppress noisy Optuna stdout
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg_path = Path("config/experiments") / f"{args.experiment}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    # Identify the boosting model in the config — we sweep the first one
    boosting_specs = [
        m for m in cfg["models"]
        if m["class"] in ("LightGBMModel", "XGBoostModel", "CatBoostModel")
    ]
    if not boosting_specs:
        console.print("[red]No LightGBM/XGBoost/CatBoost model found in the experiment.")
        return
    boost_spec = boosting_specs[0]
    model_class = boost_spec["class"]

    console.print(f"Sweeping [bold]{model_class}[/bold] on experiment "
                  f"[bold]{cfg['experiment_id']}[/bold]")
    console.print(f"Trials: {args.n_trials} | CV folds: {args.n_folds} | "
                  f"embargo: {args.embargo_days} days")

    # ----- Load panel + build feature matrix once (shared across trials) -----
    panel = load_panel(
        universe=cfg["data"]["universe"],
        start=cfg["data"]["start"],
        end=cfg["data"].get("end"),
        pit_filter=cfg["data"].get("pit_filter", False),
    )
    matrix = build_feature_matrix(
        panel,
        feature_names=cfg["features"],
        normalize_kind=cfg.get("normalize_kind", "zscore"),
        target_horizon=cfg["target_horizon"],
    )
    matrix = drop_warmup_rows(matrix, cfg["features"]).drop_nulls("y").sort("date")
    console.print(
        f"Feature matrix: {matrix.height:,} rows × "
        f"{matrix['date'].n_unique():,} dates."
    )

    # ----- Optional hold-out: truncate to dates <= max_date so Optuna sees -----
    # ----- no post-cutoff returns. The CLI walk-forward harness can then  -----
    # ----- be run separately on the full date range, and the post-cutoff  -----
    # ----- slice serves as a clean HP-free out-of-sample test.            -----
    if max_date is not None:
        before = matrix.height
        matrix = matrix.filter(pl.col("date") <= max_date)
        console.print(
            f"[bold yellow]Hold-out cutoff[/bold yellow]: restricted to "
            f"dates <= {max_date}: {matrix.height:,} rows "
            f"(dropped {before - matrix.height:,} post-cutoff rows). "
            f"Optuna will not see any data past {max_date}."
        )

    # ----- Optional regime lower bound: confine CV to dates >= train_start so the
    # sweep's folds match the regime-confined CLI training window across panels.
    ts_cfg = cfg.get("walk_forward", {}).get("train_start")
    if ts_cfg:
        before = matrix.height
        matrix = matrix.filter(pl.col("date") >= date.fromisoformat(ts_cfg))
        console.print(
            f"[bold yellow]Regime lower bound[/bold yellow]: restricted to "
            f"dates >= {ts_cfg}: {matrix.height:,} rows "
            f"(dropped {before - matrix.height:,} pre-regime rows)."
        )

    # ----- Build CV folds -----
    all_dates = sorted(matrix["date"].unique().to_list())
    folds = _purged_walk_forward_folds(all_dates, args.n_folds, args.embargo_days)
    console.print(f"Generated {len(folds)} purged walk-forward CV folds.")
    for i, (train_dates, val_dates) in enumerate(folds):
        console.print(
            f"  Fold {i}: train {len(train_dates)} dates "
            f"[{train_dates[0]} → {train_dates[-1]}], "
            f"val {len(val_dates)} dates "
            f"[{val_dates[0]} → {val_dates[-1]}]"
        )

    # ----- Optuna study -----
    objective = _objective_factory(cfg, matrix, folds, model_class, console)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ----- Report -----
    console.rule("[bold green]Best trial")
    best = study.best_trial
    table = Table()
    table.add_column("hyperparameter")
    table.add_column("value", justify="right")
    table.add_row("CV mean IC", f"{best.value:+.5f}")
    for k, v in best.params.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.6g}")
        else:
            table.add_row(k, str(v))
    console.print(table)

    # ----- Optional: write tuned YAML -----
    if args.write_tuned:
        # Suffix encodes whether this sweep used a hold-out cutoff. Keeps
        # held-out and non-held-out tuned configs from colliding.
        suffix = (
            f"_hp_pre{max_date.strftime('%Y%m%d')}"
            if max_date is not None
            else "_tuned"
        )
        tuned_cfg = copy.deepcopy(cfg)
        tuned_cfg["experiment_id"] = f"{cfg['experiment_id']}{suffix}"
        for m in tuned_cfg["models"]:
            if m["class"] == model_class:
                m["id"] = f"{m['id']}{suffix}"
                # Replace params with Optuna-selected ones, preserving any
                # original config keys not in the HP space (e.g. seeds list,
                # val_fraction, early_stopping_rounds).
                new_params = {**m.get("params", {}), **best.params}
                # Re-add the multi-seed list from the original
                if "seeds" in m.get("params", {}):
                    new_params["seeds"] = m["params"]["seeds"]
                m["params"] = new_params
        out_path = cfg_path.parent / f"{args.experiment}{suffix}.yaml"
        out_path.write_text(yaml.safe_dump(tuned_cfg, sort_keys=False))
        console.print(f"Tuned YAML written to [bold]{out_path}[/bold]")


if __name__ == "__main__":
    main()
