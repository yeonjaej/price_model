"""Typer-based CLI. Single entry point: `python -m price_model.cli <subcommand>`."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import typer
import yaml
from rich.console import Console
from rich.table import Table

from price_model.data.loaders import load_panel
from price_model.eval.metrics import compare_models
from price_model.features.base import list_features
from price_model.features.pipeline import build_feature_matrix, drop_warmup_rows
from price_model.models import build_model
from price_model.models.base import ModelConfig
from price_model.pipeline.walk_forward import join_with_realized, run_walk_forward
from price_model.serving.store import PredictionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = typer.Typer(help="price-model: cross-sectional equity return predictor")
console = Console()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@app.command("refresh-data")
def refresh_data(
    universe: str = "sp500",
    start: str = "2017-01-01",
    end: Optional[str] = None,
    tickers: Optional[str] = None,
) -> None:
    """Fetch (or extend) the cached price panel for a universe.

    --tickers AAPL,MSFT,NVDA can override the universe for ad-hoc fetches.
    """
    from price_model.data.sources import yfinance_source

    if tickers:
        names = [t.strip() for t in tickers.split(",") if t.strip()]
        panel = yfinance_source.fetch(names, start=start, end=end)
    else:
        panel = load_panel(universe=universe, start=start, end=end)
    console.print(
        f"Loaded panel: {panel.height:,} rows, {panel['ticker'].n_unique()} tickers, "
        f"{panel['date'].min()} → {panel['date'].max()}"
    )


@app.command("list-features")
def list_features_cmd() -> None:
    """List all registered features."""
    table = Table(title="Registered features")
    table.add_column("name")
    table.add_column("lookback_days")
    from price_model.features.base import FEATURE_REGISTRY
    for name in list_features():
        feat = FEATURE_REGISTRY[name]
        table.add_row(name, str(feat.lookback_days))
    console.print(table)


@app.command("run")
def run_experiment(
    experiment: Annotated[str, typer.Option("--experiment", "-e")] = "baseline",
) -> None:
    """Run a walk-forward experiment defined in config/experiments/<name>.yaml."""
    cfg_path = Path("config/experiments") / f"{experiment}.yaml"
    cfg = _load_yaml(cfg_path)

    panel = load_panel(
        universe=cfg["data"]["universe"],
        start=cfg["data"]["start"],
    )
    matrix = build_feature_matrix(
        panel,
        feature_names=cfg["features"],
        normalize_kind=cfg.get("normalize_kind", "zscore"),
        target_horizon=cfg["target_horizon"],
    )
    matrix = drop_warmup_rows(matrix, cfg["features"])

    store = PredictionStore()
    try:
        all_preds_by_model: list[pl.DataFrame] = []
        for m in cfg["models"]:
            console.rule(f"[bold]{m['id']}")
            config = ModelConfig(
                model_id=m["id"],
                feature_cols=tuple(cfg["features"]),
                target_col="y",
                params=m.get("params", {}),
            )
            model = build_model(m["class"], config)
            preds = run_walk_forward(
                matrix,
                model=model,
                feature_cols=cfg["features"],
                target_col="y",
                experiment_id=cfg["experiment_id"],
                horizon_days=cfg["target_horizon"],
                refit_freq_days=cfg["walk_forward"]["refit_freq_days"],
                embargo_days=cfg["walk_forward"]["embargo_days"],
                min_train_days=cfg["walk_forward"]["min_train_days"],
                store=store,
            )
            preds = preds.with_columns(pl.lit(m["id"]).alias("model_id"))
            all_preds_by_model.append(preds)

        if not all_preds_by_model:
            console.print("[red]No predictions produced")
            return

        joined = pl.concat(all_preds_by_model)
        eval_df = join_with_realized(joined, matrix)
        eval_df = eval_df.rename({"date": "date"})  # already named date; no-op for clarity
        summary = compare_models(eval_df, horizon_days=cfg["target_horizon"])

        console.rule("[bold green]Model comparison")
        table = Table()
        for col in summary.columns:
            table.add_column(col)
        for row in summary.iter_rows():
            table.add_row(*[str(round(v, 4)) if isinstance(v, float) else str(v) for v in row])
        console.print(table)
    finally:
        store.close()


@app.command("dashboard")
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


@app.command("show-predictions")
def show_predictions(
    model_id: Optional[str] = None,
    limit: int = 50,
) -> None:
    """Print the latest predictions from the store."""
    store = PredictionStore()
    try:
        df = store.latest_predictions(model_ids=[model_id] if model_id else None)
        if df.height == 0:
            console.print("[yellow]No predictions in store. Run an experiment first.")
            return
        console.print(df.head(limit))
    finally:
        store.close()


if __name__ == "__main__":
    app()
