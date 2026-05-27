# price-model

Cross-sectional equity return predictor. Trains on the S&P 500, deploys point + interval
forecasts for any subset (default: top 20 by market cap as of 2026-01-01), evaluates with
walk-forward backtesting, and surfaces predictions in a Streamlit dashboard.

## Design

See conversation log / docs for the full design rationale. Short version:

- **Universe**: trains on a curated 156-name large-cap subset of the S&P 500 (not the full
  ~500-name index — see "Known v0 limitations" below). Predicts for any subset at deploy
  time; the default deployment universe is the top 20 by market cap as of 2026-01-01.
- **Target**: 5-day forward excess return (return minus universe mean) — strips out the market move.
- **Features**: stock-agnostic, cross-sectionally normalized within each date — so AMD's features
  on 2026-05-26 are comparable to AAPL's on 2020-03-15 and to any other stock on any other day.
  Currently ships ten features: five base technicals (return_5d, momentum_60, vol_20, rsi_14,
  distance_ma_200), sector-relative momentum, idiosyncratic vol (residual std after rolling
  beta-adjustment), and cross-sectional ranks of momentum/vol/MA-distance.
- **Models**: pluggable via a `Model` ABC. v0 ships a zero-baseline, a last-return baseline,
  LightGBM, and an optional Chronos zero-shot foundation model. Add new models by dropping a
  file into `src/price_model/models/`.
- **Eval**: walk-forward with monthly refit and an embargo equal to the target horizon.
  Same harness across every model — comparisons are honest.
- **Storage**: every prediction (backtest, walk-forward, live) lands in a single DuckDB
  table. Dashboard, evaluation, and ensembles all read from there.
- **Dashboard**: Streamlit. Reads from the prediction store; never calls a model directly.

## Quickstart

```bash
# 1. Install
uv venv && source .venv/bin/activate     # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"               # or: pip install -e ".[dev]"

# 2. Pull a small sample of data (cached to data/raw/)
python -m price_model.cli refresh-data --tickers AAPL,MSFT,NVDA --start 2020-01-01

# 3. Run an experiment (walk-forward across all configured models)
python -m price_model.cli run --experiment baseline   # base 5 features
python -m price_model.cli run --experiment extended   # all 10 features

# 4. Launch the dashboard
python -m price_model.cli dashboard
```

### Optional: Chronos zero-shot foundation model

```bash
pip install ".[chronos]"                              # ~1 GB (torch + transformers)
python -m price_model.cli run --experiment chronos    # CPU-slow; GPU recommended
```

The Chronos experiment defaults to `amazon/chronos-t5-tiny` on the top-20 universe over
~3 years — that keeps wall-clock under an hour on CPU. Edit `config/experiments/chronos.yaml`
to point at a larger model or universe.

## Repo layout

```
config/                 # YAML configs for data, features, models, experiments
src/price_model/
  data/                 # sources, universe, loaders, splits
  features/             # registry + technical + cross-sectional + pipeline
  models/               # base ABC + baseline + boosting
  eval/                 # metrics, walk-forward report helpers
  serving/              # prediction store (DuckDB)
  pipeline/             # train, predict, walk-forward
  dashboard/            # Streamlit
  cli.py                # typer entry point
tests/                  # pytest (incl. no-leakage tests)
scripts/                # one-off ops scripts
.github/workflows/      # CI + nightly
```

## Honest framing

This is a **portfolio overlay**, not an alpha factory. Expected information coefficient on
US large-caps with daily/weekly horizons is roughly 0.02–0.05 if everything is done right.
That translates to ~50–150 bps of annual excess return over an equal-weighted benchmark —
real, but modest. If your backtest shows Sharpe > 2 on 20 mega-caps, suspect a bug before
you suspect a breakthrough.

## Known v0 limitations (tracked as TODOs)

- **Universe is 156 large-caps, not the full S&P 500.** `src/price_model/data/universes/sp500.txt`
  is a hand-curated subset of the most liquid large-cap names — roughly the top third of the
  actual index by market cap. The actual S&P 500 has ~503 constituents. The Fundamental Law
  of Active Management gives `IR ≈ IC × √breadth`; tripling the universe to the full index
  would roughly double the trading-relevant Sharpe at unchanged signal quality. Worth doing
  once the project warrants the operational complexity (more yfinance failures, manual sector
  mappings or a scrape, more heterogeneous cross-section).
- **Static universe membership → survivorship bias** in backtests. Today's "top-156" list
  over-represents historical winners; companies that fell out of the index over the test
  window aren't present. Already dropped FISV (renamed), WBA (taken private 2025),
  FI (Fiserv — Yahoo Finance API consistently 404s on this symbol; symptom of post-rename
  quote/historical endpoint disagreement), and MMC (transient yfinance failures). Proper fix
  is point-in-time membership reconstruction from Wikipedia's index-change history or a paid
  PIT data provider. Until then, all long-window backtests are upward-biased by an unknown
  but non-zero amount.
- yfinance fundamentals are sparse and not strictly point-in-time. Upgrade to Sharadar
  when fundamentals matter.
- No transaction-cost model yet — backtest Sharpes are gross, not net.
- Ensemble layer is a stub (equal-weight only). Add IVW and stacking after first results.
- Static GICS sector map. Sector reclassifications happen but are infrequent; live-data
  upgrade would pull this from FactSet history.
