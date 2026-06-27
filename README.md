# price-model

A point-in-time (PIT) corrected cross-sectional equity return predictor on the S&P 500, evaluated under strict HP-free held-out protocols at the 21-day (approximately monthly) forward horizon.

## Executive summary

A PIT-corrected, held-out 21-day cross-sectional return predictor on the S&P 500. The headline is reported **regime-confined**: trained only on the post-2022-10 bull regime (train 2022-10-10 → 2024-11-30, single refit; trees held-out-Optuna-tuned with the cutoff at 2024-12-31) and tested on 2025-01-02 → 2026-06-24 (348 dates). This is the honest "deploy when the regime resembles recent training" scenario; the expansion on longer time-scale across-regime is to be studied.

A **6-feature cross-sectional OLS regression** on documented academic anomalies (Han-He-Rapach-Zhou 2024) is the strongest model on **both** gross signal and net-of-cost return — beating long-horizon momentum and held-out-tuned LightGBM / XGBoost / CatBoost:

Long-short Sharpe is for a **top-20% / bottom-20% (quintile), equal-weighted** portfolio, computed on the **~17 non-overlapping 21-day rebalances** of the test window; net columns deduct round-trip turnover × cost (3 / 10 / 20 bp per side). **Gross IC** is the per-date rank correlation averaged over all OOS dates, but its **t-stat is Newey–West HAC-corrected (Bartlett lag 21)** to account for the 21-day return overlap — the naive daily t (≈ 10) is inflated ~√21× by that overlap and is *not* the honest figure. See [Metric definitions](#metric-definitions).

| Model (regime-confined) | Gross IC | t-stat (HAC) | Gross Sharpe | Net 3 bp | Net 10 bp | Net 20 bp | Annual turnover |
|---|---|---|---|---|---|---|---|
| **OLS, 6-feature** (Ridge ≡ Lasso) | **+0.0871** | **+2.72** | **+2.09** | **+2.05** | **+1.97** | **+1.85** | 9× |
| 36-month momentum factor | +0.0497 | +2.31 | +1.41 | +1.39 | +1.34 | +1.28 | 4× |
| LightGBM, held-out Optuna (rank panel) | +0.0689 | +1.64 | +1.06 | +1.02 | +0.93 | +0.79 | 11× |
| CatBoost, held-out Optuna (rank panel) | +0.0610 | +1.30 | +0.91 | +0.88 | +0.81 | +0.71 | 10× |
| XGBoost, held-out Optuna (rank panel) | +0.0586 | +1.29 | +0.80 | +0.76 | +0.69 | +0.58 | 10× |

Only **OLS (HAC t +2.72)** and **momentum (+2.31)** are individually significant; **none of the tuned trees clears |t| > 2 once the overlap is corrected** (+1.3 to +1.6), so their apparent edge is not distinguishable from zero on this window.

Observations:

- **OLS wins gross *and* net** — the only configuration in this project where the model beats momentum net-of-cost, and by a wide margin: gross Sharpe **+2.09** and, because the curated 6-feature book turns over only ~9×/yr, **net +1.85** at 20 bp (vs momentum +1.28 and the best tree +0.79).
- **Regularization is inert — the model is plain OLS.** On this panel the IC-scored forward-chain CV drives both Lasso and Ridge to α→0; all three (OLS ≡ Ridge ≡ Lasso) produce **identical** predictions to every digit. With ~238k training rows × 6 features the Gram matrix is hugely over-determined, so shrinkage has nothing to do — the CV-IC curve decays monotonically with α. The story is the **curated panel**, not the regularizer. (Pruning the prior 9-feature panel to 6 — dropping `idio_vol_20`, `max_return_21d`, and the anti-generalizing `beta_60` — is most of the lift.)
- **Every tuned tree trails — and none is statistically significant.** Even with held-out Optuna on each tree's *best* panel (all three prefer the 9-feature rank panel over the 14-feature engineered one), the trees net +0.6–0.8 at 20 bp, and once the 21-day overlap is HAC-corrected their IC t-stats fall to +1.3–1.6 — not distinguishable from zero. The signal is near-linear; trees over-parameterize it.
- **The honest t-stat is +2.72, not ~+10.** The naive daily IC t treats 348 overlapping-window ICs as independent; the 21-day overlap means there are really only ~17 independent monthly observations. The HAC correction (lag 21) deflates the OLS t from +10.1 to **+2.72** — still significant at p < 0.01, but a far cry from the inflated figure.

For the technical setup (PIT correction, regime-confined walk-forward with embargo, the unified temporal IC-scored CV), see [Methodology](#methodology); for why OLS beats trees, why the edge is regime-bound, why regularization is inert, and the net-of-cost analysis, see [Discussion](#discussion).

The result is **statistically rigorous, regime-conditional, and not deployable for retail investors** after bid-ask spreads, commissions, and capital-gains taxes. See [Data quality and methodological limitations](#data-quality-and-methodological-limitations) and [Scope and limitations](#scope-and-limitations) for the bounds on what these numbers mean. For the precise definitions of each metric in the table above (IC, t-stat, gross / net long-short Sharpe, annual turnover), see [Metric definitions](#metric-definitions).

## Metric definitions

The headline 21-day result is reported in terms of standard cross-sectional asset-pricing metrics, defined below. The worked example uses a small illustrative cross-section; the same formula applies at any horizon.

- **Information Coefficient (IC).** Cross-sectional ranking quality of
  the model, averaged over time.

  *Per-date IC.* On each date `t` in the evaluation window, the model
  produces a prediction for every ticker in the cross-section that has a
  realized forward excess return available at the model's target horizon
  `H`. Let `N_t` be the number of such tickers on date `t` — in the
  present universe `N_t` is on the order of 300-600. The per-date IC is
  the Spearman rank correlation between two vectors **of length N_t**:
  the vector of predictions and the vector of realized `H`-day forward
  excess returns. Note that the vectors are not length `H`; the horizon
  refers to the return target, not the number of observations.

  *Time-averaged IC.* The reported IC is the unweighted average of the
  per-date ICs across all evaluation dates (e.g., `n_dates = 348` for the
  21-day headline OOS slice 2025-01-02 → 2026-06-24). A weak ranker on
  most days plus one strong day does not produce a high IC; consistent
  ranking is required.

  *Worked example.* For 5 tickers with prediction ranks `[5,4,3,2,1]` and
  realized-return ranks `[5,3,4,2,1]` (one adjacent swap),
  `ρ = 1 − 6·Σd²/(N(N²−1)) = 1 − 12/120 = +0.90`. The headline IC of
  +0.087 is the **average of ~348 such daily correlations** (each over ~450
  tickers) — small per day, consistent across days. But consecutive days
  share 20 of their 21 forward days, so those daily ICs are heavily
  autocorrelated: there are really only **~17 independent monthly
  observations**, which is why the naive t-stat (+10.1) overstates
  significance ~√21×. The honest, overlap-corrected t-stat is **+2.72**
  (see *t-stat of IC* below).

  *Benchmarks.* On liquid US large-caps, IC = +0.02 (t > 2) is a credible
  edge; +0.10 would be exceptional; > +0.20 is not realistic out-of-sample.

- **t-stat of IC.** The naive form `mean(per-date IC) / (stdev(per-date
  IC) / √n_dates)` assumes the per-date ICs are independent — but the
  21-day forward windows overlap, so adjacent ICs are positively
  autocorrelated and the naive t overstates significance by ~√21 ≈ 4.6×.
  The reported t-stat applies a **Newey–West HAC correction** (Bartlett
  kernel, lag 21 = the overlap length): `t_HAC = mean / √Var_HAC` with
  `Var_HAC = (1/n)[γ₀ + 2·Σ_{k=1}^{21}(1 − k/22)·γ_k]` and `γ_k` the
  lag-`k` autocovariance of the per-date IC series. For the 6-feature
  6-feature panel this deflates t from **+10.1 (naive) to +2.72 (HAC)** —
  still p < 0.01, but the honest figure. |t| > 1.96 ⇒ p < 0.05;
  |t| > 2.58 ⇒ p < 0.01. (Equivalently, computing IC on the ~17
  non-overlapping rebalance dates gives a comparable t ≈ 2.8.)
- **Long-short Sharpe.** Annualized Sharpe ratio of a portfolio that is
  long the top-quintile predicted tickers (top 20%) and short the
  bottom-quintile (bottom 20%), equal-weighted within each leg,
  **rebalanced every 21 trading days and held to expiry** — the deployable
  monthly schedule. The quintile cut is `top_frac=0.2`
  (`src/price_model/eval/metrics.py`). Over the test window this is the
  ~17 **non-overlapping** 21-day long-short returns
  `mean(top.realized) − mean(bottom.realized)`, with Sharpe =
  `mean / stdev × √(252 / 21)`. Above +1.0 is the conventional bar for
  institutional tradeability; values are gross of cost unless labeled
  "net N bp."

  *Rebalancing & assumptions.* (i) dollar-neutral, equal-weight legs;
  (ii) the book is rebalanced once per 21-day horizon and the return
  series is **non-overlapping**, so — unlike a daily-formed overlapping
  estimator — there is **no autocorrelation inflation** of the Sharpe (the
  honest trade-off is a small ~17-observation sample); (iii) no financing,
  short-borrow, or capital-gains tax; (iv) annualization by `√(252/21)`;
  (v) the net columns deduct round-trip turnover × cost on the same 21-day
  clock (see *Annual turnover*).
- **Annual turnover.** One-sided fraction of portfolio capital traded
  per year. Because the prediction targets a 21-day horizon, the book is
  held over the horizon and turnover is measured on the **rebalance clock**:
  `mean(0.5 · Σ_i |w_t[i] − w_{t-21}[i]|) · (252/21)`, where `w_t` is the
  long-short weight vector and the `t-21` lag matches the 21-day holding
  period (≈ 12 rebalances/year). An annual turnover of `1.0` means the
  portfolio replaces itself once per year; `9x` means the equivalent of nine
  full-portfolio rotations per year. (A previous version measured a 1-day
  lag and annualized by 252 — charging daily churn against a 21-day return,
  which over-stated turnover ~21× and is now fixed.) The metric is
  implemented in `src/price_model/eval/turnover.py` and governs how much of a
  gross signal survives net of transaction costs: for cost level `b` bp per
  side, annual cost drag ≈ `annual_turnover · b · 2 / 10000`.
- **After-cost Sharpe at N bp.** Long-short Sharpe computed on returns
  net of transaction-cost drag. For each rebalance, gross return is
  `mean(top.realized) − mean(bottom.realized)`; net return subtracts
  `turnover_t · N / 10000 · 2` from gross (the `· 2` charges the **round
  trip** — entering and later exiting each position). The resulting
  net-return series is annualized to a Sharpe using the same `√(252/horizon)`
  convention as gross Sharpe. Reported at 3, 10, and 20 bp per side to span
  institutional → retail cost regimes.
<!-- Deflated Sharpe and cross-sectional dispersion are not used in the current
     regime-confined headline; metric definitions commented out for now (the code
     and scripts remain). Restore if these metrics re-enter the headline.

- **Deflated Sharpe ratio (DSR).** Bailey & López de Prado (2014)
  multi-test-corrected probability that a model's true Sharpe is
  positive given the observed Sharpe, the number of trials attempted,
  the sample length, and the higher moments of the return distribution.
  Defined as `P(true_Sharpe > 0 | observed)` with a higher-moment
  correction for skew and kurtosis. The threshold for "significant
  after multi-test correction" is `DSR > 0.95`. Implemented in
  `src/price_model/eval/metrics.py::deflated_sharpe_ratio`; computed
  across all project trials in `scripts/deflated_sharpe_audit.py`.
- **Cross-sectional return dispersion (regime indicator).** For each
  date, the standard deviation of daily log returns across the
  cross-section, smoothed by a 20-day rolling mean. Computed only from
  past returns (verified lookahead-safe by the universal
  truncation-invariance leakage test). Higher dispersion → more
  ticker-specific divergence; lower dispersion → more uniform market
  movement. Used as a contemporaneously-observable conditioning
  variable for both the regime-aware LightGBM feature panel and the
  walk-forward ensemble. Implemented as the `cs_return_dispersion_20`
  feature in `src/price_model/features/cross_features.py`.
-->

## Reproduction

### Step 0 — install and build the PIT universe (one-time, ~10-15 min cold)

```bash
# Core: required for everything below.
pip install -e ".[dev,classical]"

# Optional: notebook visualization stack (matplotlib + seaborn + jupyter).
# Install if you plan to run the notebooks/ directory, including the
# feature-exploration boilerplate at notebooks/05_feature_exploration.ipynb.
pip install -e ".[notebooks]"

# Scrape Wikipedia for historical S&P 500 membership and write the universe file
python -m price_model.cli build-universe --name sp500_pit --start 2017-01-01

# Fetch yfinance data for the ~700 resolving tickers (~10-15 min cold; subsequent
# runs are incremental, fetching only the new rows since the cache's max date)
python -m price_model.cli refresh-data --universe sp500_pit --start 2017-01-01
```

### Step 1 — run the regime-confined headline

One script runs the whole comparison: held-out Optuna sweeps for the three trees on both panels (rank-9 and engineered-14), then trains the tuned trees + the 6-feature linear models (OLS, plus the equivalent Lasso/Ridge — identical on this panel) + the momentum factors. Everything is **regime-confined** — trained only on 2022-10-10 → 2024-11-30 (single refit) and tested on 2025+.

```bash
# The run writes to the single-writer DuckDB store; close any Jupyter kernel
# that opened it first. ~1.5-2h at 40 trials/model (CatBoost dominates).
bash scripts/run_headline_comparison.sh 40
```

Regime confinement is pure config — three `walk_forward` fields, no code edits:

- `data.start: 2017-01-01` — full price history so the ~3-year `momentum_756` warmup is satisfied.
- `train_start: 2022-10-10` — training lower bound, applied **identically across panels** regardless of each panel's warmup (this is why the harness gained a `train_start` parameter; a single `data.start` warms the rank and engineered panels at different dates).
- `first_refit: 2025-01-02` + `refit_freq_days: 9999` — one train/test boundary; `embargo_days: 33` (calendar) covers the 21-trading-day target.

The Optuna sweeps read the same warmed matrix, so they inherit `train_start` (CV folds are in-regime) and hold out 2025+ via `--max-date 2024-12-31`.

### Step 2 — the headline net-of-cost table

```bash
PYTHONPATH=src python scripts/net_cost_regime_21d.py
```

Fits the **linear family fresh** (OLS / Ridge / Lasso on curated-6 with the temporal IC-scored CV — the store holds stale pre-fix predictions and no OLS) and reads the **trees + momentum from the store**, joins realized 21-day excess returns, and prints the **IC point estimate (all dates) with its HAC-corrected t-stat** (Newey–West, lag 21) plus gross / **3 / 10 / 20 bp net** Sharpe on the **non-overlapping 21-day rebalance schedule** — the headline convention. (`scripts/net_cost_regime.py` reports the same on the daily-overlap estimator.) Use these, **not** `scripts/compare_net_of_cost.py`, for the regime headline: the shared compare scripts intersect the full accumulated store and drop most of the new models.

### Step 3 (optional) — investigation diagnostics

The Discussion's claims reproduce via:

```bash
PYTHONPATH=src python scripts/inspect_alpha_selection.py    # prints CV folds, α grid, selected α: regularization → α≈0 (OLS)
PYTHONPATH=src python scripts/spectrum_comparison.py         # model × panel grid: OLS≡Ridge≡Lasso, linear>trees, panel dominates
PYTHONPATH=src python scripts/vol_ablation_lasso_ridge.py    # 9→6 prune; beta_60 is anti-generalizing
PYTHONPATH=src python scripts/inspect_headline_lasso_coefficients.py  # vol-cluster signs are signal, not L1 cancellation (Ridge agrees)
PYTHONPATH=src python scripts/survivorship_ablation_headline.py       # PIT-on vs survivor-snapshot, in-regime
# Deflated Sharpe (scripts/deflated_sharpe_audit.py) is parked for now — not used in the headline.
```


## Methodology

The headline result is produced by a walk-forward, cross-validated, point-in-time-corrected backtest. This section documents each component.

### Universe: S&P 500 with point-in-time membership filter

**Point-in-time (PIT) correction** restricts the cross-section evaluated on each date `t` to tickers that were *actually members of the S&P 500 index on date `t`* — not the tickers that *are* members today. The correction defends against survivorship bias: today's S&P 500 list disproportionately contains companies that survived, by selection. A backtest evaluated on today's list silently assumes the model would have "known" in 2018 to focus on the 2026 survivors.

The mechanics:

- `src/price_model/data/sources/sp500_membership.py` scrapes the Wikipedia "List of S&P 500 companies" page and its "Selected changes" table to reconstruct, for every ticker that has ever been in the index between 2014 and the present, an `(added_date, removed_date)` membership window.
- `src/price_model/data/membership.py::filter_panel_to_pit(panel)` drops every row whose `(date, ticker)` pair falls outside that ticker's membership window. A stock added on 2020-06-22 contributes only its post-2020-06-22 rows; a stock removed on 2019-04-30 contributes only its pre-2019-04-30 rows.

The PIT correction in this project is **partial**: yfinance does not return data for ~12% of historical S&P 500 tickers (SIVB, FRC, ATVI, AGN, etc. — see [Data quality and methodological limitations](#data-quality-and-methodological-limitations)). A bias-free PIT analysis requires a paid feed (CRSP or Norgate Premium Data).

### Forecast target and walk-forward training

The forecast target is the cross-sectional excess log-return over a fixed horizon. For each date `t` and ticker `i`, the target `y[t, i]` is the log-return of `i` from `t+1` to `t+H` minus the cross-sectional mean log-return over the same window. The headline uses `H = 21` trading days (approximately one month); the same framework is implemented at `H = 5` and is configurable per experiment via `target_horizon` in the YAML configs.

**Regime-confined training (this branch).** The headline trains on a single regime — the post-2022-10 bull regime — and tests on 2025+. This is one train/test block, controlled by three `walk_forward` config fields:

- `data.start: 2017-01-01` — full price history, so the ~3-year `momentum_756` warmup is satisfied.
- `train_start: 2022-10-10` — the training **lower bound**. Applied identically across feature panels regardless of each panel's warmup length (the rank panel warms at 2022-10, the 14-feature engineered panel at 2021-10; without `train_start` a single `data.start` would train them on different windows). This is the regime boundary from the time-split in `notebooks/03`.
- `first_refit: 2025-01-02` + `refit_freq_days: 9999` — a single refit, so the training block is **[2022-10-10, 2024-11-30]** and the test slice is **[2025-01-02, 2026-06-24]** (348 dates).
- `embargo_days: 33` (calendar) — covers the 21-**trading**-day (~31 calendar) target, so the last training label resolves before the test window. (This fixes the earlier `embargo_days: 22` calendar-vs-trading unit bug; `train_end = first_refit − 33 = 2024-11-30`.)

`train_start` and `first_refit` are additive harness parameters (default `None` = the original expanding-window behavior, which trains from warmup-end across all regimes — still used on `main`). For that across-regime framing, omit them and use `min_train_days`/`refit_freq_days` as before.

**Feature lookahead safety** is verified by the truncation-invariance test in `tests/test_no_leakage.py`: each registered *feature*'s value at date `t` must be unchanged when the panel is truncated to dates `≤ t`. The test is parametrized over `FEATURE_REGISTRY` and runs in CI. Note it covers features only — *target* leakage across the train/test boundary is the embargo's job (now airtight at 33 days), not this test's.

### Inner cross-validation for regularization-strength selection

Any regularization strength α (for Ridge / Lasso / ElasticNet) is selected by **one unified grader, identical to the tree sweeps: a temporal, purged, expanding-window forward-chain CV scored on per-date IC** (`src/price_model/models/_cv.py`). Within each refit's training block:

- The training dates are cut into `cv+1` contiguous chunks (default `cv=3` → 4 chunks); chunk *i* is a validation fold and all *earlier* dates form its training set, **minus a `cv_embargo`-date purge** (≥ the 21-day horizon) so the forward target cannot straddle the boundary. Expanding window, strictly causal — no future dates leak into α-selection.
- For each candidate α the model is fit on each fold's train rows and scored by **mean per-date Spearman IC** on its validation rows; the α maximizing the mean fold-IC is chosen, then the model is refit once on the full block.
- Grids are data-driven: Lasso/ElasticNet anchor to `alpha_max = max_j|x_j·ȳ|/n` (the α that zeros all coefficients); Ridge anchors to the Gram-matrix eigenvalue scale `trace(XᵀX)/p` (L2 has no `alpha_max`).

**On the curated 6-feature panel this selection drives α → 0 — the model is effectively OLS.** With ~238k training rows × 6 features the design is hugely over-determined, so shrinkage doesn't help: the CV-IC curve is flat at the low-α (OLS) end and **decays monotonically** as α bites. OLS, Lasso, and Ridge therefore produce **identical** predictions, and the headline is reported as OLS.

*(Correction to an earlier version: α used to be selected by sklearn's `LassoCV`/`RidgeCV`, which on an integer `cv` build a non-temporal `KFold(shuffle=False)`. Because the panel is sorted `["ticker","date"]`, those folds fell on ticker boundaries — a by-ticker, MSE-scored, fully-date-overlapping split that leaked future dates into HP selection. That was an accidental sklearn default, not a design choice; it is now replaced by the temporal IC grader above, so linear and tree hyperparameters are chosen the same temporally-honest way.)*

The walk-forward outer loop independently guarantees each refit's α is selected only on data preceding the deployment slice.

### Held-out Optuna protocol for LightGBM hyperparameter search

LightGBM has more hyperparameters than CV can search inside a refit. Instead, the project uses **Optuna with purged walk-forward CV** (de Prado, *Advances in Financial Machine Learning*, Ch. 7) for hyperparameter selection.

The HP-selection bias correction is provided by the `--max-date` flag in `scripts/optuna_sweep.py`: dates after the specified cutoff are excluded from the matrix before CV folds are constructed, so the chosen HPs are provably independent of any data in the post-cutoff evaluation slice. With `train_start` set, the sweep also confines its CV folds to the regime (lower bound), so HP selection is in-regime and held out from 2025+ via `--max-date 2024-12-31`.

**Regime-confined sweeps.** Each tree is swept on both panels (held-out, 2024-12-31 cutoff; CV = mean per-date IC over 3 purged forward-chain folds). The CV winner picks the deployed panel:

| Model | Panel | CV mean IC | 2025+ OOS IC |
|---|---|---|---|
| LightGBM | **rank-9** | +0.0506 | +0.0685 |
| LightGBM | eng-14 | +0.0361 | +0.0546 |
| XGBoost | **rank-9** | +0.0502 | +0.0583 |
| XGBoost | eng-14 | +0.0308 | +0.0430 |
| CatBoost | **rank-9** | +0.0513 | +0.0606 |
| CatBoost | eng-14 | +0.0296 | +0.0499 |


- **CV predicts OOS.** Every model's 2025+ OOS IC *exceeds* its CV IC (e.g. LightGBM 0.051 → 0.069) — no fragility penalty, because the CV folds and the test slice share the regime. The rank-9 panel wins CV for all three trees, so the panel choice is made honestly (on CV), not by peeking at OOS.

### Headline feature panel: 6 curated anomalies (pruned from 9)

The headline uses **6 features** — the 9-feature anomaly panel below, minus `idio_vol_20`, `max_return_21d`, and `beta_60`. The prune is justified by the collinearity and ablation analysis that follows (it raises in-regime OOS IC from +0.077 to +0.087, identically for OLS / Lasso / Ridge, which coincide on this panel). The full 9-feature superset (the prior headline `lasso_elasso_pit_h21`), curated on "one feature per economically distinct anomaly family," with **✓ = kept in the headline 6**:

| # | Feature | Anomaly family | Reference | In 6? |
|---|---|---|---|---|
| 1 | `momentum_12_1` | Cross-sectional momentum (12-month, skip-1-month) | Jegadeesh-Titman 1993, *J. Finance* 48(1) | ✓ |
| 2 | `momentum_756` | Long-horizon trend (36-month) | Moskowitz-Ooi-Pedersen 2012 (TSMOM-adjacent) | ✓ |
| 3 | `return_1d` | Short-term reversal | Lehmann 1990 / Jegadeesh 1990 | ✓ |
| 4 | `vol_ewm_20` | Low-volatility | Ang-Hodrick-Xing-Zhang 2006 | ✓ |
| 5 | `idio_vol_20` | Idiosyncratic vol (60-day beta residual) | Ang-Hodrick-Xing-Zhang 2006 | — dropped |
| 6 | `max_return_21d` | MAX (lottery-stock) effect | Bali-Cakici-Whitelaw 2011 | — dropped |
| 7 | `distance_52w_high` | 52-week-high anchoring | George-Hwang 2004 | ✓ |
| 8 | `log_dollar_volume` | Size / liquidity | Amihud 2002 | ✓ |
| 9 | `beta_60` | Market exposure control (BAB-adjacent) | Frazzini-Pedersen 2014 | — dropped |

### Net-of-cost decomposition

Gross IC and gross long-short Sharpe assume costless rebalancing. Net-of-cost Sharpe accounts for transaction-cost drag from portfolio turnover. Because the signal targets a 21-day horizon, the book is held over the horizon and **per-rebalance turnover** is the L1 distance between the long-short weight vector and its value 21 trading days earlier (`0.5 · Σ_i |w_t[i] − w_{t-21}[i]|`). Annual turnover = `mean(per-rebalance turnover) × (252/21)` (≈ 12 rebalances/year). **After-cost return** = gross return − `turnover × cost_bps / 10000 × 2`, where the `× 2` is the round trip (in and out). After-cost Sharpe is computed on the net-return series with the same `√(252 / horizon)` annualization as gross Sharpe. (Measuring turnover on a 1-day lag — the previous convention — charged daily churn against a 21-day return and over-stated turnover ~21×.)

Reported at 3 bp (institutional all-in), 10 bp (small-fund), and 20 bp (retail-equivalent). Turnover and cost-drag mechanics are implemented in `src/price_model/eval/turnover.py`. For the **regime-confined headline** use `scripts/net_cost_regime_21d.py` — it pulls each model from the store and reports the IC point estimate with its HAC-corrected t-stat plus gross/net Sharpe on the **non-overlapping 21-day rebalance schedule** (the headline convention); `scripts/net_cost_regime.py` reports the same on the daily-overlap estimator. Avoid the shared `scripts/compare_net_of_cost.py`: it intersects the full accumulated store (many overlapping historical runs) and drops most of the new regime models.

<!-- Deflated Sharpe is not used in the current regime-confined headline; subsection
     commented out for now (scripts/deflated_sharpe_audit.py remains). Restore if DSR
     re-enters the headline.

### Bailey-López de Prado deflated Sharpe (multi-test correction)

`scripts/deflated_sharpe_audit.py` applies the deflated Sharpe ratio (Bailey & López de Prado 2014) across all project models. The correction inflates the effective number of trials by the project's experiment count and accounts for skew and kurtosis of the return distribution to compute `P(true_Sharpe > 0 | observed)`. Threshold for "significant after multi-test correction" is DSR > 0.95.

The deflated-Sharpe pass list aligns with the t-stat ranking — but on the **HAC-corrected (overlap-honest)** t-stats, not the inflated naive ones. Only the 6-feature OLS (HAC t +2.72, p < 0.01) and momentum (+2.31, p < 0.05) are individually significant; the tuned trees (HAC t ≈ +1.3–1.6) are not distinguishable from zero and would **not** clear DSR. (DSR still applies the project-wide trial count, so per-model significance is necessary but not sufficient.)
-->

### Audit-driven LightGBM v3 panel

The v3 panel curation (14 features chosen from a 21-feature candidate panel via gain importance + cross-feature correlation analysis) is documented inline in `config/experiments/extended_kaggle_v3_curated.yaml`; audit script in `scripts/audit_lightgbm_features.py`.

## Discussion

The regime-confined headline finding — a 6-feature linear cross-sectional regression beats every held-out-tuned tree *and* long-horizon momentum on both gross IC and net-of-cost Sharpe, but only within the post-2022-10 regime it was trained on — raises five questions worth unpacking. This section addresses each.

### Why linear beats trees on this universe

**The 21-day cross-sectional signal on the post-2024 S&P 500 is approximately linear in feature space.** Three pieces of evidence:

1. **The momentum factors alone capture most of the signal.** mom_504, mom_756, and mom_378 individually produce IC = +0.04 to +0.05 with no model fitting; the L1 regression's incremental edge of +0.02 IC over the best single momentum factor reflects modest additional signal from non-momentum features (low-volatility, idiosyncratic volatility, MAX effect). The dominant component is a linear momentum signal that trees would have to recover via many small splits.
2. **Tree-ensemble gain-importance audits show feature concentration.** On the 14-feature v3 panel, three features produce 39% of LightGBM gain. The split structure devotes most of its capacity to a handful of features that OLS can express as a small number of coefficients. Trees' theoretical interaction-capturing advantage doesn't pay off here because the dominant signal isn't interactive.
3. **Held-out-tuned trees still fall short.** With each tree Optuna-tuned (held out at 2024-12-31) on its *best* panel, the regime-confined OOS IC is +0.069 (LightGBM) / +0.061 (CatBoost) / +0.058 (XGBoost) vs OLS's +0.087 — a ~0.02–0.03 gap.

Two caveats limit how broadly this conclusion generalizes:

- **The 6-feature panel is small and ex-ante-curated.** The linear advantage depends on this curation. On the broader 14-feature engineered panel the linear recipe is weaker — but so are the trees (every model is worse there), so curation, not model class, is the lever.
- **The post-2022-10 regime is structurally momentum-friendly.** In a regime with more leadership rotation or weaker momentum persistence, the linear-on-anomalies recipe would compress and trees might catch up. The result is robust within the studied window but has not been verified across regime changes.
The conclusion is therefore "within the post-2022-10 regime, on this universe at this horizon, a curated 6-feature OLS outperforms every tree-ensemble variant tested" — not "ML cannot succeed in cross-sectional equity prediction generally," and not "the edge is regime-robust."

### Net-of-cost: turnover, and the framing that flips the winner

Net-of-cost ranking depends on turnover, and the two training framings give opposite winners — the gap between them is itself a finding.

**Regime-confined (headline): OLS wins net.** Trained on the bull regime, the 6-feature OLS model has gross Sharpe **+2.09** and, on the 21-day rebalance clock, turns over only **~9×/yr**, so at 20 bp round-trip it retains net **+1.85**. The 36-month momentum factor is the low-turnover champion (4×, net +1.28) but its lower gross (+0.050 IC, Sharpe +1.41) can't catch it. Tuned trees net +0.6–0.8 at 20 bp (turnover 10–11×) — and none is statistically significant once the overlap is HAC-corrected (t < 2). So in-regime, **OLS wins on both gross and net** — the first configuration in this project where the model beats momentum net-of-cost.



## Data quality and methodological limitations

The primary out-of-sample estimate is honest about what it measures,
but what it measures is bounded by the data available without a paid
feed. This section enumerates those boundaries.

### Limitations of yfinance

yfinance is the only free source of daily-bar US equity data and is the
reason the project is reproducible without a paid subscription. It has
three documented failure modes that materially affect the reported
results.

1. **Delisted-ticker history is permanently lost.** When a company is
   acquired, fails, or goes private, yfinance stops returning data for
   the old symbol. SIVB (Silicon Valley Bank, failed March 2023), FRC
   (First Republic, failed May 2023), SBNY (Signature Bank, failed March
   2023), ATVI (Activision, acquired by Microsoft October 2023), AGN
   (Allergan → AbbVie 2020), and CERN (Cerner → Oracle 2022) have no
   usable pre-event history accessible via yfinance. Approximately 21
   such tickers are enumerated in the project drop list, documented
   inline in `src/price_model/data/tickers.py`.
2. **Symbol-parser fragility.** yfinance fails to fetch some single- and
   short-letter tickers due to ambiguity with currency or commodity
   symbols in its routing — for example, `K` (Kellanova), `FI` (Fiserv),
   and `DAY` (Dayforce), all currently live, exchange-listed US
   large-caps. yfinance returns "no data" after retries.
3. **Foreign listings are unreachable.** Acquired companies that
   consolidated under non-US ADRs cannot be fetched. `SIE.DE` (Siemens
   Healthineers, acquired Varian) and `MC.PA` (LVMH, acquired Tiffany)
   are examples, despite active trading on European exchanges.

**Coverage.** Of the 701 ticker symbols that Wikipedia identifies as
S&P 500 members at some point during 2017-2026, yfinance provides usable
data for 617 (~88%). The remaining ~12% are absent from the panel.

### Partial PIT reconstruction from Wikipedia

The Wikipedia "List of S&P 500 companies" page and its "Selected changes"
table are the only free source of historical index membership. The scraper
in `src/price_model/data/sources/sp500_membership.py` reconstructs the
`(ticker, added_date, removed_date)` table from those sources. Three known
incompleteness modes remain.

1. **The change log is reliable only back to approximately 2014.** Earlier
   add / remove events are absent. This does not affect the 2017-start
   evaluation window directly, but it implies that any pre-2017 extension
   (for example, applying the same machinery to a 1990s-2010s sample)
   would inherit incomplete PIT membership.
2. **Renames and continuations are ambiguous.** When a company changes
   its ticker (FB → META, RTN → RTX, FISV → FI), Wikipedia logs the event
   as a simultaneous remove and add on the same date. The scraper treats
   such pairs as continuous membership at the *new* symbol; the alias is
   resolved in `tickers.py`. The choice is correct for backtest purposes
   but implies that a researcher reading the membership table directly
   may miscount index changes by approximately 30 events over the full
   window.
3. **Wikipedia editors can be incorrect, late, or partial.** Missing
   entries (e.g., short-lived 2018 additions never logged as removals)
   were observed during the build. The membership table is not
   independently audited against a primary source.

### Ticker resolution rules

Three small lookup tables are maintained in
`src/price_model/data/tickers.py` to bridge the gaps above:

| Table | Purpose | Size |
|---|---|---|
| `TICKER_ALIASES` | Map renamed symbols to their successor (FB → META, RTN → RTX, ~70 entries documented with rename dates) | 73 entries |
| `TICKER_DROP_LIST` | Symbols with no usable yfinance data (failed banks, acquired with non-unified history, foreign listings, short-ticker parser failures) | ~25 entries |
| `SYMBOL_NORMALIZATION` | Punctuation convention (BRK.B → BRK-B) | 2 entries |

The function `resolve_ticker(symbol) -> str | None` applies all three in
order. Every ticker in every universe file is routed through it. Tests in
`tests/test_tickers.py` cover the precedence rules and known edge cases.

These tables are derived heuristically: `cli refresh-data` is run on the
expanded universe and the resulting yfinance failure log is inspected.
Each entry carries a one-line comment identifying the corporate event and
the rationale for inclusion. New entries are added when new failures are
observed.

### Sources of upward bias in the reported ICs

The regime-confined headline IC of **+0.087** (6-feature OLS on 2025+ OOS) and every other IC in the README — gross or net — should be interpreted with the following caveats. The bias sources are inherent to the data and methodology and apply uniformly across models, not to one model in isolation.

0. **Regime-confined training is best-case, by construction.** The headline trains only on the bull regime it is then deployed into. 
1. **PIT correction is partial.** Because yfinance does not return data for SIVB, FRC, ATVI, AGN, and similar delisted symbols, those tickers do not appear in the PIT panel even when Wikipedia indicates they were index members on the relevant dates. The PIT-corrected backtest therefore still excludes the worst realized losers of the 2022-2023 banking crisis. A fully PIT-correct analysis using paid data (Norgate, Polygon, or CRSP) would either misrank or correctly short SIVB at −85% on 2023-03-08; the present model does neither.
2. **Transaction costs, taxes, and slippage are partially modeled.** The 3/10/20 bp net columns model bid-ask + commission only; they exclude slippage on illiquid names, capital-gains taxes, and short-borrow. In-regime the 6-feature OLS nets +1.85 at 20 bp (turnover 9×) and momentum +1.28 (4×) — both institutional-grade gross-and-modeled-net, but the unmodeled frictions and small-account constraints still bite a retail book.
3. **Single evaluation window.** The 2025+ slice is one window. Alternative windows (2018–2020, 2020–2022, 2022–2024) would produce different ICs; the multi-window stress test is the most-needed extension and has not been performed.

### Toward a fully PIT-correct evaluation

Listed in approximate order of effort:

- **Replace yfinance with Norgate Premium Data (~$60 / month)** for survivorship-bias-free prices including delisted history. All reported ICs would change; the headline edge would likely compress somewhat, and the regime-conditional shape (strong when training matches the deployment regime, fading across regime breaks) is expected to persist.
- **Replace the Wikipedia membership source with CRSP** (paid; free
  academic access for most affiliated researchers). Provides cleaner
  pre-2014 history and eliminates scrape fragility.
- **Run multi-window robustness tests** at three different start / end
  pairs.
- **Add transaction-cost modeling** at the prediction-store layer so
  reported metrics are net of realistic costs.

## Scope and limitations

- **Not deployable for retail trading.** In-regime the 6-feature OLS model is institutional-grade on both gross (IC +0.087, Sharpe +2.09) and *modeled* net (Sharpe +1.85 at 20 bp). But the net columns model only bid-ask + commission; once you add slippage on size, capital-gains taxes, short-borrow, and the small portfolio sizes a retail account can hold, the realistic retail edge shrinks sharply. Combined with the regime-bound caveat below, this is a research result, not a trading recommendation.
- **The edge is regime-concentrated — and the headline trains inside it.** The headline confines training to the post-2022-10 regime by design, so its edge lives **entirely** in the momentum-persistent regime: on the 2025+ test slice, regime-confined training scores IC **+0.087**.
- **Single evaluation period, no multi-window robustness.** All numbers are one OOS window (2025+). A multi-window stress test (2018-2020, 2020-2022, 2022-2024, 2024-2026 in sequence) — ideally with paid PIT-correct data — is the most-needed extension and has not been performed. "Extend the regime" is the natural next step from this branch.
- **Survivorship-bias is only partially corrected.** The PIT correction excludes ~12% of historical S&P 500 members (delisted symbols yfinance can't fetch). Measured directly (`scripts/survivorship_ablation_headline.py`), inflation is ≈0 on the 2025+ slice but a sign-flip over the full sample; the absent delisted names make even that a floor. See [Data quality and methodological limitations](#data-quality-and-methodological-limitations).
- **Not a substitute for index funds.** For individual investors, decades of research show that low-cost diversified index funds outperform almost all active strategies after fees and taxes.

## Architecture

```
src/price_model/
├── data/
│   ├── sources/         # yfinance + Ken French + Wikipedia adapters
│   ├── universes/       # static universe files (sp500.txt, sp500_pit.txt)
│   ├── tickers.py       # aliases / drop list / normalization rules
│   ├── membership.py    # PIT membership lookup + filter
│   └── loaders.py       # one-call load_panel(universe, start, pit_filter)
├── features/            # technical, cross-sectional, factor-loading, anomalies
├── models/              # LightGBM, baselines, classical (ARIMA, GARCH, GBM, FF)
├── pipeline/            # walk-forward backtest harness
├── eval/                # IC, hit rate, Sharpe, bootstrap CI, time-split
├── serving/             # DuckDB prediction store
├── dashboard/           # Streamlit dashboard reading from the store
└── cli.py               # `price-model` entry point

config/experiments/      # YAML configs for each stage above
notebooks/               # diagnostic + classical + robustness + portfolio + features
tests/                   # leakage tests, PIT tests, ticker tests, contract tests
```

Notebooks (each maps to one or more README sections; together they provide interactive support for the project's claims):

- `notebooks/00_pit_ablation_study.ipynb` — PIT ablation study on the legacy v2 LightGBM panel. Quantifies survivorship-bias inflation (61–89% depending on feature-set strength). Supports Methodology + Data quality.
- `notebooks/01_prediction_diagnostics.ipynb` — Stored-prediction diagnostics on the headline 21-day models: apples-to-apples comparison, rolling 60-day IC, regime-conditional IC, net-of-cost gross-vs-net Sharpe with turnover annotation. Supports the net-of-cost and regime-conditionality analysis in Discussion.
- `notebooks/02_classical_timeseries.ipynb` — Classical EMH baselines (ARIMA / GARCH / GBM) on the per-ticker time series, contrasted with the cross-sectional linear model. Supports the case that the cross-sectional approach beats per-ticker time-series baselines.
- `notebooks/03_headline_robustness.ipynb` — Bootstrap CI / decile bucket monotonicity / time-split partition on the linear headline. Supports the regime-conditionality analysis.
- `notebooks/05_feature_exploration.ipynb` — Feature engineering boilerplate: 41-feature registry tour, distributions, correlations, fitted L1 / LightGBM importance, and a step-by-step template for adding new features. Now also includes the **panel-ablation findings** (9→6 prune, `beta_60` anti-generalizing, regularization-inert OLS≡Ridge≡Lasso, vol-cluster-is-signal). Supports Methodology (panel design) + Discussion's [Matched-grader comparison](#matched-grader-comparison-is-it-the-model-the-panel-or-the-grader).
- `notebooks/06_hp_selection_bias.ipynb` — Visualizes the **cross-regime** HP fragility: chosen HPs swinging across training cutoffs (`lambda_l1` over ~two orders of magnitude) and the held-out-vs-default inversion at 21-day. Supports Discussion's [HP-fragility subsection](#why-hyperparameter-optimization-is-fragile-to-regime-shift) — contrast with the **in-regime** sweeps in Methodology, where CV predicts OOS.

**Extras (independent of the cross-sectional ML narrative):**

- `notebooks/07_portfolio_attribution.ipynb` — Fama-French 5-factor risk decomposition + realized return attribution on a fixed portfolio. Reuses the project's FF5 data adapter and rolling-beta features for an orthogonal use case (portfolio risk, not return prediction). Independent application; the numbering jumps to 07 to signal scope drift from notebooks 00-06.

### Streamlit dashboard

`src/price_model/dashboard/` is a thin Streamlit reader over the DuckDB
prediction store. It exists to allow the model's daily output to be
inspected visually (per-date top / bottom quintile, rolling IC,
prediction vs. realized scatter) without writing a notebook for each
inspection. The dashboard is a debugging and monitoring surface, not
part of the reproduction workflow. None of the reported numbers in
this README originate from the dashboard, and skipping it does not
affect reproduction of any ablation cell. The dashboard is most useful
when extending the model and a quick sanity-check view of new runs is
desired.

## Development notes

The substantive research decisions in this project — what to measure,
what constitutes a fair test, how much survivorship bias the primary
estimate contains, which limitations are honest to ship with — are the
author's.
The engineering side of the project (test scaffolding, CI configuration,
package layout, refactors, and the translation from research intent to
reproducible implementation) was developed in close collaboration with
Claude Code. Specifically:

- The walk-forward harness, prediction store, and dashboard scaffold
  were designed and iterated in conversation with Claude Code.
- Test coverage (leakage tests, PIT-membership tests, contract tests on
  the loaders, ticker-resolver tests) was co-designed; Claude Code
  drafted many of the test cases that subsequently surfaced bugs in the
  implementation.
- The GitHub Actions CI pipeline (ruff + pyright + pytest on Python
  3.11) and the pre-commit hooks were configured with Claude Code.
- Code review, refactors, and a final pre-publish audit pass were
  conducted in collaboration with Claude Code.

The README, the experiment configs, and the ablation narrative are the
author's claims; the engineering quality of the surrounding codebase is
a product of the above collaboration.

## Citations

- **Han, Y., He, A., Rapach, D., and Zhou, G.** (2024). "Expected Stock Returns and the Cross-Section: An E-LASSO Approach." *Review of Finance*. — Panel-curation principle (one feature per economic family, rank-normalized, CV-selected α) used in the headline 6-feature linear regression (`lasso_curated6_pit_h21`), pruned from a 9-feature superset.
- **Jegadeesh, N. and Titman, S.** (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance* 48(1). — 12-1 momentum anomaly.
- **Moskowitz, T., Ooi, Y. H., and Pedersen, L. H.** (2012). "Time Series Momentum." *Journal of Financial Economics* 104(2). — Long-horizon (TSMOM) momentum.
- **Lehmann, B.** (1990). "Fads, Martingales, and Market Efficiency." *Quarterly Journal of Economics* 105(1). — 1-day reversal.
- **Ang, A., Hodrick, R., Xing, Y., and Zhang, X.** (2006). "The Cross-Section of Volatility and Expected Returns." *Journal of Finance* 61(1). — Low-vol and idiosyncratic-vol anomalies.
- **Bali, T., Cakici, N., and Whitelaw, R.** (2011). "Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns." *Journal of Financial Economics* 99(2). — MAX (lottery) effect.
- **George, T. and Hwang, C.-Y.** (2004). "The 52-Week High and Momentum Investing." *Journal of Finance* 59(5). — 52-week-high anchoring.
- **Amihud, Y.** (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects." *Journal of Financial Markets* 5(1). — Liquidity / size proxy.
- **Frazzini, A. and Pedersen, L. H.** (2014). "Betting Against Beta." *Journal of Financial Economics* 111(1). — Market-beta control.
- **Fama, E. and French, K.** (2015). "A Five-Factor Asset Pricing Model." *Journal of Financial Economics* 116(1). — FF5 baseline (reference only; not in the headline comparison due to Kenneth French release-lag truncation).
- **Zou, H. and Hastie, T.** (2005). "Regularization and Variable Selection via the Elastic Net." *Journal of the Royal Statistical Society, Series B* 67(2). — ElasticNet model class.
- **Bailey, D. and López de Prado, M.** (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management* 40(5). — Multi-test Sharpe correction.
- **DeMiguel, V., Garlappi, L., and Uppal, R.** (2009). "Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?" *Review of Financial Studies* 22(5). — The estimation-error-dominates-optimization result referenced in the Discussion's "Why HP optimization is fragile to regime shift" subsection.
- **López de Prado, M.** (2018). *Advances in Financial Machine Learning.* Wiley. — Purged walk-forward cross-validation (Ch. 7).
- **Ken French Data Library.** Daily factor returns. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Wikipedia: List of S&P 500 companies.** Historical components and change log. https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

## License

MIT.
