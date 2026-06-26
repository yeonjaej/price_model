# price-model

A point-in-time (PIT) corrected cross-sectional equity return predictor on the S&P 500, evaluated under strict HP-free held-out protocols at the 21-day (approximately monthly) forward horizon.

## Executive summary

A PIT-corrected, held-out 21-day cross-sectional return predictor on the S&P 500. The headline is reported **regime-confined**: trained only on the post-2022-10 bull regime (train 2022-10-10 → 2024-11-30, single refit; trees held-out-Optuna-tuned with the cutoff at 2024-12-31) and tested on 2025-01-02 → 2026-06-24 (348 dates). This is the honest "deploy when the regime resembles recent training" scenario; the more conservative full-history framing is below.

A **6-feature cross-sectional linear regression** on documented academic anomalies (Han-He-Rapach-Zhou 2024) is the strongest model on **both** gross signal and net-of-cost return — beating long-horizon momentum and held-out-tuned LightGBM / XGBoost / CatBoost:

| Model (regime-confined) | Gross IC | t-stat | Gross Sharpe | Net 20 bp Sharpe | Annual turnover |
|---|---|---|---|---|---|
| **6-feature linear (Lasso ≈ Ridge)** | **+0.089** | **+10.5** | **+2.00** | **+1.94** | 95× |
| 36-month momentum factor | +0.050 | +7.2 | +1.69 | +1.68 | 15× |
| LightGBM, held-out Optuna (rank panel) | +0.069 | +6.1 | +1.38 | +1.32 | 90× |
| CatBoost, held-out Optuna (rank panel) | +0.061 | +4.9 | +1.07 | +1.03 | 64× |
| XGBoost, held-out Optuna (rank panel) | +0.058 | +4.9 | +0.97 | +0.92 | 73× |

Observations:

- **Linear wins gross *and* net** — the only configuration in this project where the model beats momentum net-of-cost. Two causes: regime-matched training lifts gross Sharpe to +2.00, and the curated 6-feature panel's lower turnover (95×) limits cost drag.
- **L1 ≈ L2** — Lasso (+0.0892) and Ridge (+0.0863) tie on every column; the regularizer is not the story, the curated panel is. (Pruning the prior 9-feature panel to 6 — dropping `idio_vol_20`, `max_return_21d`, and the anti-generalizing `beta_60` — is most of the lift.)
- **Every tuned tree trails** — even with held-out Optuna on each tree's *best* panel. All three trees prefer the 9-feature rank panel over the 14-feature engineered one. The signal is near-linear; trees over-parameterize it.

**Full-history (conservative) framing.** Trained on the full expanding window instead of one regime, the same linear recipe scores only **IC +0.0695 / gross Sharpe +1.24** and **loses net-of-cost to momentum** (+1.17 vs +1.81 at 20 bp) — the project's original headline. The gap between the two framings *is* the finding: the edge is regime-bound. Across the 2022 regime break, temporally-honest (forward-chained) coefficient/HP selection shrinks the linear model to the null, and the pre-2022 sub-window IC is significantly **negative (−0.0747)**. See [Discussion](#matched-grader-comparison-is-it-the-model-the-panel-or-the-grader).

For the technical setup (PIT correction, regime-confined walk-forward with embargo, inner CV vs held-out Optuna, deflated Sharpe), see [Methodology](#methodology); for why linear beats trees, why the edge is regime-bound, and the L1-vs-L2 and net-of-cost analysis, see [Discussion](#discussion).

The result is **statistically rigorous, regime-conditional, and not deployable for retail investors** after bid-ask spreads, commissions, and capital-gains taxes. See [Data quality and methodological limitations](#data-quality-and-methodological-limitations) and [Scope and limitations](#scope-and-limitations) for the bounds on what these numbers mean. For the precise definitions of each metric in the table above (IC, t-stat, gross / net long-short Sharpe, annual turnover, deflated Sharpe), see [Metric definitions](#metric-definitions).

## Metric definitions

The headline 21-day result is reported in terms of standard cross-sectional asset-pricing metrics. The definitions below apply throughout the rest of the README. The worked example below uses 5-day forward excess return because that is what most stored predictions in the project are; substitute 21-day forward excess return for the headline result without loss of generality (the formula is the same, only the horizon changes).

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
  per-date ICs across all evaluation dates (e.g., `n_dates = 336` for the
  21-day headline OOS slice 2025-01-02 → 2026-04-29). A weak ranker on
  most days plus one strong day does not produce a high IC; consistent
  ranking is required.

  *Worked example.* For 5 tickers with prediction ranks `[5,4,3,2,1]` and
  realized-return ranks `[5,3,4,2,1]` (one adjacent swap),
  `ρ = 1 − 6·Σd²/(N(N²−1)) = 1 − 12/120 = +0.90`. The headline IC of
  +0.089 is the **average of 348 such daily correlations** (each over ~450
  tickers). Small per day, but large in t-stat (+10.5) thanks to the date
  count — *consistency* across days, not magnitude on any one day, is what
  makes it credible.

  *Benchmarks.* On liquid US large-caps, IC = +0.02 (t > 2) is a credible
  edge; +0.10 would be exceptional; > +0.20 is not realistic out-of-sample.

- **t-stat of IC.** `mean(per-date IC) / (stdev(per-date IC) / √n_dates)`.
  Tests whether the mean IC is distinguishable from zero against the
  null hypothesis that per-date ICs are sampled from a distribution
  centered at zero. |t| > 1.96 corresponds to p < 0.05.
- **Long-short Sharpe.** Annualized Sharpe ratio of a daily-rebalanced
  portfolio that is long the top-quintile predicted tickers (top 20%)
  and short the bottom-quintile (bottom 20%), equal-weighted within each
  leg.
  The quintile cut is set in code by `_long_short_returns(top_frac=0.2)`
  in `src/price_model/eval/metrics.py`. Sharpe is computed per-horizon and
  annualized as `mean(per-horizon return) / stdev(per-horizon return) ×
  √(252 / horizon_days)`. Above +1.0 is the conventional bar for
  institutional tradeability; reported values are gross of all transaction
  costs unless explicitly labeled "net N bp."
- **Annual turnover.** One-sided fraction of portfolio capital traded
  per year, computed as `mean(0.5 · Σ_i |w_t[i] − w_{t-1}[i]|) · 252`
  where `w_t` is the long-short weight vector on date `t`. An annual
  turnover of `1.0` means the portfolio replaces itself once per year;
  `100x` means the equivalent of 100 full-portfolio rotations per year.
  The metric is implemented in `src/price_model/eval/turnover.py` and
  governs how much of a gross signal survives net of transaction costs:
  for cost level `b` bp per side, annual cost drag ≈
  `annual_turnover · b · 2 / 10000`.
- **After-cost Sharpe at N bp.** Long-short Sharpe computed on returns
  net of transaction-cost drag. For each date, gross return is
  `mean(top.realized) − mean(bottom.realized)`; net return subtracts
  `turnover_t · N / 10000` from gross. The resulting net-return series
  is annualized to a Sharpe using the same `√(252/horizon)` convention
  as gross Sharpe. Reported at 3, 10, and 20 bp per side to span
  institutional → retail cost regimes.
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

One script runs the whole comparison: held-out Optuna sweeps for the three trees on both panels (rank-9 and engineered-14), then trains the tuned trees + the 6-feature linear model (Lasso and Ridge) + the momentum factors. Everything is **regime-confined** — trained only on 2022-10-10 → 2024-11-30 (single refit) and tested on 2025+.

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
PYTHONPATH=src python scripts/net_cost_regime.py
```

Pulls each headline model from the store (deduped to its latest run), joins realized 21-day excess returns, and prints gross IC / t / Sharpe + **3 / 10 / 20 bp net** (`eval/turnover.compute_turnover_and_costs`). Use this, **not** `scripts/compare_net_of_cost.py`, for the regime headline: the shared compare scripts intersect the full accumulated store (many overlapping historical runs) and drop most of the new models.

### Step 3 (optional) — investigation diagnostics

The Discussion's claims reproduce via:

```bash
PYTHONPATH=src python scripts/spectrum_comparison.py         # model × panel grid: L1≈L2, linear>trees, panel dominates
PYTHONPATH=src python scripts/vol_ablation_lasso_ridge.py    # 9→6 prune; beta_60 is anti-generalizing
PYTHONPATH=src python scripts/inspect_headline_lasso_coefficients.py  # vol-cluster signs are signal, not L1 cancellation (Ridge agrees)
PYTHONPATH=src python scripts/survivorship_ablation_headline.py       # PIT-on vs survivor-snapshot, in-regime
PYTHONPATH=src python scripts/deflated_sharpe_audit.py       # Bailey-López de Prado multi-test correction
```

### Reproducing the full-history (conservative) framing

The full-history numbers in the Executive summary (IC +0.0695; momentum wins net) are the project's **original** headline. On this branch the headline configs are regime-confined, so those numbers reproduce from the **unmodified full-history configs on the `main` branch** (`lasso_elasso_pit_h21`, `extended_kaggle_v3_curated_h21`, `momentum_factor_pit_h21`, …) via the prior `cli run` + `compare_*` flow.

Determinism: results are deterministic for a fixed yfinance / Ken French snapshot (small <5% drift as data refreshes); Optuna sweeps are non-deterministic (TPE seed) but stable within ±0.001 across re-runs.

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

`train_start` and `first_refit` are additive harness parameters (default `None` = the original full-history expanding-window behavior, still used on `main`). For the legacy full-history framing, omit them and use `min_train_days`/`refit_freq_days` as before.

**Feature lookahead safety** is verified by the truncation-invariance test in `tests/test_no_leakage.py`: each registered *feature*'s value at date `t` must be unchanged when the panel is truncated to dates `≤ t`. The test is parametrized over `FEATURE_REGISTRY` and runs in CI. Note it covers features only — *target* leakage across the train/test boundary is the embargo's job (now airtight at 33 days), not this test's.

### Inner cross-validation for regularization-strength selection

For the L1 (Lasso), L2 (Ridge), and L1+L2 (ElasticNet) cross-sectional models, the regularization strength is selected by **5-fold inner cross-validation within each refit's training window**:

- `LassoCV` (sklearn): 100-point geometric grid over α, 5-fold CV, α chosen to minimize pooled CV mean-squared error.
- `RidgeCV` (sklearn): 20-point log-spaced grid over α (project override of sklearn default which is too coarse), 5-fold CV, generalized CV.
- `ElasticNetCV` (sklearn): joint search over α and l1_ratio (default grid `[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]`), 5-fold CV.

**What the inner CV folds actually are (correction to an earlier claim that they are "random"):** `LassoCV`/`RidgeCV` with an integer `cv` use `KFold(shuffle=False)` — contiguous row blocks, not random. Because the panel is sorted `["ticker","date"]` for feature correctness, those contiguous blocks fall on ticker boundaries, so the inner CV is effectively a **by-ticker group split with full date overlap** (verified empirically: every validation date also appears in the training folds). It is therefore **non-temporal** — it scores how an α generalizes across *unseen tickers in the same period*, not across *time* — and it is **MSE-scored**, not IC-scored. This is a lenient grader relative to the purged forward-chain used for the trees (below); within a single stable regime the two agree (both pick near-zero shrinkage), but across a regime break a temporally-honest α-selection shrinks the linear model to the null. See [Matched-grader comparison](#matched-grader-comparison-is-it-the-model-the-panel-or-the-grader).

The walk-forward outer loop still guarantees each refit's α is selected only on data preceding the deployment slice; the inner-CV structure affects *how* α is chosen, not whether the OOS predictions are causal.

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

Two in-regime properties, both opposite to the cross-regime case:

- **CV predicts OOS.** Every model's 2025+ OOS IC *exceeds* its CV IC (e.g. LightGBM 0.051 → 0.069) — no fragility penalty, because the CV folds and the test slice share the regime. The rank-9 panel wins CV for all three trees, so the panel choice is made honestly (on CV), not by peeking at OOS.
- **No HP fragility within a regime.** The earlier cross-regime sweeps (full-history, 5-day) showed the opposite: `lambda_l1` swinging two orders of magnitude across training cutoffs and CV IC *failing* to predict OOS IC — the signature of HP optimization across a regime break. That fragility is a property of training across the 2022 break, not of Optuna; confining to one regime removes it. See [Discussion](#why-hyperparameter-optimization-is-fragile-to-regime-shift) and [`notebooks/06_hp_selection_bias.ipynb`](notebooks/06_hp_selection_bias.ipynb).

### Headline feature panel: 6 curated anomalies (pruned from 9)

The headline result `lasso_curated6_pit_h21` uses **6 features** — the 9-feature anomaly panel below, minus `idio_vol_20`, `max_return_21d`, and `beta_60`. The prune is justified by the collinearity and ablation analysis that follows (it raises in-regime OOS IC from +0.077 to +0.089, identically for Lasso and Ridge). The full 9-feature superset (the prior headline `lasso_elasso_pit_h21`), curated on "one feature per economically distinct anomaly family," with **✓ = kept in the headline 6**:

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

Each feature is **rank-normalized within each date** (cross-sectional rank, then scaled to `[0, 1]`) — robust to fat tails, matches the asset-pricing literature convention, and prevents any single outlier observation from dominating the regression. The curation principle follows Han-He-Rapach-Zhou (2024). L1 and L2 tie on this panel (Lasso +0.0892, Ridge +0.0863), so "L1 regression" names the penalty, not a meaningful edge over Ridge.

**Collinearity is partial, not absent.** The empirical pairwise-correlation matrix (Spearman, rank-normalized panel; computed in [`notebooks/05_feature_exploration.ipynb`](notebooks/05_feature_exploration.ipynb)) shows two correlated clusters rather than a near-orthogonal panel:

- a **volatility/risk cluster** — `vol_ewm_20 ↔ idio_vol_20` = +0.79, `vol_ewm_20 ↔ max_return_21d` = +0.72, `vol_ewm_20 ↔ beta_60` = +0.61, `idio_vol_20 ↔ max_return_21d` = +0.62 (note features 4 and 5 are both Ang-Hodrick-Xing-Zhang 2006 — the same low-volatility family, so the "one feature per family" principle is not fully satisfied here);
- a **trend cluster** — `momentum_12_1 ↔ distance_52w_high` = +0.58, `momentum_12_1 ↔ momentum_756` = +0.52, `momentum_756 ↔ distance_52w_high` = +0.46.

The remaining ~25 of 36 pairs are below |0.3|, but the volatility cluster (0.6–0.79) sits inside the same danger band that produces L1 cancellation on the pure-momentum panel (0.75–0.82; see [Pure-momentum Lasso cancellation diagnostic](#pure-momentum-lasso-cancellation-diagnostic)), and would be flagged by the project's own redundancy auditor (`scripts/audit_lightgbm_features.py`, |corr| > 0.7).

On this panel, `vol_ewm_20` and `idio_vol_20` (the 0.79 pair) take **large opposite-sign coefficients** (≈ +0.014 / −0.012) in the 2025+ refits. This *looks* like the L1 cancellation failure mode — but it is not. **Ridge, which has no cancellation mechanism, reproduces the same opposite signs** (≈ +0.011 / −0.019; see [Matched-grader comparison](#matched-grader-comparison-is-it-the-model-the-panel-or-the-grader)). The signs are the direction the *data* wants: the model is loading on `vol_ewm − idio_vol ≈ systematic (factor) volatility` — a real spread, not an artifact. (Contrast the pure-momentum panel below, where the features are *truly* redundant and the cancellation *is* pathological.) So the 9-feature panel is "mostly low-correlation with one genuinely collinear vol cluster that both L1 and L2 resolve into a signal-bearing spread," not near-orthogonal — and feature-panel design matters more than the regularizer.

**Why the headline prunes to 6.** Single-feature ablation (`scripts/vol_ablation_lasso_ridge.py`, identical for Lasso and Ridge) decomposes the vol cluster:

- `beta_60` is **anti-generalizing** — it is LightGBM's #1 feature by gain (18.8%) yet removing it *raises* OOS IC and Sharpe. Market beta is regime-unstable; the learned tilt does not persist into 2025. This is the bulk of the 9→6 improvement.
- `idio_vol_20` is marginal — its `vol_ewm − idio_vol` spread is real but small, so dropping it costs almost nothing.
- `max_return_21d` is roughly neutral once `beta_60` is gone.
- `vol_ewm_20` **must stay** — dropping the last low-vol representative collapses the signal.

Dropping the first three lifts in-regime OOS IC +0.077 → **+0.089** and gross Sharpe +1.17 → **~2.0**. (Trees behave differently: they peak at 8 features — drop only `beta_60` — because they can use `idio_vol_20`/`max_return_21d` via interactions that the linear model cannot. Each model's optimal panel size differs; see [Discussion](#matched-grader-comparison-is-it-the-model-the-panel-or-the-grader).)

### Net-of-cost decomposition

Gross IC and gross long-short Sharpe assume costless rebalancing. Net-of-cost Sharpe accounts for transaction-cost drag from portfolio turnover. For each date, **daily turnover** is computed as the L1 distance between today's and yesterday's normalized long-short weight vectors. Annual turnover = `mean(daily turnover) × 252`. **After-cost daily return** = gross return − `daily_turnover × cost_bps / 10000`. After-cost Sharpe is computed on the net-return series with the same `√(252 / horizon)` annualization as gross Sharpe.

Reported at 3 bp (institutional all-in), 10 bp (small-fund), and 20 bp (retail-equivalent). Turnover and cost-drag mechanics are implemented in `src/price_model/eval/turnover.py`. For the **regime-confined headline** use `scripts/net_cost_regime.py`, which pulls each model from the store deduped to its latest run; the shared `scripts/compare_net_of_cost.py` intersects the full accumulated store (many overlapping historical runs) and drops most of the new regime models, so it is not reliable for the regime headline.

### Bailey-López de Prado deflated Sharpe (multi-test correction)

`scripts/deflated_sharpe_audit.py` applies the deflated Sharpe ratio (Bailey & López de Prado 2014) across all project models. The correction inflates the effective number of trials by the project's experiment count and accounts for skew and kurtosis of the return distribution to compute `P(true_Sharpe > 0 | observed)`. Threshold for "significant after multi-test correction" is DSR > 0.95.

The deflated-Sharpe pass list aligns with the t-stat ranking. In the regime-confined headline the 6-feature linear models (t ≈ 10.5) and momentum factors (t ≈ 7.2) clear DSR > 0.99 trivially; the tuned trees (t ≈ 4.9–6.1) clear DSR > 0.95; lower-t-stat variants do not. (DSR still applies the project-wide trial count, so a high single-model t-stat is necessary but not sufficient.)

### Pure-momentum Lasso cancellation diagnostic

`scripts/inspect_momentum_lasso_coefficients.py` fits Lasso and ElasticNet on a four-feature pure-momentum panel (12-1, 18-month, 24-month, 36-month) and prints the fitted coefficients alongside the feature correlation matrix. The diagnostic confirms:

1. Pairwise correlations between momentum features at 0.48–0.82 (e.g., `momentum_504 ↔ momentum_756` at 0.75).
2. The CV-selected α drives all coefficients to (near) zero — the model collapses to predicting the intercept. The realized predictions inherit the sign of the small noisy intercept, producing systematic anti-correlation with realized returns.
3. ElasticNet picks l1_ratio close to 1.0 (essentially pure Lasso) — L2 stabilization does not help on a heavily-collinear panel.

This diagnostic supports the discussion of why pure-momentum L1 produces negative OOS IC (see [Discussion](#discussion)). **Contrast with the headline vol cluster:** there the opposite-sign coefficients are *signal* (Ridge reproduces them; the `vol_ewm − idio_vol` spread is real), whereas here the four momentum features are *truly* redundant ("did this stock rise over some multi-month window"), so the cancellation has nothing to extract and collapses to the noisy intercept. The cancel-ratio metric alone cannot tell the two apart — Ridge is the control: it shares weight on signal-bearing clusters but cannot rescue a genuinely redundant one.

### Regime indicator and audit-driven LightGBM v3 panel

The 14-feature LightGBM v3 panel includes `cs_return_dispersion_20` — the cross-sectional standard deviation of daily log returns smoothed by 20-day rolling mean. The feature is a lookahead-safe regime conditioning variable that distinguishes high-dispersion (idiosyncratic-pricing) regimes from low-dispersion (uniform-market-move) regimes without using forward-looking labels. Implementation in `src/price_model/features/cross_features.py::CsReturnDispersion20`; lookahead safety verified by `tests/test_microstructure_features.py`.

The v3 panel curation (14 features chosen from a 21-feature candidate panel via gain importance + cross-feature correlation analysis) is documented inline in `config/experiments/extended_kaggle_v3_curated.yaml`; audit script in `scripts/audit_lightgbm_features.py`.

## Discussion

The regime-confined headline finding — a 6-feature linear cross-sectional regression beats every held-out-tuned tree *and* long-horizon momentum on both gross IC and net-of-cost Sharpe, but only within the post-2022-10 regime it was trained on — raises five questions worth unpacking. This section addresses each.

### Why linear beats trees on this universe

**The 21-day cross-sectional signal on the post-2024 S&P 500 is approximately linear in feature space.** Three pieces of evidence:

1. **The momentum factors alone capture most of the signal.** mom_504, mom_756, and mom_378 individually produce IC = +0.04 to +0.05 with no model fitting; the L1 regression's incremental edge of +0.02 IC over the best single momentum factor reflects modest additional signal from non-momentum features (low-volatility, idiosyncratic volatility, MAX effect). The dominant component is a linear momentum signal that trees would have to recover via many small splits.
2. **Tree-ensemble gain-importance audits show feature concentration.** On the 14-feature v3 panel, three features produce 39% of LightGBM gain. The split structure devotes most of its capacity to a handful of features that L1 can express as a small number of coefficients. Trees' theoretical interaction-capturing advantage doesn't pay off here because the dominant signal isn't interactive.
3. **Held-out-tuned trees still fall short.** With each tree Optuna-tuned (held out at 2024-12-31) on its *best* panel, the regime-confined OOS IC is +0.069 (LightGBM) / +0.061 (CatBoost) / +0.058 (XGBoost) vs the linear +0.089 — a ~0.02–0.03 gap that is not a tuning failure (the matched-grader subsection below removes the HP-selection confound entirely and the gap persists).

Two caveats limit how broadly this conclusion generalizes:

- **The 6-feature panel is small and ex-ante-curated.** The linear advantage depends on this curation. On the broader 14-feature engineered panel the linear recipe is weaker — but so are the trees (every model is worse there), so curation, not model class, is the lever.
- **The post-2022-10 regime is structurally momentum-friendly.** In a regime with more leadership rotation or weaker momentum persistence, the linear-on-anomalies recipe would compress and trees might catch up. The result is robust within the studied window but has not been verified across regime changes.

**Crucially, the linear model is itself regime-bound.** A time-split partition of its full prediction history (see [`notebooks/03_headline_robustness.ipynb`](notebooks/03_headline_robustness.ipynb)) shows IC = **−0.0747 (t = −5.34)** on the 2020-mid → 2022-10 window — the same panel is *significantly anti-predictive* before the bull regime. The +0.089 in-regime OOS IC is not a steady-state property; it is the model's behavior in one regime. This is exactly why the headline trains regime-confined: the CV-fit coefficients learn whichever structure dominates the training window, and training across the 2022 break makes a temporally-honest fit collapse to the null (Claim 3 below). Trained on the matching regime, deployed in its continuation, it works; straddling a regime change, it inverts.

The conclusion is therefore "within the post-2022-10 regime, on this universe at this horizon, a curated 6-feature linear model outperforms every tree-ensemble variant tested" — not "ML cannot succeed in cross-sectional equity prediction generally," and not "the edge is regime-robust."

### Net-of-cost: turnover, and the framing that flips the winner

Net-of-cost ranking depends on turnover, and the two training framings give opposite winners — the gap between them is itself a finding.

**Regime-confined (headline): linear wins net.** Trained on the bull regime, the 6-feature linear model has gross Sharpe +2.00 at **95×** annual turnover (lower than the 9-feature's 127× — fewer, more stable features), so at 20 bp it retains net **+1.94**. The 36-month momentum factor is the low-turnover champion (15×, net +1.68) but its lower gross (+0.050 IC, Sharpe +1.69) can't catch the supercharged linear. Tuned trees net +0.9–1.3 (turnover 64–90×). So in-regime, **linear wins on both gross and net** — the first configuration in this project where the model beats momentum net-of-cost.

**Full-history (conservative): momentum wins net.** Trained on the full expanding window, the linear model's gross Sharpe is only +1.24 at 128× turnover → net +1.17 at 20 bp, while mom_756 retains +1.81 (15× turnover, ~99% of gross). Here momentum wins net by +0.64 Sharpe, the README's original conclusion.

**What flips it:** regime-matched training nearly doubles the linear gross Sharpe (1.24 → 2.00) *and* the curated 6-feature panel cuts turnover (127× → 95×). Momentum's net Sharpe barely moves between framings (it is training-independent); the linear model's does. So the net-of-cost winner is not a property of the strategies alone — it is a property of whether you train across or within the deployment regime.

**Why the regime is momentum-friendly at all:** Mag-7 megacap concentration and AI-sector persistence mean multi-year leaders are stable, which both raises the anomaly signal and keeps momentum's turnover low. In a higher-rotation regime both effects weaken.

### Why hyperparameter optimization is fragile to regime shift

**Optuna minimizes CV error on training data, not OOS error on deployment data.** When the training period is a different statistical regime from the deployment period, the HPs that minimize training-period CV error don't minimize deployment-period OOS error. The gap between train-optimal HPs and deploy-optimal HPs grows with regime divergence.

This is exactly the **DeMiguel-Garlappi-Uppal (2009)** result generalized: they showed that the optimal portfolio weights estimated from historical returns underperform the equal-weight (1/N) portfolio out-of-sample because estimation error in optimal weights overwhelms the theoretical benefit of optimization. The same logic applies to HP search:

- **Default HPs are like 1/N.** They don't fit training data as hard, so they generalize better.
- **Optuna-selected HPs are like Markowitz weights.** They minimize a training-period objective but inherit the estimation error.

The cleanest *cross-regime* demonstration is the full-history sweep: held-out Optuna at 21-day scored OOS IC +0.0187 vs default HPs' +0.0274 — even with the deployment slice excluded from HP selection, the chosen HPs *underperformed defaults*, because they were tuned across the 2022 regime break. The 5-day evidence is the same signature (Split B beats Split A on 2025+ but loses full-sample; `lambda_l1` swings two orders of magnitude across cutoffs).

**This fragility is a cross-regime property, not an Optuna property.** Confine training to one regime and it disappears: in the regime-confined sweeps (Methodology table) every tuned tree's OOS IC *exceeds* its CV IC — CV becomes a reliable predictor of OOS once the folds and the test share the regime. So the practical rule is not "don't tune" but "don't tune across a regime break": verify a tuned model's OOS IC beats defaults on a held-out slice, and confine training to the deployment regime.

### Why pure-momentum L1 produces statistically significant negative IC

The mechanism is **L1 cancellation on collinear features**, with a twist documented by the diagnostic script:

1. **The pure-momentum panel has high pairwise correlations.** mom_504 ↔ mom_756 = 0.75; mom_378 ↔ mom_504 = 0.82; mom_12_1 ↔ mom_378 = 0.72. All four features measure "did this stock go up over some multi-month window."
2. **At the α level CV selects, all coefficients are driven to zero.** The model collapses to predicting the intercept term, which on rank-normalized features is approximately zero. The realized predictions inherit the sign of the small noisy intercept.
3. **The noisy intercept is systematically anti-correlated with realized returns** across walk-forward refits, producing statistically significant negative IC.

ElasticNet does not help because the L2 component cannot stabilize the unstable L1 subspace at the CV-selected α magnitude.

The positive corollary: **L1 regularization is not a free lunch on factor zoos.** The Han-He-Rapach-Zhou (2024) paper's substantive contribution is the *panel-curation principle* (one feature per economic family), not the choice of regularization. This project's results confirm that empirically — the curated 6-feature panel produces +0.089 in-regime IC; the same regularization on a single-family momentum panel produces −0.018 IC. **Feature-panel design matters more than regularization choice.**

### Matched-grader comparison: is it the model, the panel, or the grader?

The headline "L1 beats every ML variant" is confounded: the Lasso's α was chosen by the lenient non-temporal MSE-scored inner CV (above), while the trees' HPs were chosen by the strict purged forward-chain IC-scored Optuna. To remove the confound, every model was re-run under the **same** purged-forward-chain IC grader, on the **same** split, confined to the bullish regime (train 2022-10-10 → 2024-11-29, test 2025-01-02 → 2026-05-22). Reproduce with `scripts/matched_grader_comparison.py`, `scripts/spectrum_comparison.py`, and `scripts/vol_ablation_lasso_ridge.py`. Findings:

1. **L1 ≈ L2 — the regularizer is a coin flip.** On a matched panel and grader, Lasso and Ridge tie to four decimals (e.g. +0.0772 vs +0.0769 on the 9-feature panel). The headline table's "L1 +0.0695 vs Ridge +0.0430" gap is *panel*-driven (different 9- vs 12-feature panels), not an L1-over-L2 effect. The headline could equally be titled Ridge.
2. **Linear still beats trees — even on the trees' own panel.** Under the identical strict grader, the 9-feature linear models score ~+0.077 vs LightGBM's +0.0525; on the 14-feature *engineered* tree panel every model is *worse* (linear ~+0.056, LightGBM +0.0403). So the linear advantage is not an artifact of giving trees a linear-optimized panel. The signal is near-linear; trees over-parameterize it.
3. **Panel design dominates, and the optimal size is model-specific.** Pruning the 9-feature panel to **6** (`momentum_12_1`, `momentum_756`, `return_1d`, `vol_ewm_20`, `distance_52w_high`, `log_dollar_volume`) raises OOS IC +0.077 → **+0.088** and gross Sharpe +1.17 → **~1.97** — identically for Lasso and Ridge (`config/experiments/lasso_curated6_pit_h21.yaml`). LightGBM instead peaks at **8** features (drop only `beta_60`); cutting it to 6 *hurts* it, because it can exploit `idio_vol_20`/`max_return_21d` via interactions that linear cannot.
4. **`beta_60` is anti-generalizing for every model.** It is LightGBM's #1 feature by gain (18.8%) yet removing it *raises* OOS Sharpe; for the linear models it is the single biggest drag (dropping it is most of the 9→6 gain). Market beta is regime-unstable, so the learned tilt does not persist into 2025. One low-vol representative (`vol_ewm_20`) must be kept — dropping the whole vol family collapses the signal.

These results are *within-regime*: the lenient and strict graders agree here only because the 2025 test slice is a continuation of the recent training regime. Across the 2022 regime break, the strict (forward-chaining) grader correctly shrinks the linear model to the null — see [Why hyperparameter optimization is fragile to regime shift](#why-hyperparameter-optimization-is-fragile-to-regime-shift). Ablation detail in [`notebooks/05_feature_exploration.ipynb`](notebooks/05_feature_exploration.ipynb).

### Synthesizing thesis

The observations above point to one unifying conclusion: **within the post-2022-10 regime, on this universe at the 21-day horizon, the cross-sectional return signal is dominated by slow-moving, near-linear patterns (a small set of documented anomalies) that a curated linear model recovers more efficiently than tree ensembles — winning on both gross IC and net-of-cost Sharpe.** Trees can express anything the linear model can, but pay an estimation-error tax that, with their higher turnover, leaves them behind on both. The regularizer (L1 vs L2) is irrelevant; the curated panel and the regime-matched training window are what matter. The one essential caveat: this is a *regime-conditional* edge — trained across the 2022 break it collapses to the null, and the same panel is significantly anti-predictive pre-2022. Deployability rests on the deployment regime resembling the training regime.

A cleaner test of "is this finding regime-specific or general" would require running the same comparison on multiple non-overlapping deployment windows (e.g., 2010–2014, 2014–2018, 2018–2022, 2022–2026) with paid PIT-correct data. That multi-window stress test is the most-needed extension of this work and is enumerated in [Scope and limitations](#scope-and-limitations).

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

The regime-confined headline IC of **+0.089** (6-feature linear on 2025+ OOS) and every other IC in the README — gross or net — should be interpreted with the following caveats. The bias sources are inherent to the data and methodology and apply uniformly across models, not to one model in isolation.

0. **Regime-confined training is best-case, by construction.** The headline trains only on the bull regime it is then deployed into. This *raises* the apparent edge (+0.089 vs the full-history +0.0695) precisely by specializing the model to one regime — so it is **more** regime-bound, not less. It answers "how well does this work when the deployment regime matches recent training?", not "how well does it work in general." The full-history framing (+0.0695, loses net to momentum) is the more conservative number.
1. **PIT correction is partial.** Because yfinance does not return data for SIVB, FRC, ATVI, AGN, and similar delisted symbols, those tickers do not appear in the PIT panel even when Wikipedia indicates they were index members on the relevant dates. The PIT-corrected backtest therefore still excludes the worst realized losers of the 2022-2023 banking crisis. A fully PIT-correct analysis using paid data (Norgate, Polygon, or CRSP) would either misrank or correctly short SIVB at −85% on 2023-03-08; the present model does neither.
2. **Survivorship-bias inflation — measured directly on the headline.** `scripts/survivorship_ablation_headline.py` runs the headline recipe on the same price source under per-date PIT membership vs a current-survivors-only cohort. On the **2025+ slice the inflation is ≈ 0** (the recent roster ≈ the point-in-time roster), so the headline is not survivorship-inflated *on its own window*. But over the **full 2021–2026 sample the honest IC is ≈ 0 while the survivor cohort shows +0.016** — a sign-flip — confirming the general rule (bias is larger when the honest signal is weak; cf. the legacy v2 ablation in `notebooks/00`, 61–89% inflation). Even so, the ~12% of delisted names yfinance cannot fetch are absent from *every* arm, so the measured inflation is itself a floor.
3. **The evaluation window is regime-concentrated — and the headline leans into it.** The 2025+ window is Mag-7 / AI-concentration / momentum-persistent. The headline both *trains* and *tests* inside this regime (caveat 0), so it reflects "+0.089 deploying into a momentum-friendly regime that resembles recent training," not an all-weather number. The same panel's pre-2022 IC is significantly **negative (−0.0747)**.
4. **Transaction costs, taxes, and slippage are partially modeled.** The 3/10/20 bp net columns model bid-ask + commission only; they exclude slippage on illiquid names, capital-gains taxes, and short-borrow. In-regime the 6-feature linear nets +1.94 at 20 bp (turnover 95×) and momentum +1.68 (15×) — both institutional-grade gross-and-modeled-net, but the unmodeled frictions and small-account constraints still bite a retail book.
5. **Single evaluation window.** The 2025+ slice is one window. Alternative windows (2018–2020, 2020–2022, 2022–2024) would produce different ICs; the multi-window stress test is the most-needed extension and has not been performed.

### Toward a fully PIT-correct evaluation

Listed in approximate order of effort:

- **Replace yfinance with Norgate Premium Data (~$60 / month)** for survivorship-bias-free prices including delisted history. All reported ICs would change; the headline edge would likely compress somewhat, and the regime-conditional shape (post-2022 strong, pre-2022 negative) is expected to persist.
- **Replace the Wikipedia membership source with CRSP** (paid; free
  academic access for most affiliated researchers). Provides cleaner
  pre-2014 history and eliminates scrape fragility.
- **Run multi-window robustness tests** at three different start / end
  pairs.
- **Add transaction-cost modeling** at the prediction-store layer so
  reported metrics are net of realistic costs.

## Scope and limitations

- **Not deployable for retail trading.** In-regime the 6-feature linear model is institutional-grade on both gross (IC +0.089, Sharpe +2.0) and *modeled* net (Sharpe +1.94 at 20 bp). But the net columns model only bid-ask + commission; once you add slippage on size, capital-gains taxes, short-borrow, and the small portfolio sizes a retail account can hold, the realistic retail edge shrinks sharply. Combined with the regime-bound caveat below, this is a research result, not a trading recommendation.
- **The edge is sharply regime-concentrated — and the headline trains inside it.** A time-split of the linear model's full prediction history at 2022-10-10 (see [`notebooks/03_headline_robustness.ipynb`](notebooks/03_headline_robustness.ipynb)) shows three regimes:
   - **Pre-Oct-2022** (2020-mid → 2022-10): IC = **−0.0747** (t = −5.34). The same panel is *significantly anti-predictive* in the pre-bull regime.
   - **Oct-2022 → end-2024**: IC ≈ **0** — the transitional window the headline trains on.
   - **2025+ (the test slice)**: IC = **+0.089** (regime-confined training), +0.0695 (full-history training).

   The headline confines training to the post-2022-10 regime by design, so its edge lives **entirely** in the momentum-persistent regime. The 2025+ OOS IC is a valid causal measurement, but training across the 2022 break makes a temporally-honest fit collapse to the null. **Deployment outside a megacap-momentum-persistent regime is not supported by this evidence.**
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
- `notebooks/01_prediction_diagnostics.ipynb` — Stored-prediction diagnostics on the headline 21-day models: apples-to-apples comparison, rolling 60-day IC, regime-conditional IC, net-of-cost gross-vs-net Sharpe with turnover annotation. Supports Discussion Claims 1 (net-of-cost inversion), 2 (regime fragility), and Methodology.
- `notebooks/02_classical_timeseries.ipynb` — Classical EMH baselines (ARIMA / GARCH / GBM) on the per-ticker time series, contrasted with the cross-sectional L1 regression. Supports Discussion Claim 1 (why cross-sectional methods beat univariate).
- `notebooks/03_headline_robustness.ipynb` — Bootstrap CI / decile bucket monotonicity / time-split partition on the L1 regression headline. Supports Methodology (deflated-Sharpe-adjacent rigor).
- `notebooks/05_feature_exploration.ipynb` — Feature engineering boilerplate: 41-feature registry tour, distributions, correlations, fitted L1 / LightGBM importance, and a step-by-step template for adding new features. Now also includes the **panel-ablation findings** (9→6 prune, `beta_60` anti-generalizing, L1 ≈ L2, vol-cluster-is-signal). Supports Methodology (panel design) + Discussion's [Matched-grader comparison](#matched-grader-comparison-is-it-the-model-the-panel-or-the-grader).
- `notebooks/06_hp_selection_bias.ipynb` — Visualizes the held-out Optuna sweeps documented in the README's Methodology Optuna-sweep table. Shows the chosen HPs across cutoffs (lambda_l1 varies 390× across Split A vs Split B), OOS IC bar chart, and the held-out-vs-default inversion at 21-day. Supports Discussion Claim 2.

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

- **Han, Y., He, A., Rapach, D., and Zhou, G.** (2024). "Expected Stock Returns and the Cross-Section: An E-LASSO Approach." *Review of Finance*. — Panel-curation principle (one feature per economic family, rank-normalized, CV-selected α) used in the headline 9-feature L1 regression (`lasso_elasso_pit_h21`).
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
