# price-model

A point-in-time (PIT) corrected cross-sectional equity return predictor on the S&P 500.

## Executive summary

Out-of-sample **Information Coefficient = +0.0183 (t = +4.3)** and long-short
**Sharpe = +0.86** are obtained over 905 trading days in the post-October-2022
high-dispersion regime, on a Wikipedia-reconstructed point-in-time 617-name
historical universe. The model uses 22 features:

- 13 technical baselines.
- Three documented academic anomalies (Jegadeesh-Titman 12-1 momentum,
  Hong-Lim-Stein 52-week high, Lehmann 1-day reversal).
- Six OHLCV / volume features (Parkinson high-low volatility,
  Bali-Cakici-Whitelaw MAX effect, dollar volume, abnormal turnover,
  intraday range, intraday body).

The result is statistically significant, regime-conditional, and **not
deployable for retail investors** after transaction costs, taxes, and breadth
limits — see [Scope and limitations](#scope-and-limitations).

### Metric definitions

- **Information Coefficient (IC).** Per-date Spearman rank correlation
  between predicted and realized 5-day forward excess returns, averaged
  across all dates in the evaluation window. Measures cross-sectional
  ranking quality. IC ∈ [−1, +1]; on liquid US large-caps, +0.02 with
  t-stat > 2 is considered a credible edge.
- **t-stat of IC.** `mean(daily IC) / (stdev(daily IC) / √n_dates)`. Tests
  whether the mean IC is distinguishable from zero. |t| > 1.96 corresponds
  to p < 0.05.
- **Long-short Sharpe.** Annualized Sharpe ratio of a daily-rebalanced
  portfolio that is long the top-quintile predicted names (top 20%) and
  short the bottom-quintile (bottom 20%), equal-weighted within each leg.
  The quintile cut is set in code by `_long_short_returns(top_frac=0.2)`
  in `src/price_model/eval/metrics.py`. Sharpe is computed per-horizon and
  annualized as `mean(per-horizon return) / stdev(per-horizon return) ×
  √(252 / horizon_days)`. Above +1.0 is the conventional bar for
  institutional tradeability; values reported here are gross of all
  transaction costs.

## Headline result: ablation analysis

The headline result is **+0.0183 IC (t = +4.34) on the 617-name PIT
universe with 22 features, post-October-2022**. To establish that this is
not an artifact of any single confounder, three independent dimensions are
varied — universe (with and without point-in-time correction), feature
set, and time window — and all six cells of the resulting 2×3 matrix are
reported on the strongest feature set included in the project.

The presentation order below is *not* the chronological order in which the
project was constructed. The chronological build proceeded from
technical-baseline → PIT-corrected → anomalies → OHLCV completeness; that
history is preserved in the git log. The ablation table is the
methodologically preferred ordering: one effect is isolated at a time on
the 22-feature model, rather than mixing "changed the universe" with
"changed the feature set" in the same step.

### Universe × PIT × regime ablation (22-feature model)

| | Subset universe (160 modern survivors, PIT OFF) | Headline universe (617 historical, PIT ON) |
|---|---|---|
| **Full sample** (1758 dates) | IC = +0.0142, t = +4.05, Sharpe = +0.39 | IC = +0.0055, t = +1.51, Sharpe = +0.21 |
| **Pre-Oct-2022** (853 dates) | IC = +0.0034, t = +0.64, Sharpe = −0.07 | IC = −0.0082, t = −1.38, Sharpe = −0.16 |
| **Post-Oct-2022** (905 dates) | IC = +0.0244, t = +5.37, Sharpe = +0.94 | **IC = +0.0183, t = +4.34, Sharpe = +0.86** ← headline |

Three observations follow.

**1. Survivorship-bias collapse varies sharply by regime.** Across rows:
full-sample IC collapses from +0.0142 to +0.0055 under PIT correction
(a **61% reduction**); pre-October-2022 IC **flips sign** from a noisy
+0.0034 to a clear −0.0082; post-October-2022 IC collapses only 25% (from
+0.0244 to +0.0183). Survivorship bias dominates when the underlying
signal is weak (pre-2022 noise) and contributes far less when the signal
is strong (post-2022 regime). A practitioner running the
subset / full-sample backtest would observe +0.0142 IC at t = 4.05 and
infer a credible statistically significant edge, 61% of which is selection
bias.

**2. The regime effect is large and robust to universe.** Down columns:
both universes show comparable regime shifts in Sharpe (subset:
−0.07 → +0.94; PIT: −0.16 → +0.86). The post-October-2022 regime contains
real cross-sectional signal that is not primarily a survivorship artifact.
The IC magnitudes shift similarly (subset: +0.0034 → +0.0244, roughly 7×;
PIT: −0.0082 → +0.0183, further amplified by the sign flip).

**3. Pre-October-2022 IC on the PIT universe is negative with t = −1.38.**
The 22-feature model would have actively lost on the cross-section before
the regime break, not merely been flat. The finding is sharper than "the
edge is concentrated post-2022": the edge is post-2022 *and* the opposite
disedge is pre-2022. The features behave as a regime-conditional
intensifier, amplifying whichever direction the cross-section is paying.

### Feature-set ablation

Holding the universe fixed at PIT-on and varying the feature set
(chronological order of additions):

| Features | Full sample IC | Full sample t | Post-Oct-2022 Sharpe |
|---|---|---|---|
| 13 technical only | +0.0008 | +0.24 | −0.04 |
| 13 + 3 academic anomalies | +0.0033 | +0.96 | +0.083 |
| **22 (+ 6 OHLCV / volume)** | **+0.0055** | **+1.51** | **+0.86** (regime split) |

The OHLCV / volume batch is the largest single-batch incremental
contribution. Each addition was tested on identical splits / embargo /
walk-forward settings in apples-to-apples runs; the YAML configs
(`extended_kaggle_v2_pit.yaml`, `extended_kaggle_v2_anomaly.yaml`,
`extended_kaggle_v2_ohlcv.yaml`) make this auditable.

## Reproduction

### Step 0 — install and build both universes (one-time, ~20-25 min cold)

The ablation matrix uses two universe files. Both must be prepared before
any backtest is executed.

```bash
pip install -e ".[dev,classical]"
```

**Universe A — `sp500` (the subset, used to measure survivorship bias).**
A checked-in static list of 160 modern-survivor tickers at
`src/price_model/data/universes/sp500.txt`. The name `sp500` is a misnomer
retained for backwards compatibility with existing experiment configs: the
file is *not* the full historical S&P 500 but a modern-survivor subset
(no delisted or acquired names, no PIT membership applied). No
`build-universe` step is needed; the file is checked in. Only the price
fetch is required:

```bash
# Fetch yfinance daily bars for the 160 subset names (~5-10 min cold)
python -m price_model.cli refresh-data --universe sp500 --start 2017-01-01
```

**Universe B — `sp500_pit` (the PIT-corrected headline universe).**
Wikipedia-reconstructed historical universe with point-in-time membership
applied at backtest time. The membership table and tickers list are
generated by the build step, then yfinance data is fetched for all
resolving names:

```bash
# Scrape Wikipedia for historical S&P 500 components and write the universe
python -m price_model.cli build-universe --name sp500_pit --start 2017-01-01

# Fetch yfinance data for all ~700 names that resolve (~10-15 min cold)
python -m price_model.cli refresh-data --universe sp500_pit --start 2017-01-01
```

### Step 1 — the two headline runs

The 2×3 ablation matrix is filled by two backtest runs. Each populates one
column; rows are obtained by time-splitting the predictions.

```bash
# Headline (PIT, 22 features): fills the right column.
python -m price_model.cli run -e extended_kaggle_v2_ohlcv
# Full sample → IC ≈ +0.0055, t ≈ +1.51, Sharpe ≈ +0.21

# PIT-off counterfactual (subset, 22 features): fills the left column.
python -m price_model.cli run -e extended_kaggle_v2_ohlcv_subset
# Full sample → IC ≈ +0.0142, t ≈ +4.05, Sharpe ≈ +0.39
```

### Step 2 — regime split

```bash
jupyter notebook notebooks/03_robustness.ipynb
# PIT model    post-October-2022:  IC = +0.0183, Sharpe = +0.86, t ≈ +4.34  ← headline
# PIT model    pre-October-2022:   IC = -0.0082, Sharpe = -0.16
# Subset model post-October-2022:  IC = +0.0244, Sharpe = +0.94, t ≈ +5.37
# Subset model pre-October-2022:   IC = +0.0034, Sharpe = -0.07
```

### Step 3 (optional) — the feature-set ablation runs

To reproduce the feature-set table (13 technical → 16 anomaly → 22 OHLCV):

```bash
python -m price_model.cli run -e extended_kaggle_v2_pit       # 13 technical, PIT
python -m price_model.cli run -e extended_kaggle_v2_anomaly   # 16 + anomalies, PIT
# The 22-feature run is already performed in Step 1.
```

The expected numbers above are deterministic for a fixed data snapshot.
Small drift (< 5%) is expected as yfinance updates and the Ken French
factor file refreshes monthly.
## Data quality and methodological limitations

The headline result is honest *about what it measures*, but what it
measures is bounded by the data available without a paid feed. This
section enumerates those boundaries.

### Limitations of yfinance

yfinance is the only free source of daily-bar US equity data and is the
reason the project is reproducible without a paid subscription. It has
three documented failure modes that materially affect the headline.

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

### Sources of upward bias in the headline

The headline IC of +0.0183 should be interpreted with the following
caveats.

1. **PIT correction is partial.** Because yfinance does not return data
   for SIVB, FRC, ATVI, AGN, and similar delisted symbols, those tickers
   do not appear in the PIT panel even when Wikipedia indicates they were
   index members on the relevant dates. The PIT-corrected backtest
   therefore still excludes the worst realized losers of the 2022-2023
   banking crisis. A fully PIT-correct analysis using paid data (Norgate,
   Polygon, or CRSP) would either misrank or correctly short SIVB at
   −85% on 2023-03-08; the present model does neither.
2. **The 61% bias estimate is a lower bound.** It reflects the share of
   apparent edge attributable to selection *given the data available*.
   The actual selection bias on a fully PIT-correct dataset (with
   delisted history and complete pre-2014 membership) would be larger.
   The IC drop from +0.0142 to +0.0055 on the 22-feature model is
   therefore a floor on backtest inflation, not a ceiling. For reference,
   the same effect on the original 13-feature technical-only baseline
   was an 89% collapse (+0.0075 → +0.0008); weaker feature sets exhibit
   higher survivorship-bias inflation.
3. **The post-2022 regime contains the bank-failure period (March-May
   2023).** The headline +0.0183 IC over 905 days post-October-2022 is
   computed on a cross-section that excludes the names that
   catastrophically failed in that window. A real-world model would need
   to predict (or fail to predict) those failures; the present model
   does not face that test. The accurate framing of the headline is
   "+0.0183 on the survivors of the regime, given the available data."
4. **Transaction costs, taxes, and slippage are zero in the backtest.**
   All ICs and Sharpes assume costless rebalancing. A retail investor
   faces bid-ask spreads (~5-10 bp), commissions, capital-gains tax, and
   slippage; the reported Sharpe of +0.86 is gross. After realistic
   retail costs, the after-cost Sharpe on a 10-30 name portfolio
   approaches zero. An institutional desk paying ~1-3 bp all-in could
   plausibly net a Sharpe in the 0.4-0.6 range from this signal, which
   would not constitute a standalone strategy.
5. **Single evaluation window.** The 2017-2026 window was selected as
   the range yfinance reliably covers. Alternative windows (2010-2020,
   2017-2026, 2020-2026) would produce different deltas at each stage;
   a multi-window robustness check has not been performed.

### Toward a fully PIT-correct evaluation

Listed in approximate order of effort:

- **Replace yfinance with Norgate Premium Data (~$60 / month)** for
  survivorship-bias-free prices including delisted history. All five IC
  numbers would change; the headline +0.0183 would likely fall by
  0.002-0.005, but the regime-conditional shape (post-2022 strong,
  pre-2022 negative) is expected to persist.
- **Replace the Wikipedia membership source with CRSP** (paid; free
  academic access for most affiliated researchers). Provides cleaner
  pre-2014 history and eliminates scrape fragility.
- **Run multi-window robustness tests** at three different start / end
  pairs.
- **Add transaction-cost modeling** at the prediction-store layer so
  reported metrics are net of realistic costs.

## Scope and limitations

- **Not deployable for retail trading.** After bid-ask spreads,
  commissions, and capital-gains taxes, the +0.0183 IC and gross Sharpe
  of +0.86 yield an after-cost edge close to zero for a 10-30 name
  portfolio (breadth too small) rebalanced quarterly (turnover too low).
  The result is institutional-grade gross, not retail-grade net.
- **Not a guarantee that future regimes will resemble the post-2022 one.**
  The edge is concentrated in a single regime (October 2022 → May 2026)
  characterized by Mag-7 dominance, rate-cycle dispersion, and AI-driven
  sector divergence. The same 22-feature model on pre-2022 data yields
  IC = **−0.0082**; the OHLCV / anomaly features behave as
  regime-conditional intensifiers rather than universally beneficial
  signals. Their inclusion post-2022 increases IC by approximately 58%;
  their inclusion pre-2022 increases the magnitude of the negative IC by
  approximately 50%.
- **Not a substitute for index funds.** For individual investors, decades
  of research show that low-cost diversified index funds outperform
  almost all active strategies after fees and taxes.
- **Not a complete PIT correction.** See the data-quality limitations
  above.

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
notebooks/               # diagnostic + classical + robustness + portfolio
tests/                   # leakage tests, PIT tests, ticker tests, contract tests
```

The same data infrastructure also supports
`notebooks/04_portfolio_attribution.ipynb`, which uses the Ken French
adapter directly (not the model layer) to decompose a 10-stock
equal-weight portfolio's exposures and attribute realized returns to
factors. The notebook is an independent application of the data layer
and does not depend on the predictive model.

### Streamlit dashboard

`src/price_model/dashboard/` is a thin Streamlit reader over the DuckDB
prediction store. It exists to allow the model's daily output to be
inspected visually (per-date top / bottom quintile, rolling IC,
prediction vs. realized scatter) without writing a notebook for each
inspection. The dashboard is a debugging and monitoring surface, not
part of the reproduction workflow. None of the headline numbers in this
README originate from the dashboard, and skipping it does not affect
reproduction of any ablation cell. The dashboard is most useful when
extending the model and requires a quick sanity-check view of new runs.

## Development notes

The substantive research decisions in this project — what to measure,
what constitutes a fair test, how much survivorship bias the headline
contains, which limitations are honest to ship with — are the author's.
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

- **Fama, E. and French, K.** (2015). "A Five-Factor Asset Pricing Model."
  *Journal of Financial Economics* 116(1).
- **Jegadeesh, N. and Titman, S.** (1993). "Returns to Buying Winners and
  Selling Losers." *Journal of Finance* 48(1). — 12-1 momentum anomaly.
- **Hong, H., Lim, T., and Stein, J.** (2000). "Bad News Travels Slowly:
  Size, Analyst Coverage, and the Profitability of Momentum Strategies."
  *Journal of Finance* 55(1). — 52-week-high anchoring.
- **Lehmann, B.** (1990). "Fads, Martingales, and Market Efficiency."
  *Quarterly Journal of Economics* 105(1). — 1-day reversal.
- **Ken French Data Library** — daily factor returns.
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Wikipedia: List of S&P 500 companies** — historical components and
  change log. https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

## License

MIT.
