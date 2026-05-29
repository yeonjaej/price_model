# price-model

Point-in-time-corrected cross-sectional equity return predictor on the S&P 500.

## Headline result

Out-of-sample **Information Coefficient = +0.0183 (t = +4.3)** and long-short
**Sharpe = +0.86** over 905 trading days in the post-October-2022 high-dispersion
regime, on a Wikipedia-reconstructed point-in-time 617-name historical universe
using a 22-feature model that combines technical baselines, three documented
academic anomalies (Jegadeesh-Titman 12-1 momentum, Hong-Lim-Stein 52-week high,
Lehmann 1-day reversal), and six OHLCV/volume features (Parkinson high-low
volatility, Bali-Cakici-Whitelaw MAX effect, dollar volume, abnormal turnover,
intraday range, intraday body).

The result is statistically significant, regime-conditional, and **not deployable
for retail investors** after transaction costs, taxes, and breadth limits —
[see "Honest scope" below](#honest-scope-what-this-is-not).

### Metric definitions

- **Information Coefficient (IC)**: per-date Spearman rank correlation between
  predicted and realized 5-day forward excess returns, averaged across all
  dates in the evaluation window. Measures how well the model *ranks* names
  in the cross-section. IC ∈ [-1, +1]; +0.02 with t-stat > 2 is a credible
  edge on liquid US large-caps.
- **t-stat (of IC)**: `mean(daily IC) / (stdev(daily IC) / √n_dates)`. Tests
  whether the mean IC is distinguishable from zero. |t| > 1.96 corresponds
  to p < 0.05 (5% significance).
- **Long-short Sharpe**: annualized Sharpe ratio of a daily-rebalanced
  portfolio that goes long the top-quintile predicted names (top 20%) and
  short the bottom-quintile (bottom 20%), equal-weighted within each leg.
  The quintile cut is set in code by `_long_short_returns(top_frac=0.2)` in
  `src/price_model/eval/metrics.py`. Sharpe is computed per-horizon then
  annualized: `mean(per-horizon return) / stdev(per-horizon return) × √(252
  / horizon_days)`. Above +1.0 is the conventional bar for "would actually
  trade this institutionally"; reported here gross of all transaction costs.

## How the headline was earned — three orthogonal ablations

The headline number is **+0.0183 IC (t = +4.34) on the 617-name PIT
universe with 22 features, post-October-2022**. To establish that this isn't
an artifact of any single confounder, we vary three independent dimensions
— universe (with vs. without point-in-time correction), feature set, and
time window — and report all six cells of the resulting 2×3 matrix on the
strongest feature set we ship.

This presentation order is *not* the order the project was built in. The
chronological build went technical-baseline → PIT-corrected → anomalies →
OHLCV completeness, and that history is preserved in the git log. The
ablation table below is the pedagogically correct ordering: it isolates one
effect at a time on the headline 22-feature model rather than mixing
"changed the universe" with "changed the feature set" in the same step.

### The 2×3 ablation matrix (22-feature model)

| | Subset universe (160 modern survivors, PIT OFF) | Headline universe (617 historical, PIT ON) |
|---|---|---|
| **Full sample** (1758 dates) | IC = +0.0142, t = +4.05, Sharpe = +0.39 | IC = +0.0055, t = +1.51, Sharpe = +0.21 |
| **Pre-Oct-2022** (853 dates) | IC = +0.0034, t = +0.64, Sharpe = −0.07 | IC = −0.0082, t = −1.38, Sharpe = −0.16 |
| **Post-Oct-2022** (905 dates) | IC = +0.0244, t = +5.37, Sharpe = +0.94 | **IC = +0.0183, t = +4.34, Sharpe = +0.86** ← headline |

Three independent readings fall out of this:

**1. Survivorship-bias collapse varies dramatically by regime.** Reading
across rows: full-sample IC collapses from +0.0142 to +0.0055 under PIT
correction (a **61% drop**); pre-October-2022 IC actually **flips sign**
from a noisy +0.0034 to a clear −0.0082; post-October-2022 IC collapses
only 25% (from +0.0244 to +0.0183). The bias dominates when the
underlying signal is weak (pre-2022 noise) but matters much less when
the signal is strong (post-2022 regime). A naive practitioner running
the subset/full-sample backtest would see +0.0142 IC at t = 4.05 and
conclude they had a real, statistically significant edge — 61% of which
was selection bias.

**2. Regime effect is large and robust to universe.** Reading down
columns: both universes show roughly the same regime shift in Sharpe
(subset: −0.07 → +0.94; PIT: −0.16 → +0.86). The post-October-2022
regime contains real cross-sectional signal that is *not* primarily a
survivorship artifact. The shift in IC magnitude is also similar (subset:
+0.0034 → +0.0244, about 7×; PIT: −0.0082 → +0.0183, even more dramatic
because of the sign flip).

**3. Pre-October-2022 IC on the PIT universe is negative with t = −1.38.**
The 22-feature model would have actively *lost* money on the cross-section
before the regime break, not just been flat. This is a sharper finding than
"the edge is concentrated post-2022" — it's "the edge is post-2022 AND the
opposite disedge is pre-2022." The features are a regime-conditional
intensifier; they amplify whatever direction the cross-section happens to
be paying.

### Feature-set ablation (a separate, complementary story)

Holding the universe fixed at PIT-on and reading across feature additions
(chronologically how the project was built):

| Features | Full sample IC | Full sample t | Post-Oct-2022 Sharpe |
|---|---|---|---|
| 13 technical only | +0.0008 | +0.24 | −0.04 |
| 13 + 3 academic anomalies | +0.0033 | +0.96 | +0.083 |
| **22 (+ 6 OHLCV/volume)** | **+0.0055** | **+1.51** | **+0.86** in regime split |

The OHLCV/volume batch is the largest single-batch incremental contribution.
Each addition was tested on identical splits/embargo/walk-forward settings
in apples-to-apples runs; the YAML configs (`extended_kaggle_v2_pit.yaml`,
`extended_kaggle_v2_anomaly.yaml`, `extended_kaggle_v2_ohlcv.yaml`) make
this auditable.

## Reproduce the ablations

### Step 0 — install + build both universes (one-time, ~20-25 min cold)

The ablation matrix uses two universe files. Both need to be prepared
before any backtest runs.

```bash
pip install -e ".[dev,classical]"
```

**Universe A — `sp500` (the subset, used to measure survivorship bias).**
A checked-in static list of 160 modern-survivor tickers at
`src/price_model/data/universes/sp500.txt`. The name `sp500` is a
misnomer kept for backwards-compat with existing experiment configs: it
is *not* the full historical S&P 500, just a modern-survivor subset (no
delisted / acquired names, no PIT membership applied). No `build-universe`
step is needed; the file is checked in. You only need to fetch prices:

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

The 2×3 ablation matrix is filled by exactly two backtest runs (each
populates one column; rows come from time-splitting the predictions):

```bash
# Headline (PIT, 22 features): fills the right column of the matrix
python -m price_model.cli run -e extended_kaggle_v2_ohlcv
# Full sample → IC ≈ +0.0055, t ≈ +1.51, Sharpe ≈ +0.21

# PIT-OFF counterfactual (subset, 22 features): fills the left column
python -m price_model.cli run -e extended_kaggle_v2_ohlcv_subset
# Full sample → IC ≈ +0.0142, t ≈ +4.05, Sharpe ≈ +0.39
```

### Step 2 — regime split (analyze either prediction set)

```bash
jupyter notebook notebooks/03_robustness.ipynb
# PIT model post-October-2022:  IC = +0.0183, Sharpe = +0.86, t ≈ +4.34  ← headline
# PIT model pre-October-2022:   IC = -0.0082, Sharpe = -0.16
# Subset model post-October-2022:  IC = +0.0244, Sharpe = +0.94, t ≈ +5.37
# Subset model pre-October-2022:   IC = +0.0034, Sharpe = -0.07
```

### Step 3 (optional) — the feature-set ablation runs

To reproduce the feature-set table (13 technical → 16 anomaly → 22 OHLCV):

```bash
python -m price_model.cli run -e extended_kaggle_v2_pit       # 13 technical, PIT
python -m price_model.cli run -e extended_kaggle_v2_anomaly   # 16 + anomalies, PIT
# (the 22-feature run is already done in step 1)
```

Expected numbers above are deterministic given the same data snapshot. Small
drift (<5%) is normal as yfinance updates and KF refreshes monthly.

### What about the original chronological 5-stage build?

The earlier chronological narrative (Stage 1: 13 technical on the subset,
Stage 2: PIT correction, Stage 3: + 3 anomalies, Stage 4: + 6 OHLCV, Stage 5:
regime split) is preserved in git history and in the corresponding YAML
configs (`extended_kaggle_v2.yaml`, `extended_kaggle_v2_pit.yaml`,
`extended_kaggle_v2_anomaly.yaml`, `extended_kaggle_v2_ohlcv.yaml`). Each
config is runnable in isolation. The chronological version measured the
survivorship-bias collapse on 13 technical features (89%); the 22-feature
ablation above measures the same effect on the strongest model the project
ships and is the comparison a reviewer would actually want. The 89% number
is correct *for that feature set*, and is preserved here as the historical
record of "how naive does the bias look on a deliberately weak baseline."

## Data quality and methodological limitations

This section is the most important in the README. The headline result is honest
*about what it measures*, but what it measures is bounded by data we couldn't
get for free. Reading this section is how you know exactly what's behind the
numbers above.

### What yfinance can't (or won't) give us

yfinance is the only free source of daily-bar US equity data and is what makes
this project reproducible without a paid feed. But it has three documented
failure modes that materially affect the headline:

1. **Delisted-ticker history is permanently lost.** When a company is
   acquired, fails, or goes private, yfinance stops returning data for the
   old symbol. SIVB (Silicon Valley Bank, failed March 2023), FRC (First
   Republic, failed May 2023), SBNY (Signature Bank, failed March 2023),
   ATVI (Activision, acquired by Microsoft October 2023), AGN (Allergan →
   AbbVie 2020), CERN (Cerner → Oracle 2022) — none have usable pre-event
   history accessible from yfinance. **There are ~21 such tickers in our
   drop list**, documented inline in `src/price_model/data/tickers.py`.
2. **Symbol-parser fragility.** yfinance can't fetch some single- and
   short-letter tickers due to ambiguity with currency / commodity symbols
   in its routing. Examples: `K` (Kellanova), `FI` (Fiserv), `DAY`
   (Dayforce). All three are currently live, exchange-listed US large-caps.
   yfinance returns "no data" with retries.
3. **Foreign listings unreachable.** Acquired companies that consolidated
   under non-US ADRs are inaccessible. `SIE.DE` (Siemens Healthineers,
   acquired Varian) and `MC.PA` (LVMH, acquired Tiffany) cannot be fetched
   even though they trade on European exchanges.

**Bottom line:** of the 701 ticker-symbols that Wikipedia says were S&P 500
members at some point during 2017-2026, yfinance gives us usable data for
**617** (~88%). The other ~12% are silently absent from our panels.

### Where Wikipedia gives us partial PIT

The Wikipedia "List of S&P 500 companies" page and its "Selected changes"
table are the only free source of historical index membership. The scraper in
`src/price_model/data/sources/sp500_membership.py` reconstructs the
`(ticker, added_date, removed_date)` table from those two tables. Three
known incompleteness modes:

1. **The change log is only reliable back to ~2014.** Earlier add/remove
   events are missing from the table. This doesn't affect our 2017-start
   window directly, but it means any pre-2017 reach (e.g., trying to use
   the same machinery on a 1990s-2010s sample) would have spotty PIT
   membership.
2. **Renames vs. continuations are ambiguous.** When a company changes its
   ticker (FB → META, RTN → RTX, FISV → FI), Wikipedia logs it as a
   simultaneous remove + add on the same date. The scraper treats these
   as continuous membership at the *new* symbol; the alias is resolved in
   `tickers.py`. This is the right choice but it means a researcher
   reading the membership table directly might miscount index changes by
   ~30 over the full window.
3. **Wikipedia editors can be wrong, late, or partial.** We've found
   missing entries during the build (e.g., short-lived 2018 additions
   that were never logged as removals). We don't independently audit
   against a primary source.

### The rules we created to bridge the gap

Given the above, we maintain three small lookup tables in
`src/price_model/data/tickers.py`:

| Table | Purpose | Size |
|---|---|---|
| `TICKER_ALIASES` | Map renamed symbols to their successor (FB → META, RTN → RTX, ~70 entries documented with rename dates) | 73 entries |
| `TICKER_DROP_LIST` | Symbols with no usable yfinance data (failed banks, acquired with non-unified history, foreign listings, short-ticker parser failures) | ~25 entries |
| `SYMBOL_NORMALIZATION` | Punctuation convention (BRK.B → BRK-B) | 2 entries |

The single function `resolve_ticker(symbol) -> str | None` applies all three
in order. Every ticker in every universe file goes through it. Tests in
`tests/test_tickers.py` cover the precedence rules and known edge cases.

**These tables are derived heuristically** by running `cli refresh-data` on the
expanded universe and inspecting the resulting yfinance failure log. Each entry
has a one-line comment naming the corporate event and our reason. New entries
are added when new failures are observed.

### How these limitations bias the headline upward

This is the section I'd want any reviewer of this project to read. Be skeptical
of the headline IC of +0.0183 in proportion to the following:

1. **The "PIT correction" is partial.** Because yfinance can't give us SIVB,
   FRC, ATVI, AGN, etc., those tickers don't appear in our PIT panel even
   though Wikipedia says they were index members on the relevant dates. **Our
   PIT-corrected backtest still excludes the worst realized losers of the
   2022-2023 banking crisis.** A true PIT analysis with paid data (Norgate,
   Polygon, CRSP) would have to misrank or correctly short SIVB at -85% on
   2023-03-08. We don't.
2. **The 61% bias finding is itself a lower bound.** It's the share of the
   apparent edge we *could* attribute to selection given the data we have.
   The actual selection bias on a fully-PIT-correct dataset (one with
   delisted history and complete pre-2014 membership) would be larger.
   Our IC drop from +0.0142 to +0.0055 on the 22-feature model is the
   *floor*, not the ceiling, of how much the naive backtest was inflated.
   For reference, the same effect measured on the original 13-feature
   technical-only baseline was an 89% collapse (+0.0075 → +0.0008) — the
   weaker the feature set, the worse the survivorship-bias inflation.
3. **The post-2022 regime contains the bank-failure period (Mar-May 2023).**
   Our headline +0.0183 IC over 905 days post-October-2022 is computed on a
   cross-section that excludes the names that catastrophically failed in
   that window. A real-world model would have to predict (or fail to
   predict) those failures; our model gets a free pass on them. **The
   honest read of the headline IC is "+0.0183 on the survivors of the
   regime, given our data."**
4. **Transaction costs, taxes, slippage are zero in the backtest.** All
   ICs and Sharpes assume costless rebalancing. A retail investor faces
   bid-ask spreads (~5-10bp), commissions, capital-gains tax, and
   slippage. The reported Sharpe of +0.86 is *gross*; net of realistic
   retail costs, the after-cost Sharpe on a 10-30 name portfolio is
   approximately zero. An institutional desk paying ~1-3bp all-in could
   plausibly net a Sharpe in the 0.4-0.6 range from this signal — still
   not a standalone strategy.
5. **Single random seed for the universe expansion.** We chose the
   2017-2026 evaluation window because that's what yfinance comfortably
   covers. Different windows (2010-2020 vs. 2017-2026 vs. 2020-2026) would
   produce different deltas at each stage. We didn't run the multi-window
   robustness check.

### What would need to change to deliver a *true* PIT-correct number

In rough order of effort:

- **Replace yfinance with Norgate Premium Data ($60/mo)** for survivorship-bias-free
  prices including delisted history. Would change all five IC numbers; the
  headline +0.0183 would likely fall by 0.002-0.005, but the regime-
  conditional shape (post-2022 strong, pre-2022 negative) would persist.
- **Replace the Wikipedia membership source with CRSP** (paid, academic access
  free for most affiliated researchers). Cleaner pre-2014 history, no scrape
  fragility.
- **Run multi-window robustness tests** at three different start/end pairs.
- **Add transaction-cost modeling** at the prediction-store layer, so reported
  metrics are net of realistic costs.

## Honest scope — what this is NOT

- **Not deployable for retail trading.** After bid-ask spreads, commissions,
  and capital-gains taxes, the +0.0183 IC and gross Sharpe of +0.86 produce
  approximately zero after-cost edge for an individual investor's 10-30
  name portfolio (breadth too small) rebalanced quarterly (turnover too
  low). The number is institutional-grade *gross*, not retail-grade *net*.
- **Not a guarantee future regimes will look like the post-2022 one.** The
  edge is concentrated in a single regime (Oct 2022 → May 2026) characterized
  by Mag-7 dominance, rate-cycle dispersion, and AI-driven sector divergence.
  The same 22-feature model on pre-2022 data produces IC = **−0.0082** —
  the OHLCV/anomaly features are *regime-conditional intensifiers*, not
  universally beneficial. Adding them post-2022 lifts IC by ~58%; adding
  them pre-2022 makes IC ~50% more negative.
- **Not a substitute for index funds.** For individual investors, decades of
  research show low-cost diversified index funds beat almost all active
  strategies after fees and taxes.
- **Not a complete PIT correction.** See the data limitations above.

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

The same data infrastructure also supports `notebooks/04_portfolio_attribution.ipynb`,
which uses the Ken French adapter directly (not the model layer) to decompose
a 10-stock equal-weight portfolio's exposures and attribute realized returns
to factors. **That notebook is an independent application of the same data
layer; it does not depend on the predictive model.**

### On the Streamlit dashboard

`src/price_model/dashboard/` is a thin Streamlit reader over the DuckDB
prediction store. It exists so the model's daily output can be inspected
visually (per-date top/bottom quintile, rolling IC, prediction vs. realized
scatter) without writing a notebook for each look. It is **a debugging /
monitoring surface, not part of the reproduction journey**. None of the
headline numbers in this README come from the dashboard, and skipping it
does not affect reproducing any of the four stages. It is most useful if you
want to extend the model and need a quick sanity-check view of new runs.

## Development notes

The substantive research decisions in this project — what to measure, what
constitutes a fair test, how much survivorship bias the headline contains,
which limitations are honest to ship with — are the author's. The
*engineering* side of the project (test scaffolding, CI configuration,
package layout, refactors, the bridge from "this is the idea I want to test"
to "here is a clean, reproducible implementation") was developed in heavy
collaboration with Claude Code. Specifically:

- The walk-forward harness, prediction store, and dashboard scaffold were
  designed and iterated in conversation with Claude Code.
- Test coverage (leakage tests, PIT-membership tests, contract tests on the
  loaders, ticker-resolver tests) was co-designed; Claude Code drafted many
  of the test cases that then surfaced bugs in the implementation.
- The GitHub Actions CI pipeline (ruff + pyright + pytest on Python 3.11)
  and pre-commit hooks were set up with Claude Code.
- Code review, refactors, and a final pre-publish audit pass were performed
  in collaboration with Claude Code.

Treat this README, the experiment configs, and the four-stage narrative as
the author's claims; treat the engineering quality of the surrounding
codebase as a product of that collaboration.

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
