"""Tests for the Wikipedia S&P 500 membership scaffolding.

Two layers:

1. Parser + builder tests on a synthetic Wikipedia-shaped HTML string.
   No network — the HTML is inline, the assertions verify the parser
   contract (BRK.B normalization, MultiIndex header handling, re-add
   cycles like GE 2018→re-added).

2. Public-API tests with monkeypatched `sp500_membership.fetch`, verifying
   `members_on_date`, `is_member`, and `filter_panel_to_pit` against the
   known-history scenarios:
   - TSLA NOT in the index on 2019-06-01 (joined 2020-12-21)
   - SIVB IN the index on 2023-03-08 (failed 2023-03-15)
   - SIVB NOT in the index on 2023-04-01 (post-failure)
   - META continuously in the index across the FB→META rename in 2022
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from price_model.data import membership
from price_model.data.sources import sp500_membership

# ---------------------------------------------------------------------------
# Synthetic Wikipedia HTML fixtures
# ---------------------------------------------------------------------------

# Minimal page with two tables matching Wikipedia's current shape.
# Table 0: current components. Table 1: changes log with MultiIndex header.
_SAMPLE_HTML = """
<html><body>

<table class="wikitable sortable">
<thead>
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
    <th>GICS Sub-Industry</th><th>Headquarters Location</th>
    <th>Date added</th><th>CIK</th><th>Founded</th></tr>
</thead>
<tbody>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td>
    <td>Technology Hardware</td><td>Cupertino, California</td>
    <td>1982-11-30</td><td>0000320193</td><td>1976</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td>
    <td>Multi-Sector Holdings</td><td>Omaha, Nebraska</td>
    <td>February 16, 2010</td><td>0001067983</td><td>1839</td></tr>
<tr><td>GE</td><td>General Electric</td><td>Industrials</td>
    <td>Industrial Conglomerates</td><td>Boston, Massachusetts</td>
    <td>1957-03-04</td><td>0000040545</td><td>1892</td></tr>
<tr><td>META</td><td>Meta Platforms</td><td>Communication Services</td>
    <td>Interactive Media</td><td>Menlo Park, California</td>
    <td>December 23, 2013</td><td>0001326801</td><td>2004</td></tr>
<tr><td>TSLA</td><td>Tesla, Inc.</td><td>Consumer Discretionary</td>
    <td>Automobile Manufacturers</td><td>Austin, Texas</td>
    <td>December 21, 2020</td><td>0001318605</td><td>2003</td></tr>
</tbody>
</table>

<table class="wikitable sortable">
<thead>
<tr><th rowspan="2">Date</th>
    <th colspan="2">Added</th>
    <th colspan="2">Removed</th>
    <th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th>
    <th>Ticker</th><th>Security</th></tr>
</thead>
<tbody>
<tr><td>December 21, 2020</td><td>TSLA</td><td>Tesla, Inc.</td>
    <td>AIV</td><td>Apartment Investment</td><td>Market cap promotion</td></tr>
<tr><td>June 26, 2018</td><td>WCG</td><td>WellCare Health Plans</td>
    <td>GE</td><td>General Electric</td><td>Reduced index relevance</td></tr>
<tr><td>June 7, 2024</td><td>GE</td><td>GE Aerospace</td>
    <td>WCG</td><td>WellCare</td><td>Acquired</td></tr>
<tr><td>March 15, 2023</td><td>BLDR</td><td>Builders FirstSource</td>
    <td>SIVB</td><td>SVB Financial</td><td>Receivership</td></tr>
</tbody>
</table>

</body></html>
"""


def _make_synthetic_table() -> pl.DataFrame:
    """Construct the expected (ticker, added, removed) frame directly, bypassing
    the HTML parser. Used for membership API tests so they don't depend on the
    parser's success.
    """
    return pl.DataFrame(
        {
            "ticker": ["AAPL", "BRK-B", "GE", "META", "TSLA", "SIVB", "WCG", "AIV"],
            "added": [
                date(1982, 11, 30),
                date(2010, 2, 16),
                date(2024, 6, 7),  # re-added after 2018 removal
                date(2013, 12, 23),
                date(2020, 12, 21),
                date(1970, 1, 1),  # pre-history sentinel: never current
                date(2018, 6, 26),  # never current — removed 2024
                date(1970, 1, 1),  # pre-history sentinel: removed 2020
            ],
            "removed": [
                None,
                None,
                None,
                None,
                None,
                date(2023, 3, 15),
                date(2024, 6, 7),
                date(2020, 12, 21),
            ],
        },
        schema={"ticker": pl.Utf8, "added": pl.Date, "removed": pl.Date},
    )


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_components_parser_normalizes_brk_b():
    """BRK.B in the Wikipedia source should become BRK-B for yfinance."""
    import io

    import pandas as pd

    tables = pd.read_html(io.StringIO(_SAMPLE_HTML))
    df = sp500_membership._parse_components_table(tables)
    assert "BRK-B" in df["ticker"].to_list()
    assert "BRK.B" not in df["ticker"].to_list()


def test_components_parser_extracts_all_current_members():
    import io

    import pandas as pd

    tables = pd.read_html(io.StringIO(_SAMPLE_HTML))
    df = sp500_membership._parse_components_table(tables)
    tickers = set(df["ticker"].to_list())
    assert {"AAPL", "BRK-B", "GE", "META", "TSLA"}.issubset(tickers)
    # All are current → removed should be null
    assert df.filter(pl.col("removed").is_not_null()).height == 0


def test_changes_parser_handles_multiindex_header():
    import io

    import pandas as pd

    tables = pd.read_html(io.StringIO(_SAMPLE_HTML))
    df = sp500_membership._parse_changes_table(tables)
    # 4 events in the synthetic table — all with both add and remove
    assert df.height == 4
    assert set(df.columns) == {"date", "added_ticker", "removed_ticker"}


def test_parse_date_string_handles_formats():
    parse = sp500_membership._parse_date_string
    assert parse("January 5, 2023") == date(2023, 1, 5)
    assert parse("2023-01-05") == date(2023, 1, 5)
    assert parse("Jan 5, 2023") == date(2023, 1, 5)
    assert parse("") is None
    assert parse("nan") is None
    # Year-only fallback
    assert parse("1957") == date(1957, 1, 1)


def test_build_membership_handles_re_add_cycle():
    """GE was removed in 2018, re-added in 2024. Final entry should reflect re-add."""
    import io

    import pandas as pd

    tables = pd.read_html(io.StringIO(_SAMPLE_HTML))
    components = sp500_membership._parse_components_table(tables)
    changes = sp500_membership._parse_changes_table(tables)
    table = sp500_membership._build_membership_table(components, changes)

    ge_row = table.filter(pl.col("ticker") == "GE").row(0, named=True)
    # After 2024-06-07 re-add, removed should be None and added = 2024-06-07
    assert ge_row["removed"] is None
    assert ge_row["added"] == date(2024, 6, 7)


def test_build_membership_records_sivb_removal():
    """SIVB doesn't appear in the components table (failed, no longer current).
    The builder should still register it from the changes log with the right removal date.
    """
    import io

    import pandas as pd

    tables = pd.read_html(io.StringIO(_SAMPLE_HTML))
    components = sp500_membership._parse_components_table(tables)
    changes = sp500_membership._parse_changes_table(tables)
    table = sp500_membership._build_membership_table(components, changes)

    sivb_row = table.filter(pl.col("ticker") == "SIVB").row(0, named=True)
    assert sivb_row["removed"] == date(2023, 3, 15)


# ---------------------------------------------------------------------------
# Public API tests with monkeypatched membership table
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_membership(monkeypatch):
    """Replace sp500_membership.fetch with our synthetic table and bust the lru_cache."""
    synthetic = _make_synthetic_table()
    monkeypatch.setattr(sp500_membership, "fetch", lambda *a, **kw: synthetic)
    membership._load_membership_table.cache_clear()
    yield synthetic
    membership._load_membership_table.cache_clear()


def test_tsla_not_in_index_pre_dec_2020(patched_membership):
    members = membership.members_on_date(date(2019, 6, 1))
    assert "TSLA" not in members
    assert "AAPL" in members  # sanity — Apple was definitely there


def test_tsla_in_index_post_dec_2020(patched_membership):
    members = membership.members_on_date(date(2021, 1, 1))
    assert "TSLA" in members


def test_sivb_in_index_on_failure_eve(patched_membership):
    """SIVB was in the S&P 500 until March 15, 2023. The day before, it's in."""
    assert membership.is_member("SIVB", date(2023, 3, 8))


def test_sivb_not_in_index_after_failure(patched_membership):
    """Post-removal, SIVB is out."""
    assert not membership.is_member("SIVB", date(2023, 4, 1))


def test_meta_continuous_through_rename(patched_membership):
    """META has the same canonical entry through the FB→META rename in Oct 2022.
    The membership table treats them as a single continuous record (the alias
    layer in tickers.py is what reconciles the symbol at the data layer).
    """
    # Pre-rename (when ticker was FB on the wire) and post-rename — same record
    assert membership.is_member("META", date(2014, 1, 1))
    assert membership.is_member("META", date(2024, 1, 1))


def test_filter_panel_to_pit_drops_tsla_pre_2020(patched_membership):
    """A panel with TSLA rows from 2019 should have those rows dropped."""
    panel = pl.DataFrame(
        {
            "date": [
                date(2019, 6, 1),
                date(2019, 6, 1),
                date(2021, 6, 1),
                date(2021, 6, 1),
            ],
            "ticker": ["AAPL", "TSLA", "AAPL", "TSLA"],
            "adj_close": [200.0, 250.0, 145.0, 670.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    filtered = membership.filter_panel_to_pit(panel)
    # AAPL kept twice, TSLA kept only post-2020-12-21
    by_ticker = filtered.group_by("ticker").len().sort("ticker")
    counts = {row["ticker"]: row["len"] for row in by_ticker.iter_rows(named=True)}
    assert counts["AAPL"] == 2
    assert counts["TSLA"] == 1  # only the 2021 row survived


def test_filter_panel_to_pit_preserves_extra_columns(patched_membership):
    """The filter must preserve all the panel's original columns (volume, sector, etc.)."""
    panel = pl.DataFrame(
        {
            "date": [date(2023, 1, 1), date(2023, 1, 1)],
            "ticker": ["AAPL", "META"],
            "adj_close": [150.0, 130.0],
            "volume": [10_000_000, 20_000_000],
            "sector": ["IT", "Comm"],
        }
    ).with_columns(pl.col("date").cast(pl.Date))
    filtered = membership.filter_panel_to_pit(panel)
    assert filtered.columns == panel.columns
    assert filtered.height == 2


def test_filter_panel_to_pit_requires_date_and_ticker(patched_membership):
    panel = pl.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match="requires columns"):
        membership.filter_panel_to_pit(panel)


def test_filter_panel_to_pit_handles_empty_panel(patched_membership):
    empty = pl.DataFrame(
        schema={"date": pl.Date, "ticker": pl.Utf8, "adj_close": pl.Float64}
    )
    out = membership.filter_panel_to_pit(empty)
    assert out.height == 0


# ---------------------------------------------------------------------------
# members_during_window — used by the universe builder
# ---------------------------------------------------------------------------


def test_members_during_window_includes_active_throughout(patched_membership):
    """AAPL was a member for the entire 2017-2026 range — should be in."""
    members = membership.members_during_window(date(2017, 1, 1), date(2026, 1, 1))
    assert "AAPL" in members


def test_members_during_window_includes_late_joiner(patched_membership):
    """TSLA joined 2020-12-21. A window starting in 2020 should include it."""
    members = membership.members_during_window(date(2017, 1, 1), date(2026, 1, 1))
    assert "TSLA" in members


def test_members_during_window_includes_dead_name_during_membership(patched_membership):
    """SIVB was removed 2023-03-15. Windows overlapping that period should include it."""
    members = membership.members_during_window(date(2017, 1, 1), date(2026, 1, 1))
    assert "SIVB" in members


def test_members_during_window_excludes_name_already_removed_before_window(patched_membership):
    """A window starting AFTER SIVB's removal should NOT include it."""
    members = membership.members_during_window(date(2024, 1, 1), date(2026, 1, 1))
    assert "SIVB" not in members


def test_members_during_window_excludes_name_added_after_window(patched_membership):
    """TSLA joined 2020-12-21. A window ending before then should NOT include it."""
    members = membership.members_during_window(date(2017, 1, 1), date(2020, 6, 1))
    assert "TSLA" not in members


def test_members_during_window_rejects_inverted_range(patched_membership):
    with pytest.raises(ValueError, match="start"):
        membership.members_during_window(date(2024, 1, 1), date(2023, 1, 1))
