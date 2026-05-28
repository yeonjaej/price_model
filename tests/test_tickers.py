"""Tests for ticker normalization, aliases, and exclusions."""

from __future__ import annotations

from price_model.data.tickers import (
    TICKER_ALIASES,
    TICKER_DROP_LIST,
    all_dropped_tickers,
    resolve_ticker,
)


def test_alias_resolves_renamed_symbol():
    """Common renames should resolve to their new symbol when that symbol is live."""
    assert resolve_ticker("FB") == "META"
    assert resolve_ticker("RTN") == "RTX"
    # FISV → FI: FI is itself on the drop list (yfinance short-ticker issue),
    # so the chain correctly returns None rather than a broken target.
    assert resolve_ticker("FISV") is None
    # A clean rename where the target is healthy
    assert resolve_ticker("KORS") == "CPRI"


def test_drop_list_returns_none():
    """Symbols on the drop list should resolve to None — no fetch attempted."""
    assert resolve_ticker("SIVB") is None
    assert resolve_ticker("FRC") is None
    assert resolve_ticker("TWTR") is None


def test_drop_list_precedence_over_alias():
    """If a symbol has both an alias and is on the drop list, drop wins.

    TWTR is in TICKER_ALIASES (mapped to itself for documentation) AND in
    TICKER_DROP_LIST. resolve_ticker should return None — we don't want to
    silently try to fetch a dead symbol.
    """
    # Sanity check the setup: TWTR is in both
    assert "TWTR" in TICKER_ALIASES
    assert "TWTR" in TICKER_DROP_LIST
    assert resolve_ticker("TWTR") is None


def test_alias_into_drop_list_is_dropped():
    """If an alias points to a symbol that's on the drop list, the final
    answer is None. (Currently no entries trigger this path, but the resolver
    handles it defensively.)

    Synthesize one for the test: pretend FB aliases to TWTR (which is dropped).
    """
    # Inline patch — does not modify module-level state
    original = TICKER_ALIASES.get("FB")
    try:
        TICKER_ALIASES["FB"] = "TWTR"  # TWTR is on drop list
        assert resolve_ticker("FB") is None
    finally:
        if original is None:
            TICKER_ALIASES.pop("FB", None)
        else:
            TICKER_ALIASES["FB"] = original


def test_normalization_converts_dot_to_dash():
    """BRK.B in Wikipedia/CRSP conventions should become BRK-B for yfinance."""
    assert resolve_ticker("BRK.B") == "BRK-B"
    assert resolve_ticker("BF.B") == "BF-B"


def test_normalization_applies_after_alias():
    """Aliases run before normalization. So if an aliased symbol has a dot,
    the dot becomes a dash in the final output."""
    # Synthesize: pretend XYZ aliases to BRK.B → should resolve to BRK-B
    TICKER_ALIASES["XYZ_TEST"] = "BRK.B"
    try:
        assert resolve_ticker("XYZ_TEST") == "BRK-B"
    finally:
        TICKER_ALIASES.pop("XYZ_TEST", None)


def test_unknown_symbol_passes_through():
    """A symbol not in any of the three tables should be returned unchanged."""
    assert resolve_ticker("AAPL") == "AAPL"
    assert resolve_ticker("MSFT") == "MSFT"
    assert resolve_ticker("NVDA") == "NVDA"


def test_all_dropped_tickers_includes_direct_and_aliased():
    """all_dropped_tickers should return symbols on the drop list AND symbols
    whose alias chain ends in the drop list."""
    dropped = all_dropped_tickers()
    # Direct drops
    assert "SIVB" in dropped
    assert "FRC" in dropped
    # An alias-into-drop case: TWTR is in both, so it should appear
    assert "TWTR" in dropped


def test_drop_list_no_overlap_with_alias_targets():
    """Sanity check: no alias should TARGET a drop-list symbol unless that's
    a deliberate "chain ends in a dead end" case.

    Several upstream renames lead to targets that are themselves dead
    (e.g. GGP → BPY went private, VIAB → PARA renamed to PSKY post-Skydance,
    YHOO → AABA dissolved). Each of these is intentional — putting the dead
    target on the drop list closes the loop so resolve_ticker correctly
    returns None for the upstream symbol. We whitelist the known targets
    here so accidental typos (alias to a misspelled live symbol that
    coincidentally appears in drop) still surface.
    """
    intentional_targets = {
        # Self-loop documentation markers
        "TWTR", "NLSN", "CTXS",
        # Chains that end in dead targets
        "BPY",     # GGP → BPY (went private)
        "PARA",    # VIAB → PARA (renamed to PSKY)
        "AABA",    # YHOO → AABA (dissolved)
        "SIE.DE",  # VAR → SIE.DE (German ticker, yf can't fetch)
        "FI",      # FISV → FI (yfinance short-ticker issue)
        "DAY",     # CDAY → DAY (yfinance short-ticker issue)
    }
    actual = {orig: tgt for orig, tgt in TICKER_ALIASES.items() if tgt in TICKER_DROP_LIST}
    unexpected = {orig for orig, tgt in actual.items() if tgt not in intentional_targets}
    assert not unexpected, f"Aliases target drop-list symbols unexpectedly: {unexpected}"


def test_load_universe_filters_drop_list_and_resolves_aliases(tmp_path, monkeypatch):
    """End-to-end: load_universe should never return a drop-list symbol, and
    should rewrite aliased symbols to their canonical form."""
    from price_model.data import universe

    # Write a test universe with a mix of valid, aliased, and dropped tickers
    test_file = tmp_path / "test_universe.txt"
    test_file.write_text("AAPL\nFB\nSIVB\nRTN\nMSFT\nTWTR\nBRK.B\n")

    # Patch the UNIVERSE_DIR so the loader looks in tmp_path
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    universe.load_universe.cache_clear()

    result = universe.load_universe("test_universe")
    # Expected: AAPL (passthrough), META (FB alias), RTX (RTN alias),
    # MSFT (passthrough), BRK-B (normalized). SIVB and TWTR are dropped.
    assert set(result) == {"AAPL", "META", "RTX", "MSFT", "BRK-B"}
    assert "SIVB" not in result
    assert "TWTR" not in result
    assert "FB" not in result  # aliased away
    assert "RTN" not in result  # aliased away
    universe.load_universe.cache_clear()


def test_no_circular_aliases():
    """Sanity check: no alias should map a symbol to itself (except as
    documentation marker that's also on the drop list). A true self-loop
    is fine when documented; check the unintentional case is empty.
    """
    self_loops = {orig for orig, tgt in TICKER_ALIASES.items() if orig == tgt}
    # Allowed self-loops are exactly the documentation markers also in drop list
    unexpected = self_loops - TICKER_DROP_LIST
    assert not unexpected, f"Self-aliases not on drop list: {unexpected}"
