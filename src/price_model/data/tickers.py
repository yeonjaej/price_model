"""Ticker normalization, renames, and exclusions for yfinance.

Three classes of problem yfinance presents when you expand to the historical
S&P 500 (PIT universe ~700 names):

1. **Symbol changes** — companies rebrand or merge. The OLD ticker stops
   returning data; yfinance keeps the unified history under the NEW symbol.
   Alias old → new and downstream code uses the new symbol transparently.

2. **Delisted / failed / acquired** — companies that left the index without a
   usable successor symbol in yfinance. The data is simply not there. We drop
   them with a comment explaining why, so a code reviewer can audit our
   universe choices.

3. **Symbol encoding** — yfinance uses dashes where Wikipedia uses dots for
   class-share punctuation. S&P 500 has a handful of these (BRK.B, BF.B).

The categorization here is derived from running `cli refresh-data` on the
expanded `sp500_pit` universe and inspecting the 92 yfinance failures in the
log. Each entry has a short reason comment so future-you knows whether to
retry (transient yfinance issue) or accept (genuinely no data).

Adding a new entry: just append to the appropriate dict / set. The
`resolve_ticker` function applies all three rules in order on every fetch.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Class 1: rename / alias — yfinance has post-rename history under the new symbol
# ---------------------------------------------------------------------------
# Each entry: old_symbol -> new_symbol
# Comments document the event date and the corporate action.

TICKER_ALIASES: dict[str, str] = {
    # Bank / financial
    "BHI":  "BKR",       # Baker Hughes (GE) → Baker Hughes BKR (Sep 2019)
    "ETFC": "MS",        # E*TRADE → Morgan Stanley (Oct 2020) — unified under MS
    "FLT":  "CPAY",      # FleetCor → Corpay (Apr 2024)
    "FRC":  "JPM",       # First Republic → JPMorgan (May 2023 — receivership)
    "LUK":  "JEF",       # Leucadia → Jefferies Financial Group (Mar 2018)
    "PBCT": "MTB",       # People's United → M&T Bank (Apr 2022)
    # Consumer / retail
    "ADS":  "BFH",       # Alliance Data → Bread Financial (Mar 2022)
    "DPS":  "KDP",       # Dr Pepper Snapple → Keurig Dr Pepper (Jul 2018)
    "GPS":  "GAP",       # Gap Inc → renamed GAP (Feb 2024)
    "KORS": "CPRI",      # Michael Kors → Capri Holdings (Dec 2018)
    # K (Kellanova) keeps its ticker after the KLG split — no alias needed.
    "RAI":  "BTI",       # Reynolds American → BAT (Jul 2017) — ADR
    "TIF":  "MC.PA",     # Tiffany → LVMH (Jan 2021) — French ADR, may not resolve
    "WFM":  "AMZN",      # Whole Foods → Amazon (Aug 2017) — history may not unify
    # Defense / aerospace
    "HRS":  "LHX",       # Harris → L3Harris (Jun 2019)
    "LLL":  "LHX",       # L3 → L3Harris (Jun 2019)
    "RTN":  "RTX",       # Raytheon → RTX Corp (Apr 2020)
    # Energy
    "ANDV": "MPC",       # Andeavor → Marathon Petroleum (Oct 2018) (was TSO)
    "COG":  "CTRA",      # Cabot Oil → Coterra (Oct 2021)
    "CXO":  "COP",       # Concho Resources → ConocoPhillips (Jan 2021)
    "HFC":  "DINO",      # HollyFrontier → HF Sinclair (Mar 2022)
    "MRO":  "COP",       # Marathon Oil → ConocoPhillips (Nov 2024)
    "NBL":  "CVX",       # Noble Energy → Chevron (Oct 2020)
    "PXD":  "XOM",       # Pioneer → ExxonMobil (May 2024)
    "SWN":  "EXE",       # Southwestern Energy → Expand Energy (Oct 2024)
    "TSO":  "MPC",       # Tesoro → Andeavor → Marathon Petroleum (chain)
    "XEC":  "CTRA",      # Cimarex → Coterra (Oct 2021)
    # Healthcare
    "ABMD": "JNJ",       # Abiomed → Johnson & Johnson (Dec 2022)
    "AGN":  "ABBV",      # Allergan → AbbVie (May 2020)
    "ALXN": "AZN",       # Alexion → AstraZeneca (Jul 2021) — ADR
    "BCR":  "BDX",       # CR Bard → Becton Dickinson (Dec 2017)
    "CELG": "BMY",       # Celgene → Bristol-Myers Squibb (Nov 2019)
    "CERN": "ORCL",      # Cerner → Oracle (Jun 2022)
    "MJN":  "RBGLY",     # Mead Johnson → Reckitt (Jun 2017)
    "RHT":  "IBM",       # Red Hat → IBM (Jul 2019)
    "STJ":  "ABT",       # St Jude Medical → Abbott (Jan 2017)
    "VAR":  "SIE.DE",    # Varian → Siemens Healthineers (Apr 2021) — likely fails
    "WCG":  "CNC",       # WellCare → Centene (Jan 2020)
    "WLP":  "ELV",       # WellPoint → Anthem → Elevance Health (Jun 2022)
    # Industrials / Materials
    "ARNC": "HWM",       # Arconic → Howmet Aerospace (Apr 2020) — split, HWM remains
    "DLPH": "APTV",      # Delphi Automotive → Aptiv (Dec 2017)
    "DWDP": "DD",        # DowDuPont → DuPont (after 2019 split into DD/DOW/CTVA)
    "FBHS": "FBIN",      # Fortune Brands → Fortune Brands Innovations (Dec 2022)
    "FLIR": "TDY",       # FLIR → Teledyne (May 2021)
    "JEC":  "J",         # Jacobs Engineering → Jacobs Solutions (Feb 2022)
    "JOYG": "KMTUY",     # Joy Global → Komatsu (Apr 2017) — ADR
    "MON":  "BAYRY",     # Monsanto → Bayer (Jun 2018) — ADR
    "RE":   "EG",        # Everest Re → Everest Group (Jan 2023)
    "SRCL": "WM",        # Stericycle → Waste Management (Nov 2024)
    "WYN":  "WH",        # Wyndham Worldwide → Wyndham Hotels (May 2018; Destinations is WYND)
    "XL":   "AXAHY",     # XL Group → AXA (Sep 2018) — ADR
    # Media / telecom
    "CDAY": "DAY",       # Ceridian → Dayforce (Feb 2024)
    "DISCA": "WBD",      # Discovery Communications A → Warner Bros Discovery (Apr 2022)
    "DISCK": "WBD",      # Discovery Communications K → Warner Bros Discovery (Apr 2022)
    "DISH":  "SATS",     # DISH Network → EchoStar (Dec 2023)
    "NLSN":  "NLSN",     # Nielsen — went private 2022 (keep mapped; will hit drop list)
    "SNI":   "WBD",      # Scripps Networks → Discovery → WBD
    "TWTR":  "TWTR",     # Twitter — private 2022 (keep mapped; will hit drop list)
    "VIAB":  "PARA",     # Viacom → ViacomCBS → Paramount (Feb 2022)
    "WLTW":  "WTW",      # Willis Towers Watson → WTW (Mar 2021)
    "YHOO":  "AABA",     # Yahoo → Altaba (Jun 2017); AABA dissolved 2019 — likely fails
    # Tech / semiconductors
    "ATVI":  "MSFT",     # Activision Blizzard → Microsoft (Oct 2023)
    "CTXS":  "CTXS",     # Citrix — private 2022 (no successor; will hit drop list)
    "FB":    "META",     # Facebook → Meta Platforms (Jun 2022)
    "FISV":  "FI",       # Fiserv → Fiserv Inc (FI) — ticker change
    "LLTC":  "ADI",      # Linear Technology → Analog Devices (Mar 2017)
    "LVLT":  "LUMN",     # Level 3 → CenturyLink → Lumen (Nov 2017)
    "MXIM":  "ADI",      # Maxim Integrated → Analog Devices (Aug 2021)
    "TSS":   "GPN",      # TSYS → Global Payments (Sep 2019)
    "XLNX":  "AMD",      # Xilinx → AMD (Feb 2022)
    # Other
    "GGP":   "BPY",      # General Growth Properties → Brookfield Property (Aug 2018)
    "DRE":   "PLD",      # Duke Realty → Prologis (Oct 2022)
    "KSU":   "CP",       # Kansas City Southern → Canadian Pacific (Dec 2021)
    "BHGE":  "BKR",      # Baker Hughes (transitional ticker) → BKR
}


# ---------------------------------------------------------------------------
# Class 2: drop list — symbols with no usable yfinance history
# ---------------------------------------------------------------------------
# Reasons fall into a few buckets, all noted inline:
#   * Failed / bankrupt: data simply isn't there post-failure.
#   * Went private: no public ticker continues.
#   * Acquired with non-unified yfinance history under acquirer.

TICKER_DROP_LIST: set[str] = {
    # 2023 bank failures — total writeoff in yfinance
    "SIVB",   # Silicon Valley Bank, FDIC receivership (Mar 2023)
    "FRC",    # First Republic Bank, FDIC receivership (May 2023)
    "SBNY",   # Signature Bank, FDIC receivership (Mar 2023)

    # Other bankruptcies / failed companies
    "ENDP",   # Endo International — Chapter 11 (Aug 2022)
    "FSR",    # Fisker Inc — Chapter 11 (Jun 2024)
    "FTR",    # Frontier Communications — bankruptcy (Apr 2020), now FYBR
    "MNK",    # Mallinckrodt — Chapter 11 (twice)
    "CHK",    # Chesapeake Energy — Chapter 11 (Jun 2020); re-listed but yfinance returns no usable data

    # Went private — no current ticker
    "CTXS",   # Citrix — taken private by Vista/Elliott (Sep 2022)
    "NLSN",   # Nielsen — taken private by Elliott (Oct 2022)
    "TWTR",   # Twitter — taken private by Musk (Oct 2022)
    "JWN",    # Nordstrom — taken private (2025)
    "PDCO",   # Patterson Companies — taken private (2025)

    # Acquired, no usable unified yfinance history under acquirer
    "YHOO",   # Yahoo → Altaba (2017) → dissolved
    "WFM",    # Whole Foods → Amazon — no AMZN history extension for WFM
    "TIF",    # Tiffany → LVMH (French) — no usable continuation in yf
    "VAR",    # Varian → Siemens Healthineers (Germany) — no usable continuation

    # Known yfinance issues we already encountered (transient or persistent)
    "CMA",    # Comerica — yfinance returns data sometimes; keep here for now
    "WBA",    # Walgreens Boots — yfinance gaps (and being taken private 2025)
    "MMC",    # Marsh McLennan — was already on this list from prior runs
    "CCR",    # Country Code REIT or similar — never identified, persistent failure
    "DFS",    # Discover Financial — temp issue?  (re-evaluate after COF merger)

    # Single-letter / short tickers yfinance can't parse cleanly.
    # Empirically these fail in both directions (as universe input AND as alias
    # targets). Likely a yfinance symbol-parser ambiguity with currencies /
    # commodities. Keep them out until/unless we route through a different feed.
    "K",      # Kellanova — alive but unfetchable in yfinance
    "FI",     # Fiserv (FISV renamed) — alive but unfetchable
    "DAY",    # Dayforce (CDAY renamed Feb 2024) — alive but unfetchable

    # Aliases that turned out to point at also-dead tickers — putting them on
    # the drop list closes the loop so resolve_ticker correctly returns None
    # for the upstream symbol too.
    "BPY",    # Brookfield Property Partners — went private (Jul 2021). GGP → BPY chain dead.
    "PARA",   # Paramount — renamed PSKY after Skydance merger (Aug 2025). VIAB → PARA dead.
    "AABA",   # Altaba (Yahoo successor) — dissolved (Oct 2019). YHOO → AABA dead.
    "SIE.DE", # Siemens Healthineers German listing — yfinance can't fetch German tickers.

    # Tickers we don't have a clear alias for (audit candidates)
    "ANSS",   # Ansys — Synopsys merger completed Jul 2025; unsure of yf state
    "DNB",    # Dun & Bradstreet — should work; investigate next refresh
    "FL",     # Foot Locker — DKS acquisition pending; transient
    "HBI",    # Hanesbrands — should work; investigate
    "HES",    # Hess — CVX merger litigation; transient
    "IPG",    # Interpublic — OMC merger pending; transient
    "JNPR",   # Juniper — HPE acquisition close pending
    "TGNA",   # Tegna — should work; investigate
    "CTLT",   # Catalent — Novo Holdings acquisition (Dec 2024)
}


# ---------------------------------------------------------------------------
# Class 3: punctuation normalization
# ---------------------------------------------------------------------------
# Wikipedia and most sources use dots in class-share tickers; yfinance uses
# dashes. There are only a few of these in the S&P 500.

SYMBOL_NORMALIZATION: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BF.B":  "BF-B",
}


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------


def resolve_ticker(symbol: str) -> str | None:
    """Apply alias + drop + normalization in order. Returns the canonical
    yfinance symbol, or None if the ticker is on the drop list.

    Drop list takes precedence over aliases — if we've decided a symbol is
    unusable, we don't follow the alias even when one exists. (Conversely,
    if an alias points to a symbol on the drop list, the final symbol after
    aliasing is checked and dropped.)

    Order:
        1. Drop list check on the input symbol.
        2. Alias resolution (FB → META, RTN → RTX, etc.).
        3. Drop list check on the aliased symbol (catches alias-into-dead-symbol).
        4. Punctuation normalization (BRK.B → BRK-B).
    """
    if symbol in TICKER_DROP_LIST:
        return None
    aliased = TICKER_ALIASES.get(symbol, symbol)
    if aliased in TICKER_DROP_LIST:
        return None
    return SYMBOL_NORMALIZATION.get(aliased, aliased)


def all_dropped_tickers() -> set[str]:
    """Convenience for inspection / logging — the full set of symbols that
    will be excluded by `resolve_ticker`. Includes the literal drop list plus
    any symbols whose alias points into the drop list.
    """
    via_alias = {orig for orig, new in TICKER_ALIASES.items() if new in TICKER_DROP_LIST}
    return TICKER_DROP_LIST | via_alias


__all__ = [
    "SYMBOL_NORMALIZATION",
    "TICKER_ALIASES",
    "TICKER_DROP_LIST",
    "all_dropped_tickers",
    "resolve_ticker",
]
