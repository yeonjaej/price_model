"""GICS sector mapping for the universe.

Static map. GICS reclassifications happen but are infrequent. For honest backtesting
you'd want sector membership as of each date (e.g. via FactSet history); for v0
this is good enough — it makes sector-relative features possible without buying data.

Coverage: every ticker in src/price_model/data/universes/sp500.txt. If you add a ticker
to that file, add it here too or it defaults to "Unknown" (and gets its own sector
of one, which makes sector-relative features degenerate for it).
"""

from __future__ import annotations

import polars as pl

SECTOR_MAP: dict[str, str] = {
    # Communication Services
    "CHTR": "Communication Services", "CMCSA": "Communication Services",
    "DIS": "Communication Services", "GOOG": "Communication Services",
    "GOOGL": "Communication Services", "META": "Communication Services",
    "NFLX": "Communication Services", "T": "Communication Services",
    "TMUS": "Communication Services", "VZ": "Communication Services",

    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "AZO": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "F": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "MAR": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "ORLY": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",

    # Consumer Staples
    "CL": "Consumer Staples", "COST": "Consumer Staples",
    "EL": "Consumer Staples", "KHC": "Consumer Staples",
    "KMB": "Consumer Staples", "KO": "Consumer Staples",
    "MDLZ": "Consumer Staples", "MO": "Consumer Staples",
    "PEP": "Consumer Staples", "PG": "Consumer Staples",
    "PM": "Consumer Staples", "TGT": "Consumer Staples",
    "WMT": "Consumer Staples",

    # Energy
    "COP": "Energy", "CVX": "Energy", "EOG": "Energy",
    "OXY": "Energy", "PSX": "Energy", "SLB": "Energy", "XOM": "Energy",

    # Financials
    "AFL": "Financials", "AIG": "Financials", "ALL": "Financials",
    "AXP": "Financials", "BAC": "Financials", "BK": "Financials",
    "BLK": "Financials", "BRK-B": "Financials", "C": "Financials",
    "CB": "Financials", "CME": "Financials", "COF": "Financials",
    "GS": "Financials", "ICE": "Financials", "JPM": "Financials",
    "MA": "Financials", "MCO": "Financials", "MET": "Financials",
    "MMC": "Financials", "MS": "Financials", "PGR": "Financials",
    "PNC": "Financials", "SCHW": "Financials", "SPGI": "Financials",
    "TFC": "Financials", "TRV": "Financials", "USB": "Financials",
    "V": "Financials", "WFC": "Financials",

    # Health Care
    "ABBV": "Health Care", "ABT": "Health Care", "AMGN": "Health Care",
    "BIIB": "Health Care", "BMY": "Health Care", "BSX": "Health Care",
    "CVS": "Health Care", "DHR": "Health Care", "GILD": "Health Care",
    "HUM": "Health Care", "ISRG": "Health Care", "JNJ": "Health Care",
    "LLY": "Health Care", "MDT": "Health Care", "MRK": "Health Care",
    "PFE": "Health Care", "REGN": "Health Care", "SYK": "Health Care",
    "TMO": "Health Care", "UNH": "Health Care", "VRTX": "Health Care",
    "ZTS": "Health Care",

    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "CSX": "Industrials",
    "CTAS": "Industrials", "DE": "Industrials", "EMR": "Industrials",
    "ETN": "Industrials", "FDX": "Industrials", "GD": "Industrials",
    "GE": "Industrials", "HON": "Industrials", "ITW": "Industrials",
    "LMT": "Industrials", "MMM": "Industrials", "NOC": "Industrials",
    "NSC": "Industrials", "PH": "Industrials", "ROP": "Industrials",
    "UNP": "Industrials", "UPS": "Industrials", "WM": "Industrials",

    # Information Technology
    "AAPL": "Information Technology", "ACN": "Information Technology",
    "ADBE": "Information Technology", "ADI": "Information Technology",
    "ADP": "Information Technology", "ADSK": "Information Technology",
    "AMAT": "Information Technology", "AMD": "Information Technology",
    "ANET": "Information Technology", "APH": "Information Technology",
    "AVGO": "Information Technology", "CDNS": "Information Technology",
    "CRM": "Information Technology", "CSCO": "Information Technology",
    "FI": "Information Technology",  # Fiserv (renamed from FISV in 2024)
    "IBM": "Information Technology", "INTC": "Information Technology",
    "INTU": "Information Technology", "KLAC": "Information Technology",
    "LRCX": "Information Technology", "MSFT": "Information Technology",
    "MU": "Information Technology", "NOW": "Information Technology",
    "NVDA": "Information Technology", "ORCL": "Information Technology",
    "PANW": "Information Technology", "PYPL": "Information Technology",
    "QCOM": "Information Technology", "SNPS": "Information Technology",
    "TXN": "Information Technology",

    # Materials
    "APD": "Materials", "ECL": "Materials", "FCX": "Materials",
    "LIN": "Materials", "SHW": "Materials",

    # Real Estate
    "AMT": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "PLD": "Real Estate",

    # Utilities
    "AEP": "Utilities", "D": "Utilities", "DUK": "Utilities",
    "EXC": "Utilities", "NEE": "Utilities", "SO": "Utilities",
}


def get_sector(ticker: str) -> str:
    """Return the GICS sector for a ticker, or 'Unknown' if not in the map."""
    return SECTOR_MAP.get(ticker, "Unknown")


def attach_sector(panel: pl.DataFrame) -> pl.DataFrame:
    """Add a 'sector' column to a panel based on the ticker."""
    if "sector" in panel.columns:
        return panel
    return panel.with_columns(
        pl.col("ticker")
        .map_elements(get_sector, return_dtype=pl.Utf8)
        .alias("sector")
    )
