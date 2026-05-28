"""Ken French factor-returns adapter.

Downloads the Fama-French 5-factor daily series from Dartmouth and parses it into
a polars DataFrame. This is the *only* module that touches the Dartmouth ZIP layout;
factor-loading features and the FamaFrenchFactorModel both read through this adapter.

Why direct download instead of pandas-datareader:
- Avoids the dependency entirely (pandas-datareader is heavy and routinely breaks).
- Lets us cache the raw CSV under data/raw/ alongside the yfinance parquets.
- The KF ZIP layout is stable enough that a focused parser is fine.

Source URL:
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip

Output schema:
    date     Date
    MKT_RF   Float64  market excess return (Rm - Rf)
    SMB      Float64  small minus big (size factor)
    HML      Float64  high minus low (value factor)
    RMW      Float64  robust minus weak (profitability)
    CMA      Float64  conservative minus aggressive (investment)
    RF       Float64  risk-free rate
All values are in DECIMAL units (KF reports percent; we divide by 100).
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, datetime
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
FF5_CACHE_FILENAME = "F-F_Research_Data_5_Factors_2x3_daily.csv"

# KF CSV header columns in the order they appear in the daily 5-factor file.
# We normalize "Mkt-RF" -> "MKT_RF" so it's a valid Python identifier downstream.
_KF_COL_MAP = {
    "Mkt-RF": "MKT_RF",
    "SMB": "SMB",
    "HML": "HML",
    "RMW": "RMW",
    "CMA": "CMA",
    "RF": "RF",
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_kf_csv(text: str) -> pl.DataFrame:
    """Parse a Ken French daily factor CSV (already-decoded text).

    KF CSV files start with a multi-line text block describing the data, followed
    by the header row "      ,Mkt-RF,SMB,HML,RMW,CMA,RF" and YYYYMMDD-keyed rows.
    The daily file contains a single table (no annual block at the bottom), but we
    still defensively stop at the first blank line.
    """
    lines = text.splitlines()

    # Find the header line. KF uses leading spaces before the comma, so the
    # robust test is "first line whose comma-split contains Mkt-RF".
    header_idx = -1
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split(",")]
        if "Mkt-RF" in cells:
            header_idx = i
            break
    if header_idx < 0:
        raise ValueError("Could not locate Ken French header row (expected 'Mkt-RF' column)")

    header_cells = [c.strip() for c in lines[header_idx].split(",")]
    # The first column is unnamed in KF files — it carries the YYYYMMDD date key.
    header_cells[0] = "date"

    # Collect data rows up to the first blank or non-numeric leading cell. The
    # daily file's only table is the daily factors; a blank line marks EOF or
    # an appended monthly/annual table (which the daily file doesn't have, but
    # we guard anyway).
    data_rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        cells = [c.strip() for c in line.split(",")]
        if not cells[0].isdigit():
            break
        data_rows.append(cells)

    if not data_rows:
        raise ValueError("Ken French CSV had a header but no daily rows")

    # Build the polars frame column by column to keep dtypes explicit.
    cols: dict[str, list] = {h: [] for h in header_cells}
    for row in data_rows:
        for h, v in zip(header_cells, row, strict=True):
            cols[h].append(v)

    df = pl.DataFrame(cols)

    # Parse YYYYMMDD date column
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y%m%d", strict=True),
    )
    # Cast factor columns to Float64 and convert from percent -> decimal
    factor_renames = {kf: ours for kf, ours in _KF_COL_MAP.items() if kf in df.columns}
    df = df.rename(factor_renames)
    factor_cols = list(factor_renames.values())
    df = df.with_columns([(pl.col(c).cast(pl.Float64) / 100.0) for c in factor_cols])

    return df.select(["date", *factor_cols]).sort("date")


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------


def _cache_path(raw_dir: Path) -> Path:
    return raw_dir / FF5_CACHE_FILENAME


def _download_csv() -> str:
    """Fetch the KF ZIP, unzip in-memory, and return the decoded CSV text.

    KF files use ISO-8859-1 encoding (legacy Windows / academic convention).
    """
    import urllib.request  # local import: stdlib but keep top clean

    log.info("Downloading Ken French 5-factor daily ZIP from %s", FF5_URL)
    with urllib.request.urlopen(FF5_URL, timeout=30) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # The ZIP usually contains exactly one CSV; pick it defensively.
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV inside KF zip; got: {zf.namelist()}")
        with zf.open(csv_names[0]) as f:
            return f.read().decode("iso-8859-1")


def fetch(
    start: str | date | None = None,
    end: str | date | None = None,
    raw_dir: Path | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Return the Fama-French 5-factor daily series as a polars frame.

    Cached under data/raw/F-F_Research_Data_5_Factors_2x3_daily.csv. Pass
    `force_refresh=True` to bypass the cache and re-download.

    `start` / `end` are optional inclusive bounds (YYYY-MM-DD or date).
    """
    raw_dir = raw_dir or Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(raw_dir)

    text: str
    if use_cache and cache.exists() and not force_refresh:
        text = cache.read_text(encoding="iso-8859-1")
    else:
        text = _download_csv()
        cache.write_text(text, encoding="iso-8859-1")

    df = _parse_kf_csv(text)

    if start is not None:
        start_d = start if isinstance(start, date) else datetime.fromisoformat(str(start)).date()
        df = df.filter(pl.col("date") >= pl.lit(start_d).cast(pl.Date))
    if end is not None:
        end_d = end if isinstance(end, date) else datetime.fromisoformat(str(end)).date()
        df = df.filter(pl.col("date") <= pl.lit(end_d).cast(pl.Date))

    log.info(
        "Loaded KF 5-factor daily: %d rows, %s → %s",
        df.height,
        df["date"].min(),
        df["date"].max(),
    )
    return df


def factor_columns() -> list[str]:
    """The non-RF factor columns, in canonical order. Useful for feature builders."""
    return ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
