"""Cross-sectional normalization.

For each date, transform each feature so that values are comparable across stocks
and across time. Two common choices:

- zscore: (x - mean_date) / std_date
- rank:   uniform rank in [0, 1] within the cross-section of that date

This is the step that lets a model trained on the S&P 500 universe make sensible
predictions for any individual ticker like AMD — the model sees "this stock's
momentum relative to the universe today" rather than a raw price-derived number.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import polars as pl

NormKind = Literal["zscore", "rank", "none"]


def normalize(
    panel: pl.DataFrame,
    feature_cols: Iterable[str],
    kind: NormKind = "zscore",
    date_col: str = "date",
) -> pl.DataFrame:
    """Cross-sectionally normalize each feature column within each date."""
    if kind == "none":
        return panel

    out = panel
    for col in feature_cols:
        if kind == "zscore":
            mean = pl.col(col).mean().over(date_col)
            std = pl.col(col).std().over(date_col)
            out = out.with_columns(
                ((pl.col(col) - mean) / pl.when(std == 0).then(1.0).otherwise(std)).alias(col)
            )
        elif kind == "rank":
            # rank in [0, 1] across the cross-section
            n = pl.col(col).count().over(date_col)
            out = out.with_columns((pl.col(col).rank("average").over(date_col) / n).alias(col))
        else:
            raise ValueError(f"Unknown normalization kind: {kind!r}")
    return out
