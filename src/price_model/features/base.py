"""Feature registry — the single extension point for feature engineering.

To add a new feature: write a class subclassing Feature, decorate with @register, and
add the feature name to your experiment config. The pipeline picks it up automatically;
the leakage test runs against it automatically.

Contract: `compute(panel)` returns the input panel with one additional column named
`self.name`. It must never use any data with date > the date of the row it's computing.
The leakage test in tests/test_no_leakage.py enforces this by truncating the panel and
verifying the feature value at date T is identical to the value computed on the full panel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import polars as pl


class Feature(ABC):
    """Base class for a feature."""

    name: ClassVar[str]  # column name in the output panel
    version: ClassVar[int] = 1
    inputs: ClassVar[tuple[str, ...]] = ()  # required input columns
    lookback_days: ClassVar[int] = 0  # max history needed to compute one value

    @abstractmethod
    def compute(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Return panel with one new column (self.name) added.

        The panel is assumed sorted by (ticker, date). Implementations should use
        `.over("ticker")` to keep per-ticker computations isolated.
        """
        ...


FEATURE_REGISTRY: dict[str, Feature] = {}


def register(cls: type[Feature]) -> type[Feature]:
    """Class decorator: register a Feature subclass by name."""
    if not hasattr(cls, "name"):
        raise TypeError(f"{cls.__name__} must declare a class-level `name` attribute")
    instance = cls()
    if cls.name in FEATURE_REGISTRY:
        raise ValueError(f"Duplicate feature name: {cls.name}")
    FEATURE_REGISTRY[cls.name] = instance
    return cls


def _trigger_registration() -> None:
    """Import every module that contains @register'd Feature subclasses."""
    import price_model.features.cross_features
    import price_model.features.factor_loadings
    import price_model.features.technical  # noqa: F401


def get_feature(name: str) -> Feature:
    if name not in FEATURE_REGISTRY:
        _trigger_registration()
    if name not in FEATURE_REGISTRY:
        raise KeyError(f"Feature {name!r} not registered. Known: {sorted(FEATURE_REGISTRY)}")
    return FEATURE_REGISTRY[name]


def list_features() -> list[str]:
    _trigger_registration()
    return sorted(FEATURE_REGISTRY)
