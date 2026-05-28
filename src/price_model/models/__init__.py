"""Model registry: name -> Model class.

To register a new model, add a line to MODEL_REGISTRY. The config system uses these
strings to instantiate models without import gymnastics in YAML.
"""

from __future__ import annotations

from price_model.models.base import Model, ModelConfig
from price_model.models.baseline import LastReturnPredictor, ZeroPredictor
from price_model.models.boosting import LightGBMModel
from price_model.models.classical import (
    ArimaPerTicker,
    FamaFrenchFactorModel,
    GarchVolForecaster,
    GbmMaximumLikelihood,
)
from price_model.models.foundation import ChronosZeroShot

# Classical and foundation models import cleanly without their optional deps;
# import errors are deferred to fit() / predict() time. So the registry hard-binds
# all classes; the failure mode is "extra not installed" at use time, not import.
MODEL_REGISTRY: dict[str, type[Model]] = {
    "ZeroPredictor": ZeroPredictor,
    "LastReturnPredictor": LastReturnPredictor,
    "LightGBMModel": LightGBMModel,
    "ChronosZeroShot": ChronosZeroShot,
    # Classical baselines (need [classical] extras to actually fit)
    "ArimaPerTicker": ArimaPerTicker,
    "GarchVolForecaster": GarchVolForecaster,
    "GbmMaximumLikelihood": GbmMaximumLikelihood,
    # Fama-French is pure numpy + the KF download adapter — no extras needed.
    "FamaFrenchFactorModel": FamaFrenchFactorModel,
}


def build_model(class_name: str, config: ModelConfig) -> Model:
    if class_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {class_name!r}. Known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[class_name](config)


__all__ = [
    "MODEL_REGISTRY",
    "ArimaPerTicker",
    "ChronosZeroShot",
    "FamaFrenchFactorModel",
    "GarchVolForecaster",
    "GbmMaximumLikelihood",
    "LastReturnPredictor",
    "LightGBMModel",
    "Model",
    "ModelConfig",
    "ZeroPredictor",
    "build_model",
]
