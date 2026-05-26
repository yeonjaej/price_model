"""Model registry: name -> Model class.

To register a new model, add a line to MODEL_REGISTRY. The config system uses these
strings to instantiate models without import gymnastics in YAML.
"""

from __future__ import annotations

from price_model.models.base import Model, ModelConfig
from price_model.models.baseline import LastReturnPredictor, ZeroPredictor
from price_model.models.boosting import LightGBMModel
from price_model.models.foundation import ChronosZeroShot

# ChronosZeroShot module imports cleanly without torch/chronos installed; the
# import error is deferred to first use (when _ensure_pipeline() runs). So this
# stays a hard import and the failure mode is "Chronos not installed" at predict
# time, not at registry-build time.
MODEL_REGISTRY: dict[str, type[Model]] = {
    "ZeroPredictor": ZeroPredictor,
    "LastReturnPredictor": LastReturnPredictor,
    "LightGBMModel": LightGBMModel,
    "ChronosZeroShot": ChronosZeroShot,
}


def build_model(class_name: str, config: ModelConfig) -> Model:
    if class_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {class_name!r}. Known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[class_name](config)


__all__ = [
    "MODEL_REGISTRY",
    "ChronosZeroShot",
    "LastReturnPredictor",
    "LightGBMModel",
    "Model",
    "ModelConfig",
    "ZeroPredictor",
    "build_model",
]
