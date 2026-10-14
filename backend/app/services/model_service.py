"""
CardioVision AI — Model Service Bridge
=======================================
Backward-compatible wrapper bridging ModelService to the production PredictionService.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.services.prediction_service import (
    ModelArtifactMissingError,
    PredictionService,
    get_prediction_service,
)

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when model artifacts have not been trained yet."""


class ModelService:
    """
    Backward-compatible wrapper over PredictionService.
    """

    _instance: Optional["ModelService"] = None

    def __init__(self) -> None:
        self._pred_service = get_prediction_service()

    @classmethod
    def get(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        try:
            self._pred_service.load_artifacts()
        except ModelArtifactMissingError as exc:
            raise ModelNotReadyError(str(exc)) from exc

    @property
    def is_ready(self) -> bool:
        return self._pred_service.is_ready

    @property
    def metrics(self) -> dict:
        return self._pred_service._metrics

    @property
    def model_metadata(self) -> dict:
        return self._pred_service._model_metadata

    @property
    def feature_metadata(self) -> dict:
        return self._pred_service._feature_metadata

    def artifacts_present(self) -> dict[str, bool]:
        return self._pred_service.check_artifacts_present()

    def predict(
        self,
        feature_dict: Dict[str, Any],
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.is_ready:
            self.load()

        resp = self._pred_service.predict(features=feature_dict, tasks=tasks)
        results = {}
        for task_name, pred_obj in resp.predictions.items():
            results[task_name] = {
                "probability": pred_obj.probability,
                "model_predicted_probability": pred_obj.model_predicted_probability,
                "prediction_default": int(pred_obj.probability >= 0.5),
                "prediction_tuned": pred_obj.classification_binary,
                "threshold_default": 0.5,
                "threshold_optimized": pred_obj.threshold,
                "positive_class_label": pred_obj.positive_class_label,
                "classification": pred_obj.classification,
            }
        return results


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    svc = ModelService.get()
    if not svc.is_ready:
        try:
            svc.load()
        except ModelNotReadyError:
            pass
    return svc
