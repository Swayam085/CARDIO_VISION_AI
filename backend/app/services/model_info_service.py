"""
CardioVision AI — Model Information Service
============================================
Exposes metadata, provenance, performance metrics, decision thresholds, and training details.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.schemas.model_info import ModelInfoResponse
from app.services.prediction_service import (
    PredictionService,
    get_prediction_service,
)

logger = logging.getLogger(__name__)


class ModelInfoService:
    """
    Exposes complete metadata and provenance for CardioVision AI models.
    """

    _instance: Optional["ModelInfoService"] = None

    def __init__(self, pred_service: Optional[PredictionService] = None) -> None:
        self._pred_service = pred_service or get_prediction_service()

    @classmethod
    def get_instance(cls) -> "ModelInfoService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_model_info(self) -> ModelInfoResponse:
        """
        Build and return complete model provenance and metadata response.
        """
        if not self._pred_service.is_ready:
            self._pred_service.load_artifacts()

        meta = self._pred_service._model_metadata
        metrics_data = self._pred_service._metrics
        feat_meta = self._pred_service._feature_metadata

        # Extract classifier types
        model_types = {}
        for task in ["CAD", "LAD", "LCX", "RCA"]:
            if task in self._pred_service._models:
                clf = self._pred_service._models[task].named_steps["classifier"]
                model_types[task] = type(clf).__name__
            else:
                model_types[task] = "Unknown"

        # Extract thresholds
        thresholds = {
            task: self._pred_service.get_task_threshold(task)
            for task in ["CAD", "LAD", "LCX", "RCA"]
        }

        # Dataset size
        ds_size = feat_meta.get("dataset_size", {"rows": 303, "columns": 59})

        ds_url = meta.get(
            "dataset_source",
            "Extension of Z-Alizadeh Sani Dataset (UCI ID 411) https://archive.ics.uci.edu/dataset/411/extention+of+z"
        )
        leak_ex = feat_meta.get(
            "excluded_leakage_columns",
            ["LAD", "LCX", "RCA", "Cath"]
        )

        return ModelInfoResponse(
            model_version=self._pred_service.model_version,
            model_types=model_types,
            models=model_types,
            training_dataset=ds_url,
            dataset_source=ds_url,
            dataset_size=ds_size,
            feature_count=feat_meta.get("feature_count", 55),
            validation_method=meta.get("cv_strategy", "5-Fold Stratified Cross-Validation"),
            metrics=metrics_data,
            training_timestamp=meta.get("training_date", "2026-10-02T02:58:00+00:00"),
            thresholds=thresholds,
            preprocessing_version=meta.get(
                "preprocessing_version",
                "v1.0.0 — MedianImputer + StandardScaler (numeric); MostFrequentImputer + OrdinalEncoder (categorical)"
            ),
            leakage_excluded_columns=leak_ex,
            leakage_excluded=leak_ex,
            safety_notice=meta.get(
                "safety_notice",
                "For educational and clinical decision-support purposes only. Not a standalone diagnostic tool."
            ),
        )


def get_model_info_service() -> ModelInfoService:
    """FastAPI dependency for ModelInfoService."""
    return ModelInfoService.get_instance()
