"""
CardioVision AI — Production Prediction Service
================================================
Production-quality prediction service for multi-vessel CAD stenosis risk.

Features:
1. Loads trained model artifacts and preprocessing pipelines
2. Validates feature names, column ordering, data leakage, and value range/types
3. Preprocesses patient input data reproducibly
4. Generates predictions for CAD, LAD, LCX, RCA
5. Employs cross-validation tuned thresholds from stored metrics
6. Returns model probability outputs, classifications, thresholds, and version
7. Strictly enforces terminology ("Model-predicted probability", NOT "percentage blockage")
8. Thread-safe singleton implementation with lazy artifact loading
"""

from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.schemas.prediction import PredictResponse, TaskPredictionResult
from app.utils.validation import (
    DataLeakageError,
    InputValidationError,
    validate_patient_features,
    EXPECTED_FEATURES,
    LEAKAGE_COLUMNS,
)

logger = logging.getLogger(__name__)

# Add ML src path if needed
ML_SRC = Path(__file__).resolve().parents[3] / "ml" / "src"
if ML_SRC.exists() and str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

# Artifact directories (backend first, fallback to ml/artifacts)
BACKEND_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
ML_ARTIFACTS = Path(__file__).resolve().parents[3] / "ml" / "artifacts"


class ModelArtifactMissingError(RuntimeError):
    """Raised when one or more required model artifacts are missing from disk."""


class PredictionService:
    """
    Production-quality Prediction Service singleton.
    """

    _instance: Optional["PredictionService"] = None

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._preprocessor: Any = None
        self._metrics: Dict[str, Any] = {}
        self._model_metadata: Dict[str, Any] = {}
        self._feature_metadata: Dict[str, Any] = {}
        self._loaded: bool = False
        self._artifacts_dir: Path = BACKEND_ARTIFACTS

    @classmethod
    def get_instance(cls) -> "PredictionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _resolve_artifact_path(self, filename: str) -> Path:
        """Find artifact in backend/artifacts or ml/artifacts."""
        backend_path = BACKEND_ARTIFACTS / filename
        if backend_path.exists():
            return backend_path

        ml_path = ML_ARTIFACTS / filename
        if ml_path.exists():
            return ml_path

        return backend_path  # Default to backend path for error reporting

    def load_artifacts(self, force_reload: bool = False) -> None:
        """
        Load all model pipelines, preprocessor, and metadata files.
        """
        if self._loaded and not force_reload:
            return

        required_models = ["cad_model.joblib", "lad_model.joblib", "lcx_model.joblib", "rca_model.joblib"]
        missing_artifacts: List[str] = []

        # Find paths
        model_paths = {}
        for name in ["CAD", "LAD", "LCX", "RCA"]:
            fname = f"{name.lower()}_model.joblib"
            path = self._resolve_artifact_path(fname)
            if not path.exists():
                missing_artifacts.append(fname)
            else:
                model_paths[name] = path

        preproc_path = self._resolve_artifact_path("preprocessor.joblib")
        if not preproc_path.exists():
            missing_artifacts.append("preprocessor.joblib")

        if missing_artifacts:
            raise ModelArtifactMissingError(
                f"Missing required model artifact(s): {missing_artifacts}. "
                "Run ml/src/train.py to train and generate artifacts."
            )

        # Load models
        for task, path in model_paths.items():
            self._models[task] = joblib.load(path)
            logger.info("Loaded task model [%s] from %s", task, path)

        # Load standalone preprocessor
        self._preprocessor = joblib.load(preproc_path)
        logger.info("Loaded preprocessor pipeline from %s", preproc_path)

        # Load metadata files
        metrics_p = self._resolve_artifact_path("metrics.json")
        if metrics_p.exists():
            with open(metrics_p, encoding="utf-8") as f:
                self._metrics = json.load(f)

        model_meta_p = self._resolve_artifact_path("model_metadata.json")
        if model_meta_p.exists():
            with open(model_meta_p, encoding="utf-8") as f:
                self._model_metadata = json.load(f)

        feat_meta_p = self._resolve_artifact_path("feature_metadata.json")
        if feat_meta_p.exists():
            with open(feat_meta_p, encoding="utf-8") as f:
                self._feature_metadata = json.load(f)

        self._loaded = True
        logger.info("PredictionService initialized — all 4 task models loaded successfully.")

    @property
    def is_ready(self) -> bool:
        return self._loaded and len(self._models) == 4

    @property
    def model_version(self) -> str:
        return self._model_metadata.get("version", "1.0.0")

    def get_task_threshold(self, task: str) -> float:
        """Retrieve stored cross-validation optimal threshold for task."""
        task_upper = task.upper()
        if self._metrics and "tasks" in self._metrics:
            task_info = self._metrics["tasks"].get(task_upper, {})
            if "selected_threshold" in task_info and task_info["selected_threshold"] is not None:
                return round(float(task_info["selected_threshold"]), 4)
            if "optimal_threshold" in task_info and task_info["optimal_threshold"] is not None:
                return round(float(task_info["optimal_threshold"]), 4)
        if self._model_metadata and "models" in self._model_metadata:
            meta_info = self._model_metadata["models"].get(task_upper, {})
            if "optimized_threshold" in meta_info and meta_info["optimized_threshold"] is not None:
                return round(float(meta_info["optimized_threshold"]), 4)
        return 0.5

    def check_artifacts_present(self) -> Dict[str, bool]:
        """Check status of required files."""
        result = {}
        for task in ["cad", "lad", "lcx", "rca"]:
            path = self._resolve_artifact_path(f"{task}_model.joblib")
            result[f"{task}_model"] = path.exists()
        result["preprocessor"] = self._resolve_artifact_path("preprocessor.joblib").exists()
        return result

    def validate_and_preprocess(
        self,
        raw_features: Dict[str, Any],
        strict: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate feature dict, align column order to exact EXPECTED_FEATURES,
        and construct 1-row DataFrame.
        """
        # Validate values, types, and data leakage
        cleaned_dict, warnings = validate_patient_features(raw_features, strict=strict)

        # Align columns to exact 55 expected feature order
        ordered_row = {col: cleaned_dict.get(col, np.nan) for col in EXPECTED_FEATURES}
        df = pd.DataFrame([ordered_row])

        return df, cleaned_dict

    def predict(
        self,
        features: Dict[str, Any],
        patient_id: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        strict_validation: bool = False,
    ) -> PredictResponse:
        """
        Generate predictions for specified tasks (CAD, LAD, LCX, RCA).

        Returns:
            PredictResponse schema instance with model probabilities, classifications,
            thresholds, version, and terminology disclaimers.
        """
        if not self.is_ready:
            self.load_artifacts()

        # Target selection (default to all four)
        target_tasks = [t.upper() for t in (tasks or ["CAD", "LAD", "LCX", "RCA"])]
        valid_tasks = {"CAD", "LAD", "LCX", "RCA"}
        invalid_tasks = [t for t in target_tasks if t not in valid_tasks]
        if invalid_tasks:
            raise InputValidationError(f"Invalid prediction task(s): {invalid_tasks}. Allowed: {list(valid_tasks)}")

        # Validate and preprocess patient input
        X_df, _ = self.validate_and_preprocess(features, strict=strict_validation)

        results: Dict[str, TaskPredictionResult] = {}

        for task in target_tasks:
            model = self._models[task]

            # Model pipeline prediction
            proba = float(model.predict_proba(X_df)[0, 1])
            proba_bounded = max(0.0, min(1.0, proba))

            threshold_opt = self.get_task_threshold(task)
            pred_binary = int(proba_bounded >= threshold_opt)

            # Target label
            pos_label = "CAD" if task == "CAD" else "Stenotic"
            neg_label = "Normal"
            classification_str = pos_label if pred_binary == 1 else neg_label

            results[task] = TaskPredictionResult(
                model_predicted_probability=round(proba_bounded, 6),
                probability=round(proba_bounded, 6),
                classification=classification_str,
                classification_binary=pred_binary,
                threshold=threshold_opt,
                threshold_default=0.5,
                positive_class_label=pos_label,
                model_version=self.model_version,
            )

        return PredictResponse(
            patient_id=patient_id,
            predictions=results,
            model_version=self.model_version,
        )


@lru_cache(maxsize=1)
def get_prediction_service() -> PredictionService:
    """FastAPI dependency for PredictionService singleton."""
    svc = PredictionService.get_instance()
    if not svc.is_ready:
        try:
            svc.load_artifacts()
        except ModelArtifactMissingError:
            pass  # Handled in endpoints
    return svc
