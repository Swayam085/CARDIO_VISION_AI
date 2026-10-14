"""
CardioVision AI — Model Info Schemas
=====================================
Structured Pydantic schemas exposing model metadata, training provenance,
performance metrics, thresholds, and data leakage controls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """
    Exposes complete provenance, versioning, performance metrics, and model settings.
    """
    model_version: str = Field(..., description="Model version string (e.g. '1.0.0')")
    model_types: Dict[str, str] = Field(
        ...,
        description="Classifier algorithm type per target (e.g. CAD: RandomForestClassifier, LCX: XGBClassifier)"
    )
    models: Dict[str, str] = Field(
        ...,
        description="Alias for model_types for backward compatibility"
    )
    training_dataset: str = Field(..., description="Source dataset name and URL")
    dataset_source: str = Field(..., description="Alias for training_dataset for backward compatibility")
    dataset_size: Dict[str, int] = Field(..., description="Dataset row and column counts")
    feature_count: int = Field(..., description="Number of input feature columns (55)")
    validation_method: str = Field(..., description="Cross-validation strategy description")
    metrics: Dict[str, Any] = Field(..., description="Detailed cross-validation evaluation metrics per target")
    training_timestamp: str = Field(..., description="ISO 8601 UTC timestamp of model training execution")
    thresholds: Dict[str, float] = Field(..., description="CV-tuned decision threshold per target")
    preprocessing_version: str = Field(..., description="Preprocessing pipeline specification and version")
    leakage_excluded_columns: List[str] = Field(
        ...,
        description="Forbidden leakage target columns strictly excluded from input features"
    )
    leakage_excluded: List[str] = Field(
        ...,
        description="Alias for leakage_excluded_columns for backward compatibility"
    )
    safety_notice: str = Field(..., description="Clinical safety notice and usage guidelines")
