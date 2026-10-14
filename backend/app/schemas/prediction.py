"""
CardioVision AI — Prediction Schemas
=====================================
Structured Pydantic schemas for prediction requests and responses.
Enforces strict terminology ("Model-predicted probability") and excludes
anatomical blockage terminology ("percentage blockage").
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Input request for multi-vessel CAD prediction.
    Features must contain clinical input variables (55 features).
    Target columns Cath, LAD, LCX, RCA are strictly FORBIDDEN (leakage).
    """
    patient_id: Optional[str] = Field(None, description="Optional patient identifier")
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary mapping 55 clinical feature names to patient values."
    )
    tasks: Optional[List[str]] = Field(
        None,
        description="List of targets to predict: ['CAD', 'LAD', 'LCX', 'RCA']. Default: all 4."
    )
    strict_validation: bool = Field(
        False,
        description="If True, strictly fail on missing or unexpected non-leakage features."
    )


class TaskPredictionResult(BaseModel):
    """
    Prediction result for a single vessel / target (CAD, LAD, LCX, or RCA).
    """
    model_predicted_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Statistical model probability output (0.0 to 1.0). NOT anatomical blockage %."
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Alias for model_predicted_probability for backward compatibility."
    )
    classification: str = Field(
        ...,
        description="Clinical target label ('CAD'/'Normal' or 'Stenotic'/'Normal')."
    )
    classification_binary: int = Field(
        ...,
        description="Binary classification prediction: 1 for positive class, 0 for normal."
    )
    threshold: float = Field(
        ...,
        description="Stored cross-validation tuned decision threshold for this model."
    )
    threshold_default: float = Field(
        0.5,
        description="Standard 0.5 decision threshold."
    )
    positive_class_label: str = Field(
        ...,
        description="Label corresponding to binary class 1 ('CAD' or 'Stenotic')."
    )
    model_version: str = Field(
        ...,
        description="Model version string from artifact metadata."
    )


class PredictResponse(BaseModel):
    """
    Structured multi-target prediction response for CardioVision AI.
    """
    patient_id: Optional[str] = Field(None, description="Patient identifier if provided")
    predictions: Dict[str, TaskPredictionResult] = Field(
        ...,
        description="Dictionary mapping target name ('CAD', 'LAD', 'LCX', 'RCA') to prediction result."
    )
    model_version: str = Field(..., description="Overall pipeline model version")
    terminology_notice: str = Field(
        "Model probability outputs represent statistical risk estimations derived from trained algorithms, "
        "not anatomical luminal stenosis percentages (e.g. NOT '70% blockage').",
        description="Mandatory terminology notice."
    )
    disclaimer: str = Field(
        "SAFETY NOTICE: This application provides model-based risk estimates for educational "
        "and clinical decision-support purposes only. It is NOT a medical diagnosis and does NOT "
        "replace professional clinical evaluation, coronary angiography, or physician judgment.",
        description="Mandatory medical safety disclaimer."
    )
