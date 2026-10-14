"""
CardioVision AI — Analyze Schemas
==================================
Pydantic schemas for the main POST /analyze endpoint.
Validates input payload parameters and structures the full analysis output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from app.schemas.prediction import TaskPredictionResult


class AnalyzeRequest(BaseModel):
    """
    Request payload for full CAD and vessel stenosis analysis pipeline.
    """
    patient_id: Optional[str] = Field(None, description="Optional patient identifier (e.g. PT-10024)")
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary mapping 55 clinical feature names to patient values."
    )
    include_shap: bool = Field(
        True,
        description="Whether to generate TreeSHAP feature attribution explanations."
    )
    include_vessel_risk_map: bool = Field(
        True,
        description="Whether to return vessel-specific risk scores mapped to 3D coronary anatomy."
    )
    top_n_shap: int = Field(
        10,
        ge=1,
        le=55,
        description="Number of top SHAP feature contributors to return."
    )
    strict_validation: bool = Field(
        False,
        description="If True, strictly reject payload on missing/unexpected non-leakage features."
    )

    @field_validator("features")
    @classmethod
    def validate_features_non_empty(cls, v: Any) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("Payload 'features' must be a JSON object / dictionary.")
        if not v:
            raise ValueError("Payload 'features' dictionary cannot be empty.")
        return v


class AnalyzeResponse(BaseModel):
    """
    Structured full analysis response combining CAD prediction, vessel stenosis risk,
    TreeSHAP feature attributions, and anatomical visualization disclaimers.
    """
    patient_id: Optional[str] = Field(None, description="Patient identifier if provided")
    cad_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model predicted probability of Overall CAD (Cath positive stenosis >=50%)."
    )
    predictions: Dict[str, TaskPredictionResult] = Field(
        ...,
        description="Detailed TaskPredictionResult object per target (CAD, LAD, LCX, RCA)."
    )
    vessel_risk: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping vessel branches (LAD, LCX, RCA) to model predicted probabilities."
    )
    shap_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="SHAP feature attributions for CAD prediction target."
    )
    vessel_risk_visualization_note: str = Field(
        "The 3D visualization represents vessel-level model-predicted risk scores "
        "mapped to LAD/LCX/RCA anatomy. It does NOT represent exact lesion coordinates, "
        "CT/MRI segmentation, or pixel-level anatomical data.",
        description="Explanation note for 3D coronary vessel risk mapping."
    )
    disclaimer: str = Field(
        "SAFETY NOTICE: This application provides model-based risk estimates for educational "
        "and clinical decision-support purposes only. It is NOT a medical diagnosis and does NOT "
        "replace professional clinical evaluation, coronary angiography, or physician judgment.",
        description="Mandatory medical safety disclaimer."
    )
    model_version: str = Field(..., description="Trained model version string")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC execution timestamp."
    )
