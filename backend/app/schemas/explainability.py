"""
CardioVision AI — Explainability Schemas
========================================
Structured Pydantic schemas for SHAP explainability requests and responses.
Enforces "Model Feature Contribution" labeling and disclaimers prohibiting
causality or biological mechanism claims.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    """
    Request payload for SHAP feature attribution explanation.
    """
    patient_id: Optional[str] = Field(None, description="Optional patient identifier")
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary mapping 55 clinical feature names to patient values."
    )
    target: str = Field(
        "CAD",
        description="Target model to explain: 'CAD', 'LAD', 'LCX', or 'RCA' (case-insensitive)."
    )
    top_n: int = Field(
        10,
        ge=1,
        le=55,
        description="Number of top positive, negative, and overall contributors to highlight."
    )


class FeatureContribution(BaseModel):
    """
    SHAP attribution details for a single feature.
    """
    feature: str = Field(..., description="Feature column name")
    patient_value: Any = Field(..., description="Original raw value provided for this patient")
    shap_value: float = Field(..., description="SHAP feature attribution contribution value")
    absolute_contribution: float = Field(
        ...,
        ge=0.0,
        description="Absolute value of SHAP contribution |SHAP|"
    )
    direction: str = Field(
        ...,
        description="Attribution direction: 'positive' (increases risk), 'negative' (decreases risk), or 'neutral'"
    )
    positive_negative_contribution: str = Field(
        ...,
        description="Identical to direction: 'positive', 'negative', or 'neutral'"
    )
    rank: int = Field(
        ...,
        ge=1,
        description="1-indexed rank based on absolute SHAP contribution"
    )
    feature_category: str = Field(
        "Clinical",
        description="Dataset feature group category: Demographic, Clinical, Laboratory, ECG, Echo"
    )


class ExplainResponse(BaseModel):
    """
    Structured SHAP feature contribution explanation response.
    """
    target: str = Field(..., description="Explained prediction target ('CAD', 'LAD', 'LCX', or 'RCA')")
    label: str = Field(
        "Model Feature Contribution",
        description="Official required label for SHAP explanation output."
    )
    model_predicted_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model prediction probability for this target for the given patient."
    )
    base_value: float = Field(
        ...,
        description="Explainer base value / expected model value across training distribution."
    )
    top_positive_contributors: List[FeatureContribution] = Field(
        ...,
        description="Top features driving prediction probability HIGHER (positive risk factors)."
    )
    top_negative_contributors: List[FeatureContribution] = Field(
        ...,
        description="Top features driving prediction probability LOWER (protective factors)."
    )
    top_contributors_overall: List[FeatureContribution] = Field(
        ...,
        description="Top features sorted by overall absolute impact |SHAP|."
    )
    all_contributions: List[FeatureContribution] = Field(
        ...,
        description="Complete list of all 55 features ranked by absolute SHAP contribution."
    )
    category_contributions: Optional[Dict[str, float]] = Field(
        None,
        description="Total absolute SHAP feature contribution aggregated by feature category."
    )
    disclaimer: str = Field(
        "Model Feature Contribution (SHAP) quantifies how much each input variable influenced "
        "the statistical model's prediction probability for this patient. SHAP feature contributions "
        "show model feature attributions; they do NOT prove clinical causality, biological mechanisms, "
        "or physical disease cause-and-effect.",
        description="Mandatory SHAP explanation disclaimer."
    )
