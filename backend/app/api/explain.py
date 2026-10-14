"""
CardioVision AI — Explain Router
=================================
POST /api/v1/explain

Generates TreeSHAP feature attributions for a given prediction target (CAD, LAD, LCX, RCA).
Returns ranked feature contributions, top positive/negative/overall contributors, base value,
and required disclaimers ("Model Feature Contribution").
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.explainability import ExplainRequest, ExplainResponse
from app.services.explainability_service import (
    SHAPComputationError,
    get_explainability_service,
)
from app.services.prediction_service import (
    ModelArtifactMissingError,
    get_prediction_service,
)
from app.utils.validation import DataLeakageError, InputValidationError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="SHAP feature attribution explanation",
    description=(
        "Computes TreeSHAP feature attribution values for a given patient prediction target "
        "(CAD, LAD, LCX, or RCA). Returns feature values, SHAP contributions, directions, ranks, "
        "and top positive/negative/overall contributor summaries."
    ),
    responses={
        200: {"description": "Successful SHAP feature contribution explanation"},
        422: {"description": "Input validation or target error"},
        503: {"description": "Model artifacts missing or pipeline not ready"},
    },
)
@router.post("/api/v1/explain", response_model=ExplainResponse, include_in_schema=False)
async def explain(request: ExplainRequest) -> ExplainResponse:
    pred_svc = get_prediction_service()
    if not pred_svc.is_ready:
        try:
            pred_svc.load_artifacts()
        except ModelArtifactMissingError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "model_artifacts_missing",
                    "message": str(exc),
                },
            )

    explain_svc = get_explainability_service()

    try:
        response = explain_svc.explain(
            features=request.features,
            target=request.target,
            top_n=request.top_n,
        )
        return response

    except DataLeakageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "data_leakage_violation", "message": str(exc)},
        )
    except InputValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": "input_validation_failed", "message": str(exc)},
        )
    except SHAPComputationError as exc:
        logger.error("SHAP computation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "shap_explanation_failed",
                "message": "Unable to calculate SHAP feature contributions for this model. Please try again.",
            },
        )
    except Exception as exc:
        logger.exception("Unexpected error during SHAP explanation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "explainability_failed",
                "message": "An error occurred while generating explanation. Please verify input data.",
            },
        )
