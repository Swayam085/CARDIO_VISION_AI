"""
CardioVision AI — Predict Router
=================================
POST /api/v1/predict

Accepts clinical features payload (55 variables), validates inputs and data leakage,
executes trained pipelines for CAD, LAD, LCX, RCA, and returns structured predictions,
model probabilities, classifications, CV-tuned thresholds, and version information.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.prediction_service import (
    ModelArtifactMissingError,
    get_prediction_service,
)
from app.utils.validation import DataLeakageError, InputValidationError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Multi-vessel CAD risk prediction",
    description=(
        "Predict CAD diagnosis (Cath result) and vessel-specific stenosis probabilities "
        "(LAD, LCX, RCA) from 55 clinical input features. Returns model probabilities, "
        "classifications, CV-tuned decision thresholds, and model version."
    ),
    responses={
        200: {"description": "Successful multi-vessel prediction"},
        422: {"description": "Input validation or data leakage violation"},
        503: {"description": "Model artifacts not trained or missing"},
    },
)
@router.post("/api/v1/predict", response_model=PredictResponse, include_in_schema=False)
async def predict(request: PredictRequest) -> PredictResponse:
    svc = get_prediction_service()

    if not svc.is_ready:
        try:
            svc.load_artifacts()
        except ModelArtifactMissingError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "model_artifacts_missing",
                    "message": str(exc),
                    "artifacts_status": svc.check_artifacts_present(),
                },
            )

    try:
        response = svc.predict(
            features=request.features,
            patient_id=request.patient_id,
            tasks=request.tasks,
            strict_validation=request.strict_validation,
        )
        return response

    except DataLeakageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "data_leakage_violation",
                "message": str(exc),
            },
        )
    except InputValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "input_validation_failed",
                "message": str(exc),
                "details": getattr(exc, "errors", [str(exc)]),
            },
        )
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "prediction_failed",
                "message": "An error occurred while generating predictions. Please verify input data.",
            },
        )
