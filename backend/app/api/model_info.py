"""
CardioVision AI — Model Info Router
====================================
GET /api/v1/model/info
GET /api/v1/model/features
GET /api/v1/model/health
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.model_info import ModelInfoResponse
from app.services.model_info_service import get_model_info_service
from app.services.prediction_service import (
    ModelArtifactMissingError,
    get_prediction_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata and provenance",
    description="Returns complete versioning, dataset provenance, performance metrics, and model settings.",
)
@router.get("/model-info", response_model=ModelInfoResponse, include_in_schema=False)
@router.get("/api/v1/model/info", response_model=ModelInfoResponse, include_in_schema=False)
@router.get("/api/v1/model-info", response_model=ModelInfoResponse, include_in_schema=False)
async def get_model_info() -> ModelInfoResponse:
    pred_svc = get_prediction_service()
    if not pred_svc.is_ready:
        try:
            pred_svc.load_artifacts()
        except ModelArtifactMissingError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "model_artifacts_missing", "message": str(exc)},
            )

    info_svc = get_model_info_service()
    return info_svc.get_model_info()


@router.get("/model/features", summary="Input feature list and metadata")
async def get_features():
    pred_svc = get_prediction_service()
    if not pred_svc.is_ready:
        try:
            pred_svc.load_artifacts()
        except ModelArtifactMissingError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "model_artifacts_missing"},
            )

    return pred_svc._feature_metadata


@router.get("/model/health", summary="Model readiness check")
async def model_health():
    pred_svc = get_prediction_service()
    artifacts = pred_svc.check_artifacts_present()
    all_ready = all(artifacts.values())

    return {
        "ready": pred_svc.is_ready or all_ready,
        "artifacts": artifacts,
        "all_artifacts_present": all_ready,
        "status": "ready" if (pred_svc.is_ready or all_ready) else "not_ready",
        "message": (
            "All model pipelines and preprocessors ready."
            if all_ready
            else "Model artifacts missing on disk."
        ),
    }
