"""
CardioVision AI — Production Analyze Router
============================================
POST /analyze and POST /api/v1/analyze

End-to-end multi-vessel CAD analysis endpoint orchestrating:
Patient input → Pydantic validation → Feature validation → Preprocessing →
CAD/LAD/LCX/RCA Model predictions → Probabilities → Classification →
TreeSHAP explainability → 3D Vessel risk mapping structure.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, status

from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
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
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Complete multi-vessel CAD analysis and SHAP explanation",
    description=(
        "Full cardiovascular analysis pipeline: patient input validation → "
        "feature data leakage check → preprocessor pipeline → trained CAD, LAD, LCX, RCA "
        "model inference → tuned classification → TreeSHAP feature attributions."
    ),
    responses={
        200: {"description": "Full analysis completed successfully"},
        422: {"description": "Validation error or data leakage violation"},
        503: {"description": "ML model artifacts not trained or missing"},
    },
)
@router.post("/api/v1/analyze", response_model=AnalyzeResponse, include_in_schema=False)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Executes complete CardioVision AI analysis workflow using real trained models.
    """
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
                    "artifacts_status": pred_svc.check_artifacts_present(),
                },
            )

    try:
        # 1. Run predictions for all 4 targets (CAD, LAD, LCX, RCA)
        pred_response = pred_svc.predict(
            features=request.features,
            patient_id=request.patient_id,
            tasks=["CAD", "LAD", "LCX", "RCA"],
            strict_validation=request.strict_validation,
        )

        predictions = pred_response.predictions
        cad_risk_score = predictions["CAD"].model_predicted_probability

        vessel_risk = {
            "LAD": predictions["LAD"].model_predicted_probability,
            "LCX": predictions["LCX"].model_predicted_probability,
            "RCA": predictions["RCA"].model_predicted_probability,
        }

        # 2. Run TreeSHAP explainability if requested
        shap_summary = None
        if request.include_shap:
            try:
                explain_svc = get_explainability_service()
                explain_resp = explain_svc.explain(
                    features=request.features,
                    target="CAD",
                    top_n=request.top_n_shap,
                )
                shap_summary = explain_resp.model_dump()
            except SHAPComputationError as exc:
                logger.warning("SHAP computation failed during analyze: %s", exc)
                shap_summary = {"error": "shap_unavailable", "message": str(exc)}

        return AnalyzeResponse(
            patient_id=request.patient_id,
            cad_risk_score=cad_risk_score,
            predictions=predictions,
            vessel_risk=vessel_risk,
            shap_summary=shap_summary,
            model_version=pred_svc.model_version,
        )

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
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during analyze pipeline execution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "analysis_failed",
                "message": "An error occurred during full analysis execution. Please check input features.",
            },
        )
