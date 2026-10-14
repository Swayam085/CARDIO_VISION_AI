"""
CardioVision AI — Metrics Router (Day 3)
==========================================
GET /api/v1/metrics

Returns the real computed CV metrics from metrics.json.
No fake values — sourced directly from the training output.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.services.model_service import get_model_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/metrics",
    summary="Real model evaluation metrics",
    description=(
        "Returns stratified out-of-fold cross-validation metrics for all four "
        "prediction tasks (CAD, LAD, LCX, RCA). Includes ROC-AUC, F1, accuracy, "
        "precision, recall, confusion matrices, ROC curves, and PR curves."
    ),
)
@router.get("/api/v1/metrics", include_in_schema=False)
async def get_metrics():
    svc = get_model_service()

    if not svc.is_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "error":   "ml_pipeline_not_ready",
                "message": "Models not trained yet. Run ml/src/train.py first.",
            },
        )

    metrics = svc.metrics
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail={"error": "metrics_not_found", "message": "metrics.json not loaded."},
        )

    return metrics


@router.get(
    "/metrics/summary",
    summary="Compact metrics summary for frontend display",
)
async def get_metrics_summary():
    """Returns a compact per-task summary suitable for dashboard display."""
    svc = get_model_service()

    if not svc.is_ready:
        raise HTTPException(status_code=503, detail={"error": "ml_pipeline_not_ready"})

    metrics = svc.metrics
    if not metrics or "tasks" not in metrics:
        raise HTTPException(status_code=404, detail={"error": "metrics_not_found"})

    summary = {}
    for task, data in metrics["tasks"].items():
        m = data.get("oof_metrics", {})
        md = data.get("oof_metrics_default_threshold", {})
        cv = data.get("cv_aggregate", {})
        summary[task] = {
            "selected_model":       svc.model_metadata.get("models", {}).get(task, {}).get("model_class", "unknown"),
            "roc_auc":              round(md.get("roc_auc", 0), 4),
            "f1_default":           round(md.get("f1", 0), 4),
            "f1_tuned":             round(m.get("f1", 0), 4),
            "accuracy":             round(md.get("accuracy", 0), 4),
            "precision":            round(md.get("precision", 0), 4),
            "recall":               round(md.get("recall", 0), 4),
            "cv_auc_mean":          round(cv.get("roc_auc", {}).get("mean", 0), 4),
            "cv_auc_std":           round(cv.get("roc_auc", {}).get("std", 0), 4),
            "threshold_default":    0.5,
            "threshold_optimized":  data.get("selected_threshold", 0.5),
            "class_distribution":   data.get("class_distribution", {}),
            "confusion_matrix":     data.get("confusion_matrix_default", []),
        }

    return {
        "tasks":      summary,
        "disclaimer": metrics.get("disclaimer", ""),
        "generated_at": metrics.get("generated_at", ""),
        "dataset":    metrics.get("dataset", {}),
    }
