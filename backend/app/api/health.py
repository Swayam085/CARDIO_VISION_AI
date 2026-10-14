"""
CardioVision AI — Health Router
================================
GET /health — liveness check (no ML dependencies)
"""

import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.utils.config import get_settings

router = APIRouter()
settings = get_settings()

_startup_time = datetime.now(timezone.utc)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    service: str
    version: str
    environment: str
    python_version: str
    platform: str
    uci_dataset_url: str
    anatomy_source_url: str


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
@router.get("/api/v1/health", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> HealthResponse:
    """
    Returns the current health status of the CardioVision AI backend.

    This endpoint has NO dependency on ML models — it always returns 200 OK
    as long as the server process is running.
    """
    now = datetime.now(timezone.utc)
    uptime = (now - _startup_time).total_seconds()

    return HealthResponse(
        status="ok",
        timestamp=now.isoformat(),
        uptime_seconds=round(uptime, 2),
        service=settings.app_title,
        version=settings.app_version,
        environment=settings.environment,
        python_version=sys.version.split()[0],
        platform=platform.system(),
        uci_dataset_url=settings.uci_dataset_url,
        anatomy_source_url=settings.anatomy_source_url,
    )
