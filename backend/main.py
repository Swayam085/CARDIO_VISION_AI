"""
CardioVision AI — FastAPI Application Entry Point
=================================================
AI-Powered Cardiovascular Risk Visualization & Multi-Vessel Stenosis Prediction

Backend shell for Day 1. Implements:
  - Application factory with CORS, exception handlers, lifecycle events
  - /health endpoint (live + verified)
  - Router placeholders for /predict, /analyze, /explain, /model-info, /metrics
  - No fake ML output — endpoints return proper 501 / 503 until ML pipeline ready
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Ensure backend and ml/src directories are in sys.path
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_ROOT_DIR = _BACKEND_DIR.parent
_ML_SRC_DIR = _ROOT_DIR / "ml" / "src"
if str(_ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_SRC_DIR))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, predict, analyze, explain, model_info, metrics
from app.utils.config import get_settings
from app.utils.logging_config import configure_logging

# ── Logging ─────────────────────────────────────────────────
configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# ── Lifespan ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    logger.info("CardioVision AI backend starting up…")
    logger.info("Version: %s | Environment: %s", settings.app_version, settings.environment)

    # Load ML models (non-blocking — 503 returned until ready)
    try:
        from app.services.model_service import get_model_service  # noqa: PLC0415
        svc = get_model_service()
        if svc.is_ready:
            logger.info("[OK] All ML models loaded at startup")
        else:
            logger.warning("ML models not yet trained — /predict will return 503 until ml/src/train.py is run")
    except Exception as exc:
        logger.warning("Model pre-load skipped: %s", exc)

    yield
    logger.info("CardioVision AI backend shutting down…")


# ── Application factory ──────────────────────────────────────

def create_application() -> FastAPI:
    app = FastAPI(
        title="CardioVision AI",
        description=(
            "AI-Powered Cardiovascular Risk Visualization & Multi-Vessel Stenosis Prediction. "
            "Built for Multimodal AI Hackathon 2026 — Cardiovascular Track.\n\n"
            "**SAFETY NOTICE**: This application provides model-based estimates for "
            "educational and decision-support purposes only. It is NOT a medical diagnosis "
            "and does NOT replace professional clinical evaluation or advice from a qualified "
            "healthcare professional."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────
    cors_origins = list(settings.cors_origins)
    allow_all = "*" in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else cors_origins,
        allow_origin_regex=None if allow_all else r"https?://.*\.onrender\.com",
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ────────────────────────────────────
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Request validation error on %s %s: %s", request.method, request.url, exc.errors())
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({
                "error": "validation_error",
                "message": "Invalid request payload or schema validation failure.",
                "details": exc.errors(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again later.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ── Routers ──────────────────────────────────────────────
    api_prefix = "/api/v1"
    # Root level routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, tags=["Prediction"])
    app.include_router(analyze.router, tags=["Analysis"])
    app.include_router(explain.router, tags=["Explainability"])
    app.include_router(model_info.router, tags=["Model"])
    app.include_router(metrics.router, tags=["Metrics"])

    # Prefix routers
    app.include_router(health.router, prefix=api_prefix, tags=["Health"], include_in_schema=False)
    app.include_router(predict.router, prefix=api_prefix, tags=["Prediction"], include_in_schema=False)
    app.include_router(analyze.router, prefix=api_prefix, tags=["Analysis"], include_in_schema=False)
    app.include_router(explain.router, prefix=api_prefix, tags=["Explainability"], include_in_schema=False)
    app.include_router(model_info.router, prefix=api_prefix, tags=["Model"], include_in_schema=False)
    app.include_router(metrics.router, prefix=api_prefix, tags=["Metrics"], include_in_schema=False)

    logger.info("All routers registered under root and %s", api_prefix)
    
    # ── Optional Static Frontend Serving (if built in frontend/dist) ──
    frontend_dist = _ROOT_DIR / "frontend" / "dist"
    if (frontend_dist / "index.html").exists():
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        assets_dir = frontend_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

        models_dir = frontend_dist / "models"
        if models_dir.exists():
            app.mount("/models", StaticFiles(directory=str(models_dir)), name="frontend-models")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_frontend_spa(full_path: str):
            candidate = frontend_dist / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

        logger.info("Serving compiled frontend SPA from %s", frontend_dist)

    return app


app = create_application()

# ── Dev server entry point ───────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", settings.backend_port))
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=port,
        reload=settings.backend_reload,
        log_level=settings.backend_log_level,
        app_dir=str(_BACKEND_DIR),
    )
