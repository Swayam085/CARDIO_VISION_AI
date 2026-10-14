"""Structured JSON logging configuration for CardioVision AI backend."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a structured format."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
