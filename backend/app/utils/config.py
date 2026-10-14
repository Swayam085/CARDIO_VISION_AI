"""
CardioVision AI — Application Settings
========================================
Typed configuration via Pydantic Settings, loaded from environment variables / .env
"""

from functools import lru_cache
from typing import Any, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ─────────────────────────────────────────────────
    app_title: str = "CardioVision AI"
    app_version: str = "0.1.0"
    environment: str = "development"

    # ── Backend server ───────────────────────────────────────
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_reload: bool = True
    backend_log_level: str = "info"

    # ── CORS ─────────────────────────────────────────────────
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        if isinstance(v, list):
            return [str(i).strip() for i in v if str(i).strip()]
        return v

    # ── ML Pipeline ──────────────────────────────────────────
    ml_data_dir: str = "ml/data"
    ml_artifacts_dir: str = "ml/artifacts"
    backend_artifacts_dir: str = "backend/artifacts"
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # ── Dataset (official UCI) ────────────────────────────────
    uci_dataset_url: str = "https://archive.ics.uci.edu/dataset/411/extention+of+z"
    uci_dataset_id: int = 411
    uci_dataset_name: str = "Extension of Z-Alizadeh Sani Dataset"

    # ── 3D Anatomy ───────────────────────────────────────────
    anatomy_source_url: str = "https://lifesciencedb.jp/bp3d"
    anatomy_source_name: str = "BodyParts3D / Anatomography by DBCLS"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

