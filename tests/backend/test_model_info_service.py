"""
CardioVision AI — Tests for ModelInfoService
=============================================
Tests exposure of model metadata, version, classifier types, training dataset provenance,
dataset size, feature count, validation method, metrics, thresholds, and leakage exclusions.
"""

from __future__ import annotations

import pytest

from app.schemas.model_info import ModelInfoResponse
from app.services.model_info_service import ModelInfoService


def test_model_info_response_structure():
    """Verify ModelInfoService returns complete provenance and configuration metadata."""
    svc = ModelInfoService.get_instance()
    info: ModelInfoResponse = svc.get_model_info()

    assert isinstance(info, ModelInfoResponse)
    assert info.model_version is not None
    assert isinstance(info.model_types, dict)
    assert set(info.model_types.keys()) == {"CAD", "LAD", "LCX", "RCA"}

    assert "Z-Alizadeh Sani" in info.training_dataset or "uci.edu" in info.training_dataset
    assert info.dataset_size["rows"] == 303
    assert info.feature_count == 55

    assert "Stratified" in info.validation_method
    assert "tasks" in info.metrics

    assert isinstance(info.thresholds, dict)
    for task in ["CAD", "LAD", "LCX", "RCA"]:
        assert 0.0 < info.thresholds[task] < 1.0

    assert "LAD" in info.leakage_excluded_columns
    assert "LCX" in info.leakage_excluded_columns
    assert "RCA" in info.leakage_excluded_columns
    assert "Cath" in info.leakage_excluded_columns
