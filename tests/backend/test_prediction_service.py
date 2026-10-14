"""
CardioVision AI — Tests for PredictionService
==============================================
Tests prediction service initialization, multi-vessel prediction, value ranges,
stored thresholds, data leakage guard, column ordering, invalid input validation,
and missing model handling.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from app.schemas.prediction import PredictResponse
from app.services.prediction_service import (
    ModelArtifactMissingError,
    PredictionService,
)
from app.utils.validation import DataLeakageError, InputValidationError, EXPECTED_FEATURES


@pytest.fixture
def sample_features() -> dict:
    """Provides a valid 55-feature dictionary representing a sample patient."""
    return {
        "Age": 62, "Weight": 75, "Length": 172, "Sex": "Male", "BMI": 25.3,
        "DM": 1, "HTN": 1, "Current Smoker": 0, "EX-Smoker": 1, "FH": 1,
        "Obesity": "N", "CRF": "N", "CVA": "N", "Airway disease": "N", "Thyroid Disease": "N",
        "CHF": "N", "DLP": "Y", "BP": 135, "PR": 72, "Edema": 0,
        "Weak Peripheral Pulse": "N", "Lung rales": "N", "Systolic Murmur": "N", "Diastolic Murmur": "N",
        "Typical Chest Pain": 1, "Dyspnea": "N", "Function Class": 2,
        "Atypical": "N", "Nonanginal": "N", "Exertional CP": "N", "LowTH Ang": "N",
        "Q Wave": 0, "St Elevation": 0, "St Depression": 1, "Tinversion": 1,
        "LVH": "N", "Poor R Progression": "N", "BBB": "N", "VHD": "N",
        "FBS": 110, "CR": 0.9, "TG": 180, "LDL": 130, "HDL": 42, "BUN": 16, "ESR": 15,
        "HB": 14.2, "K": 4.1, "Na": 140, "WBC": 7200, "Lymph": 30, "Neut": 62, "PLT": 240,
        "EF-TTE": 55, "Region RWMA": 0,
    }


def test_prediction_service_load_and_ready():
    """Verify PredictionService loads all trained artifacts and reaches ready state."""
    svc = PredictionService.get_instance()
    svc.load_artifacts()
    assert svc.is_ready is True
    assert svc.model_version is not None


def test_predict_all_four_targets(sample_features):
    """Verify predictions for CAD, LAD, LCX, and RCA are generated."""
    svc = PredictionService.get_instance()
    resp: PredictResponse = svc.predict(sample_features)

    assert isinstance(resp, PredictResponse)
    assert set(resp.predictions.keys()) == {"CAD", "LAD", "LCX", "RCA"}

    for target in ["CAD", "LAD", "LCX", "RCA"]:
        pred = resp.predictions[target]
        # Probability in range 0.0 to 1.0
        assert 0.0 <= pred.model_predicted_probability <= 1.0
        assert 0.0 <= pred.probability <= 1.0
        assert pred.probability == pred.model_predicted_probability

        # Classification binary is 0 or 1
        assert pred.classification_binary in (0, 1)

        # Classification label string
        if target == "CAD":
            assert pred.classification in ("CAD", "Normal")
            assert pred.positive_class_label == "CAD"
        else:
            assert pred.classification in ("Stenotic", "Normal")
            assert pred.positive_class_label == "Stenotic"

        # Stored threshold present
        assert 0.0 <= pred.threshold <= 1.0


def test_stored_model_thresholds():
    """Verify stored CV-tuned thresholds from metrics.json are populated."""
    svc = PredictionService.get_instance()
    svc.load_artifacts()

    for target in ["CAD", "LAD", "LCX", "RCA"]:
        thresh = svc.get_task_threshold(target)
        assert 0.0 < thresh < 1.0


def test_feature_ordering_and_alignment(sample_features):
    """Verify feature names and column order match exact EXPECTED_FEATURES order."""
    svc = PredictionService.get_instance()
    df, cleaned = svc.validate_and_preprocess(sample_features)

    assert list(df.columns) == EXPECTED_FEATURES
    assert len(df.columns) == 55


def test_data_leakage_rejection(sample_features):
    """Verify data leakage columns (Cath, LAD, LCX, RCA) in features raise DataLeakageError."""
    svc = PredictionService.get_instance()

    for forbidden_col in ["Cath", "LAD", "LCX", "RCA"]:
        bad_features = sample_features.copy()
        bad_features[forbidden_col] = 1

        with pytest.raises(DataLeakageError) as exc_info:
            svc.predict(bad_features)
        assert forbidden_col in str(exc_info.value) or "LEAKAGE" in str(exc_info.value)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), "-inf", "infinity", "nan", "fifty"])
def test_invalid_input_numeric_rejection(sample_features, invalid_value):
    """Verify invalid numeric values (NaN, infinity, non-numeric strings) raise InputValidationError."""
    svc = PredictionService.get_instance()
    bad_features = sample_features.copy()
    bad_features["Age"] = invalid_value

    with pytest.raises(InputValidationError):
        svc.predict(bad_features, strict_validation=True)


def test_invalid_categorical_type_rejection(sample_features):
    """Verify dict or list passed for categorical features raises InputValidationError."""
    svc = PredictionService.get_instance()
    bad_features = sample_features.copy()
    bad_features["Sex"] = {"nested": "dict"}

    with pytest.raises(InputValidationError):
        svc.predict(bad_features)


def test_missing_model_artifact_handling(monkeypatch, sample_features):
    """Verify missing model artifacts raise ModelArtifactMissingError."""
    dummy_svc = PredictionService()
    monkeypatch.setattr(dummy_svc, "_resolve_artifact_path", lambda name: Path("/nonexistent/path/" + name))

    with pytest.raises(ModelArtifactMissingError) as exc_info:
        dummy_svc.load_artifacts(force_reload=True)
    assert "Missing required model artifact" in str(exc_info.value)
