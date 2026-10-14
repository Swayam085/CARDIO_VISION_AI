"""
CardioVision AI — Tests for API Endpoints (/predict & /explain)
================================================================
Tests FastAPI endpoints for prediction, SHAP explainability, and model info.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def sample_payload() -> dict:
    return {
        "patient_id": "PT-10024",
        "features": {
            "Age": 60, "Weight": 78, "Length": 175, "Sex": "Male", "BMI": 25.5,
            "DM": 1, "HTN": 1, "Current Smoker": 0, "EX-Smoker": 1, "FH": 1,
            "Obesity": "N", "CRF": "N", "CVA": "N", "Airway disease": "N", "Thyroid Disease": "N",
            "CHF": "N", "DLP": "Y", "BP": 130, "PR": 72, "Edema": 0,
            "Weak Peripheral Pulse": "N", "Lung rales": "N", "Systolic Murmur": "N", "Diastolic Murmur": "N",
            "Typical Chest Pain": 1, "Dyspnea": "N", "Function Class": 2,
            "Atypical": "N", "Nonanginal": "N", "Exertional CP": "N", "LowTH Ang": "N",
            "Q Wave": 0, "St Elevation": 0, "St Depression": 1, "Tinversion": 1,
            "LVH": "N", "Poor R Progression": "N", "BBB": "N", "VHD": "N",
            "FBS": 115, "CR": 0.9, "TG": 175, "LDL": 125, "HDL": 45, "BUN": 15, "ESR": 14,
            "HB": 14.5, "K": 4.2, "Na": 140, "WBC": 7500, "Lymph": 32, "Neut": 60, "PLT": 250,
            "EF-TTE": 55, "Region RWMA": 0,
        },
    }


def test_api_predict_success(sample_payload):
    """Test POST /api/v1/predict returns 200 with structured multi-target prediction."""
    response = client.post("/api/v1/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["patient_id"] == "PT-10024"
    assert "predictions" in data
    assert set(data["predictions"].keys()) == {"CAD", "LAD", "LCX", "RCA"}

    cad_pred = data["predictions"]["CAD"]
    assert "model_predicted_probability" in cad_pred
    assert "probability" in cad_pred
    assert "classification" in cad_pred
    assert "threshold" in cad_pred
    assert "model_version" in cad_pred
    assert 0.0 <= cad_pred["model_predicted_probability"] <= 1.0


def test_api_predict_data_leakage_rejection(sample_payload):
    """Test POST /api/v1/predict rejects data leakage target columns with 422."""
    bad_payload = sample_payload.copy()
    bad_payload["features"]["Cath"] = "CAD"

    response = client.post("/api/v1/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "data_leakage_violation" in str(data) or "LEAKAGE" in str(data)


def test_api_explain_success(sample_payload):
    """Test POST /api/v1/explain returns 200 with SHAP feature attributions and disclaimers."""
    explain_payload = {
        "patient_id": sample_payload["patient_id"],
        "features": sample_payload["features"],
        "target": "CAD",
        "top_n": 5,
    }
    response = client.post("/api/v1/explain", json=explain_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["target"] == "CAD"
    assert data["label"] == "Model Feature Contribution"
    assert 0.0 <= data["model_predicted_probability"] <= 1.0
    assert len(data["all_contributions"]) == 55
    assert len(data["top_contributors_overall"]) <= 5

    assert "Model Feature Contribution" in data["disclaimer"]
    assert "NOT prove clinical causality" in data["disclaimer"]


def test_api_explain_all_targets(sample_payload):
    """Test POST /api/v1/explain works for all 4 targets (CAD, LAD, LCX, RCA)."""
    for target in ["CAD", "LAD", "LCX", "RCA"]:
        payload = {"features": sample_payload["features"], "target": target}
        response = client.post("/api/v1/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["target"] == target


def test_api_model_info_success():
    """Test GET /api/v1/model/info returns 200 with complete metadata."""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()

    assert "model_version" in data
    assert "model_types" in data
    assert "training_dataset" in data
    assert "thresholds" in data
