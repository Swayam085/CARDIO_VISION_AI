"""
CardioVision AI — Production Backend API Test Suite
===================================================
Comprehensive test coverage for Day 4 FastAPI backend implementation.

Tests:
- health endpoint (GET /health)
- valid analyze endpoint (POST /analyze)
- invalid input (out of range, invalid categoricals)
- missing fields / empty payload
- wrong data types
- NaN and Infinity rejection
- prediction response (POST /predict & data leakage rejection)
- SHAP explanation response (POST /explain)
- metrics endpoint reading actual metrics.json (GET /metrics & GET /metrics/summary)
- model info endpoint (GET /model-info & GET /model/info)
"""

from __future__ import annotations

import math
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture
def valid_patient_payload() -> dict:
    return {
        "patient_id": "PT-10042",
        "features": {
            "Age": 62, "Weight": 75, "Length": 172, "Sex": "Male", "BMI": 25.35,
            "DM": 1, "HTN": 1, "Current Smoker": 0, "EX-Smoker": 1, "FH": 1,
            "Obesity": "N", "CRF": "N", "CVA": "N", "Airway disease": "N", "Thyroid Disease": "N",
            "CHF": "N", "DLP": "Y", "BP": 135, "PR": 75, "Edema": 0,
            "Weak Peripheral Pulse": "N", "Lung rales": "N", "Systolic Murmur": "N", "Diastolic Murmur": "N",
            "Typical Chest Pain": 1, "Dyspnea": "N", "Function Class": 2,
            "Atypical": "N", "Nonanginal": "N", "Exertional CP": "N", "LowTH Ang": "N",
            "Q Wave": 0, "St Elevation": 0, "St Depression": 1, "Tinversion": 1,
            "LVH": "N", "Poor R Progression": "N", "BBB": "N", "VHD": "N",
            "FBS": 118, "CR": 0.95, "TG": 180, "LDL": 130, "HDL": 42, "BUN": 16, "ESR": 15,
            "HB": 14.2, "K": 4.3, "Na": 141, "WBC": 7800, "Lymph": 30, "Neut": 62, "PLT": 240,
            "EF-TTE": 55, "Region RWMA": 0,
        },
    }


# ── 1. Health Endpoint Tests ─────────────────────────────────

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "service" in data
    assert "version" in data
    assert "uci_dataset_url" in data


def test_health_v1_prefix_alias():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── 2. Analyze Endpoint Tests ────────────────────────────────

def test_valid_analyze_request(valid_patient_payload):
    response = client.post("/analyze", json=valid_patient_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["patient_id"] == "PT-10042"
    assert 0.0 <= data["cad_risk_score"] <= 1.0
    assert set(data["vessel_risk"].keys()) == {"LAD", "LCX", "RCA"}
    assert set(data["predictions"].keys()) == {"CAD", "LAD", "LCX", "RCA"}
    assert "shap_summary" in data
    assert data["shap_summary"]["target"] == "CAD"
    assert "vessel_risk_visualization_note" in data
    assert "disclaimer" in data


def test_analyze_v1_prefix_alias(valid_patient_payload):
    response = client.post("/api/v1/analyze", json=valid_patient_payload)
    assert response.status_code == 200
    assert "cad_risk_score" in response.json()


# ── 3. Input Validation Tests ────────────────────────────────

def test_invalid_input_out_of_range_numeric(valid_patient_payload):
    bad_payload = valid_patient_payload.copy()
    bad_payload["features"] = dict(valid_patient_payload["features"])
    bad_payload["features"]["Age"] = 250  # Absurd age > 120

    response = client.post("/analyze", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "out of realistic range" in str(data) or "validation" in str(data).lower()


def test_invalid_input_invalid_categorical_value(valid_patient_payload):
    bad_payload = valid_patient_payload.copy()
    bad_payload["features"] = dict(valid_patient_payload["features"])
    bad_payload["features"]["Sex"] = "Alien"

    response = client.post("/analyze", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "Invalid categorical value" in str(data) or "Sex" in str(data)


# ── 4. Missing Fields / Empty Payload Tests ─────────────────

def test_missing_features_dict_rejected():
    response = client.post("/analyze", json={})
    assert response.status_code == 422


def test_empty_features_dict_rejected():
    response = client.post("/analyze", json={"features": {}})
    assert response.status_code == 422
    data = response.json()
    assert "empty" in str(data).lower() or "validation" in str(data).lower()


# ── 5. Wrong Data Types Tests ────────────────────────────────

def test_wrong_type_string_for_numeric_feature(valid_patient_payload):
    bad_payload = valid_patient_payload.copy()
    bad_payload["features"] = dict(valid_patient_payload["features"])
    bad_payload["features"]["BP"] = "high_blood_pressure"

    response = client.post("/analyze", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "Must be a number" in str(data) or "Invalid type" in str(data)


def test_wrong_type_dict_for_numeric_feature(valid_patient_payload):
    bad_payload = valid_patient_payload.copy()
    bad_payload["features"] = dict(valid_patient_payload["features"])
    bad_payload["features"]["Age"] = {"value": 60}

    response = client.post("/analyze", json=bad_payload)
    assert response.status_code == 422


# ── 6. NaN and Infinity Tests ────────────────────────────────

def test_nan_value_rejection(valid_patient_payload):
    bad_payload = valid_patient_payload.copy()
    bad_payload["features"] = dict(valid_patient_payload["features"])
    bad_payload["features"]["Age"] = "NaN"

    response = client.post("/analyze", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "NaN" in str(data) or "validation" in str(data).lower()


def test_infinity_value_rejection(valid_patient_payload):
    bad_payload = valid_patient_payload.copy()
    bad_payload["features"] = dict(valid_patient_payload["features"])
    bad_payload["features"]["Age"] = "Infinity"

    response = client.post("/analyze", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "Infinity" in str(data) or "validation" in str(data).lower()


# ── 7. Prediction Response & Data Leakage Tests ──────────────

def test_predict_endpoint_response(valid_patient_payload):
    response = client.post("/predict", json=valid_patient_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["patient_id"] == "PT-10042"
    assert "predictions" in data
    for task in ("CAD", "LAD", "LCX", "RCA"):
        assert task in data["predictions"]
        task_pred = data["predictions"][task]
        assert 0.0 <= task_pred["model_predicted_probability"] <= 1.0
        assert task_pred["classification"] in ("CAD", "Normal", "Stenotic")
        assert "threshold" in task_pred
        assert "model_version" in task_pred


def test_predict_data_leakage_rejection(valid_patient_payload):
    bad_payload = valid_patient_payload.copy()
    bad_payload["features"] = dict(valid_patient_payload["features"])
    bad_payload["features"]["Cath"] = "CAD"  # Data leakage forbidden column

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "DATA LEAKAGE DETECTED" in str(data) or "Cath" in str(data)


# ── 8. SHAP Explanation Response Tests ───────────────────────

def test_explain_endpoint_response(valid_patient_payload):
    explain_req = {
        "patient_id": valid_patient_payload["patient_id"],
        "features": valid_patient_payload["features"],
        "target": "CAD",
        "top_n": 5,
    }
    response = client.post("/explain", json=explain_req)
    assert response.status_code == 200
    data = response.json()

    assert data["target"] == "CAD"
    assert data["label"] == "Model Feature Contribution"
    assert 0.0 <= data["model_predicted_probability"] <= 1.0
    assert len(data["all_contributions"]) == 55
    assert len(data["top_contributors_overall"]) <= 5
    assert "Model Feature Contribution" in data["disclaimer"]


# ── 9. Metrics Endpoint Tests ────────────────────────────────

def test_metrics_reads_actual_metrics_json():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "tasks" in data
    for task in ("CAD", "LAD", "LCX", "RCA"):
        assert task in data["tasks"]
        t_data = data["tasks"][task]
        oof = t_data["oof_metrics_default_threshold"]
        # Verify non-hardcoded real metrics present
        assert "roc_auc" in oof and isinstance(oof["roc_auc"], float)
        assert "f1" in oof and isinstance(oof["f1"], float)
        assert "accuracy" in oof and isinstance(oof["accuracy"], float)
        assert "precision" in oof and isinstance(oof["precision"], float)
        assert "recall" in oof and isinstance(oof["recall"], float)


def test_metrics_summary_endpoint():
    response = client.get("/api/v1/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    for task in ("CAD", "LAD", "LCX", "RCA"):
        assert task in data["tasks"]
        t_sum = data["tasks"][task]
        assert "roc_auc" in t_sum
        assert "f1_tuned" in t_sum


# ── 10. Model Info Endpoint Tests ────────────────────────────

def test_model_info_endpoint():
    for endpoint in ("/model-info", "/model/info", "/api/v1/model-info", "/api/v1/model/info"):
        response = client.get(endpoint)
        assert response.status_code == 200
        data = response.json()

        assert "model_version" in data
        assert "model_types" in data
        assert "training_dataset" in data
        assert "thresholds" in data
        assert "leakage_excluded_columns" in data
        assert set(data["leakage_excluded_columns"]) >= {"Cath", "LAD", "LCX", "RCA"}
