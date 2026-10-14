"""
CardioVision AI — Backend Tests: Health Endpoint
=================================================
Tests /health endpoint. No ML dependencies.
"""

import pytest
from fastapi.testclient import TestClient

# conftest.py at project root adds backend/ to sys.path
from main import app  # noqa: E402

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_ok():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"


def test_health_contains_required_fields():
    response = client.get("/health")
    data = response.json()
    required_fields = [
        "status", "timestamp", "uptime_seconds", "service",
        "version", "environment", "python_version", "platform",
        "uci_dataset_url", "anatomy_source_url",
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"


def test_health_uci_url_correct():
    response = client.get("/health")
    data = response.json()
    assert "411" in data["uci_dataset_url"]
    assert "uci.edu" in data["uci_dataset_url"]


def test_health_anatomy_url_correct():
    response = client.get("/health")
    data = response.json()
    assert "lifesciencedb" in data["anatomy_source_url"]


def test_health_uptime_non_negative():
    response = client.get("/health")
    data = response.json()
    assert data["uptime_seconds"] >= 0


def test_predict_returns_valid_response():
    """Predict returns 200 with real predictions when trained, or 503 when not ready."""
    response = client.post("/api/v1/predict", json={"features": {}})
    assert response.status_code in (200, 503)


def test_analyze_returns_valid_response():
    """Analyze returns 200/422/503 depending on payload validation and ML pipeline readiness."""
    response = client.post("/api/v1/analyze", json={"features": {}})
    assert response.status_code in (200, 422, 503)


def test_model_info_returns_dataset_metadata():
    """Model info returns model_metadata when models are ready, or 503."""
    response = client.get("/api/v1/model/info")
    # 200 when models loaded, 503 when not trained
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        # dataset_source should reference UCI
        assert "uci.edu" in data.get("dataset_source", "")
        # Should contain model class info for all 4 tasks
        models_info = data.get("models", {})
        for task in ("CAD", "LAD", "LCX", "RCA"):
            assert task in models_info, f"model-info missing task: {task}"


def test_model_info_leakage_excluded_present():
    """Leakage excluded columns must be in model_metadata when models are ready."""
    response = client.get("/api/v1/model/info")
    if response.status_code == 503:
        pytest.skip("Models not trained yet")
    data = response.json()
    excluded = data.get("leakage_excluded", [])
    assert "LAD" in excluded
    assert "LCX" in excluded
    assert "RCA" in excluded
    assert "Cath" in excluded


def test_metrics_returns_valid_response():
    """Metrics endpoint returns 200 with real metrics.json OR 503 if not trained."""
    response = client.get("/api/v1/metrics")
    # When models are trained: 200 with real metrics
    # When models are not trained: 503
    assert response.status_code in (200, 503), (
        f"Unexpected status {response.status_code}"
    )
    if response.status_code == 200:
        data = response.json()
        # Must have 'tasks' key with real CV data
        assert "tasks" in data, "metrics response missing 'tasks' key"
        # Must include all four targets
        for task in ("CAD", "LAD", "LCX", "RCA"):
            assert task in data["tasks"], f"Missing task '{task}' in metrics"
    else:
        # 503 = models not trained, which is valid
        data = response.json()
        assert "detail" in data
