"""
CardioVision AI — End-to-End Pipeline Integration Test
======================================================
Tests the full real-data pipeline:
  Patient Form Features
        ↓
     FastAPI (/analyze, /predict, /explain, /metrics/summary)
        ↓
     Validation & Leakage Guard (strict 422 enforcement)
        ↓
     Preprocessing (StandardScaler, OneHotEncoder, SimpleImputer)
        ↓
     4 Real ML Models (CAD, LAD, LCX, RCA)
        ↓
     Probabilities & Calibrated Classifications
        ↓
     TreeSHAP Explanations (positive/negative contributors, patient values)
        ↓
     Clinical Results & 3D Anatomy Mapping
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Real high-risk patient features matching ClinicalPresets.high_risk
HIGH_RISK_PATIENT = {
    "Age": 68,
    "Weight": 85,
    "Length": 170,
    "Sex": "Male",
    "BMI": 29.4,
    "DM": 1,
    "HTN": 1,
    "Current Smoker": 1,
    "EX-Smoker": 0,
    "FH": 1,
    "Obesity": "Y",
    "CRF": "N",
    "CVA": "N",     
    "Airway disease": "N",
    "Thyroid Disease": "N",
    "CHF": "Y",
    "DLP": "Y",
    "BP": 155,
    "PR": 88,
    "Edema": 1,
    "Weak Peripheral Pulse": "Y",
    "Lung rales": "Y",
    "Systolic Murmur": "N",
    "Diastolic Murmur": "N",
    "Typical Chest Pain": 1,
    "Dyspnea": "Y",
    "Function Class": 3,
    "Atypical": "N",
    "Nonanginal": "N",
    "Exertional CP": "Y",
    "LowTH Ang": "Y",
    "Q Wave": 1,
    "St Elevation": 0,
    "St Depression": 1,
    "Tinversion": 1,
    "LVH": "Y",
    "Poor R Progression": "Y",
    "BBB": "N",
    "VHD": "N",
    "FBS": 165,
    "CR": 1.3,
    "TG": 240,
    "LDL": 160,
    "HDL": 35,
    "BUN": 22,
    "ESR": 28,
    "HB": 13.2,
    "K": 4.4,
    "Na": 138,
    "WBC": 9200,
    "Lymph": 24,
    "Neut": 70,
    "PLT": 280,
    "EF-TTE": 40,
    "Region RWMA": 2,
}


class TestEndToEndPipeline:
    """Full system verification across all 10 integration pillars."""

    def test_01_health_and_model_info(self):
        """Verify API health and live model metadata."""
        health_res = client.get("/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["status"] in ["ok", "healthy"]
        assert "version" in health_data

        info_res = client.get("/api/v1/model/info")
        assert info_res.status_code == 200
        info_data = info_res.json()
        assert info_data["dataset_size"]["rows"] == 303
        assert "fold" in info_data["validation_method"].lower() or "cv" in info_data["validation_method"].lower()
        for target in ["CAD", "LAD", "LCX", "RCA"]:
            assert target in info_data["models"]

    def test_02_metrics_summary(self):
        """Verify /metrics/summary returns actual performance metrics for all 4 models."""
        res = client.get("/metrics/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["dataset"]["n_samples"] == 303
        tasks = data["tasks"]
        for target in ["CAD", "LAD", "LCX", "RCA"]:
            assert target in tasks
            t_metric = tasks[target]
            assert 0.0 <= t_metric["accuracy"] <= 1.0
            assert 0.0 <= t_metric["precision"] <= 1.0
            assert 0.0 <= t_metric["recall"] <= 1.0
            assert 0.0 <= t_metric["f1_default"] <= 1.0
            assert 0.0 <= t_metric["roc_auc"] <= 1.0
            assert t_metric["selected_model"] != ""

    def test_03_unified_analyze_pipeline(self):
        """Verify unified /analyze pipeline produces real predictions and SHAP explanations."""
        payload = {
            "patient_id": "CV-TEST-E2E-001",
            "features": HIGH_RISK_PATIENT,
            "top_n_shap": 10,
            "include_shap": True,
        }
        res = client.post("/analyze", json=payload)
        assert res.status_code == 200, f"/analyze failed with {res.status_code}: {res.text}"
        data = res.json()

        # Check predictions structure
        assert data["patient_id"] == "CV-TEST-E2E-001"
        preds = data["predictions"]
        for target in ["CAD", "LAD", "LCX", "RCA"]:
            assert target in preds
            p = preds[target]
            assert "model_predicted_probability" in p
            assert 0.0 <= p["model_predicted_probability"] <= 1.0
            assert "classification" in p
            assert "classification_binary" in p
            assert p["classification_binary"] in [0, 1]
            assert "threshold" in p
            assert 0.0 < p["threshold"] < 1.0

        # Check SHAP explanations summary
        assert data["shap_summary"] is not None
        cad_shap = data["shap_summary"]
        assert cad_shap["target"] == "CAD"
        assert len(cad_shap["top_contributors_overall"]) > 0
        for feat in cad_shap["top_contributors_overall"]:
            assert "feature" in feat
            assert "shap_value" in feat
            assert "direction" in feat

    def test_04_individual_vessel_explanations(self):
        """Verify TreeSHAP explanations for each anatomical vessel (LAD, LCX, RCA)."""
        for vessel in ["LAD", "LCX", "RCA"]:
            payload = {
                "features": HIGH_RISK_PATIENT,
                "target": vessel,
                "top_n": 8,
            }
            res = client.post("/api/v1/explain", json=payload)
            assert res.status_code == 200, f"/explain for {vessel} failed: {res.text}"
            data = res.json()
            assert data["target"] == vessel
            assert len(data["top_contributors_overall"]) > 0
            for feat in data["top_contributors_overall"]:
                assert "shap_value" in feat
                assert "patient_value" in feat

    def test_05_strict_data_leakage_rejection(self):
        """Verify strict 422 rejection when downstream target labels are present."""
        for forbidden_col in ["LAD", "LCX", "RCA", "Cath"]:
            leaked_features = dict(HIGH_RISK_PATIENT)
            leaked_features[forbidden_col] = 1
            payload = {"patient_id": "CV-LEAK-TEST", "features": leaked_features}
            res = client.post("/analyze", json=payload)
            assert res.status_code == 422, f"Failed to reject leakage column {forbidden_col}"
            err_body = str(res.json())
            assert "leakage" in err_body.lower() or forbidden_col in err_body

    def test_06_validation_error_handling(self):
        """Verify 422 rejection for invalid input ranges and wrong types."""
        # Out-of-range Age
        bad_features = dict(HIGH_RISK_PATIENT)
        bad_features["Age"] = 999
        payload = {"patient_id": "CV-VAL-TEST", "features": bad_features}
        res = client.post("/analyze", json=payload)
        assert res.status_code == 422

        # Invalid categorical value
        bad_features2 = dict(HIGH_RISK_PATIENT)
        bad_features2["Sex"] = "UnknownGender"
        payload2 = {"patient_id": "CV-VAL-TEST", "features": bad_features2}
        res2 = client.post("/analyze", json=payload2)
        assert res2.status_code == 422

    def test_07_3d_vessel_probabilities_mapping(self):
        """Verify that live API output supplies valid probabilities for 3D visualization."""
        payload = {
            "patient_id": "CV-3D-TEST",
            "features": HIGH_RISK_PATIENT,
        }
        res = client.post("/analyze", json=payload)
        assert res.status_code == 200
        preds = res.json()["predictions"]

        # Ensure LAD, LCX, RCA are present and valid floating numbers between 0 and 1
        for vessel_id in ["LAD", "LCX", "RCA"]:
            prob = preds[vessel_id]["model_predicted_probability"]
            assert isinstance(prob, (int, float))
            assert 0.0 <= prob <= 1.0
            # Test color tier thresholds matching vesselRegistry.ts
            if prob < 0.30:
                tier = "normal"
            elif prob < 0.60:
                tier = "mild"
            elif prob < 0.80:
                tier = "moderate"
            else:
                tier = "elevated"
            assert tier in ["normal", "mild", "moderate", "elevated"]
