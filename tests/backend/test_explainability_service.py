"""
CardioVision AI — Tests for ExplainabilityService
==================================================
Tests TreeSHAP feature attributions, contribution rankings, positive/negative summaries,
and mandated disclaimers ("Model Feature Contribution", no clinical causality claims).
"""

from __future__ import annotations

import pytest

from app.schemas.explainability import ExplainResponse, FeatureContribution
from app.services.explainability_service import ExplainabilityService
from app.utils.validation import DataLeakageError, InputValidationError


@pytest.fixture
def sample_features() -> dict:
    return {
        "Age": 65, "Weight": 80, "Length": 170, "Sex": "Male", "BMI": 27.7,
        "DM": 1, "HTN": 1, "Current Smoker": 1, "EX-Smoker": 0, "FH": 1,
        "Obesity": "Y", "CRF": "N", "CVA": "N", "Airway disease": "N", "Thyroid Disease": "N",
        "CHF": "N", "DLP": "Y", "BP": 145, "PR": 80, "Edema": 0,
        "Weak Peripheral Pulse": "N", "Lung rales": "N", "Systolic Murmur": "N", "Diastolic Murmur": "N",
        "Typical Chest Pain": 1, "Dyspnea": "Y", "Function Class": 3,
        "Atypical": "N", "Nonanginal": "N", "Exertional CP": "Y", "LowTH Ang": "N",
        "Q Wave": 1, "St Elevation": 0, "St Depression": 1, "Tinversion": 1,
        "LVH": "N", "Poor R Progression": "Y", "BBB": "N", "VHD": "N",
        "FBS": 140, "CR": 1.1, "TG": 220, "LDL": 150, "HDL": 38, "BUN": 20, "ESR": 22,
        "HB": 13.5, "K": 4.3, "Na": 138, "WBC": 8500, "Lymph": 25, "Neut": 68, "PLT": 260,
        "EF-TTE": 45, "Region RWMA": 1,
    }


@pytest.mark.parametrize("target", ["CAD", "LAD", "LCX", "RCA"])
def test_explain_all_four_targets(sample_features, target):
    """Verify SHAP attributions are computed for all four prediction targets."""
    service = ExplainabilityService.get_instance()
    response: ExplainResponse = service.explain(features=sample_features, target=target)

    assert isinstance(response, ExplainResponse)
    assert response.target == target
    assert response.label == "Model Feature Contribution"
    assert 0.0 <= response.model_predicted_probability <= 1.0

    # Total 55 features in all_contributions
    assert len(response.all_contributions) == 55

    # Check top summaries populated
    assert len(response.top_contributors_overall) <= 10
    assert len(response.top_positive_contributors) <= 10
    assert len(response.top_negative_contributors) <= 10


def test_shap_feature_contribution_structure(sample_features):
    """Verify each feature contribution object has required fields and proper types."""
    service = ExplainabilityService.get_instance()
    response: ExplainResponse = service.explain(features=sample_features, target="CAD")

    for item in response.all_contributions:
        assert isinstance(item, FeatureContribution)
        assert item.feature is not None
        assert item.patient_value is not None
        assert isinstance(item.shap_value, float)
        assert item.absolute_contribution >= 0.0
        assert item.absolute_contribution == round(abs(item.shap_value), 6)
        assert item.direction in ("positive", "negative", "neutral")
        assert item.positive_negative_contribution in ("positive", "negative", "neutral")
        assert item.rank >= 1


def test_shap_ranking_order(sample_features):
    """Verify all_contributions are sorted descending by absolute SHAP contribution with 1-based ranks."""
    service = ExplainabilityService.get_instance()
    response: ExplainResponse = service.explain(features=sample_features, target="CAD")

    prev_abs = float("inf")
    for idx, item in enumerate(response.all_contributions, start=1):
        assert item.rank == idx
        assert item.absolute_contribution <= prev_abs + 1e-9
        prev_abs = item.absolute_contribution


def test_shap_top_positive_and_negative_summaries(sample_features):
    """Verify top positive features have SHAP > 0 and top negative features have SHAP < 0."""
    service = ExplainabilityService.get_instance()
    response: ExplainResponse = service.explain(features=sample_features, target="CAD")

    for item in response.top_positive_contributors:
        assert item.shap_value > 0
        assert item.direction == "positive"

    for item in response.top_negative_contributors:
        assert item.shap_value < 0
        assert item.direction == "negative"


def test_shap_disclaimer_content(sample_features):
    """Verify mandatory SHAP disclaimer asserts model feature contribution and excludes causality claims."""
    service = ExplainabilityService.get_instance()
    response: ExplainResponse = service.explain(features=sample_features, target="CAD")

    disclaimer = response.disclaimer
    assert "Model Feature Contribution" in disclaimer
    assert "NOT prove clinical causality" in disclaimer
    assert "biological mechanisms" in disclaimer


def test_invalid_explanation_target_rejection(sample_features):
    """Verify invalid target raises InputValidationError."""
    service = ExplainabilityService.get_instance()
    with pytest.raises(InputValidationError):
        service.explain(features=sample_features, target="INVALID_TARGET")


def test_data_leakage_in_explanation_rejection(sample_features):
    """Verify passing leakage target columns to explain raises DataLeakageError."""
    service = ExplainabilityService.get_instance()
    bad_features = sample_features.copy()
    bad_features["Cath"] = "CAD"

    with pytest.raises(DataLeakageError):
        service.explain(features=bad_features, target="CAD")


def test_shap_feature_categories(sample_features):
    """Verify that all features are categorized into Demographic, Clinical, Laboratory, ECG, Echo."""
    service = ExplainabilityService.get_instance()
    response = service.explain(features=sample_features, target="CAD")
    valid_categories = {"Demographic", "Clinical", "Laboratory", "ECG", "Echo"}

    for item in response.all_contributions:
        assert item.feature_category in valid_categories, f"Unknown category: {item.feature_category} for {item.feature}"

    assert response.category_contributions is not None
    for cat in valid_categories:
        assert cat in response.category_contributions
        assert response.category_contributions[cat] >= 0.0

