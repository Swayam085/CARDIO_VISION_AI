"""
ML Leakage Audit Tests
=======================
Tests that enforce the data leakage contract.
"""

import sys
from pathlib import Path
import pytest

# Add ml/src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml" / "src"))
from leakage_audit import audit_feature_list, LeakageViolationError, FORBIDDEN_FEATURES


def test_audit_passes_clean_features():
    clean = ["Age", "BMI", "HTN", "DM", "Smoking", "BP", "PR", "Chol"]
    report = audit_feature_list(clean, strict=True)
    assert report["status"] == "PASS"
    assert report["violations"] == []


def test_audit_fails_lad_in_inputs():
    with pytest.raises(LeakageViolationError):
        audit_feature_list(["Age", "LAD", "BP"], strict=True)


def test_audit_fails_lcx_in_inputs():
    with pytest.raises(LeakageViolationError):
        audit_feature_list(["Age", "LCX", "BP"], strict=True)


def test_audit_fails_rca_in_inputs():
    with pytest.raises(LeakageViolationError):
        audit_feature_list(["Age", "RCA", "BP"], strict=True)


def test_audit_fails_cath_in_inputs():
    with pytest.raises(LeakageViolationError):
        audit_feature_list(["Age", "Cath", "BP"], strict=True)


def test_audit_non_strict_returns_violations():
    report = audit_feature_list(["Age", "LAD", "LCX", "RCA", "Cath"], strict=False)
    assert report["status"] == "FAIL"
    assert set(report["violations"]) == set(FORBIDDEN_FEATURES)


def test_forbidden_features_list_complete():
    """Verify all four forbidden features are in the constant."""
    assert "LAD" in FORBIDDEN_FEATURES
    assert "LCX" in FORBIDDEN_FEATURES
    assert "RCA" in FORBIDDEN_FEATURES
    assert "Cath" in FORBIDDEN_FEATURES


def test_clean_features_exclude_forbidden():
    all_features = ["Age", "BMI", "LAD", "LCX", "RCA", "Cath", "HTN"]
    report = audit_feature_list(all_features, strict=False)
    for forbidden in FORBIDDEN_FEATURES:
        assert forbidden not in report["clean_features"], (
            f"Forbidden feature {forbidden} appeared in clean features!"
        )
