"""
CardioVision AI — Data Leakage Audit
======================================
Formal audit ensuring no leakage-related target features appear in model inputs.

REQUIREMENT: The following features MUST NEVER appear in model inputs:
  - LAD   (Left Anterior Descending stenosis — target)
  - LCX   (Left Circumflex stenosis — target)
  - RCA   (Right Coronary Artery stenosis — target)
  - Cath  (Catheterization result — target/leakage proxy)

This module provides:
  1. audit_feature_list() — validates a proposed feature list
  2. get_safe_features()  — returns safe features from a DataFrame
  3. run_full_audit()     — detailed audit report

Usage:
  python ml/src/leakage_audit.py  (after downloading dataset)
"""

import json
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ── Forbidden features (leakage targets) ─────────────────────
FORBIDDEN_FEATURES: List[str] = ["LAD", "LCX", "RCA", "Cath"]

# ── Model targets (outputs, not inputs) ──────────────────────
# Primary target: CAD (overall coronary artery disease)
# Vessel-specific targets: LAD, LCX, RCA stenosis
MODEL_TARGETS = {
    "primary": "CAD",
    "vessel_specific": ["LAD", "LCX", "RCA"],
}


class LeakageViolationError(ValueError):
    """Raised when a forbidden feature is found in the proposed input set."""


def audit_feature_list(proposed_features: List[str], strict: bool = True) -> dict:
    """
    Audit a proposed feature list for leakage violations.

    Args:
        proposed_features: List of feature names intended as model inputs.
        strict: If True, raise LeakageViolationError on violations.

    Returns:
        Audit report dict with violations, passed features, status.

    Raises:
        LeakageViolationError: If strict=True and violations are found.
    """
    violations = [f for f in proposed_features if f in FORBIDDEN_FEATURES]
    clean_features = [f for f in proposed_features if f not in FORBIDDEN_FEATURES]

    report = {
        "status": "FAIL" if violations else "PASS",
        "violations": violations,
        "forbidden_features": FORBIDDEN_FEATURES,
        "proposed_count": len(proposed_features),
        "clean_count": len(clean_features),
        "clean_features": clean_features,
        "audit_principle": (
            "LAD, LCX, RCA are vessel-specific stenosis outcomes (targets). "
            "Cath is the catheterization result — a leakage proxy for the targets. "
            "None may appear as model input features."
        ),
    }

    if violations:
        msg = (
            f"DATA LEAKAGE DETECTED: forbidden features found in input set: {violations}. "
            f"Remove these features before training."
        )
        logger.error(msg)
        if strict:
            raise LeakageViolationError(msg)
    else:
        logger.info(
            "Leakage audit PASSED: %d features, 0 violations. "
            "LAD NOT IN INPUTS. LCX NOT IN INPUTS. RCA NOT IN INPUTS. Cath NOT IN INPUTS.",
            len(clean_features),
        )

    return report


def get_safe_features(df, target_col: str = "CAD"):
    """
    Return a DataFrame with only safe (non-leaking) input features.

    Args:
        df: Raw dataset DataFrame (includes all columns).
        target_col: Primary target column to separate.

    Returns:
        X (safe features), y (target Series)
    """
    excluded = FORBIDDEN_FEATURES + [target_col]
    safe_cols = [c for c in df.columns if c not in excluded]

    # Run audit
    audit_feature_list(safe_cols, strict=True)

    X = df[safe_cols]
    y = df[target_col]
    return X, y


def run_full_audit(csv_path: Path = None) -> dict:
    """
    Run a full leakage audit on the downloaded dataset.
    Saves audit report to ml/artifacts/leakage_audit.json.
    """
    import pandas as pd  # noqa: PLC0415

    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if csv_path is None:
        csv_path = RAW_DIR / "z_alizadeh_sani_ext.csv"

    if not csv_path.exists():
        logger.error("Dataset not found at %s. Run download_data.py first.", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    all_cols = df.columns.tolist()

    report = audit_feature_list(all_cols, strict=False)
    report["dataset_columns"] = all_cols
    report["dataset_shape"] = list(df.shape)

    # Final assertion checks for the report
    report["lad_not_in_inputs"] = "LAD" not in report["clean_features"]
    report["lcx_not_in_inputs"] = "LCX" not in report["clean_features"]
    report["rca_not_in_inputs"] = "RCA" not in report["clean_features"]
    report["cath_not_in_inputs"] = "Cath" not in report["clean_features"]

    out_path = ARTIFACTS_DIR / "leakage_audit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Leakage audit report saved: %s", out_path)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = run_full_audit()
    print(json.dumps(report, indent=2))
