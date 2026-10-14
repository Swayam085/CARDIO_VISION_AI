"""
CardioVision AI — Feature Configuration
=========================================
Single source of truth for target columns, leakage columns,
and allowed model input columns.

COLUMN NAMES VERIFIED FROM ACTUAL DATASET
------------------------------------------
The official UCI dataset (ID 411) was downloaded and inspected.
Column names are confirmed from the real data — NOT assumed.

KEY FINDING: The dataset does NOT contain a column literally named 'CAD'.
The column 'Cath' (Catheterization result) contains values 'CAD'/'Normal'
and serves as the primary binary target.

TARGETS (dataset columns to predict — never used as inputs)
-----------------------------------------------------------
  Cath  — Primary classification target: 'CAD' or 'Normal'
           (Catheterization result = CAD diagnosis)
  LAD   — Left Anterior Descending stenosis: 'Stenotic' or 'Normal'
  LCX   — Left Circumflex stenosis: 'Stenotic' or 'Normal'
  RCA   — Right Coronary Artery stenosis: 'Stenotic' or 'Normal'

LEAKAGE CONTRACT
----------------
Per UCI dataset documentation (Note 1):
  "To use this dataset only one of the LAD, LCX, RCA or Cath
   (Result of angiography) must be in dataset and the other ones
   must be eliminated for classification."

This means when using Cath as the prediction target, ALL of
LAD, LCX, RCA must be excluded from model inputs (and vice versa).
All four columns are ALWAYS excluded from input features:

  LAD   — Vessel-specific stenosis outcome (target)
  LCX   — Vessel-specific stenosis outcome (target)
  RCA   — Vessel-specific stenosis outcome (target)
  Cath  — Catheterization result (primary target for CAD classification)

INPUT FEATURES (55 columns — verified from dataset)
----------------------------------------------------
All 59 dataset columns EXCEPT the 4 target/leakage columns above.

Usage
-----
  from feature_config import (
      PRIMARY_TARGET,
      VESSEL_TARGETS,
      ALL_TARGETS,
      LEAKAGE_COLUMNS,
      FEATURE_COLUMNS,
      NUMERIC_FEATURES,
      CATEGORICAL_FEATURES,
      assert_no_leakage,
      get_feature_columns,
  )
"""

from __future__ import annotations

from typing import List

# ── Targets (verified from actual dataset) ────────────────────────────────────

# Primary classification target (Catheterization result: 'CAD' or 'Normal')
# NOTE: The column is named 'Cath' in the dataset, NOT 'CAD'
PRIMARY_TARGET: str = "Cath"

# Vessel-specific binary stenosis targets ('Stenotic' or 'Normal')
VESSEL_TARGETS: List[str] = ["LAD", "LCX", "RCA"]

# All model output targets (never used as model inputs)
ALL_TARGETS: List[str] = [PRIMARY_TARGET] + VESSEL_TARGETS  # ['Cath', 'LAD', 'LCX', 'RCA']

# ── Leakage Columns ──────────────────────────────────────────────────────────
# Per UCI Note 1: all four must be excluded when using any one as target
LEAKAGE_COLUMNS: List[str] = ["LAD", "LCX", "RCA", "Cath"]

# ── Verified Input Feature Columns (55 columns) ───────────────────────────────
# Source: actual downloaded dataset inspection
# All 59 dataset columns minus the 4 leakage/target columns

FEATURE_COLUMNS: List[str] = [
    # Demographic (5)
    "Age", "Weight", "Length", "Sex", "BMI",
    # Risk factors / Symptom & Examination (26)
    "DM", "HTN", "Current Smoker", "EX-Smoker", "FH",
    "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease",
    "CHF", "DLP", "BP", "PR", "Edema",
    "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
    "Typical Chest Pain", "Dyspnea", "Function Class",
    "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang",
    # ECG (9)
    "Q Wave", "St Elevation", "St Depression", "Tinversion",
    "LVH", "Poor R Progression", "BBB", "VHD",
    # Laboratory (13)
    "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR",
    "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT",
    # Echo (2)
    "EF-TTE", "Region RWMA",
]

# ── Feature Type Lists (verified from dataset dtypes) ─────────────────────────

NUMERIC_FEATURES: List[str] = [
    "Age", "Weight", "Length", "BMI",
    "DM", "HTN", "Current Smoker", "EX-Smoker", "FH",
    "BP", "PR", "Edema", "Typical Chest Pain", "Function Class",
    "Q Wave", "St Elevation", "St Depression", "Tinversion",
    "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR",
    "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT",
    "EF-TTE", "Region RWMA",
]

CATEGORICAL_FEATURES: List[str] = [
    "Sex",
    "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease",
    "CHF", "DLP",
    "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
    "Dyspnea",
    "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang",
    "LVH", "Poor R Progression", "BBB", "VHD",
]

# ── Feature Category Groupings ────────────────────────────────────────────────

DEMOGRAPHIC_FEATURES: List[str] = [
    "Age", "Weight", "Length", "Sex", "BMI",
]

SYMPTOM_EXAM_FEATURES: List[str] = [
    "DM", "HTN", "Current Smoker", "EX-Smoker", "FH",
    "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease",
    "CHF", "DLP", "BP", "PR", "Edema",
    "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
    "Typical Chest Pain", "Dyspnea", "Function Class",
    "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang",
]

ECG_FEATURES: List[str] = [
    "Q Wave", "St Elevation", "St Depression", "Tinversion",
    "LVH", "Poor R Progression", "BBB", "VHD",
]

LABORATORY_FEATURES: List[str] = [
    "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR",
    "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT",
]

ECHO_FEATURES: List[str] = [
    "EF-TTE", "Region RWMA",
]

# ── Leakage Assertion ─────────────────────────────────────────────────────────


def assert_no_leakage(feature_columns: List[str]) -> None:
    """
    Hard assertion: raise AssertionError if any leakage column appears
    in the proposed feature list.

    These assertions are MANDATORY before any model training.
    Per UCI documentation Note 1: LAD, LCX, RCA, Cath must not all be
    present as features simultaneously — all four are excluded from inputs.

    Args:
        feature_columns: List of column names intended as model inputs.

    Raises:
        AssertionError: If any forbidden column is found in feature_columns.
    """
    # Mandatory individual assertions
    assert "LAD" not in feature_columns, (
        "LEAKAGE VIOLATION: 'LAD' (Left Anterior Descending stenosis — vessel target) "
        "found in feature_columns. Remove it before training."
    )
    assert "LCX" not in feature_columns, (
        "LEAKAGE VIOLATION: 'LCX' (Left Circumflex stenosis — vessel target) "
        "found in feature_columns. Remove it before training."
    )
    assert "RCA" not in feature_columns, (
        "LEAKAGE VIOLATION: 'RCA' (Right Coronary Artery stenosis — vessel target) "
        "found in feature_columns. Remove it before training."
    )
    assert "Cath" not in feature_columns, (
        "LEAKAGE VIOLATION: 'Cath' (Catheterization result — primary CAD target) "
        "found in feature_columns. Remove it before training."
    )

    # Bulk check
    violations = [c for c in feature_columns if c in LEAKAGE_COLUMNS]
    assert not violations, (
        f"LEAKAGE VIOLATION: forbidden columns found in feature_columns: {violations}. "
        f"All of {LEAKAGE_COLUMNS} must be excluded from model inputs."
    )


def get_feature_columns(all_columns: List[str]) -> List[str]:
    """
    Return only the allowed input feature columns — excluding all targets
    and leakage columns.

    Args:
        all_columns: Full list of column names in the dataset.

    Returns:
        Filtered list of safe input feature columns (55 columns expected).

    Raises:
        AssertionError: Via assert_no_leakage if any leakage column slips through.
    """
    excluded = set(LEAKAGE_COLUMNS)  # LAD, LCX, RCA, Cath — all excluded
    feature_cols = [c for c in all_columns if c not in excluded]

    # Hard assertion — must not fail silently
    assert_no_leakage(feature_cols)

    return feature_cols
