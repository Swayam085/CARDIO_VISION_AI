"""
CardioVision AI — Input Validation Utility
===========================================
Validates patient feature dicts before inference and SHAP attribution.

Handles:
- Missing features
- Unexpected features
- Data leakage target columns (Cath, LAD, LCX, RCA)
- Invalid categorical values
- NaN / Infinity check
- Wrong data types (e.g. non-numeric strings for numeric features)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple
import numpy as np

# Leakage columns forbidden as model input features
LEAKAGE_COLUMNS: Set[str] = {"Cath", "LAD", "LCX", "RCA"}

# Allowed 55 feature names
EXPECTED_FEATURES: List[str] = [
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

NUMERIC_FEATURES: Set[str] = {
    "Age", "Weight", "Length", "BMI",
    "DM", "HTN", "Current Smoker", "EX-Smoker", "FH",
    "BP", "PR", "Edema", "Typical Chest Pain", "Function Class",
    "Q Wave", "St Elevation", "St Depression", "Tinversion",
    "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR",
    "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT",
    "EF-TTE", "Region RWMA",
}

CATEGORICAL_FEATURES: Set[str] = {
    "Sex",
    "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease",
    "CHF", "DLP",
    "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
    "Dyspnea",
    "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang",
    "LVH", "Poor R Progression", "BBB", "VHD",
}


NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "Age": (1.0, 120.0),
    "Weight": (10.0, 300.0),
    "Length": (50.0, 250.0),
    "BMI": (10.0, 80.0),
    "BP": (40.0, 300.0),
    "PR": (30.0, 250.0),
    "EF-TTE": (0.0, 100.0),
    "FBS": (10.0, 1000.0),
    "CR": (0.1, 30.0),
    "TG": (10.0, 2000.0),
    "LDL": (10.0, 1000.0),
    "HDL": (5.0, 200.0),
    "BUN": (1.0, 200.0),
    "ESR": (0.0, 300.0),
    "HB": (1.0, 25.0),
    "K": (1.0, 15.0),
    "Na": (50.0, 200.0),
    "WBC": (1.0, 100000.0),
    "Lymph": (0.0, 100.0),
    "Neut": (0.0, 100.0),
    "PLT": (1.0, 2000000.0),
}


class DataLeakageError(ValueError):
    """Raised when a forbidden target column is detected in input features."""


class InputValidationError(ValueError):
    """Raised when patient input fails data quality or type checks."""

    def __init__(self, message: str, errors: List[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or [message]


def validate_patient_features(
    features: Dict[str, Any],
    strict: bool = False,
    allow_missing: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate patient features dictionary for data quality, type validity, and leakage.

    Args:
        features: Flat dict of feature_name -> value
        strict: If True, raise InputValidationError on unexpected, missing, or NaN features.
        allow_missing: If True, missing features are imputed with NaN for preprocessor.

    Returns:
        (cleaned_features_dict, warnings_list)

    Raises:
        DataLeakageError: If Cath, LAD, LCX, or RCA are passed.
        InputValidationError: If NaN/inf, invalid types, or invalid values occur.
    """
    if not isinstance(features, dict):
        raise InputValidationError("Input features must be a dictionary.")

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Check Data Leakage
    leakage_found = [col for col in LEAKAGE_COLUMNS if col in features]
    if leakage_found:
        raise DataLeakageError(
            f"DATA LEAKAGE DETECTED: Features {leakage_found} are prediction targets "
            "and must NEVER be provided as input features."
        )

    # 2. Check Unexpected Features
    expected_set = set(EXPECTED_FEATURES)
    unexpected = [k for k in features.keys() if k not in expected_set]
    if unexpected:
        msg = f"Unexpected feature(s) provided: {unexpected}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    # 3. Check Missing Features
    missing = [k for k in EXPECTED_FEATURES if k not in features]
    if missing:
        msg = f"Missing required feature(s): {missing}"
        if strict and not allow_missing:
            errors.append(msg)
        else:
            warnings.append(f"Missing feature(s) will be imputed: {missing}")

    # 4. Check Values and Types
    cleaned: Dict[str, Any] = {}

    for col in EXPECTED_FEATURES:
        if col not in features:
            cleaned[col] = np.nan
            continue

        val = features[col]

        # Handle None
        if val is None:
            if strict:
                errors.append(f"Missing/null value for numeric feature '{col}' in strict mode.")
            cleaned[col] = np.nan
            continue

        # Check numeric features
        if col in NUMERIC_FEATURES:
            # Check for boolean or complex types passed to numeric
            if isinstance(val, bool):
                num_val = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                if math.isnan(val) or math.isinf(val):
                    errors.append(f"Invalid numeric value for '{col}': {val} (NaN or Infinity)")
                    continue
                num_val = float(val)
            elif isinstance(val, str):
                s_val = val.strip().lower()
                if s_val in ("nan", "none", "null", ""):
                    errors.append(f"Invalid numeric value for '{col}': '{val}' (NaN)")
                    continue
                elif s_val in ("inf", "+inf", "-inf", "infinity", "-infinity"):
                    errors.append(f"Invalid numeric value for '{col}': '{val}' (Infinity)")
                    continue
                else:
                    try:
                        num_val = float(val)
                    except ValueError:
                        errors.append(
                            f"Invalid type/value for numeric feature '{col}': '{val}'. Must be a number."
                        )
                        continue
            else:
                errors.append(f"Invalid data type for numeric feature '{col}': {type(val).__name__}")
                continue

            # Validate numeric ranges if defined
            if col in NUMERIC_RANGES:
                min_v, max_v = NUMERIC_RANGES[col]
                if num_val < min_v or num_val > max_v:
                    errors.append(
                        f"Numeric value for feature '{col}' is out of realistic range [{min_v}, {max_v}]: {num_val}"
                    )
                    continue

            cleaned[col] = num_val

        # Check categorical features
        elif col in CATEGORICAL_FEATURES:
            if isinstance(val, (dict, list)):
                errors.append(f"Invalid data type for categorical feature '{col}': {type(val).__name__}")
                continue

            # Standardize string representations
            if isinstance(val, float) and math.isnan(val):
                cleaned[col] = np.nan
            else:
                s_cat = str(val).strip()
                if col == "Sex":
                    if s_cat.upper() in ("MALE", "M", "1", "1.0"):
                        cleaned[col] = "Male"
                    elif s_cat.upper() in ("FEMALE", "F", "0", "0.0"):
                        cleaned[col] = "Female"
                    else:
                        errors.append(f"Invalid categorical value for 'Sex': '{val}'. Expected Male/Female or M/F.")
                        continue
                else:
                    if s_cat.upper() in ("Y", "YES", "1", "1.0", "TRUE"):
                        cleaned[col] = "Y"
                    elif s_cat.upper() in ("N", "NO", "0", "0.0", "FALSE"):
                        cleaned[col] = "N"
                    else:
                        # Allow original string if non-empty, else error
                        if not s_cat:
                            errors.append(f"Empty categorical value for '{col}'.")
                            continue
                        cleaned[col] = s_cat

    if errors:
        raise InputValidationError(
            f"Input validation failed with {len(errors)} error(s).",
            errors=errors,
        )

    return cleaned, warnings
