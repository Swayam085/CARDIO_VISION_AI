"""
CardioVision AI — Comprehensive ML Dataset Tests
==================================================
Tests for:
  - Dataset loads correctly
  - Expected record count (303)
  - Target columns exist (Cath, LAD, LCX, RCA)
  - Leakage columns excluded from features
  - Preprocessing pipeline works correctly
  - No accidental target leakage
  - Data types handled correctly
  - Encoded targets are binary
  - Feature config assertions work

Run:
    pytest tests/ml/test_dataset_and_preprocessing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
ML_SRC = Path(__file__).resolve().parents[2] / "ml" / "src"
RAW_DATA = Path(__file__).resolve().parents[2] / "ml" / "data" / "raw" / "z_alizadeh_sani_ext.csv"
PROCESSED_DATA = Path(__file__).resolve().parents[2] / "ml" / "data" / "processed"

sys.path.insert(0, str(ML_SRC))

from feature_config import (  # noqa: E402
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
from leakage_audit import audit_feature_list, LeakageViolationError, FORBIDDEN_FEATURES
from preprocessing import build_preprocessing_pipeline, encode_targets, load_raw_data


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load the real raw dataset (required — no synthetic fallback)."""
    if not RAW_DATA.exists():
        pytest.skip(
            f"Raw dataset not found at {RAW_DATA}. "
            "Run inspect_and_eda.py first."
        )
    return pd.read_csv(RAW_DATA)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Load Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetLoad:
    def test_dataset_file_exists(self):
        """CSV file must exist after download."""
        assert RAW_DATA.exists(), (
            f"Dataset CSV not found: {RAW_DATA}. Run inspect_and_eda.py."
        )

    def test_dataset_loads_without_error(self, raw_df):
        """Dataset must load as a non-empty DataFrame."""
        assert isinstance(raw_df, pd.DataFrame)
        assert len(raw_df) > 0, "Dataset loaded but is empty"

    def test_expected_record_count(self, raw_df):
        """Must have exactly 303 patient records (UCI specification)."""
        assert len(raw_df) == 303, (
            f"Expected 303 records, got {len(raw_df)}. "
            "Verify dataset version from UCI ID 411."
        )

    def test_expected_column_count(self, raw_df):
        """Must have exactly 59 columns (UCI specification)."""
        assert len(raw_df.columns) == 59, (
            f"Expected 59 columns, got {len(raw_df.columns)}."
        )

    def test_no_missing_values(self, raw_df):
        """UCI documentation states no missing values."""
        total_missing = raw_df.isnull().sum().sum()
        assert total_missing == 0, (
            f"Found {total_missing} missing values. "
            "UCI documentation states this dataset has no missing values."
        )

    def test_no_duplicate_rows(self, raw_df):
        """No duplicate patient records expected."""
        n_dup = raw_df.duplicated().sum()
        assert n_dup == 0, f"Found {n_dup} duplicate rows."


# ─────────────────────────────────────────────────────────────────────────────
# 2. Target Column Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetColumns:
    def test_primary_target_exists(self, raw_df):
        """Primary target 'Cath' must be in the dataset."""
        assert "Cath" in raw_df.columns, (
            "Column 'Cath' not found. This is the primary CAD target "
            "(catheterization result with values 'CAD'/'Normal')."
        )

    def test_lad_target_exists(self, raw_df):
        """LAD vessel target must exist."""
        assert "LAD" in raw_df.columns

    def test_lcx_target_exists(self, raw_df):
        """LCX vessel target must exist."""
        assert "LCX" in raw_df.columns

    def test_rca_target_exists(self, raw_df):
        """RCA vessel target must exist."""
        assert "RCA" in raw_df.columns

    def test_cath_values_are_binary(self, raw_df):
        """Cath column must contain exactly 'CAD' and 'Normal'."""
        expected = {"CAD", "Normal"}
        actual = set(raw_df["Cath"].unique())
        assert actual == expected, (
            f"Unexpected Cath values: {actual}. Expected {expected}."
        )

    def test_lad_values_are_binary(self, raw_df):
        """LAD column must contain exactly 'Stenotic' and 'Normal'."""
        expected = {"Stenotic", "Normal"}
        actual = set(raw_df["LAD"].unique())
        assert actual == expected, f"Unexpected LAD values: {actual}"

    def test_lcx_values_are_binary(self, raw_df):
        """LCX column must contain exactly 'Stenotic' and 'Normal'."""
        expected = {"Stenotic", "Normal"}
        actual = set(raw_df["LCX"].unique())
        assert actual == expected, f"Unexpected LCX values: {actual}"

    def test_rca_values_are_binary(self, raw_df):
        """RCA column must contain exactly 'Stenotic' and 'Normal'."""
        expected = {"Stenotic", "Normal"}
        actual = set(raw_df["RCA"].unique())
        assert actual == expected, f"Unexpected RCA values: {actual}"

    def test_no_column_named_cad(self, raw_df):
        """The dataset does NOT have a literal 'CAD' column — it's called 'Cath'."""
        assert "CAD" not in raw_df.columns, (
            "Unexpected 'CAD' column found. The primary target is 'Cath'. "
            "This test documents the actual column naming in the dataset."
        )

    def test_cath_cad_count(self, raw_df):
        """Cath='CAD' count should be 216 per EDA inspection."""
        cad_count = (raw_df["Cath"] == "CAD").sum()
        assert cad_count == 216, (
            f"Expected 216 CAD cases, got {cad_count}."
        )

    def test_cath_normal_count(self, raw_df):
        """Cath='Normal' count should be 87 per EDA inspection."""
        normal_count = (raw_df["Cath"] == "Normal").sum()
        assert normal_count == 87, (
            f"Expected 87 Normal cases, got {normal_count}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Leakage Prevention Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakagePrevention:
    """Critical tests ensuring leakage columns never appear in model inputs."""

    def test_lad_not_in_feature_columns(self):
        """LAD must NOT be in FEATURE_COLUMNS."""
        assert "LAD" not in FEATURE_COLUMNS, (
            "CRITICAL: 'LAD' found in FEATURE_COLUMNS — data leakage!"
        )

    def test_lcx_not_in_feature_columns(self):
        """LCX must NOT be in FEATURE_COLUMNS."""
        assert "LCX" not in FEATURE_COLUMNS, (
            "CRITICAL: 'LCX' found in FEATURE_COLUMNS — data leakage!"
        )

    def test_rca_not_in_feature_columns(self):
        """RCA must NOT be in FEATURE_COLUMNS."""
        assert "RCA" not in FEATURE_COLUMNS, (
            "CRITICAL: 'RCA' found in FEATURE_COLUMNS — data leakage!"
        )

    def test_cath_not_in_feature_columns(self):
        """Cath must NOT be in FEATURE_COLUMNS."""
        assert "Cath" not in FEATURE_COLUMNS, (
            "CRITICAL: 'Cath' found in FEATURE_COLUMNS — data leakage!"
        )

    def test_leakage_assertion_passes_for_feature_columns(self):
        """assert_no_leakage must pass for the defined FEATURE_COLUMNS."""
        # This should NOT raise
        assert_no_leakage(FEATURE_COLUMNS)

    def test_leakage_assertion_fails_if_lad_present(self):
        """assert_no_leakage must FAIL (AssertionError) if LAD is included."""
        bad_features = FEATURE_COLUMNS + ["LAD"]
        with pytest.raises(AssertionError, match="LAD"):
            assert_no_leakage(bad_features)

    def test_leakage_assertion_fails_if_lcx_present(self):
        """assert_no_leakage must FAIL if LCX is included."""
        bad_features = FEATURE_COLUMNS + ["LCX"]
        with pytest.raises(AssertionError, match="LCX"):
            assert_no_leakage(bad_features)

    def test_leakage_assertion_fails_if_rca_present(self):
        """assert_no_leakage must FAIL if RCA is included."""
        bad_features = FEATURE_COLUMNS + ["RCA"]
        with pytest.raises(AssertionError, match="RCA"):
            assert_no_leakage(bad_features)

    def test_leakage_assertion_fails_if_cath_present(self):
        """assert_no_leakage must FAIL if Cath is included."""
        bad_features = FEATURE_COLUMNS + ["Cath"]
        with pytest.raises(AssertionError, match="Cath"):
            assert_no_leakage(bad_features)

    def test_get_feature_columns_excludes_leakage(self, raw_df):
        """get_feature_columns must exclude all leakage columns."""
        feature_cols = get_feature_columns(raw_df.columns.tolist())
        for col in LEAKAGE_COLUMNS:
            assert col not in feature_cols, (
                f"LEAKAGE: '{col}' found in get_feature_columns() output!"
            )

    def test_get_feature_columns_count(self, raw_df):
        """get_feature_columns must return 55 columns (59 - 4 leakage)."""
        feature_cols = get_feature_columns(raw_df.columns.tolist())
        assert len(feature_cols) == 55, (
            f"Expected 55 feature columns, got {len(feature_cols)}"
        )

    def test_numeric_features_no_leakage(self):
        """NUMERIC_FEATURES must not contain any leakage column."""
        assert_no_leakage(NUMERIC_FEATURES)

    def test_categorical_features_no_leakage(self):
        """CATEGORICAL_FEATURES must not contain any leakage column."""
        assert_no_leakage(CATEGORICAL_FEATURES)

    # Compatibility tests with leakage_audit.py module
    def test_audit_module_detects_lad_leakage(self):
        """leakage_audit.audit_feature_list must catch LAD in inputs."""
        with pytest.raises(LeakageViolationError):
            audit_feature_list(["Age", "LAD", "BP"], strict=True)

    def test_audit_module_detects_lcx_leakage(self):
        with pytest.raises(LeakageViolationError):
            audit_feature_list(["Age", "LCX", "BP"], strict=True)

    def test_audit_module_detects_rca_leakage(self):
        with pytest.raises(LeakageViolationError):
            audit_feature_list(["Age", "RCA", "BP"], strict=True)

    def test_audit_module_detects_cath_leakage(self):
        with pytest.raises(LeakageViolationError):
            audit_feature_list(["Age", "Cath", "BP"], strict=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Target Encoding Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetEncoding:
    def test_cath_encoded_binary(self, raw_df):
        """Encoded Cath must be 0 or 1 only."""
        encoded = encode_targets(raw_df)
        cath_vals = set(encoded["Cath"].unique())
        assert cath_vals == {0, 1}, f"Cath encoding unexpected: {cath_vals}"

    def test_lad_encoded_binary(self, raw_df):
        """Encoded LAD must be 0 or 1 only."""
        encoded = encode_targets(raw_df)
        lad_vals = set(encoded["LAD"].unique())
        assert lad_vals == {0, 1}, f"LAD encoding unexpected: {lad_vals}"

    def test_lcx_encoded_binary(self, raw_df):
        """Encoded LCX must be 0 or 1 only."""
        encoded = encode_targets(raw_df)
        lcx_vals = set(encoded["LCX"].unique())
        assert lcx_vals == {0, 1}, f"LCX encoding unexpected: {lcx_vals}"

    def test_rca_encoded_binary(self, raw_df):
        """Encoded RCA must be 0 or 1 only."""
        encoded = encode_targets(raw_df)
        rca_vals = set(encoded["RCA"].unique())
        assert rca_vals == {0, 1}, f"RCA encoding unexpected: {rca_vals}"

    def test_cath_cad_maps_to_1(self, raw_df):
        """'CAD' in Cath column must encode to 1."""
        encoded = encode_targets(raw_df)
        # Find a row where Cath='CAD' and verify encoding
        cad_mask = raw_df["Cath"] == "CAD"
        assert (encoded["Cath"][cad_mask] == 1).all(), "'CAD' not mapped to 1"

    def test_cath_normal_maps_to_0(self, raw_df):
        """'Normal' in Cath column must encode to 0."""
        encoded = encode_targets(raw_df)
        normal_mask = raw_df["Cath"] == "Normal"
        assert (encoded["Cath"][normal_mask] == 0).all(), "'Normal' not mapped to 0"

    def test_no_nulls_in_encoded_targets(self, raw_df):
        """Encoded targets must have no NaN values."""
        encoded = encode_targets(raw_df)
        for target_name, target_series in encoded.items():
            assert target_series.isnull().sum() == 0, (
                f"NaN found in encoded target '{target_name}'"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Preprocessing Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingPipeline:
    def test_pipeline_builds_without_error(self):
        """Pipeline must be buildable with no errors."""
        pipeline = build_preprocessing_pipeline()
        assert pipeline is not None

    def test_pipeline_has_preprocessor_step(self):
        """Pipeline must contain a 'preprocessor' step."""
        pipeline = build_preprocessing_pipeline()
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_fits_on_training_data(self, raw_df):
        """Pipeline must fit on the feature matrix without errors."""
        X = raw_df[FEATURE_COLUMNS].copy()
        pipeline = build_preprocessing_pipeline()
        # Should not raise
        pipeline.fit(X)

    def test_pipeline_transforms_correctly(self, raw_df):
        """Transformed data must have the correct shape."""
        X = raw_df[FEATURE_COLUMNS].copy()
        pipeline = build_preprocessing_pipeline()
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)
        expected_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
        assert X_transformed.shape == (len(X), expected_cols), (
            f"Transformed shape {X_transformed.shape} != expected ({len(X)}, {expected_cols})"
        )

    def test_pipeline_no_nan_output(self, raw_df):
        """Transformed output must not contain NaN values."""
        X = raw_df[FEATURE_COLUMNS].copy()
        pipeline = build_preprocessing_pipeline()
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)
        assert not np.isnan(X_transformed).any(), (
            "NaN values found in pipeline output — imputation failed."
        )

    def test_numeric_features_are_scaled(self, raw_df):
        """StandardScaler must produce approximately zero-mean output."""
        X = raw_df[FEATURE_COLUMNS].copy()
        pipeline = build_preprocessing_pipeline()
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)
        # First len(NUMERIC_FEATURES) columns are numeric
        numeric_output = X_transformed[:, : len(NUMERIC_FEATURES)]
        # Mean should be approximately 0 (allowing some tolerance)
        assert abs(numeric_output.mean()) < 0.1, (
            f"Numeric feature mean after scaling: {numeric_output.mean():.4f} "
            "(expected ~0 for StandardScaler)"
        )

    def test_pipeline_does_not_include_leakage_columns(self, raw_df):
        """The pipeline must be constructed without any leakage columns."""
        pipeline = build_preprocessing_pipeline()
        preprocessor = pipeline.named_steps["preprocessor"]
        all_pipeline_features = []
        for _, _, features in preprocessor.transformers:
            all_pipeline_features.extend(features)
        for col in LEAKAGE_COLUMNS:
            assert col not in all_pipeline_features, (
                f"CRITICAL: '{col}' found in pipeline features — DATA LEAKAGE!"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Data Type Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataTypes:
    def test_age_is_numeric(self, raw_df):
        """Age must be a numeric column."""
        assert pd.api.types.is_numeric_dtype(raw_df["Age"]), "Age should be numeric"

    def test_bmi_is_numeric(self, raw_df):
        """BMI must be a numeric column."""
        assert pd.api.types.is_numeric_dtype(raw_df["BMI"]), "BMI should be numeric"

    def test_bp_is_numeric(self, raw_df):
        """BP must be a numeric column."""
        assert pd.api.types.is_numeric_dtype(raw_df["BP"]), "BP should be numeric"

    def test_sex_is_object(self, raw_df):
        """Sex must be a categorical (object or string) column."""
        assert pd.api.types.is_string_dtype(raw_df["Sex"]) or raw_df["Sex"].dtype == object, "Sex should be object or string dtype"

    def test_numeric_features_exist_in_dataset(self, raw_df):
        """All NUMERIC_FEATURES must be present in the dataset."""
        for col in NUMERIC_FEATURES:
            assert col in raw_df.columns, f"Numeric feature '{col}' not in dataset"

    def test_categorical_features_exist_in_dataset(self, raw_df):
        """All CATEGORICAL_FEATURES must be present in the dataset."""
        for col in CATEGORICAL_FEATURES:
            assert col in raw_df.columns, f"Categorical feature '{col}' not in dataset"

    def test_numeric_features_are_numeric_dtype(self, raw_df):
        """NUMERIC_FEATURES must have numeric dtypes in the dataset."""
        non_numeric = [
            col for col in NUMERIC_FEATURES
            if not pd.api.types.is_numeric_dtype(raw_df[col])
        ]
        assert not non_numeric, (
            f"These NUMERIC_FEATURES are not numeric in dataset: {non_numeric}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Feature Config Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureConfig:
    def test_primary_target_is_cath(self):
        """Primary target must be 'Cath' (verified column name)."""
        assert PRIMARY_TARGET == "Cath"

    def test_vessel_targets_complete(self):
        """VESSEL_TARGETS must include LAD, LCX, RCA."""
        assert set(VESSEL_TARGETS) == {"LAD", "LCX", "RCA"}

    def test_leakage_columns_complete(self):
        """LEAKAGE_COLUMNS must include all 4 forbidden columns."""
        assert set(LEAKAGE_COLUMNS) == {"LAD", "LCX", "RCA", "Cath"}

    def test_feature_columns_count_is_55(self):
        """Must have exactly 55 input features (59 total - 4 leakage)."""
        assert len(FEATURE_COLUMNS) == 55, (
            f"Expected 55 feature columns, got {len(FEATURE_COLUMNS)}"
        )

    def test_no_overlap_between_features_and_leakage(self):
        """No column should appear in both FEATURE_COLUMNS and LEAKAGE_COLUMNS."""
        overlap = set(FEATURE_COLUMNS) & set(LEAKAGE_COLUMNS)
        assert not overlap, f"Columns in both features and leakage: {overlap}"

    def test_numeric_plus_categorical_equals_total_features(self):
        """NUMERIC + CATEGORICAL must equal total FEATURE_COLUMNS count."""
        assert len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) == len(FEATURE_COLUMNS), (
            f"NUMERIC ({len(NUMERIC_FEATURES)}) + CATEGORICAL ({len(CATEGORICAL_FEATURES)}) "
            f"= {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)} != "
            f"FEATURE_COLUMNS ({len(FEATURE_COLUMNS)})"
        )
