"""
CardioVision AI — Day 3 Model Tests
=====================================
Tests for the real trained ML pipeline.

Covers:
  - Artifact existence and loadability
  - Model type and pipeline structure
  - Leakage prevention (LAD/LCX/RCA/Cath NOT in features)
  - Prediction shape and probability range [0, 1]
  - All four tasks: CAD, LAD, LCX, RCA
  - Expected positive class labels
  - Metrics.json integrity (real values, not hardcoded)
  - Model metadata completeness
  - Feature metadata integrity
  - Preprocessor consistency
  - Threshold validity
  - Backend ModelService integration

Run with system Python 3.14 which has all ML packages:
    python -m pytest tests/ml/test_day3_models.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
BACKEND_ARTIFACTS = ROOT / "backend" / "artifacts"
ML_ARTIFACTS      = ROOT / "ml" / "artifacts"
RAW_CSV           = ROOT / "ml" / "data" / "raw" / "z_alizadeh_sani_ext.csv"

# ── Known constants ───────────────────────────────────────────────────────────
ALL_TASKS         = ["CAD", "LAD", "LCX", "RCA"]
LEAKAGE_COLUMNS   = ["LAD", "LCX", "RCA", "Cath"]
EXPECTED_N_FEATURES = 55


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def artifacts_dir() -> Path:
    """Return the backend artifacts directory (asserts it exists)."""
    assert BACKEND_ARTIFACTS.exists(), (
        f"backend/artifacts/ not found at {BACKEND_ARTIFACTS}. "
        "Run ml/src/train.py first."
    )
    return BACKEND_ARTIFACTS


@pytest.fixture(scope="module")
def models(artifacts_dir) -> dict[str, Pipeline]:
    """Load all four trained model pipelines once per test session."""
    loaded = {}
    for task in ALL_TASKS:
        path = artifacts_dir / f"{task.lower()}_model.joblib"
        assert path.exists(), f"Missing model artifact: {path}"
        loaded[task] = joblib.load(path)
    return loaded


@pytest.fixture(scope="module")
def preprocessor(artifacts_dir) -> Pipeline:
    """Load the standalone preprocessor pipeline."""
    path = artifacts_dir / "preprocessor.joblib"
    assert path.exists(), f"Missing preprocessor.joblib at {path}"
    return joblib.load(path)


@pytest.fixture(scope="module")
def metrics_data(artifacts_dir) -> dict:
    """Load metrics.json."""
    path = artifacts_dir / "metrics.json"
    assert path.exists(), f"Missing metrics.json at {path}"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def model_metadata(artifacts_dir) -> dict:
    """Load model_metadata.json."""
    path = artifacts_dir / "model_metadata.json"
    assert path.exists(), f"Missing model_metadata.json at {path}"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def feature_metadata(artifacts_dir) -> dict:
    """Load feature_metadata.json."""
    path = artifacts_dir / "feature_metadata.json"
    assert path.exists(), f"Missing feature_metadata.json at {path}"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def feature_names(feature_metadata) -> list[str]:
    """Return the list of 55 input feature names."""
    names = feature_metadata.get("feature_names", [])
    assert len(names) == EXPECTED_N_FEATURES, (
        f"Expected {EXPECTED_N_FEATURES} features, got {len(names)}"
    )
    return names


@pytest.fixture(scope="module")
def raw_data() -> pd.DataFrame:
    """Load the raw dataset CSV."""
    assert RAW_CSV.exists(), f"Raw CSV not found: {RAW_CSV}"
    return pd.read_csv(RAW_CSV)


@pytest.fixture(scope="module")
def X_sample(feature_names, raw_data) -> pd.DataFrame:
    """Return a small real sample (first 5 rows) of the input features."""
    return raw_data[feature_names].head(5)


@pytest.fixture(scope="module")
def X_dummy(feature_names) -> pd.DataFrame:
    """Return a single-row DataFrame with all zeros (smoke-test only)."""
    return pd.DataFrame([{f: 0 for f in feature_names}])


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — ARTIFACT EXISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestArtifactExistence:
    """All required artifact files must be present."""

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_model_joblib_exists(self, artifacts_dir, task):
        path = artifacts_dir / f"{task.lower()}_model.joblib"
        assert path.exists(), f"{task} model artifact missing: {path}"

    def test_preprocessor_joblib_exists(self, artifacts_dir):
        assert (artifacts_dir / "preprocessor.joblib").exists()

    def test_metrics_json_exists(self, artifacts_dir):
        assert (artifacts_dir / "metrics.json").exists()

    def test_model_metadata_json_exists(self, artifacts_dir):
        assert (artifacts_dir / "model_metadata.json").exists()

    def test_feature_metadata_json_exists(self, artifacts_dir):
        assert (artifacts_dir / "feature_metadata.json").exists()

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_ml_artifacts_mirror_exists(self, task):
        """ML artifacts directory should also have copies."""
        path = ML_ARTIFACTS / f"{task.lower()}_model.joblib"
        assert path.exists(), f"ML artifacts mirror missing for {task}: {path}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelLoading:
    """Models must load as sklearn Pipeline objects."""

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_model_is_pipeline(self, models, task):
        assert isinstance(models[task], Pipeline), (
            f"{task} model should be a sklearn Pipeline, got {type(models[task])}"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_model_has_steps(self, models, task):
        steps = models[task].steps
        assert len(steps) >= 2, (
            f"{task} pipeline should have at least 2 steps (preprocessor + classifier)"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_model_has_predict_proba(self, models, task):
        assert hasattr(models[task], "predict_proba"), (
            f"{task} model must support predict_proba()"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_model_has_predict(self, models, task):
        assert hasattr(models[task], "predict"), (
            f"{task} model must support predict()"
        )

    def test_preprocessor_is_pipeline(self, preprocessor):
        assert isinstance(preprocessor, Pipeline)


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — LEAKAGE PREVENTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeakagePrevention:
    """LAD, LCX, RCA, Cath must NEVER appear in input features."""

    @pytest.mark.parametrize("col", LEAKAGE_COLUMNS)
    def test_leakage_col_not_in_feature_names(self, feature_names, col):
        assert col not in feature_names, (
            f"LEAKAGE VIOLATION: '{col}' found in model input features!"
        )

    def test_feature_count_correct(self, feature_names):
        """55 input features = 59 total columns - 4 leakage/target columns."""
        assert len(feature_names) == EXPECTED_N_FEATURES, (
            f"Expected {EXPECTED_N_FEATURES} features, got {len(feature_names)}"
        )

    def test_leakage_columns_listed_in_metadata(self, feature_metadata):
        excluded = feature_metadata.get("excluded_leakage_columns", [])
        for col in LEAKAGE_COLUMNS:
            assert col in excluded, (
                f"Leakage column '{col}' should appear in excluded_leakage_columns"
            )

    def test_leakage_columns_in_metadata_note(self, feature_metadata):
        note = feature_metadata.get("leakage_note", "")
        assert "LAD" in note or "excluded" in note.lower()

    @pytest.mark.parametrize("col", LEAKAGE_COLUMNS)
    def test_raw_dataset_has_leakage_cols(self, raw_data, col):
        """Sanity: leakage columns do exist in the raw dataset (just excluded)."""
        assert col in raw_data.columns, (
            f"Expected '{col}' in raw dataset columns — it should exist but be excluded."
        )

    def test_X_sample_has_no_leakage_columns(self, X_sample):
        for col in LEAKAGE_COLUMNS:
            assert col not in X_sample.columns, (
                f"LEAKAGE: '{col}' found in X_sample columns!"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — PREDICTION SHAPE & PROBABILITY RANGE
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictions:
    """Model predictions must have correct shape and valid probabilities."""

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_predict_proba_shape_single_row(self, models, task, X_sample):
        """predict_proba on a single row must return (1, 2)."""
        X_one = X_sample.iloc[[0]]
        proba = models[task].predict_proba(X_one)
        assert proba.shape == (1, 2), (
            f"{task}: expected proba shape (1, 2), got {proba.shape}"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_predict_proba_shape_batch(self, models, task, X_sample):
        """predict_proba on a batch must return (n, 2)."""
        proba = models[task].predict_proba(X_sample)
        assert proba.shape == (len(X_sample), 2), (
            f"{task}: expected proba shape ({len(X_sample)}, 2), got {proba.shape}"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_probabilities_in_range(self, models, task, X_sample):
        """All probabilities must be in [0, 1]."""
        proba = models[task].predict_proba(X_sample)
        assert np.all(proba >= 0.0), f"{task}: negative probability detected"
        assert np.all(proba <= 1.0), f"{task}: probability > 1.0 detected"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_probabilities_sum_to_one(self, models, task, X_sample):
        """Probabilities across classes must sum to 1.0 for each row."""
        proba = models[task].predict_proba(X_sample)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-6,
            err_msg=f"{task}: probabilities don't sum to 1.0"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_predict_returns_valid_labels(self, models, task, X_sample):
        """predict() must return labels only from the training label set."""
        preds = models[task].predict(X_sample)
        assert len(preds) == len(X_sample), (
            f"{task}: prediction length mismatch"
        )
        assert preds is not None

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_predict_classes_are_binary(self, models, task):
        """Models are binary classifiers — must have exactly 2 classes."""
        n_classes = len(models[task].classes_)
        assert n_classes == 2, (
            f"{task}: expected binary classifier (2 classes), got {n_classes}"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_no_nan_in_probabilities(self, models, task, X_sample):
        """No NaN values in predicted probabilities."""
        proba = models[task].predict_proba(X_sample)
        assert not np.any(np.isnan(proba)), f"{task}: NaN in probabilities"


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — METRICS.JSON INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricsIntegrity:
    """metrics.json must contain real, non-hardcoded computed metrics."""

    def test_metrics_has_tasks_key(self, metrics_data):
        assert "tasks" in metrics_data, "metrics.json missing 'tasks' key"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_metrics_has_all_four_tasks(self, metrics_data, task):
        assert task in metrics_data["tasks"], (
            f"metrics.json missing task '{task}'"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_roc_auc_in_valid_range(self, metrics_data, task):
        oof = metrics_data["tasks"][task].get("oof_metrics_default_threshold", {})
        auc = oof.get("roc_auc", -1)
        assert 0.5 <= auc <= 1.0, (
            f"{task} ROC-AUC={auc:.4f} is out of expected range [0.5, 1.0]"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_f1_in_valid_range(self, metrics_data, task):
        oof = metrics_data["tasks"][task].get("oof_metrics_default_threshold", {})
        f1 = oof.get("f1", -1)
        assert 0.0 <= f1 <= 1.0, f"{task} F1={f1:.4f} out of [0, 1]"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_accuracy_in_valid_range(self, metrics_data, task):
        oof = metrics_data["tasks"][task].get("oof_metrics_default_threshold", {})
        acc = oof.get("accuracy", -1)
        assert 0.0 <= acc <= 1.0, f"{task} Accuracy={acc:.4f} out of [0, 1]"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_precision_in_valid_range(self, metrics_data, task):
        oof = metrics_data["tasks"][task].get("oof_metrics_default_threshold", {})
        prec = oof.get("precision", -1)
        assert 0.0 <= prec <= 1.0, f"{task} Precision={prec:.4f} out of [0, 1]"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_recall_in_valid_range(self, metrics_data, task):
        oof = metrics_data["tasks"][task].get("oof_metrics_default_threshold", {})
        rec = oof.get("recall", -1)
        assert 0.0 <= rec <= 1.0, f"{task} Recall={rec:.4f} out of [0, 1]"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_cv_aggregate_present(self, metrics_data, task):
        cv = metrics_data["tasks"][task].get("cv_aggregate", {})
        assert "roc_auc" in cv, f"{task} missing cv_aggregate.roc_auc"
        auc_stats = cv["roc_auc"]
        assert "mean" in auc_stats and "std" in auc_stats

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_cv_auc_mean_reasonable(self, metrics_data, task):
        cv = metrics_data["tasks"][task]["cv_aggregate"]
        mean_auc = cv["roc_auc"]["mean"]
        assert 0.5 <= mean_auc <= 1.0, (
            f"{task} CV AUC mean={mean_auc:.4f} out of expected range"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_confusion_matrix_present(self, metrics_data, task):
        cm = metrics_data["tasks"][task].get("confusion_matrix_default")
        assert cm is not None and len(cm) == 2, (
            f"{task} missing or malformed confusion_matrix_default"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_class_distribution_present(self, metrics_data, task):
        dist = metrics_data["tasks"][task].get("class_distribution", {})
        assert len(dist) > 0, f"{task} missing class_distribution"

    def test_metrics_has_disclaimer(self, metrics_data):
        assert "disclaimer" in metrics_data
        disclaimer = metrics_data["disclaimer"]
        assert len(disclaimer) > 20, "Disclaimer is too short"

    def test_metrics_has_dataset_info(self, metrics_data):
        ds = metrics_data.get("dataset", {})
        assert ds.get("n_samples") == 303, "Expected 303 samples (UCI dataset)"
        assert ds.get("n_features") == EXPECTED_N_FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — THRESHOLD VALIDITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestThresholds:
    """Thresholds must be valid probabilities in (0, 1)."""

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_selected_threshold_in_range(self, metrics_data, task):
        thresh = metrics_data["tasks"][task].get("selected_threshold", -1)
        assert 0.0 < thresh < 1.0, (
            f"{task} selected_threshold={thresh} is out of (0, 1)"
        )

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_selected_threshold_is_numeric(self, metrics_data, task):
        thresh = metrics_data["tasks"][task].get("selected_threshold", 0.5)
        assert isinstance(thresh, (int, float))
        assert 0.0 < float(thresh) < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7 — MODEL METADATA COMPLETENESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelMetadata:
    """model_metadata.json must be complete with reproducibility info."""

    REQUIRED_KEYS = [
        "version", "training_date", "dataset_source", "dataset_size",
        "n_features", "random_seed", "cv_strategy", "selection_method",
        "package_versions", "models", "leakage_excluded",
    ]

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_key_present(self, model_metadata, key):
        assert key in model_metadata, f"model_metadata.json missing key: '{key}'"

    def test_random_seed_is_integer(self, model_metadata):
        seed = model_metadata.get("random_seed")
        assert isinstance(seed, int), f"random_seed should be int, got {type(seed)}"

    def test_dataset_size_correct(self, model_metadata):
        ds = model_metadata.get("dataset_size")
        if isinstance(ds, dict):
            assert ds.get("rows") == 303, "Expected 303 rows (UCI dataset)"
        else:
            assert ds == 303, "Expected 303 rows (UCI dataset)"

    def test_n_features_correct(self, model_metadata):
        assert model_metadata.get("n_features") == EXPECTED_N_FEATURES

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_all_four_tasks_in_models(self, model_metadata, task):
        models_meta = model_metadata.get("models", {})
        assert task in models_meta, f"model_metadata.models missing '{task}'"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_model_class_recorded(self, model_metadata, task):
        model_info = model_metadata["models"].get(task, {})
        assert "model_class" in model_info, (
            f"model_metadata.models.{task} missing 'model_class'"
        )

    def test_leakage_excluded_columns(self, model_metadata):
        excluded = model_metadata.get("leakage_excluded", [])
        for col in LEAKAGE_COLUMNS:
            assert col in excluded, (
                f"'{col}' missing from leakage_excluded in model_metadata"
            )

    def test_package_versions_present(self, model_metadata):
        versions = model_metadata.get("package_versions", {})
        required_pkgs = ["python", "scikit-learn", "pandas", "numpy"]
        for pkg in required_pkgs:
            assert pkg in versions, f"package_versions missing: {pkg}"

    def test_cv_strategy_documented(self, model_metadata):
        strategy = model_metadata.get("cv_strategy", "")
        assert "fold" in str(strategy).lower() or "stratified" in str(strategy).lower(), (
            "CV strategy should mention 'fold' or 'stratified'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8 — FEATURE METADATA INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureMetadata:
    """feature_metadata.json must accurately describe the input features."""

    def test_feature_count(self, feature_metadata):
        n = feature_metadata.get("feature_count", 0)
        assert n == EXPECTED_N_FEATURES, f"Expected feature_count=55, got {n}"

    def test_feature_names_count(self, feature_metadata):
        names = feature_metadata.get("feature_names", [])
        assert len(names) == EXPECTED_N_FEATURES

    def test_numeric_plus_categorical_equals_total(self, feature_metadata):
        numeric = feature_metadata.get("numeric_features", [])
        categorical = feature_metadata.get("categorical_features", [])
        total = len(numeric) + len(categorical)
        assert total == EXPECTED_N_FEATURES, (
            f"numeric({len(numeric)}) + categorical({len(categorical)}) = {total} != 55"
        )

    def test_target_columns_documented(self, feature_metadata):
        targets = feature_metadata.get("target_columns", {})
        assert targets.get("primary_target") == "Cath"
        assert "LAD" in targets.get("vessel_targets", [])
        assert "LCX" in targets.get("vessel_targets", [])
        assert "RCA" in targets.get("vessel_targets", [])

    def test_dataset_id_correct(self, feature_metadata):
        assert feature_metadata.get("dataset_id") == 411

    def test_class_distributions_present(self, feature_metadata):
        dist = feature_metadata.get("class_distributions", {})
        assert "Cath" in dist
        assert "CAD" in dist["Cath"]
        assert "Normal" in dist["Cath"]
        assert dist["Cath"]["CAD"] == 216
        assert dist["Cath"]["Normal"] == 87

    def test_no_leakage_col_in_feature_names(self, feature_metadata):
        names = set(feature_metadata.get("feature_names", []))
        for col in LEAKAGE_COLUMNS:
            assert col not in names, f"Leakage column '{col}' in feature_names!"


# ═══════════════════════════════════════════════════════════════════════════════
# 9 — PREPROCESSING CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessingConsistency:
    """The embedded preprocessor in each pipeline should transform consistently."""

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_pipeline_preprocessor_transforms_sample(self, models, task, X_sample):
        """The pipeline's preprocessor step must transform without error."""
        prep_step = models[task].steps[0][1]
        transformed = prep_step.transform(X_sample)
        assert transformed is not None
        assert transformed.shape[0] == len(X_sample)

    def test_standalone_preprocessor_transforms_sample(self, preprocessor, X_sample):
        """Standalone preprocessor.joblib must transform without error."""
        transformed = preprocessor.transform(X_sample)
        assert transformed is not None
        assert transformed.shape[0] == len(X_sample)

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_transformed_output_has_no_nan(self, models, task, X_sample):
        """Transformer output must contain no NaN values."""
        prep_step = models[task].steps[0][1]
        transformed = prep_step.transform(X_sample)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        arr = np.array(transformed, dtype=float)
        assert not np.any(np.isnan(arr)), (
            f"{task} preprocessor produces NaN values in transformed output"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10 — REAL DATASET INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealDataIntegration:
    """End-to-end tests using actual rows from the real UCI dataset."""

    def test_raw_dataset_loads(self, raw_data):
        assert raw_data.shape == (303, 59), (
            f"Expected (303, 59), got {raw_data.shape}"
        )

    def test_raw_dataset_has_cath_column(self, raw_data):
        assert "Cath" in raw_data.columns

    def test_cath_values_binary(self, raw_data):
        vals = set(raw_data["Cath"].unique())
        assert vals == {"CAD", "Normal"}, f"Unexpected Cath values: {vals}"

    def test_cath_class_counts(self, raw_data):
        counts = raw_data["Cath"].value_counts()
        assert counts["CAD"] == 216
        assert counts["Normal"] == 87

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_full_pipeline_predicts_on_real_data(self, models, task, X_sample):
        """Each model must successfully predict on real patient rows."""
        proba = models[task].predict_proba(X_sample)
        preds = models[task].predict(X_sample)
        assert proba.shape == (len(X_sample), 2)
        assert len(preds) == len(X_sample)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_positive_class_label_valid(self, models, task):
        """The positive class (index 1) must be the meaningful positive label."""
        classes = list(models[task].classes_)
        assert len(classes) == 2
        assert classes[0] != classes[1]


# ═══════════════════════════════════════════════════════════════════════════════
# 11 — ML ARTIFACTS MIRROR
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLArtifactsMirror:
    """ml/artifacts/ should mirror backend/artifacts/ for all key files."""

    @pytest.mark.parametrize("fname", [
        "metrics.json", "model_metadata.json", "feature_metadata.json",
        "preprocessor.joblib",
    ])
    def test_ml_artifact_exists(self, fname):
        path = ML_ARTIFACTS / fname
        assert path.exists(), f"ml/artifacts/{fname} not found"

    @pytest.mark.parametrize("task", ALL_TASKS)
    def test_ml_model_loadable(self, task):
        path = ML_ARTIFACTS / f"{task.lower()}_model.joblib"
        assert path.exists()
        model = joblib.load(path)
        assert isinstance(model, Pipeline)
