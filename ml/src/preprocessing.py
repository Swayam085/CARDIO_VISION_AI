"""
CardioVision AI — Preprocessing Pipeline
==========================================
Builds a reproducible sklearn preprocessing pipeline for the
Extension of Z-Alizadeh Sani Dataset (UCI ID 411).

PIPELINE DESIGN
---------------
- Numeric features: median imputation (safety net) + StandardScaler
- Categorical features: most-frequent imputation + OrdinalEncoder
- ColumnTransformer assembles all transformers
- Pipeline wraps the ColumnTransformer
- Fit only on training data (no preprocessing leakage)
- SMOTE is NOT applied here — must be done inside training folds only

LEAKAGE PREVENTION
------------------
- feature_config.LEAKAGE_COLUMNS are NEVER passed to the pipeline
- assert_no_leakage() is called before pipeline construction
- Targets (Cath, LAD, LCX, RCA) are separated before any fitting

Run:
    python ml/src/preprocessing.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "ml" / "data" / "raw" / "z_alizadeh_sani_ext.csv"
PROCESSED_DIR = ROOT / "ml" / "data" / "processed"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

for d in [PROCESSED_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Import feature config ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_config import (  # noqa: E402
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    PRIMARY_TARGET,
    VESSEL_TARGETS,
    ALL_TARGETS,
    assert_no_leakage,
)


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV dataset."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {RAW_CSV}. "
            "Run inspect_and_eda.py first."
        )
    df = pd.read_csv(RAW_CSV)
    logger.info("Loaded raw data: %s", df.shape)
    return df


def encode_targets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Encode target columns to binary integer format.

    Returns a dict mapping target name -> binary Series (0/1).

    Encoding:
      Cath:  'CAD' -> 1, 'Normal' -> 0  (primary target)
      LAD:   'Stenotic' -> 1, 'Normal' -> 0
      LCX:   'Stenotic' -> 1, 'Normal' -> 0
      RCA:   'Stenotic' -> 1, 'Normal' -> 0
    """
    targets = {}

    # Primary target: Cath
    if PRIMARY_TARGET in df.columns:
        cath_map = {"CAD": 1, "Normal": 0}
        targets[PRIMARY_TARGET] = df[PRIMARY_TARGET].map(cath_map)
        if targets[PRIMARY_TARGET].isnull().any():
            unique_vals = df[PRIMARY_TARGET].unique().tolist()
            raise ValueError(
                f"Unexpected values in '{PRIMARY_TARGET}' column: {unique_vals}. "
                f"Expected 'CAD' or 'Normal'."
            )
        logger.info(
            "Encoded '%s': %s",
            PRIMARY_TARGET,
            targets[PRIMARY_TARGET].value_counts().to_dict(),
        )

    # Vessel targets: LAD, LCX, RCA
    vessel_map = {"Stenotic": 1, "Normal": 0}
    for col in VESSEL_TARGETS:
        if col in df.columns:
            targets[col] = df[col].map(vessel_map)
            if targets[col].isnull().any():
                unique_vals = df[col].unique().tolist()
                raise ValueError(
                    f"Unexpected values in '{col}' column: {unique_vals}. "
                    f"Expected 'Stenotic' or 'Normal'."
                )
            logger.info(
                "Encoded '%s': %s", col, targets[col].value_counts().to_dict()
            )

    return targets


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build the sklearn preprocessing Pipeline.

    Architecture:
      numeric features  -> [MedianImputer] -> [StandardScaler]
      categorical features -> [MostFrequentImputer] -> [OrdinalEncoder]

    Note: The imputers are safety-nets only. UCI documentation states
    no missing values in this dataset.

    Returns:
        sklearn Pipeline with ColumnTransformer
    """
    # Verify no leakage before building pipeline
    assert_no_leakage(NUMERIC_FEATURES)
    assert_no_leakage(CATEGORICAL_FEATURES)
    assert_no_leakage(FEATURE_COLUMNS)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drop any unexpected columns
        verbose_feature_names_out=True,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])

    logger.info(
        "Pipeline built: %d numeric features + %d categorical features = %d total",
        len(NUMERIC_FEATURES),
        len(CATEGORICAL_FEATURES),
        len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
    )
    return pipeline


def get_feature_names_out(pipeline: Pipeline) -> list[str]:
    """Get output feature names from fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def preprocess_and_save(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Run the full preprocessing workflow:
    1. Encode targets
    2. Separate features from targets
    3. Assert no leakage
    4. Train/test split (stratified on primary target)
    5. Fit pipeline on train only
    6. Transform train and test
    7. Save processed artifacts

    IMPORTANT: SMOTE is NOT applied here. Resampling must happen
    inside cross-validation training folds only.

    Returns:
        Dict with preprocessing metadata
    """
    # ── Encode targets ─────────────────────────────────────────────────────────
    logger.info("Encoding target columns...")
    encoded_targets = encode_targets(df)

    # ── Separate features ──────────────────────────────────────────────────────
    X = df[FEATURE_COLUMNS].copy()
    y_primary = encoded_targets[PRIMARY_TARGET]  # Cath -> 0/1

    logger.info("Feature matrix shape: %s", X.shape)
    logger.info("Target shape: %s", y_primary.shape)

    # ── Leakage assertion (MANDATORY) ──────────────────────────────────────────
    assert_no_leakage(list(X.columns))
    logger.info("✓ Leakage assertion PASSED — no target columns in feature matrix")

    # ── Train/test split (stratified) ─────────────────────────────────────────
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y_primary))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y_primary.iloc[train_idx].reset_index(drop=True)
    y_test = y_primary.iloc[test_idx].reset_index(drop=True)

    logger.info(
        "Train split: %d records | Test split: %d records",
        len(X_train),
        len(X_test),
    )
    logger.info(
        "Train target distribution: %s", y_train.value_counts().to_dict()
    )
    logger.info(
        "Test target distribution: %s", y_test.value_counts().to_dict()
    )

    # ── Build and fit pipeline (on TRAIN only) ─────────────────────────────────
    logger.info("Building preprocessing pipeline...")
    pipeline = build_preprocessing_pipeline()

    logger.info("Fitting pipeline on TRAINING data only...")
    pipeline.fit(X_train)

    # ── Transform ──────────────────────────────────────────────────────────────
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    feature_names_out = get_feature_names_out(pipeline)

    # ── Save processed data ────────────────────────────────────────────────────
    # Training set
    train_df = pd.DataFrame(X_train_processed, columns=feature_names_out)
    train_df["target_Cath"] = y_train.values
    # Add vessel targets to train
    for col in VESSEL_TARGETS:
        if col in encoded_targets:
            train_df[f"target_{col}"] = encoded_targets[col].iloc[train_idx].values

    train_path = PROCESSED_DIR / "train.csv"
    train_df.to_csv(train_path, index=False)
    logger.info("Saved train data: %s (%d rows)", train_path, len(train_df))

    # Test set
    test_df = pd.DataFrame(X_test_processed, columns=feature_names_out)
    test_df["target_Cath"] = y_test.values
    # Add vessel targets to test
    for col in VESSEL_TARGETS:
        if col in encoded_targets:
            test_df[f"target_{col}"] = encoded_targets[col].iloc[test_idx].values

    test_path = PROCESSED_DIR / "test.csv"
    test_df.to_csv(test_path, index=False)
    logger.info("Saved test data: %s (%d rows)", test_path, len(test_df))

    # Raw feature names (before processing) for reference
    raw_train_path = PROCESSED_DIR / "train_raw.csv"
    raw_train = X_train.copy()
    raw_train["target_Cath"] = y_train.values
    raw_train.to_csv(raw_train_path, index=False)

    raw_test_path = PROCESSED_DIR / "test_raw.csv"
    raw_test = X_test.copy()
    raw_test["target_Cath"] = y_test.values
    raw_test.to_csv(raw_test_path, index=False)

    # ── Preprocessing metadata ─────────────────────────────────────────────────
    metadata = {
        "preprocessing_date": datetime.now(timezone.utc).isoformat(),
        "dataset": "Extension of Z-Alizadeh Sani (UCI ID 411)",
        "pipeline_steps": [
            "numeric: MedianImputer + StandardScaler",
            "categorical: MostFrequentImputer + OrdinalEncoder",
        ],
        "leakage_columns_excluded": LEAKAGE_COLUMNS,
        "leakage_assertion_passed": True,
        "feature_columns_input": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names_output": feature_names_out,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_fraction": test_size,
        "random_state": random_state,
        "stratified_on": PRIMARY_TARGET,
        "train_target_distribution": y_train.value_counts().to_dict(),
        "test_target_distribution": y_test.value_counts().to_dict(),
        "smote_applied": False,
        "smote_note": (
            "SMOTE was NOT applied here. It must be applied ONLY inside "
            "training folds during cross-validation to prevent leakage."
        ),
        "output_files": {
            "train_processed": str(train_path),
            "test_processed": str(test_path),
            "train_raw": str(raw_train_path),
            "test_raw": str(raw_test_path),
        },
    }

    meta_path = PROCESSED_DIR / "preprocessing_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                k: (int(v) if isinstance(v, np.integer) else v)
                for k, v in metadata.items()
            },
            f,
            indent=2,
        )
    logger.info("Preprocessing metadata saved: %s", meta_path)

    return metadata


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("CardioVision AI — Preprocessing Pipeline")
    logger.info("=" * 60)

    df = load_raw_data()
    metadata = preprocess_and_save(df)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Train records:   {metadata['train_size']}")
    print(f"Test records:    {metadata['test_size']}")
    print(f"Input features:  {len(metadata['feature_columns_input'])}")
    print(f"Output features: {len(metadata['feature_names_output'])}")
    print(f"Leakage check:   {'PASSED' if metadata['leakage_assertion_passed'] else 'FAILED'}")
    print(f"SMOTE applied:   {metadata['smote_applied']} (correct — use inside folds)")
