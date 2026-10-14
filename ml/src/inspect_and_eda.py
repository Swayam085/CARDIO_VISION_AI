"""
CardioVision AI — Dataset Inspection & EDA
============================================
Inspects the official UCI Extension of Z-Alizadeh Sani Dataset (ID 411).
Reads from: ml/data/raw/extention of Z-Alizadeh sani dataset.xlsx
Saves EDA report + CSV copy of raw data.

Run:
    python ml/src/inspect_and_eda.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "ml" / "data" / "raw"
PROCESSED_DIR = ROOT / "ml" / "data" / "processed"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

for d in [PROCESSED_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

XLSX_PATH = RAW_DIR / "extention of Z-Alizadeh sani dataset.xlsx"
CSV_PATH = RAW_DIR / "z_alizadeh_sani_ext.csv"

# ── Expected constants ─────────────────────────────────────────────────────────
EXPECTED_RECORDS = 303
EXPECTED_FEATURES_TOTAL = 59  # includes targets

# Leakage targets (must NOT be in model inputs)
LEAKAGE_COLUMNS = ["LAD", "LCX", "RCA", "Cath"]
PRIMARY_TARGET = "CAD"


def load_dataset() -> pd.DataFrame:
    """Load the dataset from the official XLSX file."""
    if not XLSX_PATH.exists():
        logger.error("XLSX not found: %s", XLSX_PATH)
        sys.exit(1)

    logger.info("Loading: %s", XLSX_PATH)
    df = pd.read_excel(XLSX_PATH, engine="openpyxl")
    logger.info("Loaded shape: %s", df.shape)

    # Save CSV copy for convenience
    df.to_csv(CSV_PATH, index=False)
    logger.info("CSV copy saved: %s", CSV_PATH)
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Full dataset inspection."""
    report = {}

    # ── Shape ─────────────────────────────────────────────────────────────────
    report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    report["columns"] = df.columns.tolist()
    report["column_count"] = len(df.columns)

    logger.info("Records: %d | Columns: %d", df.shape[0], df.shape[1])

    # ── Record count validation ────────────────────────────────────────────────
    if df.shape[0] != EXPECTED_RECORDS:
        logger.warning(
            "Record count mismatch: expected %d, got %d", EXPECTED_RECORDS, df.shape[0]
        )
    else:
        logger.info("✓ Record count matches expected: %d", EXPECTED_RECORDS)

    # ── Data types ────────────────────────────────────────────────────────────
    dtypes = df.dtypes.astype(str).to_dict()
    report["dtypes"] = dtypes

    # ── Missing values ─────────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing_values"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing[col] > 0
    }
    total_missing = int(missing.sum())
    report["total_missing_cells"] = total_missing
    if total_missing == 0:
        logger.info("✓ No missing values found (matches UCI documentation)")
    else:
        logger.warning("Missing values found: %d cells", total_missing)

    # ── Duplicate rows ─────────────────────────────────────────────────────────
    n_dup = int(df.duplicated().sum())
    report["duplicate_rows"] = n_dup
    logger.info("Duplicate rows: %d", n_dup)

    # ── Identify column types ─────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    report["numeric_columns"] = numeric_cols
    report["categorical_columns"] = object_cols

    logger.info("Numeric columns: %d", len(numeric_cols))
    logger.info("Categorical (object) columns: %d", len(object_cols))

    # ── Unique values per column ────────────────────────────────────────────────
    unique_counts = {}
    unique_values_sample = {}
    for col in df.columns:
        n_unique = int(df[col].nunique())
        unique_counts[col] = n_unique
        if n_unique <= 20:
            vals = sorted(df[col].dropna().unique().tolist())
            # Convert numpy types to Python native for JSON serialisation
            unique_values_sample[col] = [
                v.item() if hasattr(v, "item") else v for v in vals
            ]

    report["unique_value_counts"] = unique_counts
    report["unique_values_for_low_cardinality_cols"] = unique_values_sample

    # ── Categorize feature groups ─────────────────────────────────────────────
    # Based on UCI documentation: features in 4 groups + vessel targets
    all_cols = df.columns.tolist()
    report["feature_categories"] = categorize_features(all_cols)

    # ── Target identification ──────────────────────────────────────────────────
    targets_present = [c for c in [PRIMARY_TARGET] + LEAKAGE_COLUMNS if c in df.columns]
    targets_missing = [c for c in [PRIMARY_TARGET] + LEAKAGE_COLUMNS if c not in df.columns]
    report["targets_present"] = targets_present
    report["targets_missing"] = targets_missing

    if targets_missing:
        logger.warning("Expected target/leakage columns not found: %s", targets_missing)
    else:
        logger.info("✓ All expected targets present: %s", targets_present)

    # ── Target distribution ─────────────────────────────────────────────────────
    target_distributions = {}
    for tgt in targets_present:
        vc = df[tgt].value_counts().to_dict()
        # Convert numpy types
        vc = {(k.item() if hasattr(k, "item") else k): int(v) for k, v in vc.items()}
        pcts = df[tgt].value_counts(normalize=True).mul(100).round(2).to_dict()
        pcts = {(k.item() if hasattr(k, "item") else k): float(v) for k, v in pcts.items()}
        target_distributions[tgt] = {
            "counts": vc,
            "percentages": pcts,
        }
        logger.info("Target '%s' distribution: %s", tgt, vc)

    report["target_distributions"] = target_distributions

    # ── Class imbalance check ──────────────────────────────────────────────────
    if PRIMARY_TARGET in df.columns:
        cad_counts = df[PRIMARY_TARGET].value_counts()
        n_classes = len(cad_counts)
        if n_classes == 2:
            minority_pct = float(cad_counts.min() / len(df) * 100)
            if minority_pct < 30:
                logger.warning(
                    "Class imbalance detected: minority class = %.1f%%. "
                    "SMOTE/resampling strategy required (inside training folds only).",
                    minority_pct,
                )
            else:
                logger.info("Class balance: minority = %.1f%% (acceptable range)", minority_pct)
        report["class_imbalance_minority_pct"] = round(float(cad_counts.min() / len(df) * 100), 2)

    # ── CAD consistency check ─────────────────────────────────────────────────
    # UCI documentation: CAD=1 when at least one of LAD/LCX/RCA is stenotic (>=50%)
    # We check consistency but do NOT modify the official label
    all_present = all(c in df.columns for c in ["CAD", "LAD", "LCX", "RCA"])
    if all_present:
        computed_cad = ((df["LAD"] == 1) | (df["LCX"] == 1) | (df["RCA"] == 1)).astype(int)
        # Map CAD to 0/1 if it's string-like
        cad_series = df["CAD"]
        if cad_series.dtype == object:
            # Try to infer binary mapping
            unique_cad = cad_series.unique().tolist()
            logger.info("CAD unique values (raw): %s", unique_cad)
        discrepancies = 0
        try:
            # Attempt numeric comparison
            official_cad_binary = cad_series.map(
                lambda x: 1 if str(x).strip().lower() in ["cad", "1", "yes", "true"] else 0
            )
            discrepancies = int((official_cad_binary != computed_cad).sum())
            report["cad_consistency_check"] = {
                "discrepancies": discrepancies,
                "note": (
                    "Consistency check only — official CAD label is retained unchanged. "
                    f"{discrepancies} records where official CAD differs from computed "
                    "(LAD|LCX|RCA). Dataset documentation may define 'stenotic' differently."
                ),
            }
            if discrepancies > 0:
                logger.warning(
                    "CAD consistency: %d discrepancies between official label and "
                    "computed (LAD|LCX|RCA). Official label is KEPT unchanged.",
                    discrepancies,
                )
            else:
                logger.info("✓ CAD label perfectly consistent with LAD|LCX|RCA")
        except Exception as e:
            report["cad_consistency_check"] = {"error": str(e)}

    # ── Numeric summary statistics ─────────────────────────────────────────────
    if numeric_cols:
        desc = df[numeric_cols].describe()
        report["numeric_summary"] = desc.to_dict()

    # ── Suspicious value checks ────────────────────────────────────────────────
    suspicious = []

    # Age check (plausible range 0–120)
    if "Age" in df.columns:
        age_out = df[(df["Age"] < 0) | (df["Age"] > 120)]
        if len(age_out) > 0:
            suspicious.append(f"Age: {len(age_out)} values outside [0,120]")
        else:
            logger.info("✓ Age values in plausible range")

    # Blood pressure check
    for bp_col in ["BP", "Systolic BP", "Diastolic BP"]:
        if bp_col in df.columns:
            bp_out = df[(df[bp_col] < 50) | (df[bp_col] > 300)]
            if len(bp_out) > 0:
                suspicious.append(f"{bp_col}: {len(bp_out)} values outside [50,300]")

    # Cholesterol check
    if "Chol" in df.columns:
        chol_out = df[(df["Chol"] < 50) | (df["Chol"] > 700)]
        if len(chol_out) > 0:
            suspicious.append(f"Chol: {len(chol_out)} values outside [50,700]")

    # BMI check
    if "BMI" in df.columns:
        bmi_out = df[(df["BMI"] < 10) | (df["BMI"] > 70)]
        if len(bmi_out) > 0:
            suspicious.append(f"BMI: {len(bmi_out)} values outside [10,70]")

    # Ejection fraction check
    for ef_col in ["EF-TTE", "EF", "EF_TTE"]:
        if ef_col in df.columns:
            ef_out = df[(df[ef_col] < 0) | (df[ef_col] > 100)]
            if len(ef_out) > 0:
                suspicious.append(f"{ef_col}: {len(ef_out)} values outside [0,100]")

    report["suspicious_values"] = suspicious
    if suspicious:
        logger.warning("Suspicious values detected: %s", suspicious)
    else:
        logger.info("✓ No obviously suspicious values detected")

    # ── Categorical inconsistencies ────────────────────────────────────────────
    cat_issues = {}
    for col in object_cols:
        vals = df[col].dropna().unique()
        # Check for mixed case or whitespace
        stripped = [str(v).strip() for v in vals]
        if len(set(stripped)) < len(set(str(v) for v in vals)):
            cat_issues[col] = "whitespace inconsistency detected"
        # Check for unusual characters
        weird = [v for v in stripped if not str(v).replace(" ", "").replace("-", "").isalnum()]
        if weird:
            cat_issues[col] = f"unusual characters: {weird[:5]}"

    report["categorical_inconsistencies"] = cat_issues
    if cat_issues:
        logger.warning("Categorical issues: %s", cat_issues)
    else:
        logger.info("✓ No categorical inconsistencies detected")

    return report


def categorize_features(columns: list) -> dict:
    """
    Categorize columns into feature groups based on UCI documentation.

    The dataset documentation describes 4 feature groups:
    demographic, symptom/examination, ECG, laboratory/echo
    Plus vessel-specific targets: LAD, LCX, RCA, Cath, CAD
    """
    # Target/leakage columns
    target_related = ["CAD", "LAD", "LCX", "RCA", "Cath"]

    # Based on Z-Alizadeh Sani dataset documentation
    # These are approximate groupings — actual column names verified from the data
    demographic_keywords = ["Age", "Weight", "Length", "BMI", "Sex"]
    ecg_keywords = [
        "ST", "Q Wave", "T inversion", "LVH", "PR", "QRS", "QT", "QTc",
        "VHD", "BBB", "Typical Chest Pain", "Atypical", "Nonanginal",
        "Rate", "Rhythm"
    ]
    lab_echo_keywords = [
        "FBS", "Cr", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na",
        "WBC", "Lymph", "Neut", "PLT", "EF", "Region RWMA", "Wall Motion",
        "Chol", "CRP"
    ]
    symptom_exam_keywords = [
        "HTN", "DM", "Current Smoker", "Smoker", "Obesity", "CVA", "Airway disease",
        "Thyroid", "CHF", "DLP", "BP", "Edema", "Weak Peripheral", "Lung rales",
        "Systolic Murmur", "Diastolic Murmur", "Dyspnea", "Exertional CP",
        "Function Class", "Atypical", "Nonanginal", "Typical", "Exertional",
        "Low Threshold Angina", "Q Wave", "St Elevation", "St Depression",
        "Tinversion", "LVH", "Poor R Progression"
    ]

    categorized = {
        "target_and_leakage": [],
        "demographic": [],
        "symptom_and_examination": [],
        "ecg": [],
        "laboratory_and_echo": [],
        "uncategorized": [],
    }

    for col in columns:
        col_lower = col.lower().strip()

        if any(t.lower() == col_lower for t in target_related):
            categorized["target_and_leakage"].append(col)
            continue

        matched = False

        # Check demographic
        if any(k.lower() in col_lower for k in demographic_keywords):
            categorized["demographic"].append(col)
            matched = True

        # Check ECG
        if not matched:
            ecg_match = any(k.lower() in col_lower for k in ecg_keywords)
            if ecg_match:
                categorized["ecg"].append(col)
                matched = True

        # Check lab/echo
        if not matched:
            lab_match = any(k.lower() in col_lower for k in lab_echo_keywords)
            if lab_match:
                categorized["laboratory_and_echo"].append(col)
                matched = True

        # Check symptom/exam
        if not matched:
            sym_match = any(k.lower() in col_lower for k in symptom_exam_keywords)
            if sym_match:
                categorized["symptom_and_examination"].append(col)
                matched = True

        if not matched:
            categorized["uncategorized"].append(col)

    return categorized


def main():
    logger.info("=" * 60)
    logger.info("CardioVision AI — Dataset Inspection & EDA")
    logger.info("=" * 60)

    # Load
    df = load_dataset()

    # Inspect
    report = inspect_dataset(df)

    # Print column list
    logger.info("\n=== ACTUAL COLUMN NAMES (%d total) ===", len(df.columns))
    for i, col in enumerate(df.columns):
        logger.info("  [%02d] %s  (dtype: %s)", i + 1, col, df[col].dtype)

    # Log categorization
    cats = report["feature_categories"]
    logger.info("\n=== FEATURE CATEGORIZATION ===")
    for grp, cols in cats.items():
        logger.info("  %s (%d): %s", grp, len(cols), cols)

    # Save EDA report
    report_path = ARTIFACTS_DIR / "eda_report.json"
    # Ensure all values are JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(report), f, indent=2, ensure_ascii=False)
    logger.info("\nEDA report saved: %s", report_path)

    return df, report


if __name__ == "__main__":
    df, report = main()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Records:  {report['shape']['rows']}")
    print(f"Columns:  {report['shape']['columns']}")
    print(f"Targets present: {report['targets_present']}")
    print(f"Missing cells:   {report['total_missing_cells']}")
    print(f"Duplicate rows:  {report['duplicate_rows']}")
    if "class_imbalance_minority_pct" in report:
        print(f"CAD minority class: {report['class_imbalance_minority_pct']}%")
