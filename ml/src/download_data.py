"""
CardioVision AI — Dataset Download Script
==========================================
Downloads the official UCI dataset (ID 411):
  Extension of Z-Alizadeh Sani Dataset

Source:  https://archive.ics.uci.edu/dataset/411/extention+of+z
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Citation:
  Z-Alizadeh Sani, M., et al. (2022). Extension of Z-Alizadeh Sani Dataset.
  UCI Machine Learning Repository. https://doi.org/10.24432/C5QS3G

Expected:
  - 303 patient records
  - 59 features (demographic, symptom, ECG, laboratory, echo, vessel labels)
  - Target columns: CAD, LAD, LCX, RCA (MUST be excluded from model inputs)

Usage:
  python ml/src/download_data.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "ml" / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROVENANCE_FILE = RAW_DIR / "dataset_provenance.json"

# ── Dataset constants ─────────────────────────────────────────
DATASET_ID = 411
DATASET_NAME = "Extension of Z-Alizadeh Sani Dataset"
DATASET_URL = "https://archive.ics.uci.edu/dataset/411/extention+of+z"
EXPECTED_RECORDS = 303
EXPECTED_FEATURES = 59
LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
CITATION = (
    "Z-Alizadeh Sani, M., et al. (2022). Extension of Z-Alizadeh Sani Dataset. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5QS3G"
)

# Leakage targets — MUST NOT be used as model input features
LEAKAGE_TARGETS = ["LAD", "LCX", "RCA", "Cath"]


def download_dataset() -> Path:
    """
    Download the Z-Alizadeh Sani extension dataset using ucimlrepo.
    Saves as CSV and records provenance metadata.

    Returns path to saved CSV file.
    Raises RuntimeError if download fails — no silent fallback or synthetic data.
    """
    try:
        from ucimlrepo import fetch_ucirepo  # noqa: PLC0415
    except ImportError:
        logger.error("ucimlrepo not installed. Run: pip install ucimlrepo")
        sys.exit(1)

    logger.info("Fetching UCI dataset ID=%d: %s", DATASET_ID, DATASET_NAME)
    logger.info("Source URL: %s", DATASET_URL)

    try:
        dataset = fetch_ucirepo(id=DATASET_ID)
    except Exception as exc:
        logger.error("DOWNLOAD FAILED: %s", exc)
        logger.error(
            "Do NOT substitute another dataset or create synthetic records. "
            "Resolve the download error and retry."
        )
        raise RuntimeError(f"Dataset download failed: {exc}") from exc

    # ── Validate ──────────────────────────────────────────────
    X = dataset.data.features
    y = dataset.data.targets

    import pandas as pd  # noqa: PLC0415
    df = pd.concat([X, y], axis=1)

    logger.info("Downloaded: %d records, %d columns", len(df), len(df.columns))

    # Validate record count
    if len(df) != EXPECTED_RECORDS:
        logger.warning(
            "Record count mismatch: expected %d, got %d. Proceeding but verify dataset version.",
            EXPECTED_RECORDS,
            len(df),
        )

    # Leakage check — verify leakage targets are present as columns (they should be targets)
    for col in LEAKAGE_TARGETS:
        if col not in df.columns:
            logger.warning("Expected leakage target column '%s' not found in dataset.", col)

    # ── Save ──────────────────────────────────────────────────
    out_path = RAW_DIR / "z_alizadeh_sani_ext.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved: %s", out_path)

    # ── Provenance ────────────────────────────────────────────
    provenance = {
        "source": DATASET_URL,
        "dataset_name": DATASET_NAME,
        "uci_dataset_id": DATASET_ID,
        "records": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "expected_records": EXPECTED_RECORDS,
        "expected_features": EXPECTED_FEATURES,
        "license": LICENSE,
        "citation": CITATION,
        "retrieval_date": datetime.now(timezone.utc).isoformat(),
        "leakage_targets_present": [c for c in LEAKAGE_TARGETS if c in df.columns],
        "file": str(out_path),
        "download_method": "ucimlrepo Python library",
    }
    with open(PROVENANCE_FILE, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Provenance written to: %s", PROVENANCE_FILE)

    # ── Dataset variable info ─────────────────────────────────
    if hasattr(dataset, "variables"):
        var_path = RAW_DIR / "dataset_variables.json"
        try:
            dataset.variables.to_json(var_path, orient="records", indent=2)
            logger.info("Variable metadata saved: %s", var_path)
        except Exception as e:
            logger.warning("Could not save variable metadata: %s", e)

    return out_path


if __name__ == "__main__":
    out = download_dataset()
    logger.info("Dataset download complete: %s", out)
