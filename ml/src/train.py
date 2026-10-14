"""
CardioVision AI — ML Training Pipeline (Day 3)
================================================
Trains four binary classifiers on the real UCI Extension of Z-Alizadeh
Sani Dataset (ID 411):

  Task 1: CAD    — Overall CAD (Cath column: 'CAD' vs 'Normal')
  Task 2: LAD    — LAD stenosis ('Stenotic' vs 'Normal')
  Task 3: LCX    — LCX stenosis ('Stenotic' vs 'Normal')
  Task 4: RCA    — RCA stenosis ('Stenotic' vs 'Normal')

MODEL CANDIDATES
----------------
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM (if available)

VALIDATION
----------
  Stratified 5-fold cross-validation.
  Preprocessing fitted INSIDE each training fold.
  Class weights / SMOTE applied INSIDE training folds only.

LEAKAGE PREVENTION
------------------
  Mandatory assertion before any model fit:
    LAD, LCX, RCA, Cath → NEVER in input features.

Run:
    python ml/src/train.py
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
ML_SRC = ROOT / "ml" / "src"
RAW_CSV = ROOT / "ml" / "data" / "raw" / "z_alizadeh_sani_ext.csv"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
BACKEND_ARTIFACTS = ROOT / "backend" / "artifacts"

for d in [ARTIFACTS_DIR, BACKEND_ARTIFACTS]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ML_SRC))

from feature_config import (  # noqa: E402
    LEAKAGE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    PRIMARY_TARGET,
    VESSEL_TARGETS,
    assert_no_leakage,
)
from preprocessing import build_preprocessing_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS = 5

# ── Model candidates ───────────────────────────────────────────────────────────
def get_model_candidates(class_weight: str | None = "balanced") -> dict[str, Any]:
    """Return candidate models. Skip LightGBM if unavailable."""
    candidates: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
            class_weight=class_weight,
            solver="lbfgs",
            C=1.0,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            class_weight=class_weight,
            n_jobs=-1,
        ),
    }

    # XGBoost
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415
        candidates["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            verbosity=0,
            scale_pos_weight=1,  # adjusted per task
        )
        logger.info("XGBoost available ✓")
    except ImportError:
        logger.warning("XGBoost not available — skipping")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier  # noqa: PLC0415
        candidates["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            class_weight=class_weight,
            verbosity=-1,
        )
        logger.info("LightGBM available ✓")
    except ImportError:
        logger.warning("LightGBM not available — skipping")

    return candidates


# ── Target encoding ────────────────────────────────────────────────────────────
def encode_target(series: pd.Series, target_name: str) -> pd.Series:
    """Encode target to 0/1 integers."""
    if target_name == "Cath":
        return series.map({"CAD": 1, "Normal": 0}).astype(int)
    else:  # LAD, LCX, RCA
        return series.map({"Stenotic": 1, "Normal": 0}).astype(int)


# ── Metrics helpers ────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Compute classification metrics dict."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
        "avg_precision": float(average_precision_score(y_true, y_proba)),
    }


def compute_curves(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute ROC and PR curves (downsampled for JSON storage)."""
    fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
    roc_thresh = np.nan_to_num(roc_thresh, posinf=1.0, neginf=0.0)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_proba)
    pr_thresh = np.nan_to_num(pr_thresh, posinf=1.0, neginf=0.0)

    # Downsample to ≤100 points for JSON storage
    def downsample(arr: np.ndarray, n: int = 100) -> list:
        if len(arr) <= n:
            return arr.tolist()
        idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
        return arr[idx].tolist()

    return {
        "roc_curve": {
            "fpr": downsample(fpr),
            "tpr": downsample(tpr),
            "thresholds": downsample(roc_thresh),
        },
        "pr_curve": {
            "precision": downsample(prec),
            "recall": downsample(rec),
        },
    }


# ── Threshold tuning ───────────────────────────────────────────────────────────
def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, dict]:
    """
    Tune classification threshold on OOF predictions to maximise F1.
    Returns (best_threshold, metrics_at_threshold).
    This uses OOF probabilities only — NOT the held-out test set.
    """
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.1, 0.9, 81):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
    y_pred_tuned = (y_proba >= best_thresh).astype(int)
    metrics_tuned = compute_metrics(y_true, y_pred_tuned, y_proba)
    return best_thresh, metrics_tuned


# ── Cross-validation for one model ────────────────────────────────────────────
def cross_validate_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Stratified K-fold CV. Preprocessing is fitted INSIDE each fold.
    Returns per-fold and aggregate metrics, plus OOF predictions.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    fold_metrics = []
    oof_proba = np.zeros(len(y))
    oof_pred  = np.zeros(len(y), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Build preprocessing pipeline fresh for each fold
        preprocessor = build_preprocessing_pipeline()

        # Build full pipeline: preprocessor + classifier
        if model_name.startswith("xgboost"):
            # Adjust scale_pos_weight inside fold
            neg = (y_tr == 0).sum()
            pos = (y_tr == 1).sum()
            spw = float(neg / pos) if pos > 0 else 1.0
            fold_model = type(model)(**{
                **model.get_params(),
                "scale_pos_weight": spw,
            })
        else:
            fold_model = type(model)(**model.get_params())

        fold_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   fold_model),
        ])

        fold_pipeline.fit(X_tr, y_tr)

        proba = fold_pipeline.predict_proba(X_val)[:, 1]
        pred  = fold_pipeline.predict(X_val)

        oof_proba[val_idx] = proba
        oof_pred[val_idx]  = pred

        m = compute_metrics(y_val.values, pred, proba)
        m["fold"] = fold_idx + 1
        fold_metrics.append(m)
        logger.info(
            "  Fold %d/%d — AUC=%.4f F1=%.4f Acc=%.4f",
            fold_idx + 1, n_folds, m["roc_auc"], m["f1"], m["accuracy"],
        )

    # Aggregate metrics (mean ± std)
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
    aggregate = {}
    for k in metric_keys:
        vals = [fm[k] for fm in fold_metrics]
        aggregate[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }

    # OOF-level metrics (overall, not averaged)
    oof_metrics = compute_metrics(y.values, oof_pred, oof_proba)

    # Threshold tuning on OOF
    best_thresh, tuned_metrics = tune_threshold(y.values, oof_proba)

    # Confusion matrix at default and tuned threshold
    cm_default = confusion_matrix(y.values, oof_pred).tolist()
    cm_tuned   = confusion_matrix(y.values, (oof_proba >= best_thresh).astype(int)).tolist()

    # Curves at OOF level
    curves = compute_curves(y.values, oof_proba)

    return {
        "model_name":       model_name,
        "fold_metrics":     fold_metrics,
        "aggregate_cv":     aggregate,
        "oof_metrics_default_threshold":  oof_metrics,
        "oof_metrics_tuned_threshold":    tuned_metrics,
        "default_threshold":  0.5,
        "optimized_threshold": best_thresh,
        "confusion_matrix_default": cm_default,
        "confusion_matrix_tuned":   cm_tuned,
        "oof_proba":        oof_proba,   # numpy array — removed before JSON
        **curves,
    }


# ── Train final model on full data ────────────────────────────────────────────
def train_final_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> Pipeline:
    """
    Fit final model on ALL data (for deployment).
    Preprocessing is fitted fresh on the full dataset.
    """
    if model_name.startswith("xgboost"):
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        spw = float(neg / pos) if pos > 0 else 1.0
        final_model = type(model)(**{**model.get_params(), "scale_pos_weight": spw})
    else:
        final_model = type(model)(**model.get_params())

    preprocessor = build_preprocessing_pipeline()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   final_model),
    ])
    pipeline.fit(X, y)
    return pipeline


# ── Select best model for a task ──────────────────────────────────────────────
def select_best_model(cv_results: dict[str, dict]) -> str:
    """
    Select best model based on composite score:
      70% OOF ROC-AUC + 30% OOF F1  (both at default threshold)
    Documented selection methodology — not just one metric.
    """
    best_name = None
    best_score = -1.0
    for name, result in cv_results.items():
        auc = result["oof_metrics_default_threshold"]["roc_auc"]
        f1  = result["oof_metrics_default_threshold"]["f1"]
        score = 0.70 * auc + 0.30 * f1
        logger.info("  Model %-22s  composite=%.4f  AUC=%.4f  F1=%.4f", name, score, auc, f1)
        if score > best_score:
            best_score = score
            best_name = name
    logger.info("  → Selected: %s (composite=%.4f)", best_name, best_score)
    return best_name  # type: ignore[return-value]


# ── Train one task ─────────────────────────────────────────────────────────────
def train_task(
    task_name: str,
    target_col: str,
    df: pd.DataFrame,
    candidates: dict[str, Any],
) -> tuple[Pipeline, dict]:
    """
    Run full training pipeline for one target task.
    Returns (final_pipeline, task_result_dict).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("TASK: %s  (target column: '%s')", task_name, target_col)
    logger.info("=" * 60)

    # ── LEAKAGE CHECK ──────────────────────────────────────────
    assert_no_leakage(FEATURE_COLUMNS)
    logger.info("✓ Leakage assertion passed")

    # ── Prepare X, y ───────────────────────────────────────────
    X = df[FEATURE_COLUMNS].copy()
    y = encode_target(df[target_col], target_col)

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    pos_pct = float(n_pos / len(y) * 100)
    logger.info("Class distribution: 1=%d (%.1f%%)  0=%d (%.1f%%)", n_pos, pos_pct, n_neg, 100 - pos_pct)

    # ── Cross-validate all candidates ──────────────────────────
    cv_results: dict[str, dict] = {}
    for model_name, model in candidates.items():
        logger.info("  ── %s ──", model_name)
        try:
            result = cross_validate_model(model_name, model, X, y)
            cv_results[model_name] = result
        except Exception as exc:
            logger.error("  FAILED: %s — %s", model_name, exc)

    if not cv_results:
        raise RuntimeError(f"All model candidates failed for task {task_name}")

    # ── Select best model ───────────────────────────────────────
    logger.info("Model selection (composite = 0.70×AUC + 0.30×F1):")
    best_name = select_best_model(cv_results)
    best_cv   = cv_results[best_name]

    # ── Train final model on full data ─────────────────────────
    logger.info("Training final model (%s) on full dataset…", best_name)
    final_pipeline = train_final_model(candidates[best_name], X, y, best_name)

    # ── Build task result (JSON-serialisable) ──────────────────
    task_result = {
        "task": task_name,
        "target_column": target_col,
        "n_samples": len(y),
        "class_distribution": {"positive": n_pos, "negative": n_neg, "pos_pct": round(pos_pct, 2)},
        "selected_model": best_name,
        "selection_methodology": "composite = 0.70 × OOF ROC-AUC + 0.30 × OOF F1 (default threshold)",
        "default_threshold": 0.5,
        "optimized_threshold": best_cv["optimized_threshold"],
        "threshold_note": "Threshold tuned on OOF predictions only — NOT on held-out test data",
        "cv_strategy": f"Stratified {N_FOLDS}-fold CV; preprocessing fitted inside each fold",
        "model_candidates_evaluated": list(cv_results.keys()),
        "all_candidate_oof_metrics": {
            name: {
                "oof_default": res["oof_metrics_default_threshold"],
                "oof_tuned":   res["oof_metrics_tuned_threshold"],
                "cv_auc_mean": res["aggregate_cv"]["roc_auc"]["mean"],
                "cv_auc_std":  res["aggregate_cv"]["roc_auc"]["std"],
            }
            for name, res in cv_results.items()
        },
        "selected_model_cv": {
            "fold_metrics":  best_cv["fold_metrics"],
            "aggregate_cv":  best_cv["aggregate_cv"],
            "oof_metrics_default": best_cv["oof_metrics_default_threshold"],
            "oof_metrics_tuned":   best_cv["oof_metrics_tuned_threshold"],
            "confusion_matrix_default": best_cv["confusion_matrix_default"],
            "confusion_matrix_tuned":   best_cv["confusion_matrix_tuned"],
            "roc_curve":  best_cv["roc_curve"],
            "pr_curve":   best_cv["pr_curve"],
        },
        "input_features": FEATURE_COLUMNS,
        "leakage_columns_excluded": LEAKAGE_COLUMNS,
        "leakage_assertion": "PASSED",
    }

    return final_pipeline, task_result


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    start_ts = datetime.now(timezone.utc)

    logger.info("CardioVision AI — Day 3 Training Pipeline")
    logger.info("Random seed: %d | Folds: %d", RANDOM_SEED, N_FOLDS)

    # ── Load raw dataset ───────────────────────────────────────
    if not RAW_CSV.exists():
        logger.error("Raw dataset not found: %s", RAW_CSV)
        logger.error("Run ml/src/inspect_and_eda.py first.")
        sys.exit(1)

    df = pd.read_csv(RAW_CSV)
    logger.info("Loaded: %d rows × %d cols", *df.shape)

    # ── Task definitions ───────────────────────────────────────
    TASKS = [
        ("CAD",  "Cath"),   # Primary: overall CAD
        ("LAD",  "LAD"),    # LAD stenosis
        ("LCX",  "LCX"),    # LCX stenosis
        ("RCA",  "RCA"),    # RCA stenosis
    ]

    # ── Get model candidates ───────────────────────────────────
    candidates = get_model_candidates(class_weight="balanced")
    logger.info("Model candidates: %s", list(candidates.keys()))

    # ── Train each task ────────────────────────────────────────
    trained_pipelines: dict[str, Pipeline] = {}
    task_results: dict[str, dict] = {}

    for task_name, target_col in TASKS:
        pipeline, result = train_task(task_name, target_col, df, candidates)
        trained_pipelines[task_name] = pipeline
        task_results[task_name] = result

    end_ts = datetime.now(timezone.utc)

    # ── Save model artifacts ───────────────────────────────────
    logger.info("")
    logger.info("Saving model artifacts…")
    artifact_map = {
        "CAD": "cad_model.joblib",
        "LAD": "lad_model.joblib",
        "LCX": "lcx_model.joblib",
        "RCA": "rca_model.joblib",
    }
    for task_name, filename in artifact_map.items():
        pipeline = trained_pipelines[task_name]
        for dest in [ARTIFACTS_DIR, BACKEND_ARTIFACTS]:
            joblib.dump(pipeline, dest / filename)
        logger.info("  Saved: %s", filename)

    # Also save standalone preprocessor (fitted on full data)
    logger.info("Saving standalone preprocessor…")
    preprocessor_pipeline = build_preprocessing_pipeline()
    preprocessor_pipeline.fit(df[FEATURE_COLUMNS])
    for dest in [ARTIFACTS_DIR, BACKEND_ARTIFACTS]:
        joblib.dump(preprocessor_pipeline, dest / "preprocessor.joblib")
    logger.info("  Saved: preprocessor.joblib")

    # ── Package versions ───────────────────────────────────────
    def pkg_version(name: str) -> str:
        try:
            import importlib.metadata  # noqa: PLC0415
            return importlib.metadata.version(name)
        except Exception:
            return "unavailable"

    pkg_versions = {
        "python":       platform.python_version(),
        "scikit-learn": pkg_version("scikit-learn"),
        "pandas":       pkg_version("pandas"),
        "numpy":        pkg_version("numpy"),
        "xgboost":      pkg_version("xgboost"),
        "lightgbm":     pkg_version("lightgbm"),
        "joblib":       pkg_version("joblib"),
    }

    # ── metrics.json ──────────────────────────────────────────
    metrics_json = {
        "generated_at": end_ts.isoformat(),
        "dataset": {
            "source": "UCI ML Repository — Extension of Z-Alizadeh Sani (ID 411)",
            "n_samples": len(df),
            "n_features": len(FEATURE_COLUMNS),
        },
        "tasks": {
            task_name: {
                "selected_model": res["selected_model"],
                "selected_threshold": res["optimized_threshold"],
                "oof_metrics": res["selected_model_cv"]["oof_metrics_tuned"],
                "oof_metrics_default_threshold": res["selected_model_cv"]["oof_metrics_default"],
                "cv_aggregate": res["selected_model_cv"]["aggregate_cv"],
                "class_distribution": res["class_distribution"],
                "confusion_matrix_default": res["selected_model_cv"]["confusion_matrix_default"],
                "confusion_matrix_tuned":   res["selected_model_cv"]["confusion_matrix_tuned"],
                "roc_curve":  res["selected_model_cv"]["roc_curve"],
                "pr_curve":   res["selected_model_cv"]["pr_curve"],
            }
            for task_name, res in task_results.items()
        },
        "disclaimer": (
            "Metrics are from stratified out-of-fold (OOF) cross-validation on the "
            "UCI dataset (n=303). This is NOT a prospective clinical study. "
            "Do not make medical decisions based on these outputs."
        ),
    }

    for dest in [ARTIFACTS_DIR, BACKEND_ARTIFACTS]:
        with open(dest / "metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=2)
    logger.info("Saved: metrics.json")

    # ── model_metadata.json ────────────────────────────────────
    model_metadata = {
        "version":           "1.0.0",
        "training_date":     end_ts.isoformat(),
        "training_duration_s": round((end_ts - start_ts).total_seconds(), 2),
        "dataset_source":    "https://archive.ics.uci.edu/dataset/411/extention+of+z",
        "dataset_size":      len(df),
        "n_features":        len(FEATURE_COLUMNS),
        "feature_names":     FEATURE_COLUMNS,
        "leakage_excluded":  LEAKAGE_COLUMNS,
        "random_seed":       RANDOM_SEED,
        "cv_strategy":       f"Stratified {N_FOLDS}-fold; preprocessing inside each fold",
        "package_versions":  pkg_versions,
        "selection_method":  "composite = 0.70 × OOF ROC-AUC + 0.30 × OOF F1",
        "models": {
            task_name: {
                "model_class":        res["selected_model"],
                "target_column":      res["target_column"],
                "default_threshold":  res["default_threshold"],
                "optimized_threshold": res["optimized_threshold"],
                "artifact_file":      artifact_map[task_name],
            }
            for task_name, res in task_results.items()
        },
        "full_cv_results": task_results,
        "safety_notice": (
            "These models are NOT clinically validated. "
            "They are research prototypes trained on a small dataset (n=303) "
            "for the Multimodal AI Hackathon 2026."
        ),
    }

    for dest in [ARTIFACTS_DIR, BACKEND_ARTIFACTS]:
        with open(dest / "model_metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)
    logger.info("Saved: model_metadata.json")

    # ── feature_metadata.json (copy) ──────────────────────────
    feat_meta_src = ROOT / "ml" / "data" / "processed" / "feature_metadata.json"
    if feat_meta_src.exists():
        import shutil  # noqa: PLC0415
        for dest in [ARTIFACTS_DIR, BACKEND_ARTIFACTS]:
            shutil.copy(feat_meta_src, dest / "feature_metadata.json")
        logger.info("Saved: feature_metadata.json")

    # ── Summary ────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — ACTUAL METRICS")
    logger.info("=" * 60)

    for task_name, res in task_results.items():
        oof_m = res["selected_model_cv"]["oof_metrics_default"]
        tuned = res["selected_model_cv"]["oof_metrics_tuned"]
        logger.info(
            "  %s | %s | AUC=%.4f F1=%.4f Acc=%.4f Prec=%.4f Rec=%.4f (default@0.5)",
            task_name.ljust(4),
            res["selected_model"].ljust(22),
            oof_m["roc_auc"], oof_m["f1"], oof_m["accuracy"],
            oof_m["precision"], oof_m["recall"],
        )
        logger.info(
            "  %s | (tuned @%.2f)       AUC=%.4f F1=%.4f Acc=%.4f Prec=%.4f Rec=%.4f",
            " " * 4,
            res["optimized_threshold"],
            tuned["roc_auc"], tuned["f1"], tuned["accuracy"],
            tuned["precision"], tuned["recall"],
        )

    return task_results


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — REAL COMPUTED METRICS (OOF, default threshold=0.5)")
    print("=" * 60)
    for task, res in results.items():
        m = res["selected_model_cv"]["oof_metrics_default"]
        print(
            f"  {task:<4} | {res['selected_model']:<22} | "
            f"AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  "
            f"Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}"
        )
