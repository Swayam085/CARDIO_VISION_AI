"""
CardioVision AI — Production Explainability Service
===================================================
Generates TreeSHAP feature attributions and contribution rankings.

Key features:
1. Implements TreeSHAP for tree-based ensemble models (XGBoost, RandomForest, LightGBM)
   and fallback SHAP Explainers for linear models.
2. Explains predictions for all 4 targets: CAD, LAD, LCX, RCA.
3. Computes per-feature SHAP attributions, absolute contributions, directions, and 1-indexed ranks.
4. Provides top positive contributors, top negative contributors, and top overall contributors.
5. Strictly labels output as "Model Feature Contribution".
6. Includes mandatory explanation disclaimers asserting SHAP shows model contribution,
   NOT clinical causality or biological mechanism.
7. Robust error handling for SHAP failure without leaking stack traces.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap

from app.schemas.explainability import (
    ExplainResponse,
    FeatureContribution,
)
from app.services.prediction_service import (
    PredictionService,
    get_prediction_service,
)
from app.utils.validation import InputValidationError

logger = logging.getLogger(__name__)


class SHAPComputationError(RuntimeError):
    """Raised when SHAP value generation encounters an unexpected error."""


class ExplainabilityService:
    """
    Production SHAP Explainability Service singleton.
    """

    _instance: Optional["ExplainabilityService"] = None

    def __init__(self, pred_service: Optional[PredictionService] = None) -> None:
        self._pred_service = pred_service or get_prediction_service()
        self._explainers: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "ExplainabilityService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _extract_shap_values_and_base(
        self,
        classifier: Any,
        X_transformed: np.ndarray,
        task_name: str = "default",
    ) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP values and base value for binary classification target class 1.
        Uses TreeExplainer where supported, with cached SamplingExplainer / LinearExplainer fallback.
        """
        # 1. Try TreeExplainer (fastest for native tree models like XGBoost)
        if task_name in self._explainers:
            explainer = self._explainers[task_name]
            try:
                if isinstance(explainer, shap.SamplingExplainer):
                    raw_shap = explainer.shap_values(X_transformed, nsamples=100, silent=True)
                    base_val = float(np.mean(classifier.predict_proba(np.zeros((5, X_transformed.shape[1])))[:, 1]))
                else:
                    raw_shap = explainer.shap_values(X_transformed)
                    base_val = 0.0
                    if hasattr(explainer, "expected_value"):
                        exp_v = explainer.expected_value
                        if isinstance(exp_v, (list, np.ndarray)):
                            base_val = float(exp_v[1] if len(exp_v) > 1 else exp_v[0])
                        else:
                            base_val = float(exp_v)
                return self._format_shap_array(raw_shap), base_val
            except Exception as exc:
                logger.warning("Cached explainer for %s failed (%s), re-initializing.", task_name, exc)

        try:
            explainer = shap.TreeExplainer(classifier)
            raw_shap = explainer.shap_values(X_transformed)
            self._explainers[task_name] = explainer

            base_val = 0.0
            if hasattr(explainer, "expected_value"):
                exp_v = explainer.expected_value
                if isinstance(exp_v, (list, np.ndarray)):
                    base_val = float(exp_v[1] if len(exp_v) > 1 else exp_v[0])
                else:
                    base_val = float(exp_v)

            return self._format_shap_array(raw_shap), base_val

        except Exception as exc:
            logger.info(
                "TreeExplainer not directly supported for %s (%s). Falling back to resilient explainer.",
                type(classifier).__name__,
                exc,
            )
            # 2. Resilient fallback for tree ensembles on platforms where _cext is restricted
            try:
                if hasattr(classifier, "predict_proba"):
                    bg = np.zeros((5, X_transformed.shape[1]))
                    explainer = shap.SamplingExplainer(classifier.predict_proba, bg)
                    self._explainers[task_name] = explainer
                    raw_shap = explainer.shap_values(X_transformed, nsamples=100, silent=True)
                    base_val = float(np.mean(classifier.predict_proba(bg)[:, 1]))
                    return self._format_shap_array(raw_shap), base_val

                if hasattr(classifier, "coef_"):
                    explainer = shap.LinearExplainer(classifier, X_transformed)
                    self._explainers[task_name] = explainer
                    raw_shap = explainer.shap_values(X_transformed)
                    base_val = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0
                    return self._format_shap_array(raw_shap), base_val

                explainer = shap.Explainer(classifier, X_transformed)
                self._explainers[task_name] = explainer
                explanation = explainer(X_transformed)
                values = explanation.values
                base_val = 0.0
                if hasattr(explanation, "base_values"):
                    bv = explanation.base_values[0]
                    base_val = float(bv[1] if isinstance(bv, (list, np.ndarray)) and len(bv) > 1 else bv[0] if isinstance(bv, (list, np.ndarray)) else bv)
                return self._format_shap_array(values), base_val

            except Exception as fallback_exc:
                logger.error("SHAP explanation failed for classifier %s: %s", type(classifier).__name__, fallback_exc)
                raise SHAPComputationError(
                    f"Failed to generate SHAP feature attributions for model '{type(classifier).__name__}'."
                ) from fallback_exc

    @staticmethod
    def _format_shap_array(raw_shap: Any) -> np.ndarray:
        """Format raw SHAP outputs from various explainers into a 1D array."""
        if isinstance(raw_shap, list):
            # Binary classification list: [class_0, class_1]
            return np.array(raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0])
        elif hasattr(raw_shap, "ndim"):
            if raw_shap.ndim == 3:
                # Shape: (n_samples, n_features, n_classes)
                return np.array(raw_shap[0, :, 1] if raw_shap.shape[2] > 1 else raw_shap[0, :, 0])
            elif raw_shap.ndim == 2:
                # Shape: (n_samples, n_features)
                return np.array(raw_shap[0])
            else:
                return np.array(raw_shap).flatten()
        else:
            return np.array(raw_shap).flatten()

    def explain(
        self,
        features: Dict[str, Any],
        target: str = "CAD",
        top_n: int = 10,
    ) -> ExplainResponse:
        """
        Generate SHAP feature attributions for patient input and specified target.

        Args:
            features: Patient feature dict (55 features)
            target: Target task ('CAD', 'LAD', 'LCX', 'RCA')
            top_n: Number of top contributors to return in summaries

        Returns:
            ExplainResponse with structured SHAP feature contributions, ranks, and disclaimers.
        """
        if not self._pred_service.is_ready:
            self._pred_service.load_artifacts()

        task = target.upper()
        if task not in ["CAD", "LAD", "LCX", "RCA"]:
            raise InputValidationError(f"Invalid explanation target: '{target}'. Allowed targets: CAD, LAD, LCX, RCA")

        # 1. Validate input and preprocess to DataFrame
        X_df, cleaned_dict = self._pred_service.validate_and_preprocess(features)

        # 2. Extract pipeline and components
        model_pipeline = self._pred_service._models[task]
        preprocessor = model_pipeline.named_steps["preprocessor"]
        classifier = model_pipeline.named_steps["classifier"]

        # Transform features
        X_transformed = preprocessor.transform(X_df)

        # Get feature names out from preprocessor
        raw_feat_names_out = preprocessor.get_feature_names_out()
        cleaned_feature_names = [
            f.replace("numeric__", "").replace("categorical__", "")
            for f in raw_feat_names_out
        ]

        # 3. Predict probability for this target
        proba = float(model_pipeline.predict_proba(X_df)[0, 1])

        # 4. Compute SHAP values
        shap_array, base_value = self._extract_shap_values_and_base(classifier, X_transformed, task_name=task)

        # 5. Build category mapping from actual dataset metadata
        category_map = {
            "demographic": "Demographic",
            "symptom_and_examination": "Clinical",
            "laboratory": "Laboratory",
            "ecg": "ECG",
            "echo": "Echo",
        }
        feat_to_category: Dict[str, str] = {}
        if hasattr(self._pred_service, "_feature_metadata") and self._pred_service._feature_metadata:
            fg = self._pred_service._feature_metadata.get("feature_groups", {})
            for grp, f_list in fg.items():
                cat_name = category_map.get(grp, grp.capitalize())
                for f_name in f_list:
                    feat_to_category[f_name] = cat_name

        # Fallback categories if metadata not found
        fallback_demo = {"Age", "Weight", "Length", "Sex", "BMI"}
        fallback_echo = {"EF-TTE", "Region RWMA"}
        fallback_ecg = {"Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB", "VHD"}
        fallback_lab = {"FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"}

        # 6. Build per-feature contribution objects
        raw_contributions = []
        for idx, feat_name in enumerate(cleaned_feature_names):
            s_val = float(shap_array[idx]) if idx < len(shap_array) else 0.0
            abs_val = float(abs(s_val))
            patient_val = cleaned_dict.get(feat_name, features.get(feat_name, "N/A"))

            if s_val > 1e-6:
                direction = "positive"
            elif s_val < -1e-6:
                direction = "negative"
            else:
                direction = "neutral"

            # Determine category
            if feat_name in feat_to_category:
                cat = feat_to_category[feat_name]
            elif feat_name in fallback_demo:
                cat = "Demographic"
            elif feat_name in fallback_echo:
                cat = "Echo"
            elif feat_name in fallback_ecg:
                cat = "ECG"
            elif feat_name in fallback_lab:
                cat = "Laboratory"
            else:
                cat = "Clinical"

            raw_contributions.append({
                "feature": feat_name,
                "patient_value": patient_val,
                "shap_value": round(s_val, 6),
                "absolute_contribution": round(abs_val, 6),
                "direction": direction,
                "positive_negative_contribution": direction,
                "feature_category": cat,
            })

        # 7. Rank all features by absolute contribution descending
        sorted_by_abs = sorted(raw_contributions, key=lambda x: x["absolute_contribution"], reverse=True)
        all_contributions: List[FeatureContribution] = []

        cat_sums: Dict[str, float] = {
            "Demographic": 0.0,
            "Clinical": 0.0,
            "Laboratory": 0.0,
            "ECG": 0.0,
            "Echo": 0.0,
        }

        for rank, item in enumerate(sorted_by_abs, start=1):
            item["rank"] = rank
            feat_contrib = FeatureContribution(**item)
            all_contributions.append(feat_contrib)
            cat_sums[feat_contrib.feature_category] = round(
                cat_sums.get(feat_contrib.feature_category, 0.0) + feat_contrib.absolute_contribution, 6
            )

        # 8. Compute Top Summaries
        # Top Positive (SHAP > 0, highest positive first)
        pos_items = [item for item in all_contributions if item.shap_value > 0]
        pos_sorted = sorted(pos_items, key=lambda x: x.shap_value, reverse=True)[:top_n]

        # Top Negative (SHAP < 0, most negative first)
        neg_items = [item for item in all_contributions if item.shap_value < 0]
        neg_sorted = sorted(neg_items, key=lambda x: x.shap_value)[:top_n]

        # Top Overall
        top_overall = all_contributions[:top_n]

        return ExplainResponse(
            target=task,
            label="Model Feature Contribution",
            model_predicted_probability=round(proba, 6),
            base_value=round(base_value, 6),
            top_positive_contributors=pos_sorted,
            top_negative_contributors=neg_sorted,
            top_contributors_overall=top_overall,
            all_contributions=all_contributions,
            category_contributions=cat_sums,
        )


def get_explainability_service() -> ExplainabilityService:
    """FastAPI dependency for ExplainabilityService."""
    return ExplainabilityService.get_instance()
