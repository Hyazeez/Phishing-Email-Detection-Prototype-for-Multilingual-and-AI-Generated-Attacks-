from __future__ import annotations

from typing import Any

import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
from scipy import sparse

from backend.feature_engineering import (
    STRUCTURED_COLUMNS,
    build_feature_row,
)
from backend.model_service import model_service


class ExplainabilityService:
    """
    Provides:
    1. LIME word-level explanations for the complete hybrid pipeline.
    2. SHAP feature-level explanations for the XGBoost structured branch.
    """

    def __init__(self) -> None:
        self.model_service = model_service
        self._tree_explainer: shap.TreeExplainer | None = None

    def _predict_probability_matrix(
        self,
        candidate_texts: list[str],
        class_names: list[str],
        fixed_subject: str,
        fixed_body: str,
        explain_field: str,
    ) -> np.ndarray:
        """
        Called by LIME.

        Every perturbed text sample is sent through the complete hybrid
        prediction pipeline.
        """
        probability_rows: list[list[float]] = []

        for candidate_text in candidate_texts:
            if explain_field == "body":
                result = self.model_service.predict(
                    subject=fixed_subject,
                    body=candidate_text,
                )
            else:
                result = self.model_service.predict(
                    subject=candidate_text,
                    body=fixed_body,
                )

            probability_dictionary = result["probabilities"]

            probability_rows.append(
                [
                    float(probability_dictionary[class_name])
                    for class_name in class_names
                ]
            )

        return np.asarray(
            probability_rows,
            dtype=np.float32,
        )

    def explain_lime(
        self,
        subject: str,
        body: str,
        num_features: int = 10,
        num_samples: int = 100,
    ) -> dict[str, Any]:
        """
        Generate a local word-level explanation for the final hybrid model.
        """

        original_result = self.model_service.predict(
            subject=subject,
            body=body,
        )

        class_names = list(
            original_result["probabilities"].keys()
        )

        predicted_class = str(
            original_result["prediction"]
        )

        predicted_index = class_names.index(
            predicted_class
        )

        # Explain the body when available. Otherwise explain the subject.
        if body.strip():
            text_to_explain = body
            explain_field = "body"
        else:
            text_to_explain = subject
            explain_field = "subject"

        if not text_to_explain.strip():
            raise ValueError(
                "The subject and body cannot both be empty."
            )

        lime_explainer = LimeTextExplainer(
            class_names=class_names,
            random_state=42,
        )

        def classifier_function(
            candidate_texts: list[str],
        ) -> np.ndarray:
            return self._predict_probability_matrix(
                candidate_texts=list(candidate_texts),
                class_names=class_names,
                fixed_subject=subject,
                fixed_body=body,
                explain_field=explain_field,
            )

        explanation = lime_explainer.explain_instance(
            text_instance=text_to_explain,
            classifier_fn=classifier_function,
            labels=[predicted_index],
            num_features=max(1, min(num_features, 20)),
            num_samples=max(50, min(num_samples, 500)),
        )

        term_weights = explanation.as_list(
            label=predicted_index
        )

        explanation_items: list[dict[str, Any]] = []

        for term, weight in term_weights:
            numeric_weight = float(weight)

            explanation_items.append(
                {
                    "term": str(term),
                    "weight": round(numeric_weight, 6),
                    "effect": (
                        "Supports prediction"
                        if numeric_weight > 0
                        else "Opposes prediction"
                    ),
                }
            )

        return {
            "method": "LIME Text Explanation",
            "explained_field": explain_field,
            "predicted_class": predicted_class,
            "prediction_confidence": float(
                original_result["confidence"]
            ),
            "class_names": class_names,
            "local_fidelity_r2": round(
                float(explanation.score),
                4,
            ),
            "num_samples": max(
                50,
                min(num_samples, 500),
            ),
            "terms": explanation_items,
        }

    def _get_tree_explainer(
        self,
    ) -> shap.TreeExplainer:
        if self._tree_explainer is None:
            self._tree_explainer = shap.TreeExplainer(
                self.model_service.xgb_model
            )

        return self._tree_explainer

    @staticmethod
    def _clean_feature_name(
        feature_name: str,
    ) -> str:
        """
        Remove ColumnTransformer prefixes such as num__ and cat__.
        """
        cleaned_name = str(feature_name)

        if "__" in cleaned_name:
            cleaned_name = cleaned_name.split(
                "__",
                maxsplit=1,
            )[1]

        return cleaned_name

    @staticmethod
    def _extract_class_shap_values(
        shap_values: Any,
        predicted_index: int,
    ) -> np.ndarray:
        """
        Supports old and new SHAP multiclass return formats.
        """

        if isinstance(shap_values, list):
            return np.asarray(
                shap_values[predicted_index]
            )[0]

        shap_array = np.asarray(shap_values)

        # New multiclass shape:
        # samples × features × outputs
        if shap_array.ndim == 3:
            return shap_array[
                0,
                :,
                predicted_index,
            ]

        # Single-output or already selected output.
        if shap_array.ndim == 2:
            return shap_array[0]

        raise ValueError(
            f"Unsupported SHAP output shape: "
            f"{shap_array.shape}"
        )

    def explain_structured_shap(
        self,
        subject: str,
        body: str,
        top_features: int = 12,
    ) -> dict[str, Any]:
        """
        Generate SHAP values for the XGBoost structured-feature branch.
        """

        feature_frame = build_feature_row(
            subject=subject,
            body=body,
        )

        structured_frame = feature_frame[
            STRUCTURED_COLUMNS
        ]

        transformed_features = (
            self.model_service.xgb_preprocessor.transform(
                structured_frame
            )
        )

        if sparse.issparse(transformed_features):
            transformed_features = (
                transformed_features.toarray()
            )

        transformed_features = np.asarray(
            transformed_features,
            dtype=np.float32,
        )

        xgb_probabilities = (
            self.model_service.xgb_model.predict_proba(
                transformed_features
            )[0]
        )

        predicted_index = int(
            np.argmax(xgb_probabilities)
        )

        predicted_class = str(
            self.model_service.label_encoder.inverse_transform(
                [predicted_index]
            )[0]
        )

        tree_explainer = self._get_tree_explainer()

        shap_values = tree_explainer.shap_values(
            transformed_features
        )

        class_shap_values = (
            self._extract_class_shap_values(
                shap_values=shap_values,
                predicted_index=predicted_index,
            )
        )

        try:
            feature_names = list(
                self.model_service
                .xgb_preprocessor
                .get_feature_names_out()
            )
        except Exception:
            feature_names = [
                f"feature_{index}"
                for index in range(
                    transformed_features.shape[1]
                )
            ]

        feature_names = [
            self._clean_feature_name(name)
            for name in feature_names
        ]

        feature_values = transformed_features[0]

        feature_explanations: list[dict[str, Any]] = []

        for name, value, shap_value in zip(
            feature_names,
            feature_values,
            class_shap_values,
        ):
            numeric_shap_value = float(shap_value)

            feature_explanations.append(
                {
                    "feature": str(name),
                    "processed_value": round(
                        float(value),
                        6,
                    ),
                    "shap_value": round(
                        numeric_shap_value,
                        6,
                    ),
                    "effect": (
                        "Supports prediction"
                        if numeric_shap_value > 0
                        else "Opposes prediction"
                    ),
                }
            )

        feature_explanations.sort(
            key=lambda item: abs(
                item["shap_value"]
            ),
            reverse=True,
        )

        expected_value = tree_explainer.expected_value

        if np.ndim(expected_value) > 0:
            expected_array = np.asarray(
                expected_value
            ).reshape(-1)

            if predicted_index < len(expected_array):
                selected_expected_value = float(
                    expected_array[predicted_index]
                )
            else:
                selected_expected_value = float(
                    expected_array[0]
                )
        else:
            selected_expected_value = float(
                expected_value
            )

        return {
            "method": "SHAP Tree Explanation",
            "model_branch": "XGBoost structured branch",
            "predicted_class": predicted_class,
            "prediction_confidence": round(
                float(
                    xgb_probabilities[predicted_index]
                ),
                4,
            ),
            "expected_value": round(
                selected_expected_value,
                6,
            ),
            "features": feature_explanations[
                :max(1, min(top_features, 25))
            ],
        }

    def explain(
        self,
        subject: str,
        body: str,
        num_features: int = 10,
        num_samples: int = 100,
    ) -> dict[str, Any]:
        return {
            "lime_text": self.explain_lime(
                subject=subject,
                body=body,
                num_features=num_features,
                num_samples=num_samples,
            ),
            "shap_structured": (
                self.explain_structured_shap(
                    subject=subject,
                    body=body,
                    top_features=12,
                )
            ),
        }


explainability_service = ExplainabilityService()