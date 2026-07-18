from pathlib import Path
import json

import joblib
import numpy as np
import torch

from tensorflow.keras.models import load_model
from transformers import AutoModel, AutoTokenizer

from backend.feature_engineering import (
    TEXT_COLUMN,
    STRUCTURED_COLUMNS,
    RISK_MAP,
    analyze_urls_simple,
    build_feature_row,
    build_reasons,
)


# =========================================================
# File paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]

MLP_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "mlp_fusion_hybrid_v2_model.keras"
)

FUSION_SCALER_PATH = (
    BASE_DIR
    / "models"
    / "mlp_fusion_scaler.pkl"
)

XGB_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_structured_v2_model.pkl"
)

XGB_PREPROCESSOR_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_structured_preprocessor.pkl"
)

LABEL_ENCODER_PATH = (
    BASE_DIR
    / "models"
    / "hybrid_label_encoder.pkl"
)

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "hybrid"
    / "transformer_embedding_metadata.json"
)


# =========================================================
# Hybrid model service
# =========================================================
class HybridModelService:
    """
    Loads and runs the complete hybrid phishing detection model:

    Transformer embedding
        +
    XGBoost structured probabilities
        +
    Preprocessed structured features
        ->
    MLP fusion classifier
    """

    def __init__(self):
        self.loaded = False

        self.mlp_model = None
        self.fusion_scaler = None

        self.xgb_model = None
        self.xgb_preprocessor = None

        self.label_encoder = None

        self.tokenizer = None
        self.transformer_model = None

        self.transformer_model_name = None
        self.max_length = 128
        self.device = None

    # =====================================================
    # Check required files
    # =====================================================
    def get_missing_files(self) -> list[str]:
        required_files = [
            MLP_MODEL_PATH,
            FUSION_SCALER_PATH,
            XGB_MODEL_PATH,
            XGB_PREPROCESSOR_PATH,
            LABEL_ENCODER_PATH,
            METADATA_PATH,
        ]

        return [
            str(path)
            for path in required_files
            if not path.exists()
        ]

    # =====================================================
    # Load model components
    # =====================================================
    def load(self) -> None:
        if self.loaded:
            return

        missing_files = self.get_missing_files()

        if missing_files:
            formatted_files = "\n".join(missing_files)

            raise FileNotFoundError(
                "The following hybrid model files are missing:\n"
                f"{formatted_files}"
            )

        print("Loading MLP fusion model...")
        self.mlp_model = load_model(
            MLP_MODEL_PATH,
            compile=False
        )

        print("Loading fusion scaler...")
        self.fusion_scaler = joblib.load(
            FUSION_SCALER_PATH
        )

        print("Loading XGBoost model...")
        self.xgb_model = joblib.load(
            XGB_MODEL_PATH
        )

        print("Loading XGBoost preprocessor...")
        self.xgb_preprocessor = joblib.load(
            XGB_PREPROCESSOR_PATH
        )

        print("Loading label encoder...")
        self.label_encoder = joblib.load(
            LABEL_ENCODER_PATH
        )

        print("Loading transformer metadata...")

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as metadata_file:
            metadata = json.load(metadata_file)

        self.transformer_model_name = metadata.get(
            "model_name",
            "distilbert-base-multilingual-cased"
        )

        self.max_length = int(
            metadata.get("max_length", 128)
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading transformer: "
            f"{self.transformer_model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.transformer_model_name
        )

        self.transformer_model = AutoModel.from_pretrained(
            self.transformer_model_name
        )

        self.transformer_model.to(self.device)
        self.transformer_model.eval()

        self.loaded = True

        print("Hybrid model loaded successfully.")
        print("Device:", self.device)
        print("Maximum sequence length:", self.max_length)

    # =====================================================
    # Transformer mean pooling
    # =====================================================
    @staticmethod
    def mean_pooling(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        expanded_mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        summed_embeddings = torch.sum(
            last_hidden_state * expanded_mask,
            dim=1
        )

        summed_mask = torch.clamp(
            expanded_mask.sum(dim=1),
            min=1e-9
        )

        return summed_embeddings / summed_mask

    # =====================================================
    # Create transformer embedding
    # =====================================================
    def extract_transformer_embedding(
        self,
        text: str
    ) -> np.ndarray:
        if not self.loaded:
            self.load()

        text = str(text or "")

        with torch.no_grad():
            encoded = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {
                key: tensor.to(self.device)
                for key, tensor in encoded.items()
            }

            outputs = self.transformer_model(
                **encoded
            )

            embedding = self.mean_pooling(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

        return (
            embedding
            .cpu()
            .numpy()
            .astype("float32")
        )

    # =====================================================
    # Convert sparse matrix to dense float32
    # =====================================================
    @staticmethod
    def to_dense_float32(matrix) -> np.ndarray:
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()

        return np.asarray(
            matrix,
            dtype="float32"
        )

    # =====================================================
    # Validate fusion input size
    # =====================================================
    def validate_fusion_shape(
        self,
        fusion_features: np.ndarray
    ) -> None:
        expected_features = getattr(
            self.fusion_scaler,
            "n_features_in_",
            None
        )

        if (
            expected_features is not None
            and fusion_features.shape[1]
            != expected_features
        ):
            raise ValueError(
                "Fusion input feature mismatch. "
                f"Expected {expected_features} features, "
                f"but received "
                f"{fusion_features.shape[1]} features."
            )

    # =====================================================
    # Main prediction function
    # =====================================================
    def predict(
        self,
        subject: str,
        body: str
    ) -> dict:
        if not self.loaded:
            self.load()

        subject = str(subject or "")
        body = str(body or "")

        if not subject.strip() and not body.strip():
            raise ValueError(
                "Email subject and body cannot both be empty."
            )

        # ---------------------------------------------
        # Step 1: Feature engineering
        # ---------------------------------------------
        features = build_feature_row(
            subject,
            body
        )

        clean_email_text = str(
            features.iloc[0][TEXT_COLUMN]
        )

        # ---------------------------------------------
        # Step 2: Transformer embedding
        # ---------------------------------------------
        transformer_embedding = (
            self.extract_transformer_embedding(
                clean_email_text
            )
        )

        # ---------------------------------------------
        # Step 3: Structured feature preprocessing
        # ---------------------------------------------
        structured_raw = features[
            STRUCTURED_COLUMNS
        ]

        structured_features = (
            self.xgb_preprocessor.transform(
                structured_raw
            )
        )

        structured_features = (
            self.to_dense_float32(
                structured_features
            )
        )

        # ---------------------------------------------
        # Step 4: XGBoost probabilities
        # ---------------------------------------------
        xgb_probabilities = (
            self.xgb_model.predict_proba(
                structured_features
            )
            .astype("float32")
        )

        # ---------------------------------------------
        # Step 5: Build fusion input
        # ---------------------------------------------
        fusion_features = np.hstack(
            [
                transformer_embedding,
                xgb_probabilities,
                structured_features,
            ]
        ).astype("float32")

        self.validate_fusion_shape(
            fusion_features
        )

        fusion_features_scaled = (
            self.fusion_scaler.transform(
                fusion_features
            )
            .astype("float32")
        )

        # ---------------------------------------------
        # Step 6: MLP fusion prediction
        # ---------------------------------------------
        fusion_probabilities = (
            self.mlp_model.predict(
                fusion_features_scaled,
                verbose=0
            )[0]
        )

        predicted_id = int(
            np.argmax(fusion_probabilities)
        )

        prediction = (
            self.label_encoder.inverse_transform(
                [predicted_id]
            )[0]
        )

        class_names = list(
            self.label_encoder.classes_
        )

        confidence = float(
            fusion_probabilities[predicted_id]
        )

        probability_dictionary = {
            class_name: round(
                float(probability),
                4
            )
            for class_name, probability
            in zip(
                class_names,
                fusion_probabilities
            )
        }

        # ---------------------------------------------
        # Step 7: XGBoost branch result
        # ---------------------------------------------
        xgb_predicted_id = int(
            np.argmax(xgb_probabilities[0])
        )

        xgb_prediction = (
            self.label_encoder.inverse_transform(
                [xgb_predicted_id]
            )[0]
        )

        xgb_confidence = float(
            xgb_probabilities[0][
                xgb_predicted_id
            ]
        )

        # ---------------------------------------------
        # Step 8: URL analysis and explanations
        # ---------------------------------------------
        complete_email_text = (
            f"{subject} {body}"
        ).strip()

        url_report = analyze_urls_simple(
            complete_email_text
        )

        reasons = build_reasons(
            features=features,
            url_report=url_report,
            prediction=prediction,
            confidence=confidence,
            xgb_prediction=xgb_prediction,
            xgb_confidence=xgb_confidence,
        )

        risk = RISK_MAP.get(
            prediction,
            "High"
        )

        feature_dictionary = (
            features.iloc[0].to_dict()
        )

        # Convert NumPy values to normal Python values
        serializable_features = {}

        for key, value in feature_dictionary.items():
            if isinstance(value, np.integer):
                serializable_features[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_features[key] = float(value)
            else:
                serializable_features[key] = value

        # ---------------------------------------------
        # Final API response
        # ---------------------------------------------
        return {
            "prediction": str(prediction),
            "confidence": round(
                confidence,
                4
            ),
            "risk": risk,
            "detected_language": str(
                feature_dictionary.get(
                    "detected_language",
                    "Unknown"
                )
            ),
            "probabilities": probability_dictionary,
            "xgboost_prediction": str(
                xgb_prediction
            ),
            "xgboost_confidence": round(
                xgb_confidence,
                4
            ),
            "reasons": reasons,
            "url_report": url_report,
            "features": serializable_features,
            "model": {
                "architecture": (
                    "Transformer + XGBoost "
                    "+ MLP Fusion"
                ),
                "transformer": (
                    self.transformer_model_name
                ),
                "maximum_length": (
                    self.max_length
                ),
                "device": str(self.device),
                "fusion_input_shape": int(
                    fusion_features.shape[1]
                ),
            },
        }

model_service = HybridModelService()