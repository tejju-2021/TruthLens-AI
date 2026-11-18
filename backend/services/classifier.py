import logging
from typing import Dict, Any, Optional
from pathlib import Path
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.config import settings
from backend.schemas.response import ArticleClassificationResult

logger = logging.getLogger("truthlens.classifier")


class DistilBERTClassifier:
    _instance: Optional["DistilBERTClassifier"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.DISTILBERT_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        self._load_model()

    @classmethod
    def get_instance(cls) -> "DistilBERTClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        try:
            logger.info(f"Loading DistilBERT from {self.model_path} onto {self.device}...")
            if not self.model_path.exists():
                raise FileNotFoundError(f"DistilBERT checkpoint not found at {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("DistilBERT model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model: {e}")
            self._is_loaded = False
            raise

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, text: str) -> ArticleClassificationResult:
        if not self._is_loaded:
            raise RuntimeError("DistilBERT model is not loaded.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        predicted_class = int(torch.argmax(outputs.logits, dim=1).item())

        fake_prob = float(probabilities[0])
        real_prob = float(probabilities[1])

        label = "REAL" if predicted_class == 1 else "FAKE"
        confidence = real_prob if predicted_class == 1 else fake_prob

        return ArticleClassificationResult(
            model_name="DistilBERT (Fine-Tuned ISOT, max_len=256)",
            predicted_label=label,
            confidence=confidence,
            probabilities={"FAKE": fake_prob, "REAL": real_prob},
            explanation=(
                f"DistilBERT transformer classification based on ISOT dataset style patterns. "
                f"Predicts '{label}' with {confidence * 100:.2f}% confidence."
            )
        )

    def predict_simple(self, text: str) -> Dict[str, Any]:
        """Backwards compatibility helper matching original predict() schema."""
        res = self.predict(text)
        return {
            "verdict": res.predicted_label,
            "confidence": res.confidence
        }


class XGBoostClassifier:
    _instance: Optional["XGBoostClassifier"] = None

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None
    ):
        self.model_path = model_path or settings.XGBOOST_MODEL_PATH
        self.vectorizer_path = vectorizer_path or settings.TFIDF_VECTORIZER_PATH
        self.model = None
        self.vectorizer = None
        self._is_loaded = False
        self._load_model()

    @classmethod
    def get_instance(cls) -> "XGBoostClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        try:
            if not self.model_path.exists() or not self.vectorizer_path.exists():
                logger.warning(f"XGBoost model or vectorizer not found ({self.model_path}, {self.vectorizer_path})")
                self._is_loaded = False
                return

            logger.info(f"Loading TF-IDF vectorizer from {self.vectorizer_path}...")
            self.vectorizer = joblib.load(self.vectorizer_path)
            logger.info(f"Loading XGBoost model from {self.model_path}...")
            self.model = joblib.load(self.model_path)
            self._is_loaded = True
            logger.info("XGBoost model and TF-IDF vectorizer loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, text: str) -> ArticleClassificationResult:
        if not self._is_loaded:
            raise RuntimeError("XGBoost model or TF-IDF vectorizer is not loaded.")

        X_tfidf = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(X_tfidf)[0]
        predicted_class = int(np.argmax(probabilities))

        fake_prob = float(probabilities[0])
        real_prob = float(probabilities[1])

        label = "REAL" if predicted_class == 1 else "FAKE"
        confidence = real_prob if predicted_class == 1 else fake_prob

        return ArticleClassificationResult(
            model_name="TF-IDF + XGBoost (ISOT Baseline, 50k n-grams)",
            predicted_label=label,
            confidence=confidence,
            probabilities={"FAKE": fake_prob, "REAL": real_prob},
            explanation=(
                f"XGBoost gradient boosting classifier using TF-IDF n-gram features. "
                f"Predicts '{label}' with {confidence * 100:.2f}% confidence."
            )
        )
