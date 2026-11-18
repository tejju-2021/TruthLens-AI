"""
TruthLens AI Model Loader
Preserves backward compatibility while leveraging the robust DistilBERTClassifier service.
"""
from typing import Dict, Any
from backend.services.classifier import DistilBERTClassifier


def get_classifier():
    return DistilBERTClassifier.get_instance()


def predict(text: str) -> Dict[str, Any]:
    """
    Original prediction function returning {"verdict": "REAL"|"FAKE", "confidence": float}.
    Preserves exact backward compatibility.
    """
    classifier = get_classifier()
    return classifier.predict_simple(text)
