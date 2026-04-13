"""
classifier.py
-------------
Loads the pre-trained Logistic Regression model and returns
a probability score indicating how adversarial a prompt is.
"""

import os
import joblib
import numpy as np

# Default model path (relative to project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "classifier.pkl")


def load_model():
    """Load the serialized sklearn LogisticRegression classifier."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Please run: python data/dataset_loader.py  to train the model first."
        )
    return joblib.load(MODEL_PATH)


_clf = None  # lazy-loaded singleton


def get_classifier():
    """Return cached classifier instance."""
    global _clf
    if _clf is None:
        _clf = load_model()
    return _clf


def ml_score(embedding: np.ndarray) -> float:
    """
    Predict adversarial probability from an embedding vector.

    Args:
        embedding: np.ndarray of shape (1, 384)

    Returns:
        float in [0, 1] — probability that the prompt is adversarial
    """
    clf = get_classifier()
    # predict_proba returns [[prob_safe, prob_adversarial]]
    proba = clf.predict_proba(embedding)
    return float(proba[0][1])  # index 1 = adversarial class