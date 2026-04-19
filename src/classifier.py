"""
classifier.py  (v2 — upgraded)
-------------------------------
ML scoring layer. Loads the trained Logistic Regression model
and returns adversarial probability.

Upgrade from v1:
  - Calibrated probability output (Platt scaling via CalibratedClassifierCV
    is applied during training in dataset_loader.py)
  - Supports both single-text and batch scoring
  - Graceful ModelNotReady exception for cleaner error handling
"""

import os
import joblib
import numpy as np
from typing import Union

# Path resolution works whether called from project root or a subdirectory
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "models", "classifier.pkl"
)


class ModelNotReady(Exception):
    """Raised when the model file doesn't exist yet."""
    pass


_clf = None  # module-level singleton — loaded once per process


def get_classifier():
    """Return the cached classifier, loading it on first call."""
    global _clf
    if _clf is None:
        abs_path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(abs_path):
            raise ModelNotReady(
                f"Model not found at: {abs_path}\n"
                "Run:  python data/dataset_loader.py  to train it first."
            )
        _clf = joblib.load(abs_path)
    return _clf


def ml_score(embedding: np.ndarray) -> float:
    """
    Return adversarial probability for a single embedding.

    Args:
        embedding: np.ndarray of shape (1, 384) from embedding.py

    Returns:
        float [0, 1] — probability of being adversarial
    """
    clf  = get_classifier()
    prob = clf.predict_proba(embedding)
    return float(prob[0][1])


def ml_score_batch(embeddings: np.ndarray) -> np.ndarray:
    """
    Score multiple embeddings at once (more efficient than looping).

    Args:
        embeddings: np.ndarray of shape (N, 384)

    Returns:
        np.ndarray of shape (N,) with adversarial probabilities
    """
    clf  = get_classifier()
    prob = clf.predict_proba(embeddings)
    return prob[:, 1]