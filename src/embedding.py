"""
embedding.py
------------
Loads the sentence-transformer model and converts input text
into fixed-size embeddings for downstream classification.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# We use a lightweight model — fast enough for real-time inference
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazy-loaded singleton


def get_model() -> SentenceTransformer:
    """Load and cache the embedding model (loads once per session)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    """
    Convert a single string into a 384-dim embedding vector.

    Args:
        text: Raw input prompt string.

    Returns:
        np.ndarray of shape (384,)
    """
    model = get_model()
    embedding = model.encode([text], convert_to_numpy=True)
    return embedding  # shape: (1, 384) — ready for sklearn predict
