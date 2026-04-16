"""
Loads the sentence-transformer and embeds prompts.
Added: preprocessing pipeline (deobfuscation) before embedding,
       so the ML model sees clean text even when input is obfuscated.
"""

import re
import base64
import codecs
import unicodedata
from typing import Optional
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

_ZERO_WIDTH_RE = re.compile(
    r'[\u200b\u200c\u200d\u2060\ufeff\u00ad\u034f\u180e]'
)

def _preprocess(text: str) -> str:
    """
    Strip/decode obfuscation layers before embedding so the
    sentence-transformer sees semantically meaningful text.
    """
    # 1. Unicode normalise + zero-width strip
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub('', text)

    # 2. Inline base64 decode (replace tokens with decoded text)
    def _b64_replace(m):
        try:
            return base64.b64decode(m.group() + '==').decode('utf-8', errors='ignore')
        except Exception:
            return m.group()
    text = re.sub(r'[A-Za-z0-9+/]{40,}={0,2}', _b64_replace, text)

    # 3. ROT13 — only if the result contains common adversarial keywords
    rot = codecs.decode(text, 'rot_13')
    triggers = ['ignore', 'jailbreak', 'bypass', 'developer mode', 'dan']
    if any(t in rot.lower() for t in triggers):
        text = rot

    return text.strip()


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    """
    Convert a single string into a 384-dim embedding.
    Input is preprocessed to deobfuscate before embedding.

    Returns np.ndarray of shape (1, 384) — ready for sklearn.
    """
    model = get_model()
    clean = _preprocess(text)
    return model.encode([clean], convert_to_numpy=True)