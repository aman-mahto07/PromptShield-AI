"""
fusion.py
---------
Combines the ML probability score and the rule-based score
into a single final confidence score.

Strategy:
  final_score = ML_WEIGHT * ml_score + RULE_WEIGHT * rule_score

If the model file is missing (e.g., first run before training),
we fall back to rules-only mode gracefully.
"""

from typing import Tuple

# Weight allocation — adjustable
ML_WEIGHT   = 0.6
RULE_WEIGHT = 0.4

# Threshold above which we classify as ADVERSARIAL
ADVERSARIAL_THRESHOLD = 0.45


def fuse(ml_score: float, rule_score: float) -> Tuple[float, str]:
    """
    Combine ML and rule scores into a final verdict.

    Args:
        ml_score:   Float [0,1] from the logistic regression model.
        rule_score: Float [0,1] from the rule engine.

    Returns:
        final_score (float): Weighted combination.
        verdict     (str):   "ADVERSARIAL" or "SAFE"
    """
    final_score = ML_WEIGHT * ml_score + RULE_WEIGHT * rule_score
    verdict = "ADVERSARIAL" if final_score >= ADVERSARIAL_THRESHOLD else "SAFE"
    return round(final_score, 4), verdict


def fuse_rules_only(rule_score: float) -> Tuple[float, str]:
    """
    Fallback fusion when ML model is unavailable — uses rules only.

    Args:
        rule_score: Float [0,1] from the rule engine.

    Returns:
        final_score, verdict
    """
    verdict = "ADVERSARIAL" if rule_score >= ADVERSARIAL_THRESHOLD else "SAFE"
    return round(rule_score, 4), verdict
