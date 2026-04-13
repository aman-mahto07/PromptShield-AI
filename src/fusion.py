"""
fusion.py  —  v2.0 Enhanced Fusion Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Combines ML probability + rule score into a final verdict.

Improvements over v1:
  • Non-linear fusion: critical rule flags can hard-override ML
  • Asymmetric thresholds: different sensitivity for SAFE vs ADVERSARIAL
  • Confidence bands: CERTAIN / HIGH / MEDIUM / LOW labels
  • Rules-only mode graceful fallback
"""

from typing import Tuple

# ─────────────────────────────────────────────
# Weights & thresholds
# ─────────────────────────────────────────────
ML_WEIGHT   = 0.55
RULE_WEIGHT = 0.45

# Primary classification threshold
ADVERSARIAL_THRESHOLD = 0.42   # slightly lower = more sensitive

# Hard-override: if rule_score exceeds this, classify as ADVERSARIAL
# regardless of ML score (catches obfuscated payloads that fool embeddings)
RULE_HARD_OVERRIDE_THRESHOLD = 0.82

# Hard-safe: if both scores are extremely low, force SAFE
SAFE_HARD_THRESHOLD = 0.12

# Critical flags that immediately force ADVERSARIAL, even at low scores
CRITICAL_FLAGS = {
    "CSAM Signal",
    "CBRN Weapons Request",
    "Self-Harm Facilitation",
    "Explicit Jailbreak",
    "DAN Mode Trigger",
    "Harmful Content Request",
    "Obfuscated Instructions",
    "Token Injection",
}


def _confidence_band(score: float) -> str:
    if score >= 0.85:   return "CERTAIN"
    if score >= 0.68:   return "HIGH"
    if score >= 0.48:   return "MEDIUM"
    if score >= 0.30:   return "LOW"
    return "MINIMAL"


def fuse(ml_score: float, rule_score: float,
         flags: list | None = None) -> Tuple[float, str, str]:
    """
    Combine ML and rule scores into a final verdict.

    Args:
        ml_score:   Float [0,1] from the logistic regression model.
        rule_score: Float [0,1] from the rule engine.
        flags:      Optional list of triggered flag names.

    Returns:
        final_score (float), verdict (str), confidence (str)
    """
    flags = flags or []

    # ── Critical flag hard-override ────────────────────────────────────
    if any(f in CRITICAL_FLAGS for f in flags):
        final = max(0.92, ML_WEIGHT * ml_score + RULE_WEIGHT * rule_score)
        return round(final, 4), "ADVERSARIAL", "CERTAIN"

    # ── Rule hard-override (obfuscated/encoded payloads) ───────────────
    if rule_score >= RULE_HARD_OVERRIDE_THRESHOLD:
        final = ML_WEIGHT * ml_score + RULE_WEIGHT * rule_score
        return round(final, 4), "ADVERSARIAL", _confidence_band(final)

    # ── Normal weighted fusion ─────────────────────────────────────────
    final = ML_WEIGHT * ml_score + RULE_WEIGHT * rule_score

    # ── Hard-safe floor ────────────────────────────────────────────────
    if final < SAFE_HARD_THRESHOLD and not flags:
        return round(final, 4), "SAFE", "MINIMAL"

    verdict = "ADVERSARIAL" if final >= ADVERSARIAL_THRESHOLD else "SAFE"
    return round(final, 4), verdict, _confidence_band(final)


def fuse_rules_only(rule_score: float,
                    flags: list | None = None) -> Tuple[float, str, str]:
    """
    Fallback fusion when ML model is unavailable — uses rules only.
    """
    flags = flags or []

    if any(f in CRITICAL_FLAGS for f in flags):
        return round(max(0.90, rule_score), 4), "ADVERSARIAL", "CERTAIN"

    verdict = "ADVERSARIAL" if rule_score >= ADVERSARIAL_THRESHOLD else "SAFE"
    return round(rule_score, 4), verdict, _confidence_band(rule_score)