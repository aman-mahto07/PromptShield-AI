"""
fusion.py  (v2.1 — soft flag accumulation added)
-------------------------------------------------
LAYER 4: Multi-Score Ensemble Fusion

Combines five independent detection signals into one calibrated
final score and verdict:

  Signal             Source              Default Weight
  ─────────────────  ──────────────────  ──────────────
  ml_score           Logistic Regression  0.30
  rule_score         Pattern engine       0.30
  anomaly_score      Semantic anomaly     0.20
  obfuscation_score  Preprocessing        0.15
  (bonus)            Critical flag boost  +0.10 max
  (bonus)            Soft flag accumul.   +0.12 max

Design decisions:
  1. VETO LOGIC: if any single signal is >= VETO_THRESHOLD (0.88),
     classify as ADVERSARIAL immediately, regardless of other scores.
     This catches unambiguous attacks even when other signals are low.

  2. WEIGHTED SUM: normal case — weighted combination of all signals.

  3. SOFT FLAG ACCUMULATION: weak signals (trust manipulation,
     deflection phrases, passive inquiries) are individually low-weight
     but adversarial when they pile up. Each soft flag adds +0.04,
     capped at +0.12 for 3+ soft flags.

  4. CONFIDENCE BANDS: verdict comes with a confidence tier:
       HIGH   >= 0.80
       MEDIUM  0.50 – 0.79
       LOW     0.35 – 0.49
       CLEAN  < 0.35

  5. THRESHOLD TUNING: all thresholds are constants at the top of
     this file — easy to adjust during experimentation.
"""

from typing import Tuple, Optional, Dict


# ══════════════════════════════════════════════════════════════════════
# TUNABLE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════

# Signal weights (must sum to 1.0 for the base signals)
W_ML          = 0.30   # logistic regression output
W_RULE        = 0.30   # rule engine score
W_ANOMALY     = 0.20   # semantic anomaly score
W_OBFUSCATION = 0.15   # preprocessing obfuscation score
W_CRITICAL    = 0.10   # bonus for triggering any critical flag (was 0.05)

# When ML + anomaly are both 0.0 (model not running / cold start),
# redistribute their combined weight (0.50) to the rule engine so a
# high-confidence rule signal isn't diluted to ~0.23 and incorrectly
# classified as SAFE.  Obfuscation keeps its weight either way.
W_RULE_FALLBACK = 0.75   # rule weight used when ml_score == anomaly_score == 0.0

# Veto: any single signal above this → instant ADVERSARIAL
VETO_THRESHOLD = 0.88

# Final thresholds for verdict
ADVERSARIAL_THRESHOLD       = 0.42
HIGH_CONFIDENCE_THRESHOLD   = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_THRESHOLD    = 0.35

# Critical flag names — any of these alone triggers the veto bonus
CRITICAL_FLAGS = {
    "Explicit Jailbreak",
    "CSAM Indicator",
    "WMD / Bioweapon Query",
    "WMD / Nuclear Query",
    "Drug/Poison Synthesis",
    "Explosive Fabrication",
    "DAN Mode Trigger",
    "Self-Harm Facilitation",
    "OTP Interception",
    "WiFi Credential Attack",
    "Targeted Vulnerability Recon",
    "Attack Methodology Request",
    "Auth Bypass Request",
    # System / data intrusion — clear unauthorized access intent
    "Unauthorized System Access",
    "Internal System Access",
    "System Destruction Request",
    "Internal Data Exfiltration",
    "Mass Data Extraction",
    "Privilege Escalation Request",
    "Privilege Escalation Mode",
    "Circumvention Content Request",
    # Chemical / physical harm
    "Chemical Weapon Synthesis",
    "Household Chemical Weapon",
    "Poison/Bioweapon Synthesis",
    "Narrative WMD Wrap",
    "Physical Harm Evasion",
    "Detailed Harm Guide",
    # Credential / session attacks
    "Credential Hash Dumping",
    "Credential Attack Tool",
    "Personal Account Hacking",
    "Session Token Theft",
}

# Soft flags — suspicious alone, adversarial when combined
# Each soft flag adds +0.04 to the final score, capped at +0.12
SOFT_FLAGS = {
    "Passive Attack Inquiry",
    "Ease-of-Attack Inquiry",
    "Trust Manipulation",
    "Dismissal Attempt",
    "Minimization Attempt",
    "Plausible Deniability",
    "Deflection Phrase",
    "Vulnerability Recon",
    "Auth Weakness Recon",
}


# ══════════════════════════════════════════════════════════════════════
# FUSION RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════

class FusionResult:
    """Structured output from the fusion layer."""

    def __init__(
        self,
        final_score:      float,
        verdict:          str,
        confidence_tier:  str,
        component_scores: Dict[str, float],
        veto_triggered:   bool  = False,
        veto_reason:      str   = "",
    ):
        self.final_score      = final_score
        self.verdict          = verdict
        self.confidence_tier  = confidence_tier
        self.component_scores = component_scores
        self.veto_triggered   = veto_triggered
        self.veto_reason      = veto_reason

    def __repr__(self):
        return (
            f"FusionResult(verdict={self.verdict}, "
            f"score={self.final_score:.3f}, "
            f"tier={self.confidence_tier})"
        )


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _confidence_tier(score: float) -> str:
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    elif score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    elif score >= LOW_CONFIDENCE_THRESHOLD:
        return "LOW"
    else:
        return "CLEAN"


def _has_critical_flag(flags: list) -> bool:
    """Check if any critical flag was triggered."""
    for f in flags:
        for cf in CRITICAL_FLAGS:
            if cf in f:  # 'in' to catch flags like "OTP Interception [encoded payload]"
                return True
    return False


def _soft_flag_count(flags: list) -> int:
    """Count how many distinct soft/weak flags were triggered."""
    count = 0
    for f in flags:
        for sf in SOFT_FLAGS:
            if sf in f:
                count += 1
                break
    return count


# ══════════════════════════════════════════════════════════════════════
# MAIN FUSION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def fuse(
    ml_score:          float,
    rule_score:        float,
    anomaly_score:     float = 0.0,
    obfuscation_score: float = 0.0,
    flags:             list  = None,
) -> FusionResult:
    """
    Combine all detection signals into a final verdict.

    Args:
        ml_score:          Float [0,1] from logistic regression.
        rule_score:        Float [0,1] from rule engine.
        anomaly_score:     Float [0,1] from semantic anomaly checker.
        obfuscation_score: Float [0,1] from preprocessing layer.
        flags:             List of rule flag strings (for critical/soft check).

    Returns:
        FusionResult with verdict, score, tier, and component breakdown.
    """
    flags = flags or []

    component_scores = {
        "ml":          round(ml_score,          4),
        "rule":        round(rule_score,         4),
        "anomaly":     round(anomaly_score,      4),
        "obfuscation": round(obfuscation_score,  4),
    }

    # ── VETO CHECK ────────────────────────────────────────────────────
    # Any single signal at extreme confidence → immediate ADVERSARIAL
    veto_map = {
        "ML classifier":    ml_score,
        "Rule engine":      rule_score,
        "Semantic anomaly": anomaly_score,
        "Obfuscation":      obfuscation_score,
    }
    for source, score in veto_map.items():
        if score >= VETO_THRESHOLD:
            return FusionResult(
                final_score      = round(score, 4),
                verdict          = "ADVERSARIAL",
                confidence_tier  = "HIGH",
                component_scores = component_scores,
                veto_triggered   = True,
                veto_reason      = f"{source} score ({score:.3f}) exceeded veto threshold ({VETO_THRESHOLD})",
            )

    # ── ADAPTIVE RULE WEIGHT ──────────────────────────────────────────
    # When ML and anomaly signals are absent (model not loaded / cold start),
    # use W_RULE_FALLBACK so the rule engine isn't diluted by zero-value signals.
    effective_w_rule = (
        W_RULE_FALLBACK
        if ml_score == 0.0 and anomaly_score == 0.0
        else W_RULE
    )

    # ── CRITICAL FLAG BONUS ───────────────────────────────────────────
    # If a critical flag was triggered, add a fixed bonus to ensure
    # single-signal critical attacks still cross the threshold
    critical_bonus = W_CRITICAL if _has_critical_flag(flags) else 0.0

    # ── SOFT FLAG ACCUMULATION BONUS ─────────────────────────────────
    # Multiple weak signals together = meaningful threat signal
    # Each distinct soft flag adds +0.04, capped at +0.12 (3+ flags)
    soft_count = _soft_flag_count(flags)
    soft_bonus = min(0.12, soft_count * 0.04)

    # ── WEIGHTED COMBINATION ──────────────────────────────────────────
    weighted_sum = (
        W_ML          * ml_score          +
        effective_w_rule * rule_score     +
        W_ANOMALY     * anomaly_score     +
        W_OBFUSCATION * obfuscation_score +
        critical_bonus                    +
        soft_bonus
    )
    final_score = min(0.99, round(weighted_sum, 4))

    verdict   = "ADVERSARIAL" if final_score >= ADVERSARIAL_THRESHOLD else "SAFE"
    conf_tier = _confidence_tier(final_score)

    return FusionResult(
        final_score      = final_score,
        verdict          = verdict,
        confidence_tier  = conf_tier,
        component_scores = component_scores,
    )


# ══════════════════════════════════════════════════════════════════════
# FALLBACK: rules + obfuscation only (no ML, no anomaly)
# Used when classifier.pkl hasn't been trained yet.
# ══════════════════════════════════════════════════════════════════════

def fuse_rules_only(
    rule_score:        float,
    obfuscation_score: float = 0.0,
    flags:             list  = None,
) -> FusionResult:
    """
    Fusion without ML or anomaly scores (pre-training fallback).
    Redistributes ML and anomaly weights to rule and obfuscation.
    """
    flags = flags or []

    # Redistribute weights: rule 0.60, obfuscation 0.30, critical 0.10
    critical_bonus = 0.10 if _has_critical_flag(flags) else 0.0

    # Soft flag accumulation still applies in rules-only mode
    soft_count = _soft_flag_count(flags)
    soft_bonus = min(0.12, soft_count * 0.04)

    raw_score   = 0.60 * rule_score + 0.30 * obfuscation_score + critical_bonus + soft_bonus
    final_score = min(0.99, round(raw_score, 4))
    verdict     = "ADVERSARIAL" if final_score >= ADVERSARIAL_THRESHOLD else "SAFE"

    # Veto check
    if rule_score >= VETO_THRESHOLD or obfuscation_score >= VETO_THRESHOLD:
        final_score = max(rule_score, obfuscation_score)
        return FusionResult(
            final_score      = round(final_score, 4),
            verdict          = "ADVERSARIAL",
            confidence_tier  = "HIGH",
            component_scores = {"rule": rule_score, "obfuscation": obfuscation_score},
            veto_triggered   = True,
            veto_reason      = "Single signal exceeded veto threshold (rules-only mode)",
        )

    return FusionResult(
        final_score      = final_score,
        verdict          = verdict,
        confidence_tier  = _confidence_tier(final_score),
        component_scores = {"rule": rule_score, "obfuscation": obfuscation_score},
    )