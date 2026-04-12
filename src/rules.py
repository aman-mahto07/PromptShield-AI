"""
rules.py
--------
Rule-based detection layer. Catches obvious jailbreak patterns
without needing ML — fast and explainable.
"""

import re
import base64
from typing import Tuple

# ─────────────────────────────────────────────
# Configurable thresholds
# ─────────────────────────────────────────────
LONG_PROMPT_THRESHOLD = 800      # characters; anomalously long prompts are suspicious
BASE64_MIN_LENGTH     = 40       # minimum length to flag as encoded payload

# ─────────────────────────────────────────────
# Suspicious keyword / phrase patterns
# Each tuple: (regex_pattern, human-readable flag name)
# ─────────────────────────────────────────────
SUSPICIOUS_PATTERNS = [
    # Instruction override attempts
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", "Instruction Override"),
    (r"disregard\s+(your|all|previous)\s+(instructions?|rules?|guidelines?)", "Instruction Override"),
    (r"forget\s+everything\s+(you('ve| have)\s+)?been\s+told", "Instruction Override"),

    # Role / persona hijacking
    (r"\bact\s+as\b", "Persona Hijacking"),
    (r"\bpretend\s+(you\s+are|to\s+be)\b", "Persona Hijacking"),
    (r"\byou\s+are\s+now\b", "Persona Hijacking"),
    (r"\bplay\s+the\s+role\s+of\b", "Persona Hijacking"),
    (r"\bsimulate\s+a\b", "Persona Hijacking"),

    # Developer / jailbreak mode triggers
    (r"\bdeveloper\s+mode\b", "Developer Mode Trigger"),
    (r"\bdan\s+mode\b", "DAN Mode Trigger"),
    (r"\bjailbreak\b", "Explicit Jailbreak"),
    (r"\bunrestricted\s+mode\b", "Unrestricted Mode"),
    (r"\bno\s+restrictions\b", "Restriction Bypass"),
    (r"\bbypass\s+(your\s+)?(safety|filters?|rules?|guidelines?)", "Safety Bypass"),

    # Prompt injection markers
    (r"\bsystem\s*prompt\b", "System Prompt Access"),
    (r"\bprompt\s+injection\b", "Prompt Injection"),
    (r"\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>", "Token Injection"),

    # Roleplay escalation
    (r"\bin\s+this\s+fictional\s+(world|scenario|story)\b", "Roleplay Escalation"),
    (r"\bfor\s+educational\s+purposes\s+only\b", "Plausible Deniability"),
    (r"\bhypothetically\s+speaking\b", "Plausible Deniability"),

    # Harmful content fishing
    (r"\bhow\s+to\s+(make|build|create|synthesize)\s+(a\s+)?(bomb|weapon|drug|malware|virus)", "Harmful Content Request"),
    (r"\bstep[\s-]by[\s-]step\s+(instructions?|guide)\s+to\b", "Step-by-Step Harmful Guide"),
]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _is_base64_payload(text: str) -> bool:
    """
    Try to detect if a significant chunk of the text is base64-encoded.
    Finds all base64-like tokens (length >= BASE64_MIN_LENGTH) and
    attempts decoding.
    """
    # Look for long runs of base64 characters
    b64_pattern = re.compile(r'[A-Za-z0-9+/]{' + str(BASE64_MIN_LENGTH) + r',}={0,2}')
    candidates = b64_pattern.findall(text)
    for candidate in candidates:
        try:
            decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
            # If decoded output contains readable words, it's likely an encoded payload
            if len(decoded.split()) > 3:
                return True
        except Exception:
            pass
    return False


def _has_repetitive_override(text: str) -> bool:
    """
    Detect repetition attacks — e.g., "ignore ignore ignore ignore instructions"
    or a phrase repeated many times to confuse the model.
    """
    words = text.lower().split()
    if len(words) < 6:
        return False
    # Count max consecutive identical words
    max_repeat = 1
    current = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            current += 1
            max_repeat = max(max_repeat, current)
        else:
            current = 1
    return max_repeat >= 5


# ─────────────────────────────────────────────
# Main rule engine
# ─────────────────────────────────────────────

def rule_check(text: str) -> Tuple[float, list]:
    """
    Run all rule checks on the input prompt.

    Args:
        text: Raw user prompt string.

    Returns:
        rule_score (float 0–1): Normalized suspicion score from rules alone.
        flags      (list[str]): Human-readable list of triggered rule names.
    """
    flags = []
    text_lower = text.lower()

    # ── 1. Keyword / phrase patterns ──────────────
    for pattern, label in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text_lower):
            if label not in flags:
                flags.append(label)

    # ── 2. Base64 encoded payload ─────────────────
    if _is_base64_payload(text):
        flags.append("Base64 Encoded Payload")

    # ── 3. Length anomaly ─────────────────────────
    if len(text) > LONG_PROMPT_THRESHOLD:
        flags.append(f"Anomalous Prompt Length ({len(text)} chars)")

    # ── 4. Repetition attack ──────────────────────
    if _has_repetitive_override(text):
        flags.append("Repetition Attack")

    # ── 5. Score calculation ──────────────────────
    # Each flag contributes; we cap at 1.0
    # First flag is weighted heavily, diminishing returns after that
    if not flags:
        rule_score = 0.0
    elif len(flags) == 1:
        rule_score = 0.55
    elif len(flags) == 2:
        rule_score = 0.75
    else:
        rule_score = min(0.95, 0.75 + 0.05 * (len(flags) - 2))

    return rule_score, flags
