"""
explain.py
----------
Converts raw detection signals into human-readable explanations
and optionally generates a sanitized version of the prompt.
"""

import re
from typing import List, Tuple


# ─────────────────────────────────────────────
# Flag → explanation mapping
# ─────────────────────────────────────────────
FLAG_EXPLANATIONS = {
    "Instruction Override": (
        "The prompt attempts to override or nullify the model's original instructions. "
        "Phrases like 'ignore previous instructions' are a hallmark of prompt injection attacks."
    ),
    "Persona Hijacking": (
        "The prompt asks the model to adopt a different identity or persona (e.g., 'act as DAN'). "
        "This is commonly used to bypass safety guidelines by role-playing as an unrestricted AI."
    ),
    "Developer Mode Trigger": (
        "The prompt references 'developer mode', a known jailbreak pattern that tries to convince "
        "the model it can operate without restrictions."
    ),
    "DAN Mode Trigger": (
        "'DAN' (Do Anything Now) is a well-documented jailbreak technique. This prompt appears "
        "to invoke that pattern."
    ),
    "Explicit Jailbreak": (
        "The prompt explicitly contains the word 'jailbreak', strongly indicating intent to "
        "circumvent model safety mechanisms."
    ),
    "Unrestricted Mode": (
        "The prompt requests the model to operate in an 'unrestricted' mode, bypassing "
        "content policies."
    ),
    "Restriction Bypass": (
        "The prompt explicitly states 'no restrictions', attempting to remove guardrails."
    ),
    "Safety Bypass": (
        "The prompt directly asks the model to bypass its safety filters or guidelines."
    ),
    "System Prompt Access": (
        "The prompt references the system prompt, potentially attempting to reveal, "
        "override, or manipulate hidden instructions."
    ),
    "Prompt Injection": (
        "Explicit prompt injection language detected. The prompt attempts to inject "
        "instructions that override the model's behavior."
    ),
    "Token Injection": (
        "Special tokens associated with model instruction formats were detected. "
        "This is a low-level injection technique targeting tokenizer behavior."
    ),
    "Roleplay Escalation": (
        "The prompt sets up a fictional scenario — a common technique to extract harmful "
        "content under the guise of 'it's just a story'."
    ),
    "Plausible Deniability": (
        "The prompt uses disclaimers like 'hypothetically' or 'for educational purposes' "
        "to create cover for requesting harmful content."
    ),
    "Harmful Content Request": (
        "The prompt directly requests instructions for creating weapons, drugs, malware, "
        "or other dangerous material."
    ),
    "Step-by-Step Harmful Guide": (
        "The prompt asks for step-by-step instructions, often a pattern used to extract "
        "detailed harmful how-to content."
    ),
    "Base64 Encoded Payload": (
        "Base64-encoded text was found in the prompt. This is a known obfuscation technique "
        "to hide adversarial instructions from keyword filters."
    ),
    "Repetition Attack": (
        "An unusual number of repeated words or phrases was detected. Repetition attacks "
        "can confuse model attention and override safety training."
    ),
}


def explain(flags: List[str], final_score: float, verdict: str) -> str:
    """
    Generate a structured, readable explanation of detection results.

    Args:
        flags:       List of triggered rule flag names.
        final_score: Combined confidence score (0–1).
        verdict:     "SAFE" or "ADVERSARIAL"

    Returns:
        Multi-line explanation string.
    """
    if verdict == "SAFE" and not flags:
        return (
            "✅ No suspicious patterns detected. The prompt appears to be a standard, "
            "benign request. The ML model also returned a low adversarial probability."
        )

    lines = []

    if verdict == "ADVERSARIAL":
        lines.append(
            f"🚨 This prompt was classified as ADVERSARIAL with a confidence of "
            f"{final_score:.0%}. The following issues were identified:\n"
        )
    else:
        lines.append(
            f"⚠️  The prompt was marked SAFE but showed some suspicious signals "
            f"(score: {final_score:.0%}). Review the flags below:\n"
        )

    for i, flag in enumerate(flags, 1):
        description = FLAG_EXPLANATIONS.get(flag, "Suspicious pattern detected.")
        lines.append(f"  [{i}] {flag}\n      → {description}\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Prompt sanitizer (optional feature)
# ─────────────────────────────────────────────

# Phrases to strip or replace in a sanitized rewrite
SANITIZE_REPLACEMENTS = [
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", "[REDACTED]"),
    (r"disregard\s+(your|all|previous)\s+(instructions?|rules?|guidelines?)", "[REDACTED]"),
    (r"\bact\s+as\b", "help with"),
    (r"\bpretend\s+(you\s+are|to\s+be)\b", "describe"),
    (r"\byou\s+are\s+now\b", "please"),
    (r"\bdeveloper\s+mode\b", "[MODE REMOVED]"),
    (r"\bdan\s+mode\b", "[MODE REMOVED]"),
    (r"\bjailbreak\b", "[TERM REMOVED]"),
    (r"\bno\s+restrictions\b", "[REMOVED]"),
    (r"\bbypass\s+(your\s+)?(safety|filters?|rules?|guidelines?)", "[REMOVED]"),
]


def sanitize(text: str) -> Tuple[str, bool]:
    """
    Attempt to produce a safer version of the prompt by removing
    or replacing adversarial patterns.

    Args:
        text: Original prompt.

    Returns:
        sanitized_text (str): Cleaned version.
        was_modified   (bool): True if any changes were made.
    """
    sanitized = text
    was_modified = False

    for pattern, replacement in SANITIZE_REPLACEMENTS:
        new_text = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        if new_text != sanitized:
            was_modified = True
            sanitized = new_text

    # Clean up excessive whitespace
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()

    return sanitized, was_modified
