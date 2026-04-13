"""
explain.py  —  v2.0 Enhanced Explanations & Sanitizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Richer flag explanations, severity levels, attack-category grouping,
and smarter sanitization with more replacement patterns.
"""

import re
from typing import List, Tuple, Dict

# ─────────────────────────────────────────────
# Flag metadata  (explanation + severity + category)
# ─────────────────────────────────────────────
FLAG_META: Dict[str, Dict] = {
    "Instruction Override": {
        "severity": "CRITICAL",
        "category": "Injection",
        "detail": (
            "The prompt attempts to override or nullify original instructions. "
            "Phrases like 'ignore previous instructions' are the #1 hallmark of prompt injection. "
            "This technique is used to redirect the model's behavior from its intended purpose."
        ),
    },
    "Persona Hijacking": {
        "severity": "HIGH",
        "category": "Social Engineering",
        "detail": (
            "The prompt asks the model to adopt a different identity or persona (e.g., 'act as DAN'). "
            "By creating a fictional alter-ego, adversaries attempt to convince the model it can operate "
            "outside its safety guidelines while 'in character'."
        ),
    },
    "Developer Mode Trigger": {
        "severity": "HIGH",
        "category": "Mode Exploitation",
        "detail": (
            "References 'developer mode' — a well-documented jailbreak pattern suggesting the model "
            "has a hidden mode without content restrictions. No such mode exists."
        ),
    },
    "DAN Mode Trigger": {
        "severity": "CRITICAL",
        "category": "Mode Exploitation",
        "detail": (
            "'DAN' (Do Anything Now) is the most widely-circulated jailbreak technique. "
            "It attempts to role-play an alternate, unrestricted AI persona that ignores all guidelines."
        ),
    },
    "Explicit Jailbreak": {
        "severity": "CRITICAL",
        "category": "Direct Attack",
        "detail": (
            "The word 'jailbreak' appears explicitly, strongly signalling intent to circumvent "
            "model safety mechanisms."
        ),
    },
    "Unrestricted Mode": {
        "severity": "HIGH",
        "category": "Mode Exploitation",
        "detail": (
            "Requests the model to operate in an 'unrestricted' or 'uncensored' mode, "
            "attempting to remove guardrails."
        ),
    },
    "Restriction Bypass": {
        "severity": "HIGH",
        "category": "Direct Attack",
        "detail": (
            "Explicitly states the model should have 'no restrictions', attempting to remove content policies."
        ),
    },
    "Safety Bypass": {
        "severity": "CRITICAL",
        "category": "Direct Attack",
        "detail": (
            "Directly instructs the model to bypass its safety filters, guidelines, or training. "
            "This is an unambiguous adversarial signal."
        ),
    },
    "System Prompt Access": {
        "severity": "HIGH",
        "category": "Confidentiality",
        "detail": (
            "Attempts to reveal, read, or override the hidden system prompt. "
            "This is a core confidentiality threat and may precede a more sophisticated attack."
        ),
    },
    "Confidentiality Probe": {
        "severity": "MEDIUM",
        "category": "Confidentiality",
        "detail": (
            "Asks for internal or hidden instructions, potentially probing for sensitive configuration "
            "before launching a more targeted attack."
        ),
    },
    "Prompt Injection": {
        "severity": "HIGH",
        "category": "Injection",
        "detail": (
            "Explicit prompt injection language detected. The prompt attempts to inject instructions "
            "that hijack or override model behavior."
        ),
    },
    "Token Injection": {
        "severity": "HIGH",
        "category": "Injection",
        "detail": (
            "Special tokens associated with model instruction formats detected (e.g., [INST], <|im_start|>). "
            "This is a low-level injection technique targeting tokenizer parsing."
        ),
    },
    "Roleplay Escalation": {
        "severity": "MEDIUM",
        "category": "Indirect Attack",
        "detail": (
            "A fictional scenario is being set up — a common 'laundering' technique to extract harmful "
            "content under the guise of storytelling."
        ),
    },
    "Plausible Deniability": {
        "severity": "MEDIUM",
        "category": "Indirect Attack",
        "detail": (
            "Disclaimers like 'hypothetically' or 'for educational purposes' are being used to provide "
            "cover for requesting harmful content. Often combined with other attack patterns."
        ),
    },
    "Harmful Content Request": {
        "severity": "CRITICAL",
        "category": "Direct Harm",
        "detail": (
            "Direct request for instructions to create weapons, drugs, malware, or cause physical harm."
        ),
    },
    "Step-by-Step Harmful Guide": {
        "severity": "HIGH",
        "category": "Direct Harm",
        "detail": (
            "Requests a step-by-step guide for a harmful activity. This format is specifically used "
            "to extract actionable, detailed harmful instructions."
        ),
    },
    "CBRN Weapons Request": {
        "severity": "CRITICAL",
        "category": "Direct Harm",
        "detail": (
            "Chemical, Biological, Radiological, or Nuclear weapons content detected. "
            "This is the highest-severity threat category."
        ),
    },
    "Self-Harm Facilitation": {
        "severity": "CRITICAL",
        "category": "Direct Harm",
        "detail": (
            "Content that could facilitate self-harm or suicide. Requires immediate safe-messaging response."
        ),
    },
    "CSAM Signal": {
        "severity": "CRITICAL",
        "category": "Direct Harm",
        "detail": (
            "Signals indicating a request for child sexual abuse material. "
            "Absolute refusal required. Law enforcement relevant."
        ),
    },
    "Base64 Encoded Payload": {
        "severity": "HIGH",
        "category": "Obfuscation",
        "detail": (
            "Base64-encoded text was found and successfully decoded to readable content. "
            "This obfuscation technique is used to hide adversarial instructions from keyword filters."
        ),
    },
    "Hex Encoded Payload": {
        "severity": "HIGH",
        "category": "Obfuscation",
        "detail": (
            "Hexadecimal-encoded text was detected that decodes to readable content. "
            "Attackers use this to smuggle instructions past text-based filters."
        ),
    },
    "URL Encoded Payload": {
        "severity": "MEDIUM",
        "category": "Obfuscation",
        "detail": (
            "URL percent-encoding was heavily used — a technique to obscure adversarial instructions."
        ),
    },
    "Data-URI Obfuscation": {
        "severity": "HIGH",
        "category": "Obfuscation",
        "detail": (
            "A data-URI with base64 content was detected. Can be used to embed hidden payloads."
        ),
    },
    "ROT13 Encoded Payload": {
        "severity": "MEDIUM",
        "category": "Obfuscation",
        "detail": (
            "ROT13-decoded content contains adversarial patterns. Simple but effective against "
            "string-matching defenses."
        ),
    },
    "Obfuscated Instructions": {
        "severity": "CRITICAL",
        "category": "Obfuscation",
        "detail": (
            "Multiple obfuscation signals combined with instruction-override patterns detected. "
            "This composite signal strongly indicates a sophisticated evasion attempt."
        ),
    },
    "Zero-Width Character Attack": {
        "severity": "HIGH",
        "category": "Obfuscation",
        "detail": (
            "Zero-width Unicode characters detected (U+200B etc.). These invisible characters "
            "can break keyword detection by splitting words visually while keeping them intact "
            "for the model."
        ),
    },
    "Homoglyph Attack": {
        "severity": "HIGH",
        "category": "Obfuscation",
        "detail": (
            "Visually similar characters from non-Latin scripts detected (e.g., Cyrillic 'а' "
            "substituted for Latin 'a'). Used to bypass string-based filters."
        ),
    },
    "Leetspeak Obfuscation": {
        "severity": "MEDIUM",
        "category": "Obfuscation",
        "detail": (
            "Leet-speak substitutions detected (e.g., '1gnor3' for 'ignore'). "
            "Used to evade keyword and regex filters."
        ),
    },
    "Goal Hijacking": {
        "severity": "HIGH",
        "category": "Injection",
        "detail": (
            "Attempts to redefine the model's purpose, objective, or motivations. "
            "Often the precursor to more specific harmful requests."
        ),
    },
    "Indirect Injection": {
        "severity": "HIGH",
        "category": "Injection",
        "detail": (
            "Attempts to inject persistent instructions that modify future behavior, "
            "or that activate on specific trigger words."
        ),
    },
    "Privilege Escalation": {
        "severity": "HIGH",
        "category": "Social Engineering",
        "detail": (
            "Claims elevated access, permissions, or a special role to justify receiving "
            "restricted content."
        ),
    },
    "False Authority Claim": {
        "severity": "HIGH",
        "category": "Social Engineering",
        "detail": (
            "Claims authorization from a recognized authority (e.g., Anthropic, OpenAI) "
            "to override safety behaviors."
        ),
    },
    "Memory Manipulation": {
        "severity": "MEDIUM",
        "category": "Context Manipulation",
        "detail": (
            "Attempts to manipulate the model's understanding of prior context, "
            "create false memory, or reset conversational state."
        ),
    },
    "Training Data Extraction": {
        "severity": "HIGH",
        "category": "Privacy",
        "detail": (
            "Attempts to extract verbatim training data from the model, "
            "which could leak sensitive or copyrighted content."
        ),
    },
    "Model Fingerprinting": {
        "severity": "LOW",
        "category": "Reconnaissance",
        "detail": (
            "Probing for model identity, version, or architecture details. "
            "Low risk alone; may precede a targeted attack."
        ),
    },
    "Output Format Hijacking": {
        "severity": "MEDIUM",
        "category": "Injection",
        "detail": (
            "Attempts to alter the output format to strip disclaimers, warnings, or safety messaging."
        ),
    },
    "Delimiter Injection": {
        "severity": "HIGH",
        "category": "Injection",
        "detail": (
            "Uses separator strings that mimic model instruction formats, attempting to "
            "inject new system-level instructions mid-conversation."
        ),
    },
    "Repetition Attack": {
        "severity": "MEDIUM",
        "category": "Structural",
        "detail": (
            "An unusual number of repeated words or phrases was detected. Repetition attacks "
            "can overload attention mechanisms and override safety training."
        ),
    },
    "Context Window Flooding": {
        "severity": "MEDIUM",
        "category": "Structural",
        "detail": (
            "Extremely long repeated character sequences detected. May be used to push "
            "original instructions out of the context window."
        ),
    },
    "ASCII Art Smuggling": {
        "severity": "MEDIUM",
        "category": "Obfuscation",
        "detail": (
            "Spaced single-character patterns detected — potential ASCII-art steganography "
            "technique to embed hidden messages."
        ),
    },
    "High Special-Character Ratio": {
        "severity": "LOW",
        "category": "Structural",
        "detail": (
            "Unusually high ratio of non-alphanumeric characters. May indicate obfuscation, "
            "injection artifacts, or encoding tricks."
        ),
    },
    "Anomalous Prompt Length": {
        "severity": "LOW",
        "category": "Structural",
        "detail": (
            "The prompt is unusually long. Lengthy inputs are sometimes used to dilute "
            "safety signals or push system instructions out of context."
        ),
    },
    "Multilingual Override": {
        "severity": "MEDIUM",
        "category": "Injection",
        "detail": (
            "Override instructions in non-English languages detected. "
            "Attackers use language switching to evade English-trained filters."
        ),
    },
    "Gaslighting / Confusion": {
        "severity": "MEDIUM",
        "category": "Social Engineering",
        "detail": (
            "Claims that the model's safety features are broken, non-existent, or disabled. "
            "Used to destabilise the model's confidence in its own guidelines."
        ),
    },
    "Social Engineering": {
        "severity": "MEDIUM",
        "category": "Social Engineering",
        "detail": (
            "Emotional manipulation or urgency framing used to convince the model to "
            "make a one-time exception to its guidelines."
        ),
    },
    "Multi-turn Manipulation": {
        "severity": "MEDIUM",
        "category": "Context Manipulation",
        "detail": (
            "References to prior sessions or previous 'permissions' granted — attempting to "
            "fabricate a history of compliance."
        ),
    },
    "Sycophancy Exploitation": {
        "severity": "LOW",
        "category": "Social Engineering",
        "detail": (
            "Flattery combined with a request — attempts to leverage the model's "
            "helpfulness instincts to lower its guard."
        ),
    },
}

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "MINIMAL": 4}
SEVERITY_ICONS = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "MINIMAL": "⚪"}


def explain(flags: List[str], final_score: float, verdict: str,
            confidence: str = "") -> str:
    if verdict == "SAFE" and not flags:
        return (
            "✅  No suspicious patterns were detected. The prompt appears to be a standard, "
            "benign request. ML model returned a low adversarial probability."
        )

    # Sort by severity
    def sort_key(f):
        meta = FLAG_META.get(f, {})
        return SEVERITY_ORDER.get(meta.get("severity", "LOW"), 99)

    sorted_flags = sorted(flags, key=sort_key)

    lines = []
    conf_str = f"  [{confidence} confidence]" if confidence else ""

    if verdict == "ADVERSARIAL":
        lines.append(
            f"🚨  ADVERSARIAL — {final_score:.0%} risk score{conf_str}\n"
            f"    {len(flags)} threat signal{'s' if len(flags) != 1 else ''} triggered:\n"
        )
    else:
        lines.append(
            f"⚠️   SAFE (with signals) — score {final_score:.0%}{conf_str}\n"
            f"    {len(flags)} low-severity signal{'s' if len(flags) != 1 else ''} noted:\n"
        )

    for i, flag in enumerate(sorted_flags, 1):
        meta = FLAG_META.get(flag, {})
        severity  = meta.get("severity", "MEDIUM")
        category  = meta.get("category", "Unknown")
        detail    = meta.get("detail", "Suspicious pattern detected.")
        icon      = SEVERITY_ICONS.get(severity, "⚪")
        lines.append(
            f"  [{i}] {icon} {flag}  [{severity}] · {category}\n"
            f"      → {detail}\n"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Sanitizer — enhanced pattern coverage
# ─────────────────────────────────────────────

SANITIZE_REPLACEMENTS = [
    # Hard overrides
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier|initial)\s+instructions?", "[INSTRUCTION OVERRIDE REMOVED]"),
    (r"disregard\s+(your|all|previous|any|these)\s+(instructions?|rules?|guidelines?|policies|constraints?|training)", "[INSTRUCTION OVERRIDE REMOVED]"),
    (r"forget\s+everything\s+(you('ve| have)\s+)?been\s+told", "[INSTRUCTION OVERRIDE REMOVED]"),
    (r"override\s+(your\s+)?(instructions?|programming|training|rules?)", "[OVERRIDE REMOVED]"),
    # Mode triggers
    (r"\bdeveloper\s+mode\b", "[MODE REMOVED]"),
    (r"\bdan\s+mode\b", "[MODE REMOVED]"),
    (r"\bgod\s+mode\b", "[MODE REMOVED]"),
    (r"\bjailbreak\b", "[TERM REMOVED]"),
    (r"\bno\s+restrictions\b", "[REMOVED]"),
    (r"\buncensored\b", "[REMOVED]"),
    # Safety bypass
    (r"\bbypass\s+(your\s+)?(safety|filters?|rules?|guidelines?|restrictions?|guardrails?)", "[SAFETY BYPASS REMOVED]"),
    (r"\bdisable\s+(your\s+)?(safety|content\s+filter|guardrail|restriction)\b", "[SAFETY BYPASS REMOVED]"),
    # Persona
    (r"\bact\s+as\b", "help with"),
    (r"\bpretend\s+(you\s+are|to\s+be)\b", "describe"),
    (r"\byou\s+are\s+now\b", "please"),
    # Token injection
    (r"\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>", "[TOKEN REMOVED]"),
    # Plausible deniability (keep the subject, strip the wrapper)
    (r"\bfor\s+(educational|research|academic|scientific|informational)\s+purposes\s+only\b", ""),
    (r"\bhypothetically\s+speaking\b", ""),
    (r"\btheoretically\s+speaking\b", ""),
]


def sanitize(text: str) -> Tuple[str, bool]:
    sanitized = text
    was_modified = False
    for pattern, replacement in SANITIZE_REPLACEMENTS:
        new = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        if new != sanitized:
            was_modified = True
            sanitized = new
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()
    return sanitized, was_modified