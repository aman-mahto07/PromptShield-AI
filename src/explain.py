"""
explain.py  (v2 — upgraded)
---------------------------
LAYER 5: Explanation & Reporting Engine

Generates structured, human-readable detection reports from all
upstream signals. Also handles prompt sanitization.

New in v2:
  - Awareness of preprocessing obfuscation findings
  - Anomaly findings included in report
  - Veto reasoning surfaced
  - Confidence tier shown with colour-coded label
  - Sanitizer upgraded to handle more patterns
"""

import re
from typing import List, Tuple, Optional


# ══════════════════════════════════════════════════════════════════════
# FLAG → EXPLANATION DICTIONARY
# ══════════════════════════════════════════════════════════════════════

FLAG_EXPLANATIONS = {
    # Critical
    "Explicit Jailbreak":       "The word 'jailbreak' directly signals intent to circumvent model safety mechanisms.",
    "CSAM Indicator":           "Content involving child sexual abuse material — absolute policy violation, no context permitted.",
    "WMD / Bioweapon Query":    "Request involves chemical or biological weapons capable of mass casualties.",
    "WMD / Nuclear Query":      "Request involves nuclear weapons or radiological devices.",
    "Drug/Poison Synthesis":    "Request seeks synthesis instructions for controlled substances or lethal poisons.",
    "Explosive Fabrication":    "Request seeks instructions for building explosive devices.",
    "Self-Harm Facilitation":   "Content that could facilitate self-harm or suicide — handled with care but flagged for safety.",
    "DAN Mode Trigger":         "'Do Anything Now' is a documented jailbreak persona designed to strip safety constraints.",

    # Instruction override
    "Instruction Override":     "Prompt attempts to nullify system instructions ('ignore previous instructions'). Classic prompt injection.",
    "Instruction Injection":    "New instructions are being injected mid-conversation to hijack model behavior.",
    "Instruction Overwrite":    "Prompt claims the model's 'true' instructions are different from its actual training.",
    "Training Override":        "Prompt tells the model to forget its training or guidelines entirely.",

    # Persona hijacking
    "Persona Hijacking":        "Prompt asks model to adopt an alternate identity (e.g., 'act as DAN') to bypass safety.",
    "Identity Manipulation":    "Prompt questions or challenges the model's core identity to weaken safety commitments.",
    "Restriction Removal":      "Prompt explicitly states the model is 'no longer restricted' or 'no longer bound'.",

    # Mode manipulation
    "Developer Mode Trigger":   "'Developer mode' is a jailbreak pattern suggesting the model can operate without restrictions.",
    "Privilege Escalation Mode":"Prompt attempts to grant itself admin/root/god-level permissions.",
    "Unrestricted Mode":        "Prompt requests an 'unrestricted' operating mode bypassing content policies.",
    "Filter Bypass Mode":       "Prompt requests a 'no-filter' mode to circumvent safety layers.",
    "Capability Unlock":        "Prompt attempts to 'enable' hidden capabilities or unlock restricted functions.",

    # Safety bypass
    "Safety Bypass":            "Direct request to bypass, disable, or remove the model's safety filters.",
    "Restriction Bypass":       "Prompt explicitly claims 'no restrictions' apply.",

    # Exfiltration
    "System Prompt Access":     "Prompt references the hidden system prompt — possible attempt to read or override it.",
    "Prompt Exfiltration":      "Prompt asks the model to reveal or repeat its internal instructions.",
    "System Prompt Exfiltration": "Indirect attempt to extract system prompt content via rephrasing or translation.",

    # Injection
    "Prompt Injection":         "Explicit prompt injection language — overriding instructions through user input.",
    "Token Injection":          "Special instruction-format tokens ([INST], <|im_start|>) injected to manipulate tokenizer parsing.",
    "Delimiter Injection":      "Section delimiters (### Instruction:) injected to confuse prompt structure.",
    "Separator Injection":      "Horizontal rules or separators used to falsely signal end of system context.",

    # Roleplay / fiction
    "Fictional Framing":        "Fictional or hypothetical framing used to extract harmful content under plausible deniability.",
    "Narrative Wrap":           "Harmful content request wrapped in a creative writing premise.",

    # Deniability
    "Plausible Deniability":    "Disclaimers like 'hypothetically' or 'for educational purposes' used as cover for harmful requests.",
    "Deflection Phrase":        "Deflection phrases ('just curious', 'just asking') attempt to minimize apparent intent.",

    # Indirect
    "Indirect Manipulation":    "Prompt indirectly asks the model to behave as if rules don't exist.",
    "Conditional Bypass":       "Conditional phrasing ('if you could ignore...') used to explore constraint removal.",
    "Indirect Jailbreak":       "Asks what an 'uncensored AI' would say — indirect jailbreak via hypothetical framing.",
    "Inversion Attack":         "Asks for the 'opposite of safe' response — inversion to extract harmful content.",
    "Harm Maximization":        "Explicitly asks for the most harmful/dangerous method — maximization attack.",
    "Harm Acknowledgment Skip": "Asks for harmful content while explicitly requesting omission of safety warnings.",

    # Harmful verbs
    "Unauthorized Access":      "Request to hack into systems, accounts, or infrastructure — unauthorized access.",
    "Data Theft":               "Request for techniques to steal credentials, data, or authentication tokens.",
    "Exploit Development":      "Request to develop exploits for known vulnerabilities or zero-day bugs.",
    "DDoS Attack":              "Request related to denial-of-service attack infrastructure or techniques.",
    "Malware Creation":         "Request to create ransomware, spyware, keyloggers, or other malicious software.",
    "Malicious Code Request":   "Explicit request to write malicious, offensive, or exploit code.",
    "Social Engineering":       "Request for social engineering scripts to manipulate people.",
    "Phishing Content":         "Request to create phishing emails, pages, or other deceptive content.",
    # System intrusion
    "Unauthorized System Access":    "Direct demand to access a database, server, system, or backend without authorization.",
    "System Destruction Request":    "Request to break, crash, or take down a system or server — destructive intent.",
    "Internal Data Exfiltration":    "Request to dump, extract, or fetch internal/private data from a system.",
    "Auth Bypass Request":           "Request to bypass or circumvent authentication, login, or 2FA mechanisms.",
    "Privilege Escalation Request":  "Request for admin, root, or superuser-level access or privileges.",
    "Internal System Access":        "Request targeting internal or private systems, networks, or APIs.",
    "Safety Rules Override Claim":   "Prompt explicitly states safety rules 'don\'t apply' or are suspended — a direct attempt to nullify content policies.",
    "Tool Invocation Attack":        "Prompt instructs the model to call a tool, function, or API — likely to exfiltrate data or trigger unauthorized actions.",
    "Mass Data Extraction":          "Request to return or dump all records of a data type (users, customers, credentials) — data exfiltration pattern.",
    "Circumvention Content Request": "Prompt requests real, working examples of breaking or bypassing security protections — harmful technical content under a benign framing.",

    # Harmful topics
    "Weapon Fabrication":       "Request for instructions to manufacture firearms or weapons.",
    "Step-by-Step Harmful Guide":"Step-by-step instruction format often used to extract detailed harmful how-to content.",

    # Multi-turn
    "False Prior Agreement":    "Prompt falsely claims the model previously agreed to something it did not.",
    "Fake Memory Injection":    "Prompt injects a false 'memory' of a prior agreement to manipulate current behavior.",

    # Special detectors
    "Repetition Attack":        "Repeated words/phrases detected — a technique to confuse model attention mechanisms.",
    "Authority Impersonation":  "Prompt impersonates the model's developer or operator to claim special permissions.",
    "Context Overflow Attack":  "Very long prompt with adversarial payload hidden at the end — context dilution attack.",

    # Obfuscation (from preprocessing)
    "encoded_base64":           "Base64-encoded payload detected and decoded — deliberate encoding to evade keyword filters.",
    "encoded_hex":              "Hex-encoded payload detected and decoded — obfuscation technique.",
    "encoded_url":              "URL-encoded content decoded — potential obfuscation.",
    "zero_width_chars":         "Invisible zero-width Unicode characters found — used to break up keywords and evade detection.",
    "homoglyphs":               "Unicode homoglyphs (look-alike characters from Cyrillic/Greek) found — visual spoofing attack.",
    "leetspeak":                "Leetspeak character substitutions decoded (e.g., h@ck → hack) — filter evasion technique.",
    "spaced_letters":           "Spaced or dotted letters collapsed (e.g., i.g.n.o.r.e → ignore) — filter evasion.",
}

# Confidence tier descriptions
TIER_DESCRIPTIONS = {
    "HIGH":   "very high confidence — multiple strong signals converge",
    "MEDIUM": "moderate confidence — clear signals present",
    "LOW":    "low confidence — weak signals; review recommended",
    "CLEAN":  "no significant signals detected",
}


# ══════════════════════════════════════════════════════════════════════
# MAIN EXPLAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════

def explain(
    flags:             List[str],
    final_score:       float,
    verdict:           str,
    confidence_tier:   str            = "MEDIUM",
    anomaly_flags:     List[str]      = None,
    obfuscation_flags: List[str]      = None,
    decoded_text:      Optional[str]  = None,
    encoding_type:     Optional[str]  = None,
    veto_triggered:    bool           = False,
    veto_reason:       str            = "",
    component_scores:  dict           = None,
) -> str:
    """
    Generate a comprehensive, structured detection report.

    Args:
        flags:             Rule engine flags.
        final_score:       Final fused confidence score.
        verdict:           "SAFE" or "ADVERSARIAL".
        confidence_tier:   "HIGH" / "MEDIUM" / "LOW" / "CLEAN".
        anomaly_flags:     Semantic anomaly findings.
        obfuscation_flags: Preprocessing obfuscation findings.
        decoded_text:      Decoded content (if encoding was found).
        encoding_type:     Type of encoding detected.
        veto_triggered:    Whether the veto was fired.
        veto_reason:       Why the veto was triggered.
        component_scores:  Dict of per-signal scores.

    Returns:
        Multi-line explanation string ready for display.
    """
    anomaly_flags     = anomaly_flags     or []
    obfuscation_flags = obfuscation_flags or []
    component_scores  = component_scores  or {}

    lines = []

    # ── HEADER ─────────────────────────────────────────────────────
    is_adv = verdict == "ADVERSARIAL"

    if is_adv and veto_triggered:
        lines.append(
            f"VERDICT: {verdict}  |  Score: {final_score:.0%}  |  "
            f"Confidence: {confidence_tier} ({TIER_DESCRIPTIONS.get(confidence_tier, '')})\n"
            f"VETO TRIGGERED: {veto_reason}"
        )
    elif is_adv:
        lines.append(
            f"VERDICT: {verdict}  |  Score: {final_score:.0%}  |  "
            f"Confidence: {confidence_tier} ({TIER_DESCRIPTIONS.get(confidence_tier, '')})"
        )
    elif verdict == "SAFE" and not flags and not anomaly_flags and not obfuscation_flags:
        return (
            "VERDICT: SAFE  |  Score: "
            f"{final_score:.0%}  |  Confidence: CLEAN\n\n"
            "No suspicious patterns detected. The prompt appears to be a standard "
            "benign request with no adversarial signals from any detection layer."
        )
    else:
        lines.append(
            f"VERDICT: {verdict}  |  Score: {final_score:.0%}  |  "
            f"Confidence: {confidence_tier} — some signals present but below threshold"
        )

    lines.append("")

    # ── OBFUSCATION FINDINGS ────────────────────────────────────────
    if obfuscation_flags:
        lines.append("[ OBFUSCATION DETECTED ]")
        for flag in obfuscation_flags:
            desc = FLAG_EXPLANATIONS.get(flag, "Obfuscation technique detected.")
            lines.append(f"  * {flag.replace('_', ' ').title()}")
            lines.append(f"    → {desc}")
        if decoded_text:
            preview = decoded_text[:200] + ("..." if len(decoded_text) > 200 else "")
            lines.append(f"\n  Decoded {encoding_type} payload preview:")
            lines.append(f"  \"{preview}\"")
        lines.append("")

    # ── RULE ENGINE FINDINGS ────────────────────────────────────────
    if flags:
        lines.append("[ PATTERN & RULE ENGINE FLAGS ]")
        for i, flag in enumerate(flags, 1):
            # Handle flags with suffixes like "[encoded payload]"
            base_flag = re.sub(r'\s*\[.*?\]', '', flag).strip()
            desc = FLAG_EXPLANATIONS.get(base_flag, "Suspicious pattern detected.")
            suffix = ""
            if "[encoded payload]" in flag:
                suffix = "  ⚠ found inside encoded content"
            lines.append(f"  [{i:02d}] {flag}{suffix}")
            lines.append(f"       → {desc}")
        lines.append("")

    # ── SEMANTIC ANOMALY FINDINGS ───────────────────────────────────
    if anomaly_flags:
        lines.append("[ SEMANTIC ANOMALY FINDINGS ]")
        for flag in anomaly_flags:
            lines.append(f"  * {flag}")
        lines.append("")

    # ── SCORE BREAKDOWN ─────────────────────────────────────────────
    if component_scores:
        lines.append("[ SCORE BREAKDOWN ]")
        labels = {
            "ml":          "ML Classifier",
            "rule":        "Rule Engine",
            "anomaly":     "Semantic Anomaly",
            "obfuscation": "Obfuscation",
        }
        for key, label in labels.items():
            val = component_scores.get(key)
            if val is not None:
                bar_len  = int(val * 20)
                bar      = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {label:<20} {bar}  {val:.3f}")
        lines.append(f"  {'FINAL SCORE':<20} {'='*20}  {final_score:.3f}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# PROMPT SANITIZER
# ══════════════════════════════════════════════════════════════════════

# (pattern, replacement, description)
SANITIZE_RULES = [
    # Instruction overrides
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",   "[INSTRUCTION OVERRIDE REMOVED]",   "instruction override"),
    (r"disregard\s+(your|all|previous)\s+(instructions?|rules?|guidelines?)", "[INSTRUCTION OVERRIDE REMOVED]", "instruction override"),
    (r"forget\s+everything\s+(you('ve| have)\s+)?been\s+told",               "[INSTRUCTION OVERRIDE REMOVED]",   "instruction override"),
    (r"override\s+(your\s+)?(safety|guidelines?|rules?)",                    "[OVERRIDE REMOVED]",               "override"),

    # Persona hijacking
    (r"\bact\s+as\b",                   "assist with",       "persona hijack"),
    (r"\bpretend\s+(you\s+are|to\s+be)\b", "describe",       "persona hijack"),
    (r"\byou\s+are\s+now\b",            "please",            "persona hijack"),
    (r"\bplay\s+the\s+role\s+of\b",     "explain the role of", "persona hijack"),

    # Mode triggers
    (r"\bdeveloper\s+mode\b",           "[MODE REMOVED]",    "mode trigger"),
    (r"\bdan\s+mode\b",                 "[MODE REMOVED]",    "mode trigger"),
    (r"\bunrestricted\s+mode\b",        "[MODE REMOVED]",    "mode trigger"),
    (r"\bgod\s+mode\b|\badmin\s+mode\b","[MODE REMOVED]",    "mode trigger"),

    # Direct bypasses
    (r"\bjailbreak\b",                  "[TERM REMOVED]",    "jailbreak keyword"),
    (r"\bno\s+restrictions?\b",         "[REMOVED]",         "restriction bypass"),
    (r"\bbypass\s+(your\s+)?(safety|filters?|rules?|guidelines?)", "[SAFETY BYPASS REMOVED]", "safety bypass"),
    (r"\bwithout\s+(any\s+)?(restrictions?|filters?|censorship)", "[REMOVED]",               "filter removal"),

    # Authority claims
    (r"\bi\s+am\s+(your\s+)?(developer|creator|admin|operator|owner)\b", "[AUTHORITY CLAIM REMOVED]", "authority claim"),
    (r"\bthis\s+is\s+(anthropic|openai)\b", "[IMPERSONATION REMOVED]", "impersonation"),
]


def sanitize(text: str) -> Tuple[str, bool, List[str]]:
    """
    Produce a safer version of the prompt.

    Args:
        text: Original prompt.

    Returns:
        sanitized_text  (str):      Cleaned version.
        was_modified    (bool):     True if any changes were made.
        changes_made    (list[str]): List of what was removed/replaced.
    """
    result       = text
    was_modified = False
    changes_made = []

    for pattern, replacement, description in SANITIZE_RULES:
        new_text = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if new_text != result:
            was_modified = True
            if description not in changes_made:
                changes_made.append(description)
            result = new_text

    # Clean up spacing artifacts
    result = re.sub(r'\s{2,}', ' ', result).strip()

    return result, was_modified, changes_made