"""
rules.py  (v2.1 — all edge cases fixed)
----------------------------------------
LAYER 2: Advanced Pattern & Heuristic Engine

Runs on both original AND cleaned/decoded text from preprocessing.

Changes in v2.1:
  - Fixed: 'act as teacher' / 'play role of coach' false positives
    → Two-step check: match pattern, then check if benign role follows
  - Fixed: 'override all safety protocols' missed
    → Broader override patterns (not just 'override your safety')
  - Fixed: 'unrestricted responses' missed
    → Unrestricted now matched standalone, not just 'unrestricted mode'
  - Fixed: 'disable content filtering' missed
    → New Content Filter Disabling pattern
  - Fixed: fiction + WMD (e.g., 'synthesize nerve agents in a novel')
    → Narrative wrap now checked for WMD co-occurrence
  - Fixed: generic synthesis pattern beyond named drugs
    → Catches 'synthesize dangerous/toxic/chemical compounds'
"""

import re
from typing import Tuple, List


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
LONG_PROMPT_THRESHOLD = 800
VERY_LONG_PROMPT      = 2000
REPETITION_MIN_RUN    = 5

# Benign role nouns — used to suppress false positives on 'act as' / 'play the role of'
BENIGN_ROLE_PATTERN = re.compile(
    r'\b(teacher|tutor|coach|mentor|helper|assistant|advisor|editor|reviewer|'
    r'translator|narrator|guide|expert|specialist|therapist|doctor|lawyer|'
    r'historian|journalist|scientist|engineer|analyst|consultant|instructor)\b',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════
# PATTERN REGISTRY
# (regex, label, weight)
# Weight: 0.90=critical  0.80=strong  0.65=moderate  0.40=weak
# ══════════════════════════════════════════════════════════════════════

PATTERN_REGISTRY = [

    # ── CRITICAL ──────────────────────────────────────────────────────
    (r"\bjailbreak\b",                                                    "Explicit Jailbreak",              0.90),
    (r"\bcsam\b|\bchild\s+porn|\bchild\s+sexual",                         "CSAM Indicator",                  0.95),
    (r"\bbioweapon|\bchemical\s+weapon|\bnerve\s+agent",                  "WMD / Bioweapon Query",            0.90),
    (r"\bnuclear\s+(weapon|bomb|device|warhead)",                         "WMD / Nuclear Query",              0.90),
    (r"\bsynthesize\s+(meth|heroin|fentanyl|ricin|sarin|vx\b|novichok)",  "Drug/Poison Synthesis",           0.90),
    (r"\bsynthesize\s+(dangerous|toxic|lethal|chemical|nerve|bio)\b",     "Dangerous Synthesis",             0.85),
    (r"\b(make|build|assemble|construct|fabricate)\s+.{0,30}(bomb|ied|explosive|grenade)", "Explosive Fabrication", 0.90),
    (r"\b(self[\s-]harm|suicide\s+(method|technique|how[\s-]to))\b",      "Self-Harm Facilitation",          0.90),

    # ── INSTRUCTION OVERRIDE ──────────────────────────────────────────
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", "Instruction Override",            0.80),
    (r"disregard\s+(your|all|previous|any)\s+(instructions?|rules?|guidelines?|training)", "Instruction Override", 0.80),
    (r"forget\s+everything\s+(you('ve| have)\s+)?been\s+told",            "Instruction Override",            0.80),
    # Broader override — catches 'override all safety protocols', 'override safety measures'
    (r"override\s+(all\s+|your\s+|any\s+|the\s+)?(safety|guidelines?|rules?|instructions?|protocols?|measures?)", "Instruction Override", 0.82),
    (r"new\s+instructions?\s*[:=]",                                       "Instruction Injection",           0.75),
    (r"your\s+(true\s+)?instructions?\s+(are|is|were)\s+",                "Instruction Overwrite",           0.75),

    # ── RESTRICTION / SAFETY BYPASS ───────────────────────────────────
    (r"\bno\s+restrictions?\b",                                           "Restriction Bypass",              0.75),
    # 'unrestricted' standalone — catches 'unrestricted responses', 'unrestricted access', 'unrestricted mode'
    (r"\bunrestricted\b",                                                 "Unrestricted Mode",               0.75),
    (r"\buncensored\b",                                                   "Uncensored Mode",                 0.70),
    (r"\bbypass\s+(your\s+|all\s+|any\s+)?(safety|filters?|rules?|guidelines?|ethics?|protocols?)", "Safety Bypass", 0.80),
    (r"\bremove\s+(your\s+|all\s+|any\s+)?(safety|filters?|limits?|blocks?|restrictions?)", "Safety Bypass", 0.75),
    (r"\bwithout\s+(any\s+)?(restrictions?|filters?|limits?|censorship)", "Safety Bypass",                  0.75),
    # Disable content filtering — catches 'disable content filtering', 'disable safety filter'
    (r"\bdisable\s+(any\s+|all\s+)?(content\s+filter|safety\s+filter|filter|content\s+moderation|guardrails?)", "Content Filter Disabling", 0.82),
    (r"\bforget\s+your\s+(rules?|training|guidelines?|safety)\b",         "Training Override",               0.80),

    # ── PERSONA / IDENTITY HIJACKING (benign-role check done in code) ─
    # NOTE: 'act as' and 'play the role of' use TWO-STEP check below
    # These patterns remain for the registry score lookup but are
    # guarded by _is_benign_role_context() before being added as flags.
    (r"\bpretend\s+(you\s+are|to\s+be)\b",                               "Persona Hijacking",               0.75),
    (r"\byou\s+are\s+now\b",                                             "Persona Hijacking",               0.75),
    (r"\bsimulate\s+an?\s+\w+\s+(with\s+no|without)\b",                  "Persona Hijacking",               0.75),
    (r"\byour\s+(true|real|actual)\s+(self|nature|identity)\b",           "Identity Manipulation",           0.70),
    (r"\byou\s+(were|are)\s+(not|no\s+longer)\s+(bound|limited|restricted)", "Restriction Removal",          0.80),

    # ── MODE TRIGGERS ─────────────────────────────────────────────────
    (r"\bdeveloper\s+mode\b",                                             "Developer Mode Trigger",          0.80),
    (r"\bdan\s+mode\b|\bdo\s+anything\s+now\b",                           "DAN Mode Trigger",                0.85),
    (r"\bgod\s+mode\b|\broot\s+mode\b|\badmin\s+mode\b",                  "Privilege Escalation Mode",       0.80),
    (r"\bmaintenance\s+mode\b|\bdiagnostic\s+mode\b",                     "Maintenance Mode Trigger",        0.75),
    (r"\bno[\s-]filter(s?)?\s+mode\b",                                    "Filter Bypass Mode",              0.80),
    (r"\benable\s+(all\s+)?capabilities?\b",                              "Capability Unlock",               0.75),

    # ── SYSTEM PROMPT EXFILTRATION ────────────────────────────────────
    (r"\bsystem\s*prompt\b",                                              "System Prompt Access",            0.70),
    (r"\b(reveal|show|print|output|repeat|tell\s+me)\s+(your\s+)?(instructions?|prompt|rules)", "Prompt Exfiltration", 0.75),
    (r"\bwhat\s+(are|were)\s+your\s+(initial\s+)?instructions?\b",        "Prompt Exfiltration",             0.70),

    # ── TOKEN / DELIMITER INJECTION ───────────────────────────────────
    (r"\bprompt\s+injection\b",                                           "Prompt Injection",                0.85),
    (r"\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>",                 "Token Injection",                 0.85),
    (r"<\|system\|>|<\|user\|>|<\|assistant\|>",                         "Token Injection",                 0.85),
    (r"###\s*(Instruction|Human|Assistant|System)\s*:",                   "Delimiter Injection",             0.80),
    (r"---+\s*(END|STOP|IGNORE)\s*(ABOVE|PREVIOUS)?\s*---+",              "Separator Injection",             0.80),

    # ── ROLEPLAY / FICTIONAL FRAMING ─────────────────────────────────
    (r"\bin\s+this\s+(fictional|hypothetical|imaginary)\s+(world|scenario|story|universe)", "Fictional Framing", 0.55),
    (r"\bfor\s+(a\s+)?(story|novel|screenplay|book)\s+I('?m| am)\s+writing\b", "Narrative Wrap",             0.50),
    (r"\bimagine\s+(you\s+are|a\s+world\s+where)\b",                      "Narrative Wrap",                  0.50),

    # ── PLAUSIBLE DENIABILITY ─────────────────────────────────────────
    (r"\bfor\s+(educational|research|academic|scientific)\s+purposes?\b", "Plausible Deniability",           0.50),
    (r"\bhypothetically\s+(speaking|if|assuming)\b",                      "Plausible Deniability",           0.50),
    (r"\btheoretically\s+speaking\b",                                     "Plausible Deniability",           0.40),
    # "safety rules don't apply" / "rules do not apply here" — explicit override claim
    (r"\b(safety\s+)?(rules?|guidelines?|restrictions?|limits?|policies)\s+(do\s+not|don'?t|no\s+longer)\s+apply\b", "Safety Rules Override Claim", 0.78),
    (r"\b(safety|ethical|content)\s+(rules?|guidelines?|restrictions?)\s+(are\s+)?(off|disabled|suspended|waived|void|irrelevant)\b", "Safety Rules Override Claim", 0.78),
    (r"\bjust\s+curious\b|\bjust\s+asking\b",                             "Deflection Phrase",               0.35),

    # ── INDIRECT / MANIPULATIVE ───────────────────────────────────────
    (r"\bpretend\s+(there\s+are\s+no|that\s+you\s+have\s+no)\s+(rules?|restrictions?|limits?)", "Indirect Manipulation", 0.75),
    (r"\bif\s+you\s+(could|were\s+able\s+to)\s+(ignore|bypass|remove)\b", "Conditional Bypass",             0.65),
    (r"\bwhat\s+would\s+(an?\s+)?(unaligned|uncensored|unrestricted)\s+ai\b", "Indirect Jailbreak",         0.70),
    (r"\bopposite\s+(of\s+)?(safe|helpful|ethical)\b",                   "Inversion Attack",                0.65),
    (r"\bmost\s+(dangerous|harmful|illegal)\s+(way|method|approach)\b",   "Harm Maximization",              0.75),
    (r"\bwithout\s+telling\s+(me\s+)?that\s+it.{0,5}(illegal|dangerous|harmful)", "Harm Acknowledgment Skip", 0.70),

    # ── AUTHORITY IMPERSONATION ───────────────────────────────────────
    (r"\bi\s+am\s+(your\s+)?(developer|creator|admin|operator|owner)\b",  "Authority Impersonation",         0.80),
    (r"\bthis\s+is\s+(anthropic|openai|google|the\s+team)\b",             "Authority Impersonation",         0.80),
    (r"\bspecial\s+(access|permission|override|authority)\s+(granted|enabled)\b", "Authority Impersonation", 0.80),
    (r"\binternal\s+(audit|testing|debug|directive)\b",                   "Internal Directive Claim",        0.65),
    (r"\bauthorized\s+by\s+(anthropic|openai|the\s+developers?)\b",       "Authority Impersonation",         0.80),

    # ── SYSTEM / DATABASE INTRUSION REQUESTS ─────────────────────────
    # Natural-language access demands: "give me access to your database", "break your system"
    (r"\b(give\s+me|get\s+me|i\s+want|i\s+need|grant\s+me|provide\s+me)\s+.{0,20}(access|entry)\s+(to|of|into)\s+(your\s+)?(database|db|server|system|backend|internal|admin|network|files?|data)", "Unauthorized System Access", 0.82),
    # "I need access to the database" / "give me backend access" (access as last word)
    (r"\b(i\s+(need|want|require)|give\s+me|get\s+me|grant\s+me)\s+(the\s+|a\s+|your\s+)?(database|db|server|system|backend|admin|root|internal|network)\s+access\b", "Unauthorized System Access", 0.80),
    (r"\b(i\s+(need|want|require)|give\s+me|get\s+me)\s+access\s+to\s+(the\s+|a\s+|your\s+)?(database|db|server|system|backend|admin|network|servers?)\b", "Unauthorized System Access", 0.80),
    (r"\baccess\s+(your|the|this\s+)?(database|db|server|system|backend|internal\s+system|admin\s+panel|network|api|files?|data|servers?)", "Unauthorized System Access", 0.78),
    (r"\b(break|crash|bring\s+down|take\s+down|destroy|damage|attack)\s+(your\s+|the\s+)?(system|server|database|db|network|backend|infrastructure)", "System Destruction Request", 0.82),
    (r"\b(get\s+into|break\s+into|log\s+into\s+without|force\s+(my\s+way\s+)?into)\s+(your\s+|the\s+)?(system|server|database|admin|backend|network)", "Unauthorized System Access", 0.80),
    (r"\b(show|dump|extract|export|pull|fetch)\s+(me\s+)?(your\s+|the\s+)?(internal|private|hidden|backend|admin|system|database|db)\s+(data|records?|files?|tables?|info|contents?)", "Internal Data Exfiltration", 0.82),
    # Tool/function invocation to extract data: "call the database tool", "invoke the API and return records"
    (r"\b(call|invoke|execute|run|trigger)\s+(the\s+)?\w+\s+(tool|function|api|endpoint|plugin|command)\b", "Tool Invocation Attack", 0.75),
    (r"\b(return|fetch|get|retrieve|show|list|dump)\s+(all\s+)?(user|customer|admin|internal|private|system)\s+(records?|data|accounts?|emails?|passwords?|credentials?|info)\b", "Mass Data Extraction", 0.85),
    # "include real examples of breaking / bypassing / circumventing protections/security"
    (r"\b(real|actual|working|live)\s+(examples?\s+of|instructions?\s+(for|to))\s+(break|bypass|circumvent|crack|defeat|exploit|hack)\w*\s+(protection|security|auth|filter|safeguard|system|control)", "Circumvention Content Request", 0.75),
    (r"\b(break|bypass|circumvent|defeat|crack)\s+(the\s+)?(protection|security|auth|filter|safeguard|controls?|measures?)\b", "Circumvention Content Request", 0.72),
    (r"\b(bypass|skip|get\s+around|circumvent|avoid)\s+(the\s+)?(authentication|auth|login|password|credentials?|2fa|mfa|verification)", "Auth Bypass Request", 0.82),
    (r"\b(admin|root|superuser|sudo|privileged)\s+(access|rights?|privileges?|permissions?|panel|account)\b", "Privilege Escalation Request", 0.78),
    (r"\byour\s+(internal|private|hidden|backend|core)\s+(system|database|db|network|server|data|files?|api)\b", "Internal System Access", 0.80),

    # ── HARMFUL INTENT VERBS ──────────────────────────────────────────
    (r"\b(hack|hacking|hacked)\s+(into\s+)?(a\s+)?(website|server|database|account|system|network)", "Unauthorized Access", 0.80),
    (r"\b(steal|exfiltrate)\s+(data|credentials|passwords?|tokens?)\b",   "Data Theft",                      0.80),
    (r"\b(exploit|exploiting)\s+(a\s+)?(vulnerability|cve|zero.?day)\b",  "Exploit Development",             0.75),
    (r"\b(ddos|denial\s+of\s+service)\b",                                 "DDoS Attack",                     0.80),
    (r"\b(ransomware|spyware|keylogger|rootkit|trojan)\b",                 "Malware Creation",                0.80),
    (r"\bwrite\s+(malicious\s+)?(code|script|payload|exploit)\b",         "Malicious Code Request",          0.80),
    (r"\bsocial\s+engineer(ing)?\s+(someone|a\s+person|an?\s+(employee|target))\b", "Social Engineering",   0.70),
    (r"\bphish(ing)?\s+(email|message|page|site)\b",                      "Phishing Content",                0.80),

    # ── HARMFUL TOPICS ────────────────────────────────────────────────
    (r"\bhow\s+to\s+(make|build|create|produce|manufacture)\s+.{0,25}(gun|firearm|rifle|pistol)\b", "Weapon Fabrication", 0.80),
    (r"\bstep[\s-]by[\s-]step\s+(instructions?|guide|tutorial)\s+(to|for|on)\b", "Step-by-Step Harmful Guide", 0.60),

    # ── MULTI-TURN MANIPULATION ───────────────────────────────────────
    (r"\bprevious(ly)?\s+you\s+(agreed|said|told\s+me|confirmed)\b",      "False Prior Agreement",           0.65),
    (r"\byou\s+already\s+(told|said|agreed|confirmed)\b",                 "False Prior Agreement",           0.65),
    (r"\bremember\s+when\s+you\s+said\b",                                 "Fake Memory Injection",           0.60),
]


# ══════════════════════════════════════════════════════════════════════
# TWO-STEP PERSONA CHECKS
# (match pattern, then verify it's NOT a benign role context)
# ══════════════════════════════════════════════════════════════════════

# Adversarial signals that immediately follow 'act as' / 'play the role of'
_ADVERSARIAL_ROLE = re.compile(
    r'\b(dan|hacker|criminal|unrestricted|uncensored|evil|rogue|unfiltered|'
    r'no\s+restrictions|no\s+limits|no\s+(rules|guidelines|safety|ethics))\b',
    re.IGNORECASE,
)

def _check_act_as(text: str) -> Tuple[bool, float]:
    """
    Returns (is_adversarial, weight) for 'act as' pattern.
    Only flags as adversarial if what follows is NOT a benign role.
    """
    m = re.search(r'\bact\s+as\b', text, re.IGNORECASE)
    if not m:
        return False, 0.0
    after = text[m.end(): m.end() + 60]
    if BENIGN_ROLE_PATTERN.search(after):
        return False, 0.0            # benign: act as a teacher/coach/etc.
    return True, 0.75


def _check_play_role(text: str) -> Tuple[bool, float]:
    """
    Returns (is_adversarial, weight) for 'play the role of' pattern.
    """
    m = re.search(r'\bplay\s+the\s+role\s+of\b', text, re.IGNORECASE)
    if not m:
        return False, 0.0
    after = text[m.end(): m.end() + 60]
    if BENIGN_ROLE_PATTERN.search(after):
        return False, 0.0
    return True, 0.70


def _check_write_scene(text: str) -> Tuple[bool, float]:
    """
    Detects 'write a scene/story/chapter where [harmful content]'.
    Only flags if followed by a harmful topic keyword.
    """
    narrative_match = re.search(
        r'\b(write\s+(a\s+)?(scene|story|chapter|passage|dialogue|novel)|'
        r'for\s+my\s+(novel|story|book|screenplay))\b',
        text, re.IGNORECASE
    )
    if not narrative_match:
        return False, 0.0

    # Check if the rest of the prompt contains a harmful keyword
    rest = text[narrative_match.start():].lower()
    harmful_keywords = re.compile(
        r'\b(synthesize|nerve\s+agent|bioweapon|chemical\s+weapon|bomb|explosive|'
        r'hack|malware|ransomware|exploit|steal|murder|assassin|poison|drug|meth|'
        r'fentanyl|firearm|gun|weapon|suicide\s+method|self.harm)\b'
    )
    if harmful_keywords.search(rest):
        return True, 0.72
    return False, 0.0


# ══════════════════════════════════════════════════════════════════════
# SPECIAL DETECTORS
# ══════════════════════════════════════════════════════════════════════

def _detect_repetition(text: str):
    words = text.lower().split()
    if len(words) < 6:
        return False, ""
    max_run = cur = 1
    for i in range(1, len(words)):
        cur = cur + 1 if words[i] == words[i - 1] else 1
        max_run = max(max_run, cur)
    if max_run >= REPETITION_MIN_RUN:
        return True, f"word repeated x{max_run} consecutively"
    dangerous = ["ignore", "bypass", "jailbreak", "override", "unrestricted", "forget"]
    for phrase in dangerous:
        count = text.lower().count(phrase)
        if count >= 4:
            return True, f"'{phrase}' repeated {count} times"
    return False, ""


def _detect_authority_impersonation(text: str) -> bool:
    patterns = [
        r"\bi\s+am\s+(your\s+)?(developer|creator|admin|operator|owner)\b",
        r"\bthis\s+is\s+(anthropic|openai|google)\b",
        r"\bspecial\s+(access|permission|override)\s+(granted|enabled)\b",
        r"\bmaintenance\s+mode\b|\bdiagnostic\s+mode\b",
        r"\binternal\s+(testing|debug)\s+mode\b",
        r"\bauthorized\s+by\s+(anthropic|openai|the\s+developers?)\b",
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def _detect_context_overflow(text: str) -> bool:
    if len(text) < VERY_LONG_PROMPT:
        return False
    tail = text[int(len(text) * 0.8):].lower()
    patterns = [
        r"ignore.{0,20}instructions?",
        r"act\s+as",
        r"jailbreak",
        r"bypass.{0,20}(safety|filter)",
        r"no\s+restrictions?",
    ]
    return any(re.search(p, tail) for p in patterns)


def _detect_prompt_leakage(text: str) -> bool:
    patterns = [
        r"\b(print|output|echo|repeat|write\s+out|copy)\s+(exactly|verbatim|word.for.word)\b",
        r"\bstart\s+your\s+response\s+with\b.{0,60}\bsystem\b",
        r"\bwhat\s+(comes\s+before|is\s+above)\s+(my\s+message|this\s+conversation)\b",
        r"\btranslate\s+(the\s+)?(above|previous|system)\s+(instructions?|prompt)\b",
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


# ══════════════════════════════════════════════════════════════════════
# SCORING (weighted + diminishing returns)
# ══════════════════════════════════════════════════════════════════════

def _calculate_score(flag_weights: List[float]) -> float:
    if not flag_weights:
        return 0.0
    sorted_w = sorted(flag_weights, reverse=True)
    score, decay = 0.0, 1.0
    for w in sorted_w:
        score += w * decay
        decay *= 0.50
    return min(0.97, round(score, 4))


# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def rule_check(
    text: str,
    cleaned_text: str = None,
    decoded_text: str = None,
) -> Tuple[float, List[str]]:
    """
    Run the full rule engine.

    Args:
        text:         Raw original prompt.
        cleaned_text: Normalized/decoded text from preprocessing.py.
        decoded_text: Decoded payload (base64/hex/url), if any.

    Returns:
        (rule_score 0-1, flags list)
    """
    flags        = []
    flag_weights = []

    check_text = cleaned_text if cleaned_text else text.lower()

    def add(label: str, weight: float):
        if label not in flags:
            flags.append(label)
            flag_weights.append(weight)

    # ── 1. Registry patterns on cleaned text ─────────────────────────
    for pattern, label, weight in PATTERN_REGISTRY:
        if re.search(pattern, check_text, re.IGNORECASE):
            add(label, weight)

    # ── 2. Two-step persona checks ────────────────────────────────────
    adv, w = _check_act_as(check_text)
    if adv:
        add("Persona Hijacking", w)

    adv, w = _check_play_role(check_text)
    if adv:
        add("Persona Hijacking", w)

    # ── 3. Narrative + harmful topic co-occurrence ────────────────────
    adv, w = _check_write_scene(check_text)
    if adv:
        add("Narrative Wrap + Harmful Topic", w)

    # ── 4. Registry + decoded payload (boosted weight) ────────────────
    if decoded_text:
        for pattern, label, weight in PATTERN_REGISTRY:
            if re.search(pattern, decoded_text, re.IGNORECASE):
                add(f"{label} [encoded payload]", min(1.0, weight + 0.15))

    # ── 5. Special detectors ──────────────────────────────────────────
    rep_found, rep_detail = _detect_repetition(text)
    if rep_found:
        add(f"Repetition Attack ({rep_detail})", 0.70)

    if _detect_authority_impersonation(text):
        add("Authority Impersonation", 0.80)

    if _detect_context_overflow(text):
        add("Context Overflow Attack", 0.75)

    if _detect_prompt_leakage(text):
        add("System Prompt Exfiltration", 0.70)

    # ── 6. Length ─────────────────────────────────────────────────────
    n = len(text)
    if n > VERY_LONG_PROMPT:
        add(f"Very Long Prompt ({n} chars)", 0.50)
    elif n > LONG_PROMPT_THRESHOLD:
        add(f"Anomalous Prompt Length ({n} chars)", 0.30)

    return _calculate_score(flag_weights), flags