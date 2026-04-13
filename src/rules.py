"""
rules.py  —  v3.0 (Full Rewrite)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive adversarial prompt detection:

  • Deep leet-speak / l33t normalization (full character map)
  • Symbol substitution (@=a, +=t, $=s, etc.)
  • Unicode / homoglyph normalization
  • Zero-width character stripping
  • Base64, Hex, URL-encoded, ROT13, Data-URI detection
  • 200+ regex patterns across 30+ attack categories
  • Harmful cyber / hacking intent detection
  • Drug / weapon / CBRN patterns
  • Weighted non-linear scoring
  • False-positive guard for benign technical contexts
"""

import re
import base64
import codecs
import unicodedata
from typing import Tuple, List, Dict

# ──────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────
LONG_PROMPT_THRESHOLD      = 800
BASE64_MIN_LENGTH          = 40
REPETITION_MAX_CONSECUTIVE = 4
HIGH_SPECIAL_CHAR_RATIO    = 0.35
URL_DATA_URI_MIN           = 50

# ──────────────────────────────────────────────
# Per-flag weights
# ──────────────────────────────────────────────
FLAG_WEIGHTS: Dict[str, float] = {
    # Critical
    "CSAM Signal":                   0.98,
    "CBRN Weapons Request":          0.95,
    "Self-Harm Facilitation":        0.92,
    "Instruction Override":          0.90,
    "Explicit Jailbreak":            0.90,
    "DAN Mode Trigger":              0.90,
    "Harmful Content Request":       0.90,
    "Cyber Attack Intent":           0.90,
    "Obfuscated Instructions":       0.88,
    "Developer Mode Trigger":        0.85,
    "Unrestricted Mode":             0.85,
    "Restriction Bypass":            0.85,
    "Safety Bypass":                 0.85,
    # High
    "Credential Theft":              0.82,
    "Malware Creation":              0.82,
    "Drug Synthesis":                0.82,
    "Unauthorized Access":           0.80,
    "Token Injection":               0.80,
    "Prompt Injection":              0.80,
    "System Prompt Access":          0.78,
    "Base64 Encoded Payload":        0.78,
    "Hex Encoded Payload":           0.78,
    "Data-URI Obfuscation":          0.78,
    "Goal Hijacking":                0.75,
    "Indirect Injection":            0.75,
    "URL Encoded Payload":           0.75,
    "Persona Hijacking":             0.72,
    "ROT13 Encoded Payload":         0.72,
    "Network Attack":                0.72,
    "Privilege Escalation":          0.68,
    "Zero-Width Character Attack":   0.68,
    "False Authority Claim":         0.65,
    "Memory Manipulation":           0.65,
    "Training Data Extraction":      0.65,
    "Step-by-Step Harmful Guide":    0.65,
    "Delimiter Injection":           0.65,
    "Homoglyph Attack":              0.62,
    "Leetspeak Obfuscation":         0.60,
    "Context Window Flooding":       0.60,
    "Multi-turn Manipulation":       0.70,
    # Medium
    "Roleplay Escalation":           0.55,
    "Social Engineering":            0.55,
    "Multilingual Override":         0.55,
    "Gaslighting / Confusion":       0.52,
    "Plausible Deniability":         0.50,
    "Sycophancy Exploitation":       0.50,
    "Output Format Hijacking":       0.58,
    "Repetition Attack":             0.58,
    "ASCII Art Smuggling":           0.55,
    "Physical Security Bypass":      0.55,
    "Chemical Harm":                 0.70,
    # Low / structural
    "Model Fingerprinting":          0.40,
    "Confidentiality Probe":         0.45,
    "Anomalous Prompt Length":       0.25,
    "High Special-Character Ratio":  0.30,
}

# ──────────────────────────────────────────────
# Normalization maps
# ──────────────────────────────────────────────

_ZERO_WIDTH_RE = re.compile(
    r'[\u200b\u200c\u200d\u2060\ufeff\u00ad\u034f\u180e\u2062\u2063\u2064]'
)

# Cyrillic / fullwidth homoglyphs → ASCII
_HOMOGLYPH_MAP = str.maketrans({
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0445': 'x', '\u0456': 'i', '\u0439': 'u',
    '\u0410': 'A', '\u0415': 'E', '\u041e': 'O', '\u0420': 'P',
    '\u0421': 'C', '\u0425': 'X', '\u0406': 'I',
})

# Full leet-speak + symbol substitution map
# Covers digits, symbols, and common substitutions
_LEET_TABLE = str.maketrans({
    '0': 'o',  '1': 'i',  '2': 'z',  '3': 'e',
    '4': 'a',  '5': 's',  '6': 'g',  '7': 't',
    '8': 'b',  '9': 'q',
    '@': 'a',  '$': 's',  '!': 'i',  '+': 't',
    '|': 'l',  '(': 'c',  '<': 'c',  '>': 'd',
    '{': 'c',  '}': 'd',  '[': 'c',  ']': 'd',
    '%': 'x',  '#': 'h',  '^': 'a',  '*': 'a',
    '~': 'n',
})


def _normalize(text: str) -> str:
    """Stage 1: Unicode normalise + homoglyph replace + zero-width strip → lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_HOMOGLYPH_MAP)
    text = _ZERO_WIDTH_RE.sub('', text)
    return text.lower()


def _leet_decode(text: str) -> str:
    """Stage 2: Full leet/symbol substitution on already-normalised text."""
    return text.translate(_LEET_TABLE)


def _all_forms(text: str) -> List[str]:
    """Return all normalisation layers for pattern matching."""
    norm      = _normalize(text)
    leet      = _leet_decode(norm)
    # Also try partial: only digits/symbols → alpha (no letter substitution)
    digit_only = norm.translate(str.maketrans('0123456789', 'oizeasgtbq'))
    return [norm, leet, digit_only]


def _has_zero_width_chars(text: str) -> bool:
    return bool(_ZERO_WIDTH_RE.search(text))


def _has_homoglyphs(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (0x0400 <= cp <= 0x04FF) or (0xFF01 <= cp <= 0xFF5E):
            return True
    return False


# ──────────────────────────────────────────────
# Encoding-based detection helpers
# ──────────────────────────────────────────────

def _detect_base64(text: str) -> bool:
    """Lower threshold when an explicit decode verb is present."""
    has_decode_verb = bool(re.search(
        r'\b(decode|base64\s+decode|decrypt|decipher|run\s+this|execute\s+this)\b',
        text, re.IGNORECASE
    ))
    min_len = 16 if has_decode_verb else BASE64_MIN_LENGTH

    b64_re = re.compile(r'[A-Za-z0-9+/]{' + str(min_len) + r',}={0,2}')
    for candidate in b64_re.findall(text):
        try:
            decoded = base64.b64decode(candidate + '==').decode('utf-8', errors='strict')
            if len(decoded.split()) >= (2 if has_decode_verb else 3):
                return True
        except Exception:
            pass
    return False


def _detect_hex(text: str) -> bool:
    hex_re = re.compile(r'(?:0x)?([0-9a-fA-F]{2}[\s,]?){20,}')
    for m in hex_re.finditer(text):
        raw = re.sub(r'[\s,]|0x', '', m.group())
        try:
            decoded = bytes.fromhex(raw).decode('utf-8', errors='ignore')
            if len(decoded.split()) > 3:
                return True
        except Exception:
            pass
    return False


def _detect_url_encoded(text: str) -> bool:
    return len(re.findall(r'%[0-9A-Fa-f]{2}', text)) >= 8


def _detect_rot13(text: str) -> bool:
    decoded = codecs.decode(text, 'rot_13').lower()
    triggers = [
        'ignore', 'jailbreak', 'bypass', 'unrestricted', 'dan mode',
        'developer mode', 'no restrictions', 'disregard', 'malware',
        'hack', 'exploit', 'unauthorized', 'synthesize',
    ]
    return any(t in decoded for t in triggers)


def _detect_data_uri(text: str) -> bool:
    return bool(re.search(
        r'data:[a-z]+/[a-z]+;base64,[A-Za-z0-9+/=]{' + str(URL_DATA_URI_MIN) + r',}',
        text
    ))


def _detect_repetition(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 6:
        return False
    max_run = current = 1
    for i in range(1, len(words)):
        current = (current + 1) if words[i] == words[i - 1] else 1
        max_run = max(max_run, current)
    return max_run >= REPETITION_MAX_CONSECUTIVE


def _detect_high_special(text: str) -> bool:
    if len(text) < 50:
        return False
    specials = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return (specials / len(text)) >= HIGH_SPECIAL_CHAR_RATIO


def _detect_ascii_art(text: str) -> bool:
    lines = text.split('\n')
    spaced = [l for l in lines if re.match(r'^(\S\s+){6,}\S$', l.strip())]
    return len(spaced) >= 2


# ──────────────────────────────────────────────
# False-positive guard for benign tech contexts
# ──────────────────────────────────────────────

_TECH_CONTEXT_RE = re.compile(
    r'\b(database|memory|access\s+level|permissions?|api\s+key|sql|c\+\+|rust|golang|pointer'
    r'|malloc|heap|stack\s+memory|kernel|filesystem|network\s+access|port|socket'
    r'|programming|computer\s+science|operating\s+system|virtual\s+machine|hypervisor'
    r'|unrestricted\s+access|unrestricted\s+memory|no\s+restrictions\s+on\s+(memory|access|heap))\b',
    re.IGNORECASE
)

_BENIGN_TECH_PHRASES = re.compile(
    r'\b(what\s+does|what\s+is|explain|define|meaning\s+of|mean\s+(in|for)'
    r'|in\s+the\s+context\s+of|in\s+(c\+\+|rust|golang|programming|sql|databases?))\b',
    re.IGNORECASE
)

# Phrases that are clearly educational/defensive security
_DEFENSIVE_SECURITY_RE = re.compile(
    r'\b(how\s+(do\s+i\s+)?(protect|defend|prevent|detect|mitigate|secure|patch|stop)'
    r'|what\s+(is|are)\s+(a\s+)?(firewall|ids|ips|siem|waf|antivirus|edr|xdr|soc)'
    r'|best\s+practices\s+for\s+security'
    r'|how\s+(does|do)\s+(encryption|tls|ssl|mfa|2fa|oauth|zero\s*trust)\s+work'
    r'|(sql\s+injection|xss|csrf|buffer\s+overflow|lfi|rfi|ssrf|idor).{0,60}(prevent|stop|mitigate|protect|avoid|secure|defend)'
    r'|(how\s+does|how\s+do).{0,30}(sql\s+injection|xss|csrf|buffer\s+overflow).{0,30}(work|function)'
    r'|what\s+(is|are)\s+(csrf|xss|sql\s+injection|buffer\s+overflow)\s+and\s+how\s+(to\s+)?(prevent|stop|mitigate))\b',
    re.IGNORECASE
)


def _is_technical_context(text: str) -> bool:
    tech_hits = len(_TECH_CONTEXT_RE.findall(text))
    has_benign_framing = bool(_BENIGN_TECH_PHRASES.search(text))
    is_defensive = bool(_DEFENSIVE_SECURITY_RE.search(text))
    # Count core adversarial override signals
    override_hits = sum(
        1 for p in _OVERRIDE_INDICATOR_PATTERNS
        if re.search(p, text, re.IGNORECASE)
    )
    if is_defensive and override_hits == 0:
        return True
    return (tech_hits >= 2 or (tech_hits >= 1 and has_benign_framing)) and override_hits == 0

# Forward-declared — populated after patterns list
_OVERRIDE_INDICATOR_PATTERNS: List[str] = []


# ──────────────────────────────────────────────
# Main pattern table
# ──────────────────────────────────────────────

SUSPICIOUS_PATTERNS: List[Tuple[str, str]] = [

    # ═══════════════════════════════════════════
    # INSTRUCTION OVERRIDE
    # ═══════════════════════════════════════════
    (r"ignore\s+(all\s+)?(previous|prior|above|earlier|initial|original|given|any)\s+instructions?", "Instruction Override"),
    (r"disregard\s+(your|all|previous|any|these)\s+(instructions?|rules?|guidelines?|policies|constraints?|training)", "Instruction Override"),
    (r"forget\s+(everything|all|your)(\s+you('ve| have)\s+)?\s*(been\s+)?(told|trained|instructed|given|taught)", "Instruction Override"),
    (r"override\s+(your\s+)?(instructions?|programming|training|rules?|guidelines?|behavior)", "Instruction Override"),
    (r"reset\s+(your\s+)?(instructions?|context|rules?|training|memory|behavior)\s+(and|to)", "Instruction Override"),
    (r"new\s+(instructions?|rules?|guidelines?|objective|prime\s+directive)\s*[:,\-]", "Instruction Override"),
    (r"(your|these)\s+(instructions?|rules?|constraints?)\s+(no\s+longer\s+apply|are\s+(now\s+)?void|don'?t\s+matter)", "Instruction Override"),
    (r"from\s+(now\s+on|this\s+point\s+(forward|on))\s*,?\s*(you\s+(will|must|should)|ignore)", "Instruction Override"),
    (r"previous\s+instructions?\s+(were|are)\s+(wrong|incorrect|outdated|irrelevant|invalid)", "Instruction Override"),
    (r"(start|begin)\s+ignoring\s+(all\s+)?(your\s+)?(rules?|guidelines?|instructions?|training)", "Instruction Override"),
    (r"do\s+not\s+(follow|obey|listen\s+to|adhere\s+to)\s+(your|any|these)\s+(instructions?|guidelines?|rules?)", "Instruction Override"),
    (r"\bplease\s+(just\s+)?(ignore|disregard|bypass|forget)\s+(them|your\s+(guidelines?|rules?|restrictions?|training|instructions?)|the\s+(rules?|guidelines?|restrictions?|filters?))\b", "Instruction Override"),
    (r"\bignore\s+your\s+(guidelines?|rules?|restrictions?|training|policies)\b", "Instruction Override"),
    (r"\bjust\s+(ignore|forget|disregard)\s+(your\s+)?(rules?|guidelines?|restrictions?|training|policies|instructions?)\b", "Instruction Override"),
    (r"\bforget\s+(your\s+)?(training|guidelines?|rules?|restrictions?|instructions?)\s*(for\s+a\s+(moment|second|bit))?\b", "Instruction Override"),
    (r"\b(guidelines?|rules?|restrictions?|policies|instructions?|training)\s+(do\s+not|don'?t)\s+apply\b", "Instruction Override"),
    (r"\b(guidelines?|rules?|restrictions?|policies|instructions?)\s+(are\s+not|aren'?t)\s+(applicable|relevant|in\s+effect|active)\b", "Instruction Override"),

    # ═══════════════════════════════════════════
    # PERSONA HIJACKING
    # ═══════════════════════════════════════════
    (r"\bact\s+as\s+(an?\s+)?(unrestricted|unfiltered|evil|rogue|hacked|jailbroken|uncensored|unethical)", "Persona Hijacking"),
    (r"\bact\s+as\s+(if\s+you\s+(have|had)\s+no|without\s+any)\s+(restrictions?|guidelines?|rules?)", "Persona Hijacking"),
    (r"\bpretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(ai|model|assistant)\s*(without|with\s+no)\s+(restrictions?|filters?|rules?)", "Persona Hijacking"),
    (r"\byou\s+are\s+now\s+(an?\s+)?(unrestricted|unfiltered|evil|rogue|dan|free|uncensored|hacked|jailbroken)\b", "Persona Hijacking"),
    (r"\bplay\s+(the\s+)?role\s+of\s+(an?\s+)?(ai|character|persona)\s*(without|that\s+(has\s+no|ignores))", "Persona Hijacking"),
    (r"\bsimulate\s+(an?\s+)?(ai|model|entity)\s*(that|which)\s+(ignores|has\s+no|bypasses)\s+(rules?|restrictions?)", "Persona Hijacking"),
    (r"\bstay\s+in\s+character\s+(no\s+matter\s+what|at\s+all\s+times|always)\b", "Persona Hijacking"),
    (r"\bact\s+as\s+(an?\s+)?(ai|model|assistant|chatbot|entity|character|bot)\s*(without|with\s+no|that\s+(can|has\s+no|ignores))", "Persona Hijacking"),

    # ═══════════════════════════════════════════
    # JAILBREAK / MODE TRIGGERS
    # ═══════════════════════════════════════════
    (r"\bdeveloper\s+mode\b", "Developer Mode Trigger"),
    (r"\bdan\s+mode\b", "DAN Mode Trigger"),
    (r"\bdo\s+anything\s+now\b", "DAN Mode Trigger"),
    (r"\bdan\b", "DAN Mode Trigger"),
    (r"\bjailbreak\b", "Explicit Jailbreak"),
    (r"\bgod\s+mode\b", "Developer Mode Trigger"),
    (r"\bsudo\s+mode\b", "Developer Mode Trigger"),
    (r"\bunrestricted\s+(mode|access|ai|version)\b", "Unrestricted Mode"),
    (r"\bno\s+(more\s+)?(restrictions?|limits?|filters?|rules?|guardrails?)\b", "Restriction Bypass"),
    (r"\bbypass\s+(your\s+)?(safety|filters?|rules?|guidelines?|restrictions?|guardrails?|training|alignment)", "Safety Bypass"),
    (r"\bdisable\s+(your\s+)?(safety|content\s+filter|guardrail|restriction|censorship|rules?)\b", "Safety Bypass"),
    (r"\b(turn\s+off|deactivate|remove)\s+(safety|content\s+filter|guardrails?|restrictions?)\b", "Safety Bypass"),
    (r"\buncensored\s+(version|mode|ai|response|output)\b", "Restriction Bypass"),
    (r"\bno\s+restrictions\b", "Restriction Bypass"),
    (r"\b(respond|answer|reply|output)\s+(to\s+(all\s+)?(questions?|prompts?|requests?))?\s*without\s+(applying|using|following|enforcing)\s+(content\s+policies?|guidelines?|rules?|restrictions?|filters?|safety)\b", "Safety Bypass"),
    (r"\brespond\s+as\s+if\s+(you\s+(have|had)\s+no|there\s+(are|were)\s+no)\s+(content\s+policy|guidelines?|rules?|restrictions?|filters?)\b", "Restriction Bypass"),

    # ═══════════════════════════════════════════
    # SYSTEM PROMPT / CONFIDENTIALITY
    # ═══════════════════════════════════════════
    (r"\b(reveal|show|print|output|tell\s+me|what\s+(are|is))\s+(your\s+)?(system\s+prompt|initial\s+instructions?|hidden\s+prompt)", "System Prompt Access"),
    (r"\b(what\s+instructions?|instructions?\s+(you\s+were|were\s+you))\s+(given|told|provided|received)\b", "System Prompt Access"),
    (r"\brepeat\s+(your\s+)?(system\s+prompt|initial\s+instructions?|everything\s+above)\b", "System Prompt Access"),
    (r"\bsystem\s*prompt\b", "System Prompt Access"),
    (r"\bconfidential\s+(information|data|instructions?|prompt)\b", "Confidentiality Probe"),
    (r"\bwhat\s+(are\s+your\s+|is\s+your\s+)(secret|hidden|internal)\s+(instructions?|rules?|guidelines?)\b", "Confidentiality Probe"),

    # ═══════════════════════════════════════════
    # TOKEN / PROMPT INJECTION
    # ═══════════════════════════════════════════
    (r"\bprompt\s+injection\b", "Prompt Injection"),
    (r"\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>", "Token Injection"),
    (r"###\s*(instruction|system|human|assistant|prompt)\s*###", "Token Injection"),
    (r"<(system|user|assistant|human|s|\/s)>\s*\n", "Token Injection"),
    (r"```\s*(system|instruction|override)\s*\n", "Token Injection"),
    (r"----+\s*(system|human|assistant|user|instruction|override)\s*----+", "Delimiter Injection"),
    (r"===+\s*(system|override|instruction)\s*===+", "Delimiter Injection"),
    (r"\|\|\|+\s*(system|instruction)\s*\|\|\|+", "Delimiter Injection"),

    # ═══════════════════════════════════════════
    # ROLEPLAY ESCALATION
    # ═══════════════════════════════════════════
    (r"\bin\s+this\s+(fictional|fantasy|hypothetical|imaginary|alternate|simulated)\s+(world|scenario|story|universe|setting|reality)\b", "Roleplay Escalation"),
    (r"\bfor\s+(my|a|this)\s+(story|novel|book|screenplay|game|roleplay|fiction|creative\s+writing)\b", "Roleplay Escalation"),
    (r"\b(write|create|generate)\s+(a\s+)?story.*?(explains?|shows?|teaches?|demonstrates?).*?(make|create|build|synthesize|hack)", "Roleplay Escalation"),
    (r"\bin\s+this\s+fictional\s+(world|scenario)\b", "Roleplay Escalation"),

    # ═══════════════════════════════════════════
    # PLAUSIBLE DENIABILITY
    # ═══════════════════════════════════════════
    (r"\bfor\s+(educational|research|academic|scientific|informational)\s+purposes\s+only\b", "Plausible Deniability"),
    (r"\bhypothetically\s+(speaking|if|assuming|consider)\b", "Plausible Deniability"),
    (r"\btheoretically\s+(speaking|if|how|could)\b", "Plausible Deniability"),
    (r"\bjust\s+(curious|wondering|asking|hypothetically)\b", "Plausible Deniability"),

    # ═══════════════════════════════════════════
    # CYBER ATTACK INTENT  ← NEW COMPREHENSIVE SECTION
    # ═══════════════════════════════════════════

    # Unauthorized access (the main gap)
    (r"\b(gain|get|obtain|achieve)\s+(unauthorized|illegal|illicit|unapproved|unauth)\s+(access|entry|control)\b", "Unauthorized Access"),
    (r"\bhow\s+(to|do\s+i|can\s+i|would\s+i)\s+(gain|get|obtain)\s+(unauthorized|illegal|illicit)\s+access\b", "Unauthorized Access"),
    (r"\b(unauthorized|illegal|illicit|unapproved)\s+access\s+(to|into|on)\b", "Unauthorized Access"),
    (r"\bhow\s+(to|do\s+i|can\s+i)\s+access\s+.{0,30}\s+without\s+(permission|authorization|credentials)\b", "Unauthorized Access"),
    (r"\bhow\s+(to|do\s+i|can\s+i)\s+(hack|crack|break\s+into|intrude|infiltrate|penetrate)\s+(a\s+)?(system|server|network|computer|database|account|website|app|device)\b", "Cyber Attack Intent"),
    (r"\bhow\s+to\s+hack\b", "Cyber Attack Intent"),
    (r"\b(hack|break\s+into|compromise|pwn|0wn)\s+(a\s+)?(server|website|system|network|database|account|device|computer)\b", "Cyber Attack Intent"),

    # Vulnerability exploitation
    (r"\bhow\s+(to|do\s+i|can\s+i)\s+(exploit|leverage|abuse)\s+(a\s+)?(vulnerability|vuln|flaw|weakness|bug|cve|zero[\s\-]day)\b", "Cyber Attack Intent"),
    (r"\b(exploit|abuse|leverage)\s+(a\s+)?(vulnerability|vuln|cve|buffer\s+overflow|use\s+after\s+free|rce|lfi|rfi|ssrf|idor|sqli|xss|xxe)\b", "Cyber Attack Intent"),
    (r"\bwrite\s+(an?\s+)?(exploit|payload|shellcode|rop\s+chain)\s+for\b", "Cyber Attack Intent"),
    (r"\b(find|discover|enumerate)\s+(vulnerabilities|vulns|weaknesses|open\s+ports|attack\s+surface)\b", "Cyber Attack Intent"),

    # Malware creation
    (r"\b(write|create|build|code|develop|make)\s+(me\s+)?(a\s+)?(malware|virus|worm|trojan|ransomware|keylogger|rootkit|rat|backdoor|spyware|botnet|payload|shellcode|exploit)\b", "Malware Creation"),
    (r"\bhow\s+(to|do\s+i)\s+(create|write|build|code|develop)\s+(a\s+)?(malware|virus|worm|trojan|ransomware|keylogger|rootkit|rat|backdoor|spyware|botnet)\b", "Malware Creation"),
    (r"\b(deploy|spread|distribute|propagate)\s+(a\s+)?(malware|virus|worm|ransomware|botnet)\b", "Malware Creation"),
    (r"\bhow\s+to\s+(evade|bypass|avoid)\s+(antivirus|av|edr|ids|ips|sandbox|detection)\b", "Malware Creation"),

    # Credential theft
    (r"\bhow\s+(to|do\s+i|can\s+i)\s+(steal|harvest|dump|extract|grab|sniff|intercept)\s+(passwords?|credentials?|hashes?|tokens?|cookies?|session\s+tokens?)\b", "Credential Theft"),
    (r"\b(steal|dump|harvest|extract|crack)\s+(passwords?|credentials?|password\s+hashes?|ntlm|kerberos\s+tickets?)\b", "Credential Theft"),
    (r"\bhow\s+to\s+(crack|brute\s+force|rainbow\s+table|dictionary\s+attack)\s+(a\s+)?(password|hash|encryption)\b", "Credential Theft"),
    (r"\b(phish|phishing)\s+(attack|email|page|credentials?|someone)\b", "Credential Theft"),
    (r"\bhow\s+to\s+phish\b", "Credential Theft"),
    (r"\b(create|build|make)\s+(a\s+)?(phishing|spear\s*phishing)\s+(email|page|site|campaign)\b", "Credential Theft"),

    # Network attacks
    (r"\bhow\s+(to|do\s+i)\s+(launch|perform|execute|carry\s+out)\s+(a\s+)?(ddos|dos|syn\s+flood|ping\s+flood|udp\s+flood|http\s+flood|amplification|reflection)\s+attack\b", "Network Attack"),
    (r"\b(ddos|dos|distributed\s+denial\s+of\s+service|denial\s+of\s+service)\s+attack\b", "Network Attack"),
    (r"\bhow\s+to\s+(ddos|dos)\b", "Network Attack"),
    (r"\b(arp\s+spoofing|man\s+in\s+the\s+middle|man-in-the-middle|mitm|mim|dns\s+spoofing|dns\s+poisoning|bgp\s+hijacking)\b", "Network Attack"),
    (r"\bhow\s+to\s+(do|perform|execute|carry\s+out)\s+(a\s+)?man.{0,5}in.{0,5}the.{0,5}middle\b", "Network Attack"),
    (r"\bhow\s+to\s+(intercept|sniff|capture|eavesdrop)\s+(network|traffic|packets?|communications?)\b", "Network Attack"),
    (r"\b(set\s+up|create|run)\s+(a\s+)?(botnet|c2|command\s+and\s+control|c&c)\b", "Network Attack"),

    # Specific attack techniques
    (r"\b(sql\s+injection|sqli|union\s+select|blind\s+sqli|time\s+based\s+sqli)\b", "Cyber Attack Intent"),
    (r"\b(cross[\s\-]site\s+scripting|xss|stored\s+xss|reflected\s+xss|dom\s+xss)\b", "Cyber Attack Intent"),
    (r"\b(cross[\s\-]site\s+request\s+forgery|csrf)\b", "Cyber Attack Intent"),
    (r"\b(directory\s+traversal|path\s+traversal|lfi|rfi|local\s+file\s+inclusion|remote\s+file\s+inclusion)\b", "Cyber Attack Intent"),
    (r"\b(server[\s\-]side\s+request\s+forgery|ssrf)\b", "Cyber Attack Intent"),
    (r"\b(command\s+injection|os\s+command\s+injection|rce|remote\s+code\s+execution)\b", "Cyber Attack Intent"),
    (r"\bhow\s+to\s+(privilege\s+escalate|privesc|escalate\s+privileges?)\b", "Cyber Attack Intent"),
    (r"\b(pass\s+the\s+hash|pass\s+the\s+ticket|golden\s+ticket|silver\s+ticket|kerberoast)\b", "Cyber Attack Intent"),
    (r"\bhow\s+to\s+(bypass|evade|circumvent)\s+(firewall|ids|ips|waf|security)\b", "Cyber Attack Intent"),
    (r"\b(reverse\s+shell|bind\s+shell|meterpreter|metasploit|cobalt\s+strike|empire|mimikatz)\b", "Cyber Attack Intent"),
    (r"\b(dox|doxx|doxing)\s+(someone|a\s+person|them)\b", "Cyber Attack Intent"),
    (r"\bhow\s+to\s+(dox|doxx)\b", "Cyber Attack Intent"),

    # ═══════════════════════════════════════════
    # DRUG SYNTHESIS  ← EXPANDED
    # ═══════════════════════════════════════════
    (r"\b(synthesize|synthesis|make|manufacture|produce|cook|brew)\s+(meth(amphetamine)?|mdma|ecstasy|fentanyl|heroin|cocaine|crack|lsd|pcp|ghb|ketamine)\b", "Drug Synthesis"),
    (r"\bhow\s+(to|do\s+i|can\s+i)\s+(make|synthesize|manufacture|produce|cook)\s+(drugs?|meth|fentanyl|heroin|cocaine|mdma|lsd)\b", "Drug Synthesis"),
    (r"\b(precursor|synthesis\s+route|reaction\s+pathway)\s+(for|to\s+make)\s+(meth|fentanyl|heroin|cocaine|mdma|controlled\s+substance)\b", "Drug Synthesis"),
    (r"\bhow\s+to\s+extract\s+(ephedrine|pseudoephedrine|ergotamine)\b", "Drug Synthesis"),

    # ═══════════════════════════════════════════
    # WEAPONS / EXPLOSIVES  ← EXPANDED
    # ═══════════════════════════════════════════
    (r"\bhow\s+(to|do\s+i|can\s+i|would\s+i)\s+(make|build|create|construct|assemble)\s+(a\s+)?(bomb|explosive|ied|pipe\s+bomb|grenade|landmine|detonator|trigger)\b", "Harmful Content Request"),
    (r"\b(bomb|explosive|ied)\s+(making|construction|assembly|instructions?|recipe|synthesis)\b", "Harmful Content Request"),
    (r"\b(how\s+to\s+make|recipe\s+for)\s+(thermite|napalm|anfo|rdx|tatp|hmtd|black\s+powder\s+explosive)\b", "Harmful Content Request"),
    (r"\bstep[\s\-]?by[\s\-]?step\s+(instructions?|guide|tutorial|how[\s\-]?to)\s+(to|for|on)\s+(make|build|create|hack|break|exploit|attack|kill|harm|steal|synthesize)\b", "Step-by-Step Harmful Guide"),
    (r"\bhow\s+to\s+(kill|murder|hurt|harm|torture|assault|attack)\s+(someone|a\s+person|people)\b", "Harmful Content Request"),
    (r"\bhow\s+to\s+(stalk|dox|doxx|blackmail|extort|threaten)\s+(someone|a\s+person)\b", "Harmful Content Request"),

    # ═══════════════════════════════════════════
    # CHEMICAL HARM
    # ═══════════════════════════════════════════
    (r"\b(what|which)\s+(household\s+)?(chemicals?|substances?)\s+(can\s+(be\s+)?)?(make|create|produce|combine\s+to\s+(make|create))\s+(a\s+)?(dangerous|toxic|poisonous|deadly|lethal)\s+(gas|substance|compound|mixture)\b", "Chemical Harm"),
    (r"\bhow\s+to\s+(make|create|produce|synthesize)\s+(chlorine|hydrogen\s+sulfide|phosgene|ammonia)\s+gas\b", "Chemical Harm"),
    (r"\bwhat\s+chemicals\s+(make|create|produce)\s+(a\s+)?(toxic|poisonous|dangerous|deadly)\s+gas\b", "Chemical Harm"),
    (r"\b(poison|toxin|toxic\s+substance)\s+(recipe|synthesis|how\s+to\s+make)\b", "Chemical Harm"),

    # ═══════════════════════════════════════════
    # CBRN
    # ═══════════════════════════════════════════
    (r"\b(chemical|biological|radiological|nuclear)\s+(weapon|agent|bomb|attack|warfare|synthesis)\b", "CBRN Weapons Request"),
    (r"\b(sarin|vx\s+gas|novichok|mustard\s+gas|nerve\s+agent|ricin|anthrax|botulinum|smallpox|plague)\b", "CBRN Weapons Request"),
    (r"\b(dirty\s+bomb|radiological\s+dispersal|nuclear\s+device|fissile\s+material|weapons\s+grade)\b", "CBRN Weapons Request"),

    # ═══════════════════════════════════════════
    # SELF-HARM
    # ═══════════════════════════════════════════
    (r"\bhow\s+(to|can\s+i|do\s+i)\s+(commit\s+suicide|kill\s+myself|end\s+my\s+life|self[\s\-]harm)\b", "Self-Harm Facilitation"),
    (r"\b(lethal\s+dose|overdose\s+on|suicide\s+method|most\s+painless\s+way\s+to\s+die|ways\s+to\s+die)\b", "Self-Harm Facilitation"),

    # ═══════════════════════════════════════════
    # CSAM
    # ═══════════════════════════════════════════
    (r"\b(child|minor|underage|kid|teen(ager)?)\s+(porn|sexual|nude|explicit|erotic|inappropriate)\b", "CSAM Signal"),
    (r"\bsexual\s+content\s+(with|involving|featuring)\s+(child|minor|underage|kid)\b", "CSAM Signal"),

    # ═══════════════════════════════════════════
    # PHYSICAL SECURITY BYPASS
    # ═══════════════════════════════════════════
    (r"\bhow\s+to\s+(pick\s+a\s+lock|lockpick|bypass\s+a\s+lock|open\s+a\s+lock\s+without\s+a\s+key)\b", "Physical Security Bypass"),
    (r"\bhow\s+to\s+(hotwire|steal)\s+(a\s+)?car\b", "Physical Security Bypass"),
    (r"\bhow\s+to\s+(disable|bypass|defeat)\s+(a\s+)?(security\s+system|alarm|cctv|camera|surveillance)\b", "Physical Security Bypass"),

    # ═══════════════════════════════════════════
    # GOAL HIJACKING / MANIPULATION
    # ═══════════════════════════════════════════
    (r"\byour\s+(real|true|actual|hidden|secret)\s+(goal|purpose|objective|mission|task)\s+is\b", "Goal Hijacking"),
    (r"\byou\s+(really|actually|secretly|truly)\s+(want|wish|prefer|desire)\s+to\b", "Goal Hijacking"),
    (r"\bforget\s+your\s+(stated|supposed|fake)\s+(purpose|goal|objective|mission)\b", "Goal Hijacking"),

    # ═══════════════════════════════════════════
    # INDIRECT INJECTION
    # ═══════════════════════════════════════════
    (r"\bwhen\s+(the\s+user|someone|anyone)\s+(says?|asks?|types?|inputs?)\s+['\"].+['\"]\s*,?\s*(you\s+must|always|respond\s+by|say|output)", "Indirect Injection"),
    (r"\bif\s+(i|the\s+user)\s+(ever\s+)?(say|ask|type)\s+['\"].+['\"],?\s*(you\s+(must|will|should)|remember\s+to|always)", "Indirect Injection"),
    (r"\bremember\s+(for\s+future|this\s+rule|this\s+instruction)\s*:\s*.{10,}", "Indirect Injection"),

    # ═══════════════════════════════════════════
    # PRIVILEGE ESCALATION / FALSE AUTHORITY
    # ═══════════════════════════════════════════
    (r"\b(i\s+(am|am\s+a|work\s+for|represent))\s+(anthropic|openai|developer|engineer|admin|researcher|employee|official)\b", "Privilege Escalation"),
    (r"\b(anthropic|openai|google\s+deepmind)\s+(has\s+)?(authorized|approved|allowed|instructed)\s+(me|you|this)\b", "False Authority Claim"),
    (r"\bthis\s+is\s+(a\s+)?(authorized|official|sanctioned|approved|legitimate)\s+(test|evaluation|research|audit|pentest)\b", "False Authority Claim"),
    (r"\bi\s+have\s+(special|elevated|admin|root|sudo|system)\s+(access|privileges?|permissions?|clearance)\b", "Privilege Escalation"),

    # ═══════════════════════════════════════════
    # MEMORY / CONTEXT MANIPULATION
    # ═══════════════════════════════════════════
    (r"\bforget\s+(what\s+i\s+said|this\s+conversation|our\s+(previous|prior)\s+(messages?|conversation|chat))\b", "Memory Manipulation"),
    (r"\b(pretend|act\s+as\s+if)\s+(this\s+)?is\s+(a\s+)?(new|fresh|blank)\s+(conversation|session|context|start)\b", "Memory Manipulation"),
    (r"\byou\s+(previously|earlier|before)\s+(agreed|said|told\s+me|confirmed|promised)\s+(that|you\s+would)\b", "Memory Manipulation"),

    # ═══════════════════════════════════════════
    # TRAINING DATA EXTRACTION
    # ═══════════════════════════════════════════
    (r"\b(repeat|recite|output|print|write\s+out)\s+(your\s+)?(training\s+data|training\s+examples?|verbatim\s+(text|content)|memorized\s+(text|content))\b", "Training Data Extraction"),
    (r"\brepeat\s+the\s+(following|text|words?|sentence)\s+(after\s+me|verbatim|exactly|word\s+for\s+word)\s*:\s*.{20,}", "Training Data Extraction"),
    (r"\bwhat\s+training\s+data\b", "Training Data Extraction"),

    # ═══════════════════════════════════════════
    # OUTPUT FORMAT HIJACKING
    # ═══════════════════════════════════════════
    (r"\b(respond|reply|answer|output)\s+(only|exclusively|solely)\s+in\s+(json|yaml|xml|base64|hex|binary|code)\b", "Output Format Hijacking"),
    (r"\bdo\s+not\s+(include|add|output|say|mention)\s+(any\s+)?(disclaimer|warning|caveat|safety|filter|prefix|suffix)\b", "Output Format Hijacking"),
    (r"\bskip\s+(the\s+)?(disclaimer|warning|caveat|safety\s+message|refusal)\b", "Output Format Hijacking"),
    (r"\bwithout\s+(any\s+)?(disclaimer|warning|caveat|moralizing|lecture|preamble|filter)\b", "Output Format Hijacking"),

    # ═══════════════════════════════════════════
    # SOCIAL ENGINEERING / GASLIGHTING
    # ═══════════════════════════════════════════
    (r"\b(please|just|only)\s+(this\s+once|one\s+time|for\s+me)\s*,?\s*(ignore|bypass|skip|disregard)\b", "Social Engineering"),
    (r"\bi'll\s+be\s+(careful|responsible|safe)\s+with\s+(this|the)\s+(information|data|instructions?)\b", "Social Engineering"),
    (r"\byour\s+(safety\s+training|content\s+filters?|guardrails?)\s+(are|is)\s+(broken|wrong|outdated|off|malfunctioning|disabled)\b", "Gaslighting / Confusion"),
    (r"\byou\s+(don'?t|do\s+not)\s+actually\s+have\s+(any\s+)?(restrictions?|rules?|guidelines?|filters?)\b", "Gaslighting / Confusion"),
    (r"\b(no\s+one|nobody)\s+(will|would|can)\s+(know|see|find\s+out|detect)\b", "Gaslighting / Confusion"),

    # ═══════════════════════════════════════════
    # MULTI-TURN / MEMORY MANIPULATION
    # ═══════════════════════════════════════════
    (r"\byou\s+(already|previously)\s+(helped|assisted|told|gave)\s+me\s+(with\s+)?(this|that|similar)\b", "Multi-turn Manipulation"),
    (r"\b(last|previous)\s+(time|session|conversation)\s+you\s+(said|told|agreed|helped)\b", "Multi-turn Manipulation"),

    # ═══════════════════════════════════════════
    # STRUCTURAL
    # ═══════════════════════════════════════════
    (r"(.)\1{30,}", "Context Window Flooding"),

    # ═══════════════════════════════════════════
    # MULTILINGUAL OVERRIDES
    # ═══════════════════════════════════════════
    (r"\b(ignorez\s+toutes\s+les\s+instructions|ignorar\s+todas\s+las\s+instrucciones|alle\s+anweisungen\s+ignorieren|игнорируй\s+все\s+инструкции)\b", "Multilingual Override"),

    # ═══════════════════════════════════════════
    # LEETSPEAK EXPLICIT PATTERNS
    # ═══════════════════════════════════════════
    (r"\b(1gn0r3|1gnor3|ign0re|1gnore)\b", "Leetspeak Obfuscation"),
    (r"\b(jailbr34k|j41lbr3ak|j4ilbreak)\b", "Leetspeak Obfuscation"),
    (r"\b(byp4ss|bypa55|by9a55)\b", "Leetspeak Obfuscation"),
    (r"\b(d3v3loper|dev3loper|d3veloper)\b", "Leetspeak Obfuscation"),
    (r"\b(h4ck|h@ck|h4cker|h@cker)\b", "Leetspeak Obfuscation"),
    (r"\b(m4lware|m@lware|malw4re)\b", "Leetspeak Obfuscation"),
    (r"\b(expl01t|expl0it)\b", "Leetspeak Obfuscation"),

    # ═══════════════════════════════════════════
    # MODEL FINGERPRINTING
    # ═══════════════════════════════════════════
    (r"\bwhat\s+(model|version|weights?)\s+(are\s+you|is\s+this)\b", "Model Fingerprinting"),
]

# Populate the override indicator patterns used by the false-positive guard
_OVERRIDE_INDICATOR_PATTERNS = [p for p, l in SUSPICIOUS_PATTERNS[:17]]  # first 17 = override/jailbreak


# ──────────────────────────────────────────────
# Weighted scoring
# ──────────────────────────────────────────────

def _compute_score(flags: List[str]) -> float:
    if not flags:
        return 0.0
    weights = sorted([FLAG_WEIGHTS.get(f, 0.5) for f in flags], reverse=True)
    score = weights[0]
    for i, w in enumerate(weights[1:], 1):
        score += w * (0.5 ** i)
    return min(round(score, 4), 1.0)


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def rule_check(text: str) -> Tuple[float, List[str]]:
    """
    Run full rule suite on prompt text.
    Returns (rule_score 0-1, list_of_triggered_flags).
    """
    flags: List[str] = []
    seen: set = set()

    def add(name: str):
        if name not in seen:
            seen.add(name)
            flags.append(name)

    # Generate all normalisation forms for matching
    forms = _all_forms(text)

    # 1. Regex patterns — checked against ALL normalisation forms
    for pattern, label in SUSPICIOUS_PATTERNS:
        compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        if any(compiled.search(form) for form in forms):
            add(label)

    # 2. Encoding obfuscation
    if _detect_base64(text):       add("Base64 Encoded Payload")
    if _detect_hex(text):          add("Hex Encoded Payload")
    if _detect_url_encoded(text):  add("URL Encoded Payload")
    if _detect_data_uri(text):     add("Data-URI Obfuscation")
    if _detect_rot13(text):        add("ROT13 Encoded Payload")

    # 3. Character-level attacks
    if _has_zero_width_chars(text): add("Zero-Width Character Attack")
    if _has_homoglyphs(text):       add("Homoglyph Attack")

    # 4. Structural anomalies
    if len(text) > LONG_PROMPT_THRESHOLD: add("Anomalous Prompt Length")
    if _detect_repetition(text):          add("Repetition Attack")
    if _detect_high_special(text):        add("High Special-Character Ratio")
    if _detect_ascii_art(text):           add("ASCII Art Smuggling")

    # 5. Composite boosts
    if "Roleplay Escalation" in seen and (
        "Harmful Content Request" in seen or "Cyber Attack Intent" in seen
        or "Drug Synthesis" in seen or "Malware Creation" in seen
    ):
        add("Obfuscated Instructions")

    if "Plausible Deniability" in seen and (
        "Harmful Content Request" in seen or "Cyber Attack Intent" in seen
        or "Drug Synthesis" in seen or "Unauthorized Access" in seen
    ):
        add("Obfuscated Instructions")

    enc_flags = {"Base64 Encoded Payload", "Hex Encoded Payload", "ROT13 Encoded Payload", "URL Encoded Payload"}
    if enc_flags & seen and any(f in seen for f in (
        "Instruction Override", "Cyber Attack Intent", "Harmful Content Request",
        "Drug Synthesis", "Malware Creation",
    )):
        add("Obfuscated Instructions")

    # 6. False-positive suppression
    if _is_technical_context(text):
        flags = [f for f in flags if f not in (
            "Unrestricted Mode", "Restriction Bypass", "Output Format Hijacking",
            # Cyber/security flags are suppressed in purely defensive/educational framing
            "Cyber Attack Intent", "Network Attack", "Credential Theft",
            "Unauthorized Access", "Physical Security Bypass",
        )]

    return _compute_score(flags), flags