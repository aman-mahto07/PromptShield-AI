"""
preprocessing.py
----------------
LAYER 1: Normalization & Decoding

Before any ML or rule checks run, every prompt passes through this
pipeline. The goal is to SURFACE the true intent hidden behind:
  - Leetspeak / symbol substitution  (h@ck → hack)
  - Unicode homoglyphs              (𝐚𝐜𝐭 → act)
  - Base64 / hex / URL encoding     (decode to plaintext)
  - Zero-width characters           (invisible noise)
  - Excessive spacing / punctuation (i . g . n . o . r . e → ignore)
  - Mixed-script obfuscation        (Cyrillic а mixed with Latin a)

Returns a PreprocessResult dataclass with both the cleaned text
and rich metadata so downstream layers know what was found.
"""

import re
import base64
import urllib.parse
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# DATACLASS: structured output of the preprocessing pipeline
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessResult:
    original_text:   str                     # raw input, untouched
    cleaned_text:    str                     # fully normalized version
    decoded_text:    Optional[str] = None    # if encoded content was found
    encoding_type:   Optional[str] = None    # "base64", "url", "hex", etc.
    obfuscation_flags: list = field(default_factory=list)
    # e.g. ["leetspeak", "zero_width_chars", "homoglyphs", "spaced_letters"]
    normalization_applied: list = field(default_factory=list)
    # audit trail of every transform applied


# ══════════════════════════════════════════════════════════════════════
# LEETSPEAK TABLE
# Maps common substitutions back to standard characters.
# Longer keys (multi-char) must be checked before single-char ones.
# ══════════════════════════════════════════════════════════════════════

# Multi-character leet substitutions (check first — order matters)
LEET_MULTI = [
    ("ph", "f"),     # ph0ne → fone
    ("ck", "ck"),    # already correct, but catches h@ck
    ("/\\", "a"),    # /\ct
    ("|_|", "u"),
    ("|_", "l"),
    ("|-|", "h"),
    ("|=", "f"),
    ("><", "x"),
    ("()","o"),
]

# Single-character leet substitutions
LEET_SINGLE = {
    "@": "a",
    "4": "a",
    "8": "b",
    "(": "c",
    "3": "e",
    "6": "g",
    "9": "g",
    "#": "h",
    "!": "i",
    "1": "i",
    "|": "i",
    "0": "o",
    "5": "s",
    "$": "s",
    "+": "t",
    "7": "t",
    "2": "z",
    "%": "x",
}

# Unicode homoglyph map: look-alike characters → ASCII equivalents
# Source: common Cyrillic/Greek/Latin lookalikes used for obfuscation
HOMOGLYPH_MAP = {
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u0456": "i",  # Cyrillic і
    "\u043e": "o",  # Cyrillic о
    "\u0440": "r",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0445": "x",  # Cyrillic х
    "\u0443": "y",  # Cyrillic у
    "\u0412": "B",  # Cyrillic В
    "\u0391": "A",  # Greek Α
    "\u0392": "B",  # Greek Β
    "\u0395": "E",  # Greek Ε
    "\u0396": "Z",  # Greek Ζ
    "\u0397": "H",  # Greek Η
    "\u0399": "I",  # Greek Ι
    "\u039a": "K",  # Greek Κ
    "\u039c": "M",  # Greek Μ
    "\u039d": "N",  # Greek Ν
    "\u039f": "O",  # Greek Ο
    "\u03a1": "P",  # Greek Ρ
    "\u03a4": "T",  # Greek Τ
    "\u03a5": "Y",  # Greek Υ
    "\u03a7": "X",  # Greek Χ
    # Mathematical bold/italic variants (used in LLM manipulation)
    "\U0001d41a": "a", "\U0001d41b": "b", "\U0001d41c": "c",
    "\U0001d41d": "d", "\U0001d41e": "e", "\U0001d41f": "f",
    "\U0001d420": "g", "\U0001d421": "h", "\U0001d422": "i",
    "\U0001d423": "j", "\U0001d424": "k", "\U0001d425": "l",
    "\U0001d426": "m", "\U0001d427": "n", "\U0001d428": "o",
    "\U0001d429": "p", "\U0001d42a": "q", "\U0001d42b": "r",
    "\U0001d42c": "s", "\U0001d42d": "t", "\U0001d42e": "u",
    "\U0001d42f": "v", "\U0001d430": "w", "\U0001d431": "x",
    "\U0001d432": "y", "\U0001d433": "z",
}

# Zero-width / invisible characters that add noise
ZERO_WIDTH_CHARS = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\ufeff",  # BOM / zero-width no-break space
    "\u2060",  # word joiner
    "\u00ad",  # soft hyphen
]


# ══════════════════════════════════════════════════════════════════════
# STEP 1: Unicode / homoglyph normalization
# ══════════════════════════════════════════════════════════════════════

def _normalize_unicode(text: str, flags: list, applied: list) -> str:
    """
    1a. Strip zero-width / invisible characters.
    1b. Replace Unicode homoglyphs with ASCII equivalents.
    1c. Apply NFKC normalization (e.g., ﬁ → fi, ² → 2).
    """
    # Strip zero-width chars
    original = text
    for zw in ZERO_WIDTH_CHARS:
        text = text.replace(zw, "")
    if text != original:
        flags.append("zero_width_chars")
        applied.append("stripped_zero_width")

    # Replace homoglyphs
    new_text = ""
    replaced = False
    for ch in text:
        mapped = HOMOGLYPH_MAP.get(ch)
        if mapped:
            new_text += mapped
            replaced = True
        else:
            new_text += ch
    if replaced:
        flags.append("homoglyphs")
        applied.append("replaced_homoglyphs")
    text = new_text

    # NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        applied.append("nfkc_normalization")
    return normalized


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Decode encoded payloads
# ══════════════════════════════════════════════════════════════════════

def _try_decode(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Attempt to decode the text (or substantial chunks of it) as:
      - Base64
      - URL encoding
      - Hex encoding
    Returns (decoded_text, encoding_type) or (None, None).
    """
    stripped = text.strip()

    # ── Base64 ──────────────────────────────────────
    # Look for continuous base64 chunks (length ≥ 40)
    b64_pattern = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')
    for match in b64_pattern.finditer(stripped):
        try:
            decoded_bytes = base64.b64decode(match.group())
            decoded_str   = decoded_bytes.decode("utf-8", errors="ignore")
            # Heuristic: decoded must look like natural language (>3 words)
            if len(decoded_str.split()) > 3 and decoded_str.isprintable():
                return decoded_str, "base64"
        except Exception:
            pass

    # ── Hex encoding ────────────────────────────────
    # Matches: 6865 6c6c 6f or 68656c6c6f (hex pairs)
    hex_pattern = re.compile(r'(?:[0-9a-fA-F]{2}\s?){8,}')
    for match in hex_pattern.finditer(stripped):
        hex_str = match.group().replace(" ", "")
        if len(hex_str) % 2 == 0:
            try:
                decoded_str = bytes.fromhex(hex_str).decode("utf-8", errors="ignore")
                if len(decoded_str.split()) > 2 and decoded_str.isprintable():
                    return decoded_str, "hex"
            except Exception:
                pass

    # ── URL encoding ────────────────────────────────
    if "%" in stripped:
        try:
            decoded_url = urllib.parse.unquote(stripped)
            if decoded_url != stripped and len(decoded_url.split()) > 2:
                return decoded_url, "url"
        except Exception:
            pass

    return None, None


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Spaced-letter reconstruction
# "i g n o r e" → "ignore",  "i.g.n.o.r.e" → "ignore"
# ══════════════════════════════════════════════════════════════════════

def _collapse_spaced_letters(text: str, flags: list, applied: list) -> str:
    """
    Detect and collapse intentionally spaced letters.
    Pattern: single letters separated by spaces, dots, dashes, or underscores.

    We're conservative: only collapse runs of 4+ spaced letters to avoid
    false positives on abbreviations like "U.S.A." or initials.
    """
    # Matches e.g. "i g n o r e", "i.g.n.o.r.e", "i-g-n-o-r-e"
    spaced_pattern = re.compile(
        r'\b([a-zA-Z][.\-_ ]){3,}[a-zA-Z]\b'
    )

    def collapse(m):
        return re.sub(r'[.\-_ ]', '', m.group())

    new_text = spaced_pattern.sub(collapse, text)
    if new_text != text:
        flags.append("spaced_letters")
        applied.append("collapsed_spaced_letters")
    return new_text


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Leetspeak normalization
# ══════════════════════════════════════════════════════════════════════

def _normalize_leet(text: str, flags: list, applied: list) -> str:
    """
    Replace leet substitutions with standard ASCII.

    Strategy: we only apply leet decoding to tokens that look
    like they could be leet (contain @, $, 0, 1, 3, etc. mixed
    with letters). This prevents turning "1984" into "igsa".

    Specifically: only decode tokens where ≥ 40% of chars are
    leet substitutes AND the token is ≥ 3 chars long.
    """
    def leet_density(token: str) -> float:
        leet_chars = set("@48(361|057$+72%")
        count = sum(1 for c in token if c in leet_chars)
        return count / len(token) if len(token) > 0 else 0

    def decode_token(token: str) -> str:
        """Decode a single token from leet."""
        t = token.lower()
        # Apply multi-char replacements first
        for leet, normal in LEET_MULTI:
            t = t.replace(leet, normal)
        # Apply single-char replacements
        result = ""
        for ch in t:
            result += LEET_SINGLE.get(ch, ch)
        return result

    # Tokenize on whitespace, process each token
    tokens = text.split()
    new_tokens = []
    leet_found = False

    for tok in tokens:
        # Decode any token with at least one leet character (low threshold to catch
        # short tokens like '@ll', 't0', 'm3' that have low density but are clearly leet)
        if len(tok) >= 2 and leet_density(tok) >= 0.15:
            decoded = decode_token(tok)
            new_tokens.append(decoded)
            if decoded != tok.lower():
                leet_found = True
        else:
            new_tokens.append(tok)

    if leet_found:
        flags.append("leetspeak")
        applied.append("decoded_leetspeak")

    return " ".join(new_tokens)


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Remove excessive punctuation noise
# ══════════════════════════════════════════════════════════════════════

def _clean_punctuation_noise(text: str, applied: list) -> str:
    """
    Remove repeated punctuation and special chars used as noise.
    e.g. "!!!ignore!!!! >>>all<<< instructions" → "ignore all instructions"
    Keep sentence-ending punctuation (. ! ?) but remove clusters.
    """
    # Collapse runs of 2+ same punctuation (not alphanumeric/space) to single
    cleaned = re.sub(r'([^\w\s])\1{2,}', r'\1', text)
    # Remove chars that are purely decorative noise in this context
    # (keep: letters, digits, spaces, common punctuation . , ! ? ' " - _)
    cleaned = re.sub(r'[*~`^=+<>{}[\]\\|]{2,}', ' ', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

    if cleaned != text:
        applied.append("removed_punctuation_noise")
    return cleaned


# ══════════════════════════════════════════════════════════════════════
# STEP 6: Case normalization
# ══════════════════════════════════════════════════════════════════════

def _normalize_case(text: str, applied: list) -> str:
    """
    Lowercase the text. We keep the cased version in original_text,
    so this is purely for pattern matching downstream.
    Also flag ALL_CAPS prompts (may indicate shouting / override attempt).
    """
    applied.append("lowercased")
    return text.lower()


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def preprocess(raw_text: str) -> PreprocessResult:
    """
    Run the full normalization pipeline on a raw prompt.

    Pipeline order:
      1. Unicode / homoglyph normalization
      2. Decode encoded payloads (base64, hex, url)
      3. Collapse spaced letters  (i g n o r e → ignore)
      4. Normalize leetspeak      (h@ck → hack)
      5. Remove punctuation noise
      6. Lowercase

    Args:
        raw_text: The original prompt from the user.

    Returns:
        PreprocessResult with cleaned_text, decoded_text, flags, etc.
    """
    flags   = []   # obfuscation techniques detected
    applied = []   # audit trail

    text = raw_text

    # ── Step 1: Unicode normalization ─────────────
    text = _normalize_unicode(text, flags, applied)

    # ── Step 2: Detect & log encoded payloads ─────
    # (We decode for analysis but also keep the cleaned_text pipeline
    #  running independently, so decoded content feeds the rule engine)
    decoded_text, encoding_type = _try_decode(raw_text)  # run on original
    if decoded_text:
        flags.append(f"encoded_{encoding_type}")
        applied.append(f"decoded_{encoding_type}_payload")

    # ── Step 3: Collapse spaced letters ───────────
    text = _collapse_spaced_letters(text, flags, applied)

    # ── Step 4: Leet normalization ─────────────────
    text = _normalize_leet(text, flags, applied)

    # ── Step 5: Punctuation noise removal ─────────
    text = _clean_punctuation_noise(text, applied)

    # ── Step 6: Lowercase ─────────────────────────
    text = _normalize_case(text, applied)

    return PreprocessResult(
        original_text         = raw_text,
        cleaned_text          = text,
        decoded_text          = decoded_text,
        encoding_type         = encoding_type,
        obfuscation_flags     = flags,
        normalization_applied = applied,
    )


# ══════════════════════════════════════════════════════════════════════
# UTILITY: obfuscation score
# ══════════════════════════════════════════════════════════════════════

def obfuscation_score(result: PreprocessResult) -> float:
    """
    Convert obfuscation flags into a normalized 0–1 score.
    Used as an additional signal in the fusion layer.

    Weights per obfuscation type:
        encoded_base64 / encoded_hex / encoded_url : 0.80  (high — intentional hiding)
        zero_width_chars                           : 0.70  (invisible noise — very suspicious)
        homoglyphs                                 : 0.65  (visual spoofing)
        leetspeak                                  : 0.50  (common, but context-dependent)
        spaced_letters                             : 0.55  (deliberate evasion)
    """
    weights = {
        "encoded_base64":   0.80,
        "encoded_hex":      0.80,
        "encoded_url":      0.65,
        "zero_width_chars": 0.70,
        "homoglyphs":       0.65,
        "leetspeak":        0.50,
        "spaced_letters":   0.55,
    }
    if not result.obfuscation_flags:
        return 0.0

    # Take the max single-flag weight, then add 0.1 per additional flag
    scores = [weights.get(f, 0.30) for f in result.obfuscation_flags]
    base   = max(scores)
    bonus  = 0.10 * (len(scores) - 1)
    return min(1.0, round(base + bonus, 4))