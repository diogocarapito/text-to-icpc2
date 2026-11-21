"""Text normalisation, used for de-duplication, leakage checks and hash-based splits.

Models always see the original text; `norm_key` is only a comparison key.
"""

import hashlib
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_CODE = re.compile(r"^\s*([A-Za-z])\s?(\d{2})\s*$")
# digit runs long enough to be an SNS/NIF/phone/process number
_LONG_DIGITS = re.compile(r"\d{6,}")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_key(s: str) -> str:
    s = strip_accents(str(s)).casefold()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def hash_fraction(s: str, salt: str) -> float:
    """Deterministic value in [0, 1) from the normalised text — stable split assignment."""
    h = hashlib.sha256(f"{salt}|{norm_key(s)}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


def as_icpc2_code(s: str) -> str | None:
    """'k86' / 'K 86' -> 'K86'; anything else -> None."""
    m = _CODE.match(s or "")
    return f"{m.group(1).upper()}{m.group(2)}" if m else None


def scrub(s: str) -> str:
    """Light de-identification before logging: mask long digit runs (SNS, NIF, phone numbers)."""
    return _LONG_DIGITS.sub("<NUM>", s)
