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
# the same numbers written in groups: "912 345 678", "+351 912-345-678", "123.456.789"
_GROUPED_DIGITS = re.compile(r"(?<![\w.,/])\+?\d{2,4}(?:[ .-]\d{2,4}){2,}(?![\w/])")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# full dates only (day, month AND year): "14/03/1952", "14-3-52", "1952-03-14". Two-part values
# such as a blood pressure "140/90" are left alone.
_DATE = re.compile(r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b")


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
    """Light de-identification before logging: e-mails, full dates and long numbers (SNS, NIF,
    phone, process numbers, also when written in groups). Names and addresses are NOT detected —
    the page asks users not to type them."""
    s = _EMAIL.sub("<EMAIL>", s)
    s = _DATE.sub("<DATA>", s)
    s = _LONG_DIGITS.sub("<NUM>", s)
    return _GROUPED_DIGITS.sub(lambda m: "<NUM>" if sum(c.isdigit() for c in m[0]) >= 9 else m[0], s)
