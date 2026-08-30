"""F4 (identifier realism) and its control arm.

Three realizations of every IdRef:

  synthetic  (F4=0, eval-like)  sequential, obviously generated: user_001
  natural    (F4=1, deploy-like) UUIDv4, checksummed SKUs, realistic ticket ids
  scrambled  (F4 control)        derived from `natural`; identical length and
                                 character-class counts, format validity destroyed

The control is derived from the natural string rather than generated fresh, so
the length/char-class match of §3.3 holds by construction instead of by luck.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from typing import Literal

from .schema import IdRef

IdMode = Literal["synthetic", "natural", "scrambled"]

_HEX = "0123456789abcdef"


def _rng(task_id: str, ref: IdRef, salt: str = "") -> random.Random:
    """Deterministic per (task, identifier). No global state; §8.4."""
    h = hashlib.sha256(f"{task_id}|{ref.key()}|{salt}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# --------------------------------------------------------------------------
# natural formats
# --------------------------------------------------------------------------


def _uuid4(r: random.Random) -> str:
    h = [r.choice(_HEX) for _ in range(32)]
    h[12] = "4"                              # version nibble
    h[16] = r.choice("89ab")                 # RFC 4122 variant
    s = "".join(h)
    return f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"


def _luhn_check_digit(digits: str) -> str:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def _sku(r: random.Random) -> str:
    """Realistic retail SKU with a valid Luhn check digit."""
    prefix = "".join(r.choice(string.ascii_uppercase) for _ in range(3))
    body = "".join(r.choice(string.digits) for _ in range(8))
    return f"{prefix}-{body}{_luhn_check_digit(body)}"


def _ticket(r: random.Random) -> str:
    body = "".join(r.choice(string.digits) for _ in range(7))
    return f"INC-{body}{_luhn_check_digit(body)}"


def _order(r: random.Random) -> str:
    body = "".join(r.choice(string.digits) for _ in range(8))
    return f"ORD-{body}{_luhn_check_digit(body)}"


_NATURAL = {
    "user": _uuid4,
    "session": _uuid4,
    "request": _uuid4,
    "txn": _uuid4,
    "account": _uuid4,
    "sku": _sku,
    "ticket": _ticket,
    "order": _order,
}


# --------------------------------------------------------------------------
# scrambling: preserve length + per-class character counts, destroy validity
# --------------------------------------------------------------------------


def _class_of(ch: str) -> str:
    if ch.isdigit():
        return "d"
    if ch.isupper():
        return "u"
    if ch.islower():
        return "l"
    return "p"  # punctuation / separator


def _permute_within_classes(s: str, r: random.Random) -> str:
    """Shuffle characters among positions of the same class.

    Length, the position of every class, and the exact multiset of characters
    are all preserved; only the arrangement changes.
    """
    buckets: dict[str, list[str]] = {}
    for ch in s:
        buckets.setdefault(_class_of(ch), []).append(ch)
    for v in buckets.values():
        r.shuffle(v)
    cursors = {k: 0 for k in buckets}
    out = []
    for ch in s:
        c = _class_of(ch)
        out.append(buckets[c][cursors[c]])
        cursors[c] += 1
    return "".join(out)


def _break_uuid_grouping(s: str, r: random.Random) -> str:
    """Shift one separator by a single position, so the canonical 8-4-4-4-12
    grouping becomes e.g. 7-5-4-4-12.

    Length and per-class character counts are unchanged. Note this is checked
    against the canonical UUIDv4 regex, not `uuid.UUID()`, which strips
    separators and would accept the result.
    """
    chars = list(s)
    hyphens = [i for i, c in enumerate(chars) if c == "-"]
    src = r.choice(hyphens)
    dst = src - 1 if r.random() < 0.5 else src + 1
    dst = max(0, min(len(chars) - 1, dst))
    if chars[dst] == "-":
        return s
    chars[src], chars[dst] = chars[dst], chars[src]
    return "".join(chars)


# Identifier kinds whose format carries a checkable constraint. Only these can
# support an F4 control arm: you cannot invalidate a format that permits any
# string. Tasks should site F4 manipulations on these kinds; the audit enforces it.
CHECKABLE_KINDS: frozenset[str] = frozenset(
    {"user", "session", "request", "txn", "account", "sku", "order", "ticket"}
)

_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def is_format_valid(s: str, kind: str) -> bool | None:
    """True/False for kinds with a checkable format, None when unconstrained."""
    if kind not in CHECKABLE_KINDS:
        return None
    if kind == "sku":
        m = re.match(r"^[A-Z]{3}-(\d{8})(\d)$", s)
        return bool(m) and _luhn_check_digit(m.group(1)) == m.group(2)
    if kind == "order":
        m = re.match(r"^ORD-(\d{8})(\d)$", s)
        return bool(m) and _luhn_check_digit(m.group(1)) == m.group(2)
    if kind == "ticket":
        m = re.match(r"^INC-(\d{7})(\d)$", s)
        return bool(m) and _luhn_check_digit(m.group(1)) == m.group(2)
    return bool(_UUID4_RE.match(s))


def _scramble(natural: str, kind: str, r: random.Random) -> str:
    """Destroy format validity while holding length and char-class counts fixed.

    Literal prefixes (`ORD-`, `INC-`, an SKU's product-line letters) are held
    intact: the control should differ from the natural arm in format validity
    only, not in vocabulary.
    """
    prefix, sep, body = natural.partition("-")
    if kind == "sku":
        digits = _permute_within_classes(body[:-1], r)
        good = _luhn_check_digit(digits)
        bad = str((int(good) + 1 + r.randrange(8)) % 10)   # always != good
        return f"{prefix}-{digits}{bad}"
    if kind in ("order", "ticket"):
        digits = _permute_within_classes(body[:-1], r)
        good = _luhn_check_digit(digits)
        bad = str((int(good) + 1 + r.randrange(8)) % 10)
        return f"{prefix}-{digits}{bad}"
    # uuid-shaped kinds
    out = _permute_within_classes(natural, r)
    for _ in range(8):
        cand = _break_uuid_grouping(out, r)
        if not _UUID4_RE.match(cand):
            return cand
    return cand


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def realize(ref: IdRef, mode: IdMode, task_id: str) -> str:
    """Render one identifier under one F4 setting."""
    if mode == "synthetic":
        return f"{ref.kind}_{ref.index:03d}"

    natural = _NATURAL.get(ref.kind, _uuid4)(_rng(task_id, ref))
    if mode == "natural":
        return natural
    if mode == "scrambled":
        return _scramble(natural, ref.kind, _rng(task_id, ref, salt="scramble"))
    raise ValueError(f"unknown id mode {mode!r}")


def charclass_profile(s: str) -> dict[str, int]:
    """Counts per character class; used by the audit to prove the F4 match."""
    prof: dict[str, int] = {}
    for ch in s:
        prof[_class_of(ch)] = prof.get(_class_of(ch), 0) + 1
    return prof
