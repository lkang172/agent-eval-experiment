"""Pluggable token counting.

Length matching (§8.1) is defined on tokens, not characters. The real tokenizer
is only available once the model environment exists, so counts carry an
`exact` flag and nothing downstream may silently treat an estimate as exact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TOKENIZER = None
_TRIED = False

# Ungated mirrors of the Llama-3.1 tokenizer, in preference order.
CANDIDATES = [
    os.environ.get("RVR_TOKENIZER", ""),
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]


@dataclass(frozen=True)
class TokenCount:
    n: int
    exact: bool
    source: str


def _load():
    global _TOKENIZER, _TRIED
    if _TRIED:
        return _TOKENIZER
    _TRIED = True
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    for name in CANDIDATES:
        if not name:
            continue
        try:
            _TOKENIZER = (AutoTokenizer.from_pretrained(name), name)
            return _TOKENIZER
        except Exception:
            continue
    return None


def count(text: str) -> TokenCount:
    tk = _load()
    if tk is None:
        # Conservative proxy; flagged inexact so audits report it as such.
        return TokenCount(max(1, round(len(text) / 3.6)), False, "char/3.6 estimate")
    tok, name = tk
    return TokenCount(len(tok.encode(text, add_special_tokens=False)), True, name)


def is_exact() -> bool:
    return _load() is not None
