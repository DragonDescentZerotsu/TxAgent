"""Text normalization and phrase matching helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable


_SEPARATOR_RE = re.compile(r"[-/_,;:()\[\]{}]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Normalize text for ChEMBL assay keyword matching."""
    if value is None:
        return ""
    text = str(value).lower()
    text = _SEPARATOR_RE.sub(" ", text)
    text = text.replace(".", " ")
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def contains_phrase(text: str, phrase: str) -> bool:
    """Return true when normalized phrase occurs as token-bounded text."""
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(normalized_phrase) + r"(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def match_phrases(text: str, phrases: Iterable[str]) -> list[str]:
    """Return original phrases whose normalized form occurs in text."""
    normalized_text = normalize_text(text)
    return [phrase for phrase in phrases if contains_phrase(normalized_text, phrase)]


def join_text_parts(*parts: object) -> str:
    """Normalize and join multiple text fields for rule matching."""
    return normalize_text(" ".join(str(part) for part in parts if part is not None))
