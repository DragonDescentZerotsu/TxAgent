"""Fresh, parent-local similarity neighborhoods for second-pass LLM review."""

from __future__ import annotations

from collections import defaultdict
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


MAX_POSTING = 1000


def _text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\W+", " ", normalized).strip()


def _search_tokens(group: dict) -> set[str]:
    tokens: set[str] = set()
    for value in [str(group["label"]), *group["values"]]:
        words = _text_key(value).split()
        tokens.update(word for word in words if len(word) > 1)
        # Full-name and abbreviated assay labels can then share a posting.
        for length in (2, 3):
            tokens.update(
                "".join(word[0] for word in words[start : start + length])
                for start in range(len(words) - length + 1)
                if all(len(word) > 2 for word in words[start : start + length])
            )
    return tokens


def _nearby_batches(groups: list[dict], batch_size: int) -> list[list[dict]]:
    if len(groups) <= batch_size:
        return [groups]
    search_text = [
        _text_key(str(group["label"])) or _text_key(group["values"][0]) or str(group["group_id"])
        for group in groups
    ]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), dtype=np.float32).fit_transform(search_text)
    postings: dict[str, list[int]] = defaultdict(list)
    member_tokens = [_search_tokens(group) for group in groups]
    for index, tokens in enumerate(member_tokens):
        for token in tokens:
            postings[token].append(index)

    remaining = set(range(len(groups)))
    batches = []
    cursor = 0
    while remaining:
        while cursor not in remaining:
            cursor += 1
        seed = cursor
        useful = sorted(
            (token for token in member_tokens[seed] if 1 < len(postings[token]) <= MAX_POSTING),
            key=lambda token: (len(postings[token]), token),
        )[:4]
        candidates = sorted(({index for token in useful for index in postings[token]} & remaining) - {seed})
        if candidates:
            scores = np.asarray(matrix[candidates].dot(matrix[seed].T).toarray()).ravel()
            ranked = sorted(
                zip(candidates, scores),
                key=lambda item: (
                    -float(item[1])
                    - sum(1 / len(postings[token]) for token in useful if token in member_tokens[item[0]]),
                    item[0],
                ),
            )
            chosen = [seed, *(index for index, _ in ranked[: batch_size - 1])]
        else:
            chosen = [seed]
        # Fill sparse neighborhoods to avoid a separate request for each singleton.
        filler = cursor + 1
        chosen_set = set(chosen)
        while len(chosen) < batch_size and filler < len(groups):
            if filler in remaining and filler not in chosen_set:
                chosen.append(filler)
            filler += 1
        remaining.difference_update(chosen)
        batches.append([groups[index] for index in chosen])
    return batches


def batch_groups(groups: list[dict], *, batch_size: int = 40) -> list[list[dict]]:
    """Batch every provisional group once, across Pass-1 boundaries but not parents."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    by_parent: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for group in groups:
        parent = str(group["parent_id"])
        group_id = str(group["group_id"])
        values = group["values"]
        if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
            raise ValueError(f"group {group_id} must contain canonical values")
        identity = (parent, group_id)
        if identity in seen:
            raise ValueError(f"duplicate provisional group: {identity}")
        seen.add(identity)
        by_parent[parent].append(group)

    batches = []
    for parent in sorted(by_parent):
        ordered = sorted(
            by_parent[parent],
            key=lambda group: (
                _text_key(str(group["label"])),
                _text_key(min(group["values"])),
                str(group["group_id"]),
            ),
        )
        batches.extend(_nearby_batches(ordered, batch_size))
    return batches
