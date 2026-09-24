"""Match explicit identifiers in private reasoning against a saved prompt index.

The input is a reasoning string and the exact ``reasoning_reference_index``
stored with its request; the output is the matched stable IDs, occurrence count,
and unknown or ambiguous labels. Progressive diagnostics and post-run prediction
analysis share this neutral matcher. It recognizes visible molecule, evidence
group, ``Cxx``, and ``Record N-N`` labels without inferring names or chemistry.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


MOLECULE_RE = re.compile(r"\b(molecule\s+[1-9][0-9]*)\b", re.IGNORECASE)
GROUP_RE = re.compile(
    r"\b((?:evidence\s+group|semantic\s+bucket)\s+[1-9][0-9]*)\b",
    re.IGNORECASE,
)
CARD_RE = re.compile(
    r"\b(C[0-9]{2,}|Record\s+[1-9][0-9]*-[1-9][0-9]*)\b",
    re.IGNORECASE,
)


def match_references(
    reasoning: str, index: list[Mapping[str, Any]],
) -> tuple[set[str], int, list[str], list[str], int]:
    """Match case-insensitive exact labels, leaving duplicate labels unresolved."""
    occurrences = [
        match.group(1)
        for pattern in (MOLECULE_RE, GROUP_RE, CARD_RE)
        for match in pattern.finditer(reasoning)
    ]
    by_label: dict[str, list[str]] = {}
    display: dict[str, str] = {}
    for row in index:
        key = str(row["visible_id"]).casefold()
        by_label.setdefault(key, []).append(str(row["stable_id"]))
        display[key] = str(row["visible_id"])
    found: set[str] = set()
    unknown: set[str] = set()
    ambiguous: set[str] = set()
    matched_occurrences = ambiguous_occurrences = 0
    for occurrence in occurrences:
        key = occurrence.casefold()
        matches = by_label.get(key, [])
        if len(matches) == 1:
            found.add(matches[0])
            matched_occurrences += 1
        elif len(matches) > 1:
            ambiguous.add(display[key])
            ambiguous_occurrences += 1
        else:
            unknown.add(occurrence)
    return (
        found,
        matched_occurrences,
        sorted(unknown),
        sorted(ambiguous),
        ambiguous_occurrences,
    )
