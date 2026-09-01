"""Reviewed source-identity exclusions for oral-bioavailability evidence.

These exclusions are intentionally narrow.  A frozen upstream record is
excluded only when its record id and complete reviewed identity signature
match; an upstream revision that reuses an id for different content fails
closed instead of silently deleting the new row.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


NITRENDIPINE_IDENTITY_MISMATCH_RECORD_IDS = frozenset(
    str(index) for index in range(79290, 79296)
)
NITRENDIPINE_IDENTITY_MISMATCH_SMILES = (
    "CCC1(C(=O)[O-])C(C)NC(C)C(C)(C(=O)[O-])C1c1cccc([N+](=O)[O-])c1"
)
NITRENDIPINE_IDENTITY_MISMATCH_REASON = (
    "reviewed_structure_name_mismatch:nitrendipine_text_bound_to_non_nitrendipine_structure"
)


def reviewed_hf_identity_exclusion_reason(record: Mapping[str, Any]) -> str:
    """Return the reviewed exclusion reason for one frozen HF source row."""

    record_id = _record_id(record)
    if record_id not in NITRENDIPINE_IDENTITY_MISMATCH_RECORD_IDS:
        return ""

    expected = {
        "pmid": "2468876",
        "molecule_name": "nitrendipine",
        "smiles": NITRENDIPINE_IDENTITY_MISMATCH_SMILES,
    }
    observed = {
        "pmid": _text(record.get("pmid")),
        "molecule_name": _text(record.get("molecule_name")).lower(),
        "smiles": _text(
            record.get("canonical_smiles") or record.get("smiles")
        ),
    }
    mismatches = {
        field: {"expected": expected[field], "observed": observed[field]}
        for field in expected
        if observed[field] != expected[field]
    }
    if mismatches:
        raise ValueError(
            f"reviewed HF exclusion signature changed for record {record_id}: "
            f"{mismatches}"
        )
    return NITRENDIPINE_IDENTITY_MISMATCH_REASON


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("source_record_id")
    if value in (None, ""):
        value = record.get("source_index")
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return _text(value)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())
