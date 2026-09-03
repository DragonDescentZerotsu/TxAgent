"""Build Starling evidence indices after excluding all benchmark eval parents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    identity_from_record,
    normalize_molecule_identity,
)
from data.processing.evidence_library.evidence_library import (
    build_and_write_starling_index,
)


def identity_key(record: Mapping[str, Any]) -> str:
    """Return the benchmark-compatible normalized parent key for a record."""
    identity = identity_from_record(record)
    if not (identity.parent_inchi_key or identity.parent_smiles):
        identity = normalize_molecule_identity(str(record.get("drug") or ""))
    return identity.parent_inchi_key or identity.parent_smiles


def load_heldout_identity_keys(path: str | Path) -> set[str]:
    """Load and validate the parent identities emitted by the benchmark builder."""
    keys: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            stored_identity = row.get("molecule_identity") or {}
            version = stored_identity.get("normalizer_version")
            if version and version != IDENTITY_NORMALIZER_VERSION:
                raise ValueError(
                    f"{path}:{line_number} uses identity normalizer {version!r}, "
                    f"expected {IDENTITY_NORMALIZER_VERSION!r}"
                )
            stored_key = str(row.get("molecule_identity_key") or "").strip()
            computed_key = identity_key(row)
            if not computed_key:
                raise ValueError(f"{path}:{line_number} has no usable parent identity")
            if stored_key and stored_key != computed_key:
                raise ValueError(
                    f"{path}:{line_number} stores parent identity {stored_key!r}, "
                    f"but {IDENTITY_NORMALIZER_VERSION} computes {computed_key!r}"
                )
            key = stored_key or computed_key
            keys.add(key)
    if not keys:
        raise ValueError(f"{path} contains no held-out molecule identities")
    return keys


def filter_heldout_evidence_rows(
    rows: Iterable[Mapping[str, Any]],
    heldout_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove every evidence row whose normalized parent is in valid or test."""
    kept: list[dict[str, Any]] = []
    excluded_keys: set[str] = set()
    unresolved = 0
    n_input = 0
    for source_row in rows:
        n_input += 1
        row = dict(source_row)
        key = identity_key(row)
        if not key:
            unresolved += 1
            continue
        if key in heldout_keys:
            excluded_keys.add(key)
            continue
        kept.append(row)

    residual = {key for row in kept if (key := identity_key(row)) in heldout_keys}
    if residual:
        raise AssertionError(f"held-out parent leakage remains for {len(residual)} identities")
    return kept, {
        "n_input_evidence_rows": n_input,
        "n_output_evidence_rows": len(kept),
        "n_excluded_evidence_rows": n_input - len(kept),
        "n_excluded_unresolved_evidence_rows": unresolved,
        "n_heldout_parent_identities": len(heldout_keys),
        "n_matched_heldout_parent_identities": len(excluded_keys),
        "n_unresolved_evidence_rows": unresolved,
        "n_residual_heldout_parent_identities": 0,
        "zero_parent_overlap": True,
    }


def build_heldout_starling_index(
    *,
    source_evidence_jsonl: str | Path,
    heldout_labels_jsonl: str | Path,
    out_dir: str | Path,
    index_version: str,
    evidence_filename: str,
    index_filename: str,
    meta_filename: str,
    workers: int = 1,
    progress_every: int = 10000,
) -> dict[str, Any]:
    """Filter one full-source evidence file and build a leakage-audited index."""
    source_path = Path(source_evidence_jsonl)
    heldout_path = Path(heldout_labels_jsonl)
    rows = _read_jsonl(source_path)
    heldout_keys = load_heldout_identity_keys(heldout_path)
    filtered_rows, filter_stats = filter_heldout_evidence_rows(rows, heldout_keys)
    source = {
        "type": "starling_heldout_parent_filtered_index",
        "source_evidence_jsonl": str(source_path),
        "source_evidence_sha256": _sha256_file(source_path),
        "heldout_labels_jsonl": str(heldout_path),
        "heldout_labels_sha256": _sha256_file(heldout_path),
        "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
        "zero_parent_overlap": True,
    }
    return build_and_write_starling_index(
        filtered_rows,
        out_dir=out_dir,
        index_version=index_version,
        source=source,
        source_stats={"heldout_filter": filter_stats},
        evidence_filename=evidence_filename,
        index_filename=index_filename,
        meta_filename=meta_filename,
        workers=workers,
        progress_every=progress_every,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
