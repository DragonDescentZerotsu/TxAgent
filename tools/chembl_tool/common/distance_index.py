"""Build a ChEMBL-only superset index from frozen base and distance extensions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rdkit import DataStructs


def merge_neighbor_indices(
    base_index: Mapping[str, Any],
    extension_index: Mapping[str, Any],
    *,
    index_version: str,
    source_release: str,
    workers: int = 1,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Merge two prebuilt indices without reprocessing the frozen base rows."""
    del workers, progress_every
    base_groups = set(base_index.get("group_to_molecule_indices") or {})
    extension_groups = set(extension_index.get("group_to_molecule_indices") or {})
    overlap = base_groups & extension_groups
    if overlap:
        raise ValueError(f"Base and extension source groups overlap: {sorted(overlap)}")
    if not extension_groups or any(not group.startswith("Distance H") for group in extension_groups):
        raise ValueError("Extension index must contain only `Distance H1/H2.*` source groups.")

    if dict(base_index.get("fingerprint") or {}) != dict(extension_index.get("fingerprint") or {}):
        raise ValueError("Base and extension indices use different fingerprint contracts.")
    _validate_chembl_rows(_unique_evidence_rows(base_index))
    _validate_chembl_rows(_unique_evidence_rows(extension_index))

    base_entries = _molecule_entries(base_index)
    extension_entries = _molecule_entries(extension_index)
    molecules: list[dict[str, Any]] = []
    fingerprints: list[Any] = []
    evidence_by_molecule_group: dict[str, dict[str, list[dict[str, Any]]]] = {}
    group_to_molecule_indices: dict[str, list[int]] = {
        group_id: [] for group_id in sorted(base_groups | extension_groups)
    }
    for molecule_id in sorted(set(base_entries) | set(extension_entries)):
        base_entry = base_entries.get(molecule_id)
        extension_entry = extension_entries.get(molecule_id)
        if base_entry is not None and extension_entry is not None:
            if DataStructs.TanimotoSimilarity(base_entry[1], extension_entry[1]) != 1.0:
                raise ValueError(f"Fingerprint mismatch for shared molecule `{molecule_id}`.")
        primary = base_entry or extension_entry
        assert primary is not None
        source_molecule, source_fp, _ = primary
        combined_groups: dict[str, list[dict[str, Any]]] = {}
        for entry in (base_entry, extension_entry):
            if entry is None:
                continue
            for group_id, rows in entry[2].items():
                if group_id in combined_groups:
                    raise ValueError(
                        f"Molecule `{molecule_id}` has duplicate evidence group `{group_id}` across indices."
                    )
                combined_groups[group_id] = rows
        molecule_index = len(molecules)
        for group_id in sorted(combined_groups):
            group_to_molecule_indices[group_id].append(molecule_index)
        molecule = dict(source_molecule)
        molecule["groups"] = sorted(combined_groups)
        molecule["n_evidence_rows"] = sum(len(rows) for rows in combined_groups.values())
        molecules.append(molecule)
        fingerprints.append(source_fp)
        evidence_by_molecule_group[molecule_id] = combined_groups

    merged = {
        "version": index_version,
        "fingerprint": dict(base_index.get("fingerprint") or {}),
        "molecules": molecules,
        "fingerprints": fingerprints,
        "group_to_molecule_indices": group_to_molecule_indices,
        "evidence_by_molecule_group": evidence_by_molecule_group,
    }
    merged["source"] = {
        "type": "ChEMBL",
        "dataset": "ChEMBL",
        "release": source_release,
        "mixed_sources": False,
        "levels": ["D", "C", "H1", "H2"],
    }
    merged["distance_index"] = {
        "base_index_version": base_index.get("version"),
        "extension_index_version": extension_index.get("version"),
        "base_group_ids": sorted(base_groups),
        "extension_group_ids": sorted(extension_groups),
    }
    return merged


def _unique_evidence_rows(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for molecule_id in sorted(index.get("evidence_by_molecule_group") or {}):
        groups = index["evidence_by_molecule_group"][molecule_id]
        for group_id in sorted(groups):
            for row in groups[group_id]:
                row_id = id(row)
                if row_id in seen:
                    continue
                seen.add(row_id)
                rows.append(row)
    return rows


def _molecule_entries(
    index: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], Any, dict[str, list[dict[str, Any]]]]]:
    molecules = list(index.get("molecules") or [])
    fingerprints = list(index.get("fingerprints") or [])
    if len(molecules) != len(fingerprints):
        raise ValueError("Index molecule/fingerprint arrays have different lengths.")
    evidence = index.get("evidence_by_molecule_group") or {}
    entries = {}
    for molecule, fingerprint in zip(molecules, fingerprints, strict=True):
        molecule_id = str(molecule.get("molecule_chembl_id") or "")
        if not molecule_id or molecule_id in entries:
            raise ValueError(f"Invalid or duplicate molecule ID in index: `{molecule_id}`.")
        entries[molecule_id] = (molecule, fingerprint, evidence.get(molecule_id, {}))
    return entries


def _validate_chembl_rows(rows: list[dict[str, Any]]) -> None:
    invalid_sources = sorted(
        {
            str(row.get("evidence_source") or "").strip()
            for row in rows
            if str(row.get("evidence_source") or "").strip().lower() not in {"", "chembl"}
        }
    )
    if invalid_sources:
        raise ValueError(f"Distance index cannot mix non-ChEMBL evidence: {invalid_sources}")
