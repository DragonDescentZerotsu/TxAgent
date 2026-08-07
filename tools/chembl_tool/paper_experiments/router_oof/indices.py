"""Fold-specific direct Starling evidence indices for router OOF queries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

from tools.chembl_tool.common.evidence_contract import minimal_evidence_from_row
from tools.chembl_tool.common.json_utils import write_json_atomic
from tools.chembl_tool.common.molecule_identity import IDENTITY_NORMALIZER_VERSION
from tools.chembl_tool.common.starling.heldout_index import (
    identity_key,
    load_heldout_identity_keys,
)

from .contract import TaskSpec, fold_root
from .io import sha256_file as _sha256_file


def build_fold_index(
    spec: TaskSpec,
    *,
    output_root: str | Path,
    fold: int,
    source_index: str | Path | None = None,
    workers: int = 1,
    progress_every: int = 10000,
    source_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_root = fold_root(output_root, spec.task, fold)
    fold_manifest_path = current_root / "fold_manifest.json"
    if not fold_manifest_path.exists():
        raise FileNotFoundError(fold_manifest_path)
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    source_path = Path(source_index or spec.source_index)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    heldout_path = Path(fold_manifest["paths"]["heldout_molecule_labels"])
    out_dir = current_root / "evidence" / spec.direct_index_name
    del workers, progress_every  # Existing fingerprints are intentionally reused.
    return build_heldout_index_from_frozen_index(
        spec,
        source_index=source_path,
        heldout_labels=heldout_path,
        out_dir=out_dir,
        fold=fold,
        source_payload=source_payload,
    )


def build_heldout_index_from_frozen_index(
    spec: TaskSpec,
    *,
    source_index: str | Path,
    heldout_labels: str | Path,
    out_dir: str | Path,
    fold: int,
    source_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter an audited full index without reparsing/re-fingerprinting its JSONL."""
    source_path = Path(source_index)
    heldout_path = Path(heldout_labels)
    heldout_keys = load_heldout_identity_keys(heldout_path)
    if source_payload is None:
        source_payload = load_frozen_index(source_path)
    molecules = source_payload.get("molecules") or []
    fingerprints = source_payload.get("fingerprints") or []
    if len(molecules) != len(fingerprints):
        raise ValueError(f"Molecule/fingerprint mismatch in {source_path}")

    keep_old_indices: list[int] = []
    excluded_keys: set[str] = set()
    unresolved = 0
    for old_index, molecule in enumerate(molecules):
        key = _cached_identity_key(molecule)
        if not key:
            unresolved += 1
            continue
        if key in heldout_keys:
            excluded_keys.add(key)
            continue
        keep_old_indices.append(old_index)

    old_to_new = {old: new for new, old in enumerate(keep_old_indices)}
    kept_molecules = [molecules[index] for index in keep_old_indices]
    kept_fingerprints = [fingerprints[index] for index in keep_old_indices]
    kept_ids = {str(row["molecule_chembl_id"]) for row in kept_molecules}
    evidence_by_molecule = {
        molecule_id: groups
        for molecule_id, groups in (source_payload.get("evidence_by_molecule_group") or {}).items()
        if molecule_id in kept_ids
    }
    n_role_corrections = _correct_skin_index_roles(spec, evidence_by_molecule)
    group_to_indices = {
        group: [old_to_new[index] for index in indices if index in old_to_new]
        for group, indices in (source_payload.get("group_to_molecule_indices") or {}).items()
    }
    residual = {_cached_identity_key(row) for row in kept_molecules} & heldout_keys
    if residual:
        raise AssertionError(f"Held-out parent leakage remains for {len(residual)} identities")

    source_sha = _sha256_file(source_path)
    heldout_sha = _sha256_file(heldout_path)
    filtered = {
        **source_payload,
        "version": f"{spec.direct_index_name}.router_oof_fold{fold:02d}.v1",
        "molecules": kept_molecules,
        "fingerprints": kept_fingerprints,
        "group_to_molecule_indices": group_to_indices,
        "evidence_by_molecule_group": evidence_by_molecule,
        "source": {
            "type": "frozen_index_heldout_parent_filtered",
            "source_index": str(source_path),
            "source_index_sha256": source_sha,
            "upstream_source": source_payload.get("source") or {},
            "heldout_labels_jsonl": str(heldout_path),
            "heldout_labels_sha256": heldout_sha,
            "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
            "zero_parent_overlap": True,
        },
    }
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / spec.index_filename
    _write_pickle_atomic(index_path, filtered)
    stats = {
        "n_input_index_molecules": len(molecules),
        "n_output_index_molecules": len(kept_molecules),
        "n_excluded_index_molecules": len(molecules) - len(kept_molecules),
        "n_excluded_unresolved_molecules": unresolved,
        "n_heldout_parent_identities": len(heldout_keys),
        "n_matched_heldout_parent_identities": len(excluded_keys),
        "n_residual_heldout_parent_identities": 0,
        "n_skin_role_corrections": n_role_corrections,
        "zero_parent_overlap": True,
    }
    meta = {
        "index_version": filtered["version"],
        "n_index_molecules": len(kept_molecules),
        "groups": sorted(group_to_indices),
        "fingerprint": filtered.get("fingerprint") or {},
        "source": filtered["source"],
        "source_stats": {"heldout_filter": stats},
        "construction": "filter_existing_frozen_index_reuse_fingerprints.v1",
        "paths": {"index_pkl": str(index_path)},
    }
    write_json_atomic(output_dir / spec.meta_filename, meta)
    return meta


def load_frozen_index(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen index is not a mapping: {path}")
    return payload


def _cached_identity_key(molecule: dict[str, Any]) -> str:
    cache_field = "_router_parent_identity_key"
    if cache_field not in molecule:
        molecule[cache_field] = identity_key(molecule)
    return str(molecule[cache_field])


def _correct_skin_index_roles(
    spec: TaskSpec,
    evidence_by_molecule: dict[str, Any],
) -> int:
    if spec.task != "skin_reaction":
        return 0
    corrected = 0
    seen = 0
    for groups in evidence_by_molecule.values():
        for row in groups.get("Mechanism.skin_exposure", []):
            seen += 1
            record = minimal_evidence_from_row(row)
            annotations = record.setdefault("annotations", {})
            if annotations.get("evidence_role") != "context_modifier":
                annotations["evidence_role"] = "context_modifier"
                corrected += 1
            row["evidence_role"] = "context_modifier"
            row["minimal_evidence"] = record
    if not seen:
        raise ValueError("Filtered Skin index contains no Mechanism.skin_exposure evidence")
    return corrected


def _write_pickle_atomic(path: Path, payload: Any) -> None:
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
