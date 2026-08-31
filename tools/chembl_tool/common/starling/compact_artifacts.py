"""Compact persisted records, relational evidence, and portable neighbor indices.

Every function here is task-agnostic.  The task supplies a
:class:`CompactArtifactProfile` carrying its artifact/index version strings, the
human-readable evidence-source label, and its source-column contract.  Task
modules bind one profile and re-export thin wrappers.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import DataStructs

from tools.chembl_tool.common.evidence_contract import attach_minimal_evidence
from tools.chembl_tool.common.starling.assay_transfer_measurements import (
    DISPLAY_INVERSE_LOG10,
    DISPLAY_INVERSE_LOGIT,
    display_scalar_value,
)
from tools.chembl_tool.common.starling.evidence_library import starling_molecule_id
from tools.chembl_tool.common.starling.normalization.audit import write_parquet
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256, stable_id
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
    fingerprint_metadata,
)


# These values are either duplicated JSON, derivable from retained columns, or
# constant for an entire artifact and therefore belong in manifests/registries.
BANNED_PERSISTED_FIELDS = frozenset(
    {
        "source_payload_json",
        "evidence_context_json",
        "llm_source_fields_json",
        "llm_source_contract_json",
        "cleaning_version",
        "normalization_version",
        "measurement_normalization_version",
        "spacing_and_spelling_version",
        "source_scalar_rule_version",
        "auxiliary_attachment_version",
        "report_type_normalization_version",
        "source_column_contract_version",
        "organization_version",
        "source_revision",
        "source_path",
        "source_sha256",
        "unit_dimension_json",
        "spacing_and_spelling_reason",
        "source_scalar_rule_reason",
        "duplicate_source_record_ids",
        "assay_tier",
        "endpoint_group",
        "evidence_role",
        "target_pref_name",
    }
)


@dataclass(frozen=True)
class CompactArtifactProfile:
    """Per-task identity for the compact artifact and index families."""

    task_id: str
    artifact_version: str
    index_version: str
    evidence_source_label: str
    source_columns: Mapping[str, tuple[str, ...]]
    llm_source_projection: Callable[[Mapping[str, Any]], dict[str, Any]]
    record_contract_version: str = ""
    source_contract_version: str = ""


def compact_persisted_records(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove only frozen, derivable, or duplicated infrastructure fields."""
    return [compact_persisted_record(row) for row in rows]


def compact_persisted_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compact one row without requiring a second corpus-sized list."""
    return {
        str(field): value
        for field, value in row.items()
        if field not in BANNED_PERSISTED_FIELDS
    }


def assert_compact_schema(rows: Sequence[Mapping[str, Any]]) -> None:
    found = sorted(
        BANNED_PERSISTED_FIELDS
        & {str(field) for row in rows for field in row}
    )
    if found:
        raise ValueError(f"compact artifact contains banned persisted fields: {found}")


def build_relational_evidence_catalog(
    records: Sequence[Mapping[str, Any]],
    *,
    max_record_examples: int = 6,
    family_resolver: (
        Callable[[str, str, Mapping[str, Any] | None], Any] | None
    ) = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build molecule-family summaries plus references to finalized records.

    ``assay_tier``, ``endpoint_group``, ``evidence_role`` and ``target_pref_name``
    are derivable, so ``compact_persisted_records`` strips them before the
    organize stage is written.  Records reloaded from that artifact -- which is
    what happens whenever the index stage is resumed rather than run end to end
    -- therefore no longer carry them.  Passing ``family_resolver`` re-derives
    them so a resumed build and a full build produce the same catalog.
    """
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("retrieval_eligible"):
            grouped[
                (str(record.get("canonical_smiles") or ""), str(record.get("group_id") or ""))
            ].append(record)

    families: list[dict[str, Any]] = []
    bridge: list[dict[str, Any]] = []
    for (smiles, group_id), group_records in sorted(grouped.items()):
        record_id_field = (
            "canonical_record_id"
            if "canonical_record_id" in group_records[0]
            else "normalized_record_id"
        )
        evidence_id = stable_id("molecule_family_evidence", smiles, group_id)
        molecule_id = starling_molecule_id(smiles)
        ranked = sorted(group_records, key=_representative_sort_key)
        representative_ids = _representative_ids(ranked, max_record_examples)
        representative_rank = {
            record_id: rank
            for rank, record_id in enumerate(representative_ids, start=1)
        }
        source_record_count = sum(
            int(record.get("source_record_count") or 1) for record in group_records
        )
        scalar_count = sum(
            int(record.get("source_record_count") or 1)
            for record in group_records
            if record.get("finite_scalar_value") is not None
        )
        confidences = [
            float(record["confidence"])
            for record in group_records
            if record.get("confidence") is not None
        ]
        endpoint_counts = Counter(
            str(record.get("endpoint_name") or "unspecified")
            for record in group_records
        )
        source_ids = sorted(
            {
                source_id
                for record in group_records
                for source_id in (
                    _json_list(record.get("source_ids_json"))
                    or [str(record.get("source_id") or "")]
                )
                if source_id
            }
        )
        retrieval_source_ids = sorted(
            {
                str(record.get("retrieval_source_id") or "")
                for record in group_records
                if record.get("retrieval_source_id")
            }
        )
        source_names = sorted(
            {
                str(record.get("source_name") or "")
                for record in group_records
                if record.get("source_name")
            }
        )
        family_fields = _family_fields(group_records[0], family_resolver)
        families.append(
            {
                "evidence_id": evidence_id,
                "molecule_id": molecule_id,
                "canonical_smiles": smiles,
                "group_id": group_id,
                "assay_tier": family_fields["assay_tier"],
                "endpoint_group": family_fields["endpoint_group"],
                "evidence_role": family_fields["evidence_role"],
                "target_pref_name": family_fields["target_pref_name"],
                "source_record_count": source_record_count,
                "collapsed_record_count": len(group_records),
                "source_numeric_record_count": scalar_count,
                "source_qualitative_record_count": source_record_count - scalar_count,
                "median_confidence": (
                    round(statistics.median(confidences), 4) if confidences else None
                ),
                "source_ids": source_ids,
                "retrieval_source_ids": retrieval_source_ids,
                "source_names": source_names,
                "endpoint_counts": [
                    {"endpoint": endpoint, "count": count}
                    for endpoint, count in endpoint_counts.most_common(8)
                ],
                "uncertainty": (
                    ["qualitative_or_non_scalar_records_present"]
                    if scalar_count != source_record_count
                    else []
                ),
            }
        )
        for order, record in enumerate(ranked):
            record_id = str(record.get(record_id_field) or "")
            if not record_id:
                raise ValueError(f"evidence catalog record lacks {record_id_field}")
            bridge.append(
                {
                    "evidence_id": evidence_id,
                    record_id_field: record_id,
                    "record_order": order,
                    "representative_rank": representative_rank.get(record_id),
                }
            )
    return families, bridge


_FAMILY_FIELDS = (
    "assay_tier",
    "endpoint_group",
    "evidence_role",
    "target_pref_name",
)


def _family_fields(
    record: Mapping[str, Any],
    family_resolver: (
        Callable[[str, str, Mapping[str, Any] | None], Any] | None
    ),
) -> dict[str, str]:
    """Read the family labels off a record, re-deriving any the artifact dropped."""
    resolved = {
        field: str(record.get(field) or "") for field in _FAMILY_FIELDS
    }
    if all(resolved.values()) or family_resolver is None:
        return resolved
    assignment = family_resolver(
        str(record.get("source_id") or ""),
        str(record.get("endpoint_name") or ""),
        record,
    )
    if assignment is None:
        return resolved
    for field in _FAMILY_FIELDS:
        if not resolved[field]:
            resolved[field] = str(getattr(assignment, field, "") or "")
    return resolved


def write_compact_neighbor_index(
    *,
    profile: CompactArtifactProfile,
    families: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    workers: int = 1,
    progress_every: int = 0,
    standardized_by_molecule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a portable index without embedding any evidence records."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    thin_evidence = [
        {
            "evidence_id": family["evidence_id"],
            "molecule_chembl_id": family["molecule_id"],
            "canonical_smiles": family["canonical_smiles"],
            "group_id": family["group_id"],
            "evidence_source": profile.evidence_source_label,
        }
        for family in families
    ]
    index = build_neighbor_index(
        thin_evidence,
        index_version=profile.index_version,
        workers=workers,
        progress_every=progress_every,
        standardized_by_molecule=standardized_by_molecule,
    )
    evidence_ids = {
        (str(family["molecule_id"]), str(family["group_id"])): str(
            family["evidence_id"]
        )
        for family in families
    }
    retrieval_sources_by_evidence = {
        str(family["evidence_id"]): _list_value(
            family.get("retrieval_source_ids")
        )
        for family in families
    }
    molecules: list[dict[str, Any]] = []
    for molecule_index, molecule in enumerate(index["molecules"]):
        identity = dict(molecule.get("molecule_identity") or {})
        molecules.append(
            {
                "molecule_index": molecule_index,
                "molecule_id": molecule["molecule_chembl_id"],
                "canonical_smiles": molecule["canonical_smiles"],
                "standard_inchi_key": molecule.get("standard_inchi_key"),
                "parent_smiles": identity.get("parent_smiles"),
                "parent_inchi_key": identity.get("parent_inchi_key"),
                "parent_connectivity_key": identity.get("parent_connectivity_key"),
                "component_parent_inchi_keys": list(
                    identity.get("component_parent_inchi_keys") or []
                ),
                "identity_status": identity.get("status"),
            }
        )
    memberships: list[dict[str, Any]] = []
    for group_id, molecule_indices in sorted(index["group_to_molecule_indices"].items()):
        for molecule_index in molecule_indices:
            molecule_id = str(index["molecules"][molecule_index]["molecule_chembl_id"])
            memberships.append(
                {
                    "group_id": group_id,
                    "molecule_index": molecule_index,
                    "molecule_id": molecule_id,
                    "evidence_id": evidence_ids[(molecule_id, group_id)],
                    "retrieval_source_ids": retrieval_sources_by_evidence[
                        evidence_ids[(molecule_id, group_id)]
                    ],
                }
            )

    fingerprint_bytes = b"".join(
        DataStructs.BitVectToBinaryText(fingerprint)
        for fingerprint in index["fingerprints"]
    )
    packed = np.frombuffer(fingerprint_bytes, dtype=np.uint8).reshape(
        len(index["fingerprints"]), 256
    )

    molecules_path = target / "molecules.parquet"
    membership_path = target / "group_membership.parquet"
    fingerprints_path = target / "fingerprints.npz"
    write_parquet(molecules_path, molecules)
    write_parquet(membership_path, memberships)
    np.savez_compressed(
        fingerprints_path,
        packed_fingerprints=packed,
        fingerprint_size=np.asarray([2048], dtype=np.int32),
        bitorder=np.asarray(["little"]),
    )
    manifest = {
        "index_version": profile.index_version,
        "artifact_version": profile.artifact_version,
        "task_id": profile.task_id,
        "molecules": len(molecules),
        "memberships": len(memberships),
        "groups": sorted(index["group_to_molecule_indices"]),
        "fingerprint": fingerprint_metadata(),
        "files": {
            name: {"path": name, "sha256": file_sha256(target / name)}
            for name in (
                "molecules.parquet",
                "group_membership.parquet",
                "fingerprints.npz",
            )
        },
        "validations": {
            "one_fingerprint_per_molecule": packed.shape[0] == len(molecules),
            "fingerprints_are_2048_bits": packed.shape[1] == 256,
            "evidence_not_embedded": True,
        },
    }
    if profile.record_contract_version:
        manifest["record_contract_version"] = profile.record_contract_version
    if profile.source_contract_version:
        manifest["source_contract_version"] = profile.source_contract_version
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_compact_neighbor_index(
    index_dir: str | Path,
    *,
    profile: CompactArtifactProfile,
    evidence_dir: str | Path | None = None,
    records_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load compact files and hydrate prompt evidence once in memory.

    The disk index contains only molecule identities, fingerprints, and
    evidence IDs.  Finalized records remain the sole record-level source of
    truth; this loader joins their representative IDs when the index is loaded,
    before any query is served.
    """
    import pandas as pd
    from rdkit import DataStructs

    index_root = Path(index_dir)
    manifest = json.loads((index_root / "manifest.json").read_text(encoding="utf-8"))
    artifact_root = index_root.parent
    manifest_evidence_dir = str(manifest.get("evidence_dir") or "").strip()
    manifest_records_path = str(manifest.get("records_path") or "").strip()
    catalog_root = (
        Path(evidence_dir)
        if evidence_dir
        else (
            (index_root / manifest_evidence_dir).resolve()
            if manifest_evidence_dir
            else artifact_root / "04_evidence_catalog"
        )
    )
    final_records = (
        Path(records_path)
        if records_path
        else (
            (index_root / manifest_records_path).resolve()
            if manifest_records_path
            else artifact_root / "03_records" / "records.parquet"
        )
    )
    molecules_rows = pd.read_parquet(index_root / "molecules.parquet").to_dict(
        orient="records"
    )
    membership_rows = pd.read_parquet(
        index_root / "group_membership.parquet"
    ).to_dict(orient="records")
    family_rows = pd.read_parquet(
        catalog_root / "molecule_families.parquet"
    ).to_dict(orient="records")
    bridge_frame = pd.read_parquet(
        catalog_root / "molecule_family_records.parquet"
    )
    representative_frame = bridge_frame[
        bridge_frame["representative_rank"].notna()
    ].copy()
    bridge_rows = representative_frame.to_dict(orient="records")

    with np.load(index_root / "fingerprints.npz", allow_pickle=False) as payload:
        packed = payload["packed_fingerprints"]
        size = int(payload["fingerprint_size"][0])
        bitorder = str(payload["bitorder"][0])
    unpacked = np.unpackbits(packed, axis=1, bitorder=bitorder)[:, :size]
    fingerprints: list[DataStructs.ExplicitBitVect] = []
    for bits in unpacked:
        fingerprint = DataStructs.ExplicitBitVect(size)
        fingerprint.SetBitsFromList(np.flatnonzero(bits).astype(int).tolist())
        fingerprints.append(fingerprint)

    molecules = [_runtime_molecule(row) for row in molecules_rows]
    group_to_molecule_indices: dict[str, list[int]] = defaultdict(list)
    membership_by_evidence: dict[str, tuple[str, int, str]] = {}
    for row in membership_rows:
        group_id = str(row["group_id"])
        molecule_index = int(row["molecule_index"])
        group_to_molecule_indices[group_id].append(molecule_index)
        membership_by_evidence[str(row["evidence_id"])] = (
            str(row["molecule_id"]),
            molecule_index,
            group_id,
        )

    bridge_record_id_field = (
        "canonical_record_id"
        if bridge_rows and "canonical_record_id" in bridge_rows[0]
        else "normalized_record_id"
    )
    representative_ids = {
        str(row[bridge_record_id_field]) for row in bridge_rows
    }
    records_by_id = _load_representative_records(
        final_records, representative_ids, profile.source_columns
    )
    bridge_by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bridge_rows:
        bridge_by_evidence[str(row["evidence_id"])].append(row)

    evidence_by_molecule_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for raw_family in family_rows:
        family = _clean_nan_values(raw_family)
        evidence_id = str(family["evidence_id"])
        membership = membership_by_evidence.get(evidence_id)
        if membership is None:
            raise ValueError(f"compact catalog evidence is absent from index: {evidence_id}")
        molecule_id, _, group_id = membership
        links = sorted(
            bridge_by_evidence.get(evidence_id, []),
            key=lambda row: int(row.get("record_order") or 0),
        )
        representative_records: list[tuple[int, str, dict[str, Any]]] = []
        for link in links:
            record_id = str(link[bridge_record_id_field])
            record = records_by_id.get(record_id)
            if record is None:
                raise ValueError(
                    f"compact evidence references missing finalized record: {record_id}"
                )
            rank = link.get("representative_rank")
            if rank is not None and not _is_nan(rank):
                representative_records.append((int(rank), record_id, record))
        ordered_representatives = sorted(representative_records)
        evidence = _hydrate_family_evidence(
            family,
            [record for _, _, record in ordered_representatives],
            profile,
        )
        # Runtime-only bridge used to join frozen Stage 07 Morgan representatives
        # to their cached assay-transfer scores.  The leading underscore keeps
        # these source identifiers out of minimal_evidence.v1 and LLM prompts.
        evidence["_representative_record_ids"] = [
            record_id for _, record_id, _ in ordered_representatives
        ]
        evidence_by_molecule_group[molecule_id][group_id] = [evidence]

    if len(molecules) != len(fingerprints):
        raise ValueError(
            "compact index molecule/fingerprint mismatch: "
            f"{len(molecules)} != {len(fingerprints)}"
        )
    return {
        "version": manifest.get("index_version", profile.index_version),
        "fingerprint": manifest.get("fingerprint", {}),
        "source": {
            "name": profile.evidence_source_label,
            "artifact_version": manifest.get("artifact_version"),
        },
        "molecules": molecules,
        "fingerprints": fingerprints,
        "group_to_molecule_indices": {
            group: sorted(indices)
            for group, indices in sorted(group_to_molecule_indices.items())
        },
        "evidence_by_molecule_group": dict(evidence_by_molecule_group),
    }


def validate_compact_neighbor_index(
    index_dir: str | Path,
    *,
    profile: CompactArtifactProfile,
    evidence_dir: str | Path,
) -> dict[str, int]:
    """Validate a built index without hydrating prompt evidence or RDKit bits."""
    import pyarrow.parquet as pq

    root = Path(index_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("index_version") != profile.index_version:
        raise ValueError("compact index version differs from its profile")
    if manifest.get("artifact_version") != profile.artifact_version:
        raise ValueError("compact artifact version differs from its profile")
    for filename, metadata in (manifest.get("files") or {}).items():
        path = root / str(filename)
        if not path.is_file() or file_sha256(path) != str(metadata.get("sha256") or ""):
            raise ValueError(f"compact index file hash mismatch: {filename}")

    molecule_table = pq.read_table(
        root / "molecules.parquet",
        columns=["molecule_index", "molecule_id"],
    )
    membership_table = pq.read_table(
        root / "group_membership.parquet",
        columns=["group_id", "molecule_index", "molecule_id", "evidence_id"],
    )
    family_table = pq.read_table(
        Path(evidence_dir) / "molecule_families.parquet",
        columns=["evidence_id", "molecule_id", "group_id"],
    )
    molecule_rows = molecule_table.to_pylist()
    membership_rows = membership_table.to_pylist()
    family_rows = family_table.to_pylist()
    molecule_ids = [str(row["molecule_id"]) for row in molecule_rows]
    molecule_indices = [int(row["molecule_index"]) for row in molecule_rows]
    if molecule_indices != list(range(len(molecule_rows))):
        raise ValueError("compact molecule indices are not contiguous")
    if len(molecule_ids) != len(set(molecule_ids)):
        raise ValueError("compact index repeats a molecule ID")
    family_by_evidence = {
        str(row["evidence_id"]): (str(row["molecule_id"]), str(row["group_id"]))
        for row in family_rows
    }
    if len(family_by_evidence) != len(family_rows):
        raise ValueError("compact evidence catalog repeats an evidence ID")
    for row in membership_rows:
        index = int(row["molecule_index"])
        if index < 0 or index >= len(molecule_rows):
            raise ValueError("compact membership has an invalid molecule index")
        molecule_id = str(row["molecule_id"])
        group_id = str(row["group_id"])
        if molecule_ids[index] != molecule_id:
            raise ValueError("compact membership molecule index/ID mismatch")
        if family_by_evidence.get(str(row["evidence_id"])) != (
            molecule_id,
            group_id,
        ):
            raise ValueError("compact membership has an invalid evidence reference")
    with np.load(root / "fingerprints.npz", allow_pickle=False) as payload:
        packed = payload["packed_fingerprints"]
        size = int(payload["fingerprint_size"][0])
        bitorder = str(payload["bitorder"][0])
    if packed.shape != (len(molecule_rows), 256) or size != 2048 or bitorder != "little":
        raise ValueError("compact fingerprint matrix shape/metadata mismatch")
    groups = sorted({str(row["group_id"]) for row in membership_rows})
    if groups != sorted(str(value) for value in manifest.get("groups", [])):
        raise ValueError("compact index group inventory mismatch")
    if len(molecule_rows) != int(manifest.get("molecules", -1)):
        raise ValueError("compact index molecule count mismatch")
    if len(membership_rows) != int(manifest.get("memberships", -1)):
        raise ValueError("compact index membership count mismatch")
    return {
        "molecules": len(molecule_rows),
        "memberships": len(membership_rows),
        "families": len(family_rows),
    }


def _runtime_molecule(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "molecule_chembl_id": str(row.get("molecule_id") or ""),
        "canonical_smiles": str(row.get("canonical_smiles") or ""),
        "standard_inchi_key": str(row.get("standard_inchi_key") or ""),
        "molecule_identity": {
            "parent_smiles": _none_if_nan(row.get("parent_smiles")),
            "parent_inchi_key": _none_if_nan(row.get("parent_inchi_key")),
            "parent_connectivity_key": _none_if_nan(
                row.get("parent_connectivity_key")
            ),
            "component_parent_inchi_keys": _list_value(
                row.get("component_parent_inchi_keys")
            ),
            "status": _none_if_nan(row.get("identity_status")),
        },
    }


def _hydrate_family_evidence(
    family: Mapping[str, Any],
    representatives: Sequence[Mapping[str, Any]],
    profile: CompactArtifactProfile,
) -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    for record in representatives:
        projection = (
            _collapsed_source_projection(record)
            if record.get("collapsed_record_id")
            else profile.llm_source_projection(record)
        )
        display = None
        if (
            not record.get("collapsed_record_id")
            and record.get("measurement_unit_mapping_status") == "mapped"
        ):
            value = record.get("measurement_resolution_input_measurement")
            unit = record.get("measurement_resolution_input_unit")
            origin = record.get("measurement_resolution_origin")
            if value not in (None, "") and unit not in (None, "") and origin:
                display = {
                    "value": str(value),
                    "unit": str(unit),
                    "origin": str(origin),
                }
        condition_group = record.get("condition_group")
        if condition_group == "no_reported_external_condition":
            condition_group = None
        progressive_card = {
            "endpoint_type": record.get("canonical_endpoint_name")
            or record.get("endpoint_name"),
            "reported_value": record.get("display_measurement_text")
            or record.get("canonical_measurement_text"),
            "reported_units": record.get("display_unit_text")
            or record.get("canonical_unit_text"),
            "assay_context": record.get("canonical_assay_context")
            or record.get("assay_model")
            or record.get("study_context"),
            "species_context": record.get("canonical_species_context")
            or record.get("species"),
            "qualifying_conditions": record.get("qualifying_conditions")
            or (condition_group if record.get("collapsed_record_id") else None),
            "support_text": record.get("support_text"),
        }
        examples.append(
            {
                "source_contract": {
                    key: value
                    for key, value in projection.items()
                    if key != "source_fields"
                },
                "source_fields": projection["source_fields"],
                **(
                    {"resolved_measurement_display": display}
                    if display is not None
                    else {}
                ),
                "_progressive_card": {
                    key: value
                    for key, value in progressive_card.items()
                    if value not in (None, "")
                },
            }
        )
    source_names = [str(value) for value in _list_value(family.get("source_names"))]
    endpoint_counts = _list_value(family.get("endpoint_counts"))
    standard_type = "; ".join(
        f"{entry.get('endpoint')}={entry.get('count')}"
        for entry in endpoint_counts
        if isinstance(entry, Mapping)
    )
    row = {
        "evidence_id": family.get("evidence_id"),
        "molecule_chembl_id": family.get("molecule_id"),
        "canonical_smiles": family.get("canonical_smiles"),
        "assay_chembl_id": f"STARLING_NORMALIZED_{str(family.get('group_id') or '').upper()}",
        "assay_tier": family.get("assay_tier"),
        "endpoint_group": family.get("endpoint_group"),
        "group_id": family.get("group_id"),
        "standard_type": standard_type,
        "standard_relation": "",
        "standard_value": "",
        "standard_units": "",
        "activity_comment": (
            f"Normalized Starling summary over {int(family.get('source_record_count') or 0)} retained records "
            f"from {len(source_names)} source(s)"
        ),
        "assay_description": "",
        "target_pref_name": family.get("target_pref_name"),
        "confidence_score": family.get("median_confidence"),
        "evidence_source": profile.evidence_source_label,
        "evidence_role": family.get("evidence_role"),
        "evidence_scope": {},
        "transferability": "not_assessed",
        "uncertainty": _list_value(family.get("uncertainty")),
        "source_record_count": family.get("source_record_count"),
        "retrieval_source_ids": _list_value(family.get("retrieval_source_ids")),
        "source_numeric_record_count": family.get("source_numeric_record_count"),
        "source_qualitative_record_count": family.get(
            "source_qualitative_record_count"
        ),
        "source_record_examples": examples,
        "source_qualitative_examples": [],
        "source_names": source_names,
    }
    return attach_minimal_evidence(row)


def _load_representative_records(
    records_path: Path,
    record_ids: set[str],
    source_columns: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, Any]]:
    """Scan Parquet by batch and materialize only prompt representatives."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(records_path)
    available = set(parquet.schema_arrow.names)
    record_id_field = (
        "canonical_record_id"
        if "canonical_record_id" in available
        else "normalized_record_id"
    )
    columns = [
        column
        for column in (
            record_id_field,
            "source_id",
            "collapsed_record_id",
            "retrieval_source_id",
            "aggregation_method",
            "aggregation_status",
            "aggregate_counts_json",
            "aggregate_min",
            "aggregate_q1",
            "aggregate_q3",
            "aggregate_max",
            "measurement_unit_mapping_status",
            "measurement_resolution_input_measurement",
            "measurement_resolution_input_unit",
            "measurement_resolution_origin",
            "canonical_endpoint_name",
            "canonical_measurement_text",
            "canonical_unit_text",
            "finite_scalar_value",
            "condition_group",
            "condition_scope",
            "condition_key_status",
            "condition_atoms_json",
            "canonical_pair_fields_json",
            "display_measurement_text",
            "display_scalar_value",
            "display_unit_text",
            "display_transform_id",
            "source_record_count",
            "deduplicated_source_record_count",
            *sorted({field for fields in source_columns.values() for field in fields}),
        )
        if column in available
    ]
    wanted = pa.array(sorted(record_ids), type=pa.string())
    output: dict[str, dict[str, Any]] = {}
    for batch in parquet.iter_batches(columns=columns, batch_size=32_768):
        ids = batch.column(batch.schema.get_field_index(record_id_field))
        selected = batch.filter(pc.is_in(ids, value_set=wanted))
        for row in selected.to_pylist():
            record_id = str(row.get(record_id_field) or "")
            output[record_id] = row
    missing = sorted(record_ids - set(output))
    if missing:
        raise ValueError(
            f"compact representatives reference {len(missing)} missing record(s); "
            f"first={missing[0]}"
        )
    return output


def _collapsed_source_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    """Expose aggregate evidence without direct-vote or rejection bookkeeping."""
    retrieval_source = str(record.get("retrieval_source_id") or "")
    evidence_partition = {
        "direct_vote": "direct_outcome",
        "direct_residual": "contextual_direct_outcome",
        "indirect": "indirect_experimental_evidence",
    }.get(retrieval_source, "experimental_evidence")
    counts = _json_object(record.get("aggregate_counts_json"))
    direct = retrieval_source in {"direct_vote", "direct_residual"}
    method = record.get("aggregation_method")
    if direct and method == "direct_binary_vote":
        method = "consensus"
    measurement = {
        "value": record.get("display_measurement_text"),
        "numeric_value": record.get("display_scalar_value"),
        "unit": record.get("display_unit_text"),
        "method": method,
        "status": record.get("aggregation_status"),
        "range": {
            "minimum": display_scalar_value(
                record.get("aggregate_min"), str(record.get("display_transform_id") or "")
            ),
            "q1": display_scalar_value(
                record.get("aggregate_q1"), str(record.get("display_transform_id") or "")
            ),
            "q3": display_scalar_value(
                record.get("aggregate_q3"), str(record.get("display_transform_id") or "")
            ),
            "maximum": display_scalar_value(
                record.get("aggregate_max"), str(record.get("display_transform_id") or "")
            ),
        }
        if record.get("aggregate_min") is not None
        else None,
    }
    if record.get("display_transform_id") == DISPLAY_INVERSE_LOG10:
        measurement["aggregation_scale"] = "log10"
    elif record.get("display_transform_id") == DISPLAY_INVERSE_LOGIT:
        measurement["aggregation_scale"] = "logit"
    if not direct:
        measurement["category_counts"] = counts or None
    pair_context = _json_object(record.get("canonical_pair_fields_json"))
    source_fields = {
        "evidence_partition": evidence_partition,
        "endpoint": record.get("canonical_endpoint_name"),
        "aggregated_measurement": measurement,
        "canonical_context": (
            {
                "condition_group": record.get("condition_group"),
                "condition_scope": record.get("condition_scope"),
                "condition_atoms": _json_list(record.get("condition_atoms_json")),
            }
            if direct
            else {
                "pair_fields": pair_context,
            }
        ),
        "source_record_count": record.get("source_record_count"),
    }
    return {
        "contract_version": "collapsed_record_prompt.v3",
        "source_id": str(record.get("source_id") or ""),
        "source_name": "collapsed normalized experimental evidence",
        "source_or_simply_cleaned": {key: True for key in source_fields},
        "source_fields": source_fields,
    }


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _json_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _clean_nan_values(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): None if _is_nan(value) else value
        for key, value in row.items()
    }


def _is_nan(value: Any) -> bool:
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _none_if_nan(value: Any) -> Any:
    return None if _is_nan(value) else value


def _list_value(value: Any) -> list[Any]:
    if value is None or _is_nan(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _representative_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(
            record.get("canonical_endpoint")
            or record.get("canonical_endpoint_name")
            or record.get("endpoint_name")
            or ""
        ),
        -(float(record["confidence"]) if record.get("confidence") is not None else 0.0),
        str(record.get("source_id") or ""),
        int(record.get("source_row_number") or 0),
    )


def _representative_ids(
    ranked: Sequence[Mapping[str, Any]], limit: int
) -> list[str]:
    selected: list[Mapping[str, Any]] = []
    endpoints: set[str] = set()
    record_id_field = (
        "canonical_record_id"
        if ranked and "canonical_record_id" in ranked[0]
        else "normalized_record_id"
    )
    for record in ranked:
        endpoint = str(
            record.get("canonical_endpoint")
            or record.get("canonical_endpoint_name")
            or record.get("endpoint_name")
            or ""
        )
        if endpoint not in endpoints:
            selected.append(record)
            endpoints.add(endpoint)
        if len(selected) >= limit:
            break
    for record in ranked:
        if len(selected) >= limit:
            break
        if record not in selected:
            selected.append(record)
    return [str(record.get(record_id_field) or "") for record in selected]


__all__ = [
    "BANNED_PERSISTED_FIELDS",
    "CompactArtifactProfile",
    "assert_compact_schema",
    "build_relational_evidence_catalog",
    "compact_persisted_records",
    "load_compact_neighbor_index",
    "validate_compact_neighbor_index",
    "write_compact_neighbor_index",
]
