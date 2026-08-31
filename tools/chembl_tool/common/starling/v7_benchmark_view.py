"""Paper retrieval views derived from active canonical v7 pair-bucket records.

Held-out filtering can remove exact parents, both direct retrieval partitions,
or every record sharing a validation/test scaffold. All current v7 paper views
consume the active deduplicated Stage-3 records; no lineage owns a private dedup.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow.parquet as pq

from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    bemis_murcko_scaffold,
    normalize_molecule_identity,
)
from tools.chembl_tool.common.starling.compact_artifacts import (
    build_relational_evidence_catalog,
    validate_compact_neighbor_index,
    write_compact_neighbor_index,
)
from tools.chembl_tool.common.starling.build_runtime import (
    INCOMPLETE_BUILD_FILENAME,
    complete_build_session_active,
    starling_build_session,
)
from tools.chembl_tool.common.starling.heldout_index import (
    load_heldout_identity_keys,
)
from tools.chembl_tool.common.starling.normalization.audit import (
    read_parquet_records,
    write_parquet,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.normalization.cleaning import stable_id
from tools.chembl_tool.common.starling.normalization.organization import (
    is_retrieval_eligible,
)
from tools.chembl_tool.common.starling.normalization.task_policy import (
    StarlingTaskPolicy,
)
from tools.chembl_tool.common.starling.split_downstream import (
    CORE_PAIR_BUCKET_STAGE,
    SplitDownstreamSpec,
    evidence_catalog_projection_columns,
)


VIEW_VERSION = "starling_v7_benchmark_view.v4"
RECORDS_MANIFEST_VERSION = "starling_v7_benchmark_records.v1"
EVIDENCE_MANIFEST_VERSION = "starling_v7_benchmark_evidence.v1"
AUDIT_VERSION = "starling_v7_benchmark_audit.v1"
FULL_VIEW = "full"
ALL_PARENT_FILTER = "all_parents"
DIRECT_SOURCE_ONLY_FILTER = "direct_source_only"
ALL_SCAFFOLD_FILTER = "all_scaffolds"
HELDOUT_FILTER_MODES = (
    ALL_PARENT_FILTER,
    DIRECT_SOURCE_ONLY_FILTER,
    ALL_SCAFFOLD_FILTER,
)


def build_v7_benchmark_view(
    *,
    policy: StarlingTaskPolicy,
    normalized_root: str | Path,
    heldout_labels_jsonl: str | Path,
    train_labels_jsonl: str | Path | None = None,
    out_dir: str | Path,
    benchmark_split: str,
    view: str = FULL_VIEW,
    heldout_filter_mode: str = ALL_PARENT_FILTER,
    downstream_spec: SplitDownstreamSpec | None = None,
    workers: int = 1,
    progress_every: int = 10_000,
    max_record_examples: int = 6,
) -> dict[str, Any]:
    source_root = Path(normalized_root)
    with starling_build_session(source_root, mark_incomplete=False):
        if (
            (source_root / INCOMPLETE_BUILD_FILENAME).exists()
            and not complete_build_session_active(source_root)
        ):
            raise RuntimeError(
                f"normalized source has an incomplete build: {source_root}"
            )
        return _build_v7_benchmark_view(
            policy=policy,
            normalized_root=source_root,
            heldout_labels_jsonl=heldout_labels_jsonl,
            train_labels_jsonl=train_labels_jsonl,
            out_dir=out_dir,
            benchmark_split=benchmark_split,
            view=view,
            heldout_filter_mode=heldout_filter_mode,
            downstream_spec=downstream_spec,
            workers=workers,
            progress_every=progress_every,
            max_record_examples=max_record_examples,
        )


def _build_v7_benchmark_view(
    *,
    policy: StarlingTaskPolicy,
    normalized_root: str | Path,
    heldout_labels_jsonl: str | Path,
    train_labels_jsonl: str | Path | None = None,
    out_dir: str | Path,
    benchmark_split: str,
    view: str = FULL_VIEW,
    heldout_filter_mode: str = ALL_PARENT_FILTER,
    downstream_spec: SplitDownstreamSpec | None = None,
    workers: int = 1,
    progress_every: int = 10_000,
    max_record_examples: int = 6,
) -> dict[str, Any]:
    """Build one self-contained compact index from canonical v7 records."""
    source_root = Path(normalized_root)
    records_path = source_root / CORE_PAIR_BUCKET_STAGE / "records.parquet"
    source_manifest_path = source_root / CORE_PAIR_BUCKET_STAGE / "manifest.json"
    if not records_path.is_file():
        raise FileNotFoundError(
            f"active Stage-3 records are required before paper-view build: {records_path}"
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    expected_records_hash = (source_manifest.get("outputs") or {}).get(
        "records.parquet"
    )
    actual_records_hash = file_sha256(records_path)
    if expected_records_hash != actual_records_hash:
        raise ValueError("active Stage-3 records differ from their manifest")
    heldout_path = Path(heldout_labels_jsonl)
    target = Path(out_dir)
    if view != FULL_VIEW:
        raise ValueError(f"unsupported v7 benchmark view: {view}")
    if heldout_filter_mode not in HELDOUT_FILTER_MODES:
        raise ValueError(f"unsupported held-out filter mode: {heldout_filter_mode}")
    if heldout_filter_mode == DIRECT_SOURCE_ONLY_FILTER:
        if downstream_spec is None:
            raise ValueError("direct_source_only requires a downstream task specification")
        if downstream_spec.task_id != policy.task_id:
            raise ValueError(
                "held-out filter task mismatch: "
                f"policy={policy.task_id}, downstream={downstream_spec.task_id}"
            )
    schema = set(pq.read_schema(records_path).names)
    if "canonical_record_id" not in schema:
        raise ValueError(f"paper view requires v7 canonical records: {records_path}")

    profile = policy.compact_profile_for_contract(policy.record_contract.version)
    columns = evidence_catalog_projection_columns(
        schema,
        policy=policy,
        v7=True,
    )
    columns.update(
        field
        for source_fields in profile.source_columns.values()
        for field in source_fields
        if field in schema
    )
    records = read_parquet_records(records_path, columns=sorted(columns))
    for record in records:
        record["retrieval_eligible"] = is_retrieval_eligible(
            record,
            endpoint_identity_required_sources=(
                policy.endpoint_identity_required_sources
            ),
        )
    heldout_keys = load_heldout_identity_keys(heldout_path)
    heldout_scaffolds = _load_heldout_scaffolds(heldout_path)
    filtered, filter_stats = _filter_records(
        records,
        heldout_keys=heldout_keys,
        heldout_scaffolds=heldout_scaffolds,
        view_predicate=_view_predicate(view),
        heldout_filter_mode=heldout_filter_mode,
        downstream_spec=downstream_spec,
    )
    gold_swap = None
    if train_labels_jsonl is not None:
        filtered, gold_swap = _swap_direct_votes_for_gold(
            filtered,
            Path(train_labels_jsonl),
            task_id=policy.task_id,
        )
    families, bridge = build_relational_evidence_catalog(
        filtered,
        max_record_examples=max_record_examples,
        family_resolver=policy.family_resolver,
    )

    candidate = target.with_name(f".{target.name}.candidate.{os.getpid()}")
    if candidate.exists():
        shutil.rmtree(candidate)
    records_dir = candidate / "06_records"
    evidence_dir = candidate / "07_molecule_evidence"
    index_dir = candidate / "08_neighbor_index"
    audit_dir = candidate / "09_audits"
    records_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    records_file = records_dir / "records.parquet"
    families_file = evidence_dir / "molecule_families.parquet"
    bridge_file = evidence_dir / "molecule_family_records.parquet"
    write_parquet(records_file, filtered)
    write_parquet(families_file, families)
    write_parquet(bridge_file, bridge)
    records_manifest = {
        "version": RECORDS_MANIFEST_VERSION,
        "task_id": policy.task_id,
        "benchmark_split": benchmark_split,
        "view": view,
        "heldout_filter_mode": heldout_filter_mode,
        "records": len(filtered),
        "records_file": {
            "path": "records.parquet",
            "sha256": file_sha256(records_file),
        },
        "source_v7_records": {
            "path": str(records_path),
            "sha256": actual_records_hash,
            "stage": CORE_PAIR_BUCKET_STAGE,
            "manifest_sha256": file_sha256(source_manifest_path),
        },
        "heldout_labels": {
            "path": str(heldout_path),
            "sha256": file_sha256(heldout_path),
        },
        "heldout_filter": filter_stats,
        **({"gold_label_swap_for_direct_labels": gold_swap} if gold_swap else {}),
    }
    (records_dir / "manifest.json").write_text(
        json.dumps(records_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    evidence_manifest = {
        "version": EVIDENCE_MANIFEST_VERSION,
        "task_id": policy.task_id,
        "benchmark_split": benchmark_split,
        "view": view,
        "records_manifest": "../06_records/manifest.json",
        "families": len(families),
        "record_references": len(bridge),
        "files": {
            "molecule_families.parquet": file_sha256(families_file),
            "molecule_family_records.parquet": file_sha256(bridge_file),
        },
    }
    (evidence_dir / "manifest.json").write_text(
        json.dumps(evidence_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    index_manifest = write_compact_neighbor_index(
        profile=profile,
        families=families,
        output_dir=index_dir,
        workers=workers,
        progress_every=progress_every,
    )
    index_manifest.update(
        {
            "evidence_dir": "../07_molecule_evidence",
            "evidence_manifest": "../07_molecule_evidence/manifest.json",
            "records_path": "../06_records/records.parquet",
            "records_manifest": "../06_records/manifest.json",
            "benchmark_view": view,
            "benchmark_split": benchmark_split,
            "heldout_filter_mode": heldout_filter_mode,
            "heldout_filter": filter_stats,
            "source_v7_records": records_manifest["source_v7_records"],
            "heldout_labels": {
                "path": str(heldout_path),
                "sha256": file_sha256(heldout_path),
            },
            **({"gold_label_swap_for_direct_labels": gold_swap} if gold_swap else {}),
            "identity_normalizer_version": IDENTITY_NORMALIZER_VERSION,
        }
    )
    (index_dir / "manifest.json").write_text(
        json.dumps(index_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = validate_compact_neighbor_index(
        index_dir,
        profile=profile,
        evidence_dir=evidence_dir,
    )
    audit = {
        "version": AUDIT_VERSION,
        "task_id": policy.task_id,
        "benchmark_split": benchmark_split,
        "view": view,
        "heldout_filter_mode": heldout_filter_mode,
        "heldout_filter": filter_stats,
        **({"gold_label_swap_for_direct_labels": gold_swap} if gold_swap else {}),
        "index_validation": validation,
        "validations": {
            "zero_filter_scope_parent_overlap": filter_stats[
                "zero_filter_scope_parent_overlap"
            ],
            "zero_heldout_scaffold_overlap": (
                heldout_filter_mode != ALL_SCAFFOLD_FILTER
                or filter_stats["zero_heldout_scaffold_overlap"]
            ),
            "lineage_local_records": True,
            "lineage_local_molecule_evidence": True,
            "lineage_local_neighbor_index": True,
        },
    }
    (audit_dir / "heldout_overlap.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": VIEW_VERSION,
        "task_id": policy.task_id,
        "benchmark_split": benchmark_split,
        "view": view,
        "heldout_filter_mode": heldout_filter_mode,
        "source_v7_records": index_manifest["source_v7_records"],
        "heldout_labels": index_manifest["heldout_labels"],
        "heldout_filter": filter_stats,
        **({"gold_label_swap_for_direct_labels": gold_swap} if gold_swap else {}),
        "records": len(filtered),
        "families": len(families),
        "record_references": len(bridge),
        "index": index_manifest,
        "validation": validation,
        "stage_manifests": {
            "06_records": "06_records/manifest.json",
            "07_molecule_evidence": "07_molecule_evidence/manifest.json",
            "08_neighbor_index": "08_neighbor_index/manifest.json",
            "09_audits": "09_audits/heldout_overlap.json",
        },
        "files": {
            "records": file_sha256(records_file),
            "records_manifest": file_sha256(records_dir / "manifest.json"),
            "families": file_sha256(families_file),
            "bridge": file_sha256(bridge_file),
            "evidence_manifest": file_sha256(evidence_dir / "manifest.json"),
            "index_manifest": file_sha256(index_dir / "manifest.json"),
            "audit": file_sha256(audit_dir / "heldout_overlap.json"),
        },
    }
    (candidate / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _publish_candidate(candidate, target)
    return manifest


def _view_predicate(view: str) -> Callable[[Mapping[str, Any]], bool]:
    if view != FULL_VIEW:
        raise ValueError(f"unsupported v7 benchmark view: {view}")
    return lambda record: True


def _swap_direct_votes_for_gold(
    records: list[dict[str, Any]],
    train_labels_path: Path,
    *,
    task_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace reconstructed direct votes with frozen training labels only."""
    if not train_labels_path.is_file():
        raise FileNotFoundError(train_labels_path)
    direct_votes = [
        row for row in records if row.get("retrieval_source_id") == "direct_vote"
    ]
    direct_groups = {str(row.get("group_id") or "") for row in direct_votes}
    direct_groups.discard("")
    if len(direct_groups) != 1:
        raise ValueError(
            f"gold swap requires exactly one direct group, found {sorted(direct_groups)}"
        )
    direct_group = next(iter(direct_groups))
    retained = [
        row for row in records if row.get("retrieval_source_id") != "direct_vote"
    ]
    preserved_counts = Counter(
        str(row.get("retrieval_source_id") or "") for row in retained
    )
    labels = [
        json.loads(line)
        for line in train_labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    benchmark_ids = [str(row.get("benchmark_row_id") or "") for row in labels]
    if not labels or any(not value for value in benchmark_ids):
        raise ValueError("training gold rows require benchmark_row_id")
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("training gold contains duplicate benchmark_row_id values")

    endpoint, outcomes = {
        "bbb_martins": (
            "meaningful_systemic_cns_access",
            {0: "restricted or poor CNS access", 1: "meaningful CNS access"},
        ),
        "bioavailability_ma": (
            "oral_bioavailability_f20",
            {0: "oral bioavailability below 20%", 1: "oral bioavailability at least 20%"},
        ),
    }.get(task_id, ("direct_outcome", {0: "negative", 1: "positive"}))
    synthetic = []
    for position, label in enumerate(labels, start=1):
        outcome = int(label.get("Y"))
        if outcome not in {0, 1}:
            raise ValueError("training gold labels must be binary")
        benchmark_id = str(label["benchmark_row_id"])
        smiles = str(
            (label.get("molecule_identity") or {}).get("canonical_smiles")
            or label.get("drug")
            or ""
        )
        if not smiles:
            raise ValueError(f"training gold row lacks a molecule: {benchmark_id}")
        condition_atoms = list(label.get("condition_atoms") or [])
        record_id = stable_id("conditioned_benchmark_gold", task_id, benchmark_id)
        synthetic.append(
            {
                "canonical_record_id": record_id,
                "collapsed_record_id": record_id,
                "source_id": "conditioned_benchmark_gold",
                "source_name": "conditioned benchmark training label",
                "source_row_number": position,
                "source_record_id": benchmark_id,
                "canonical_smiles": smiles,
                "group_id": direct_group,
                "retrieval_eligible": True,
                "retrieval_source_id": "direct_vote",
                "canonicalization_status": "valid",
                "canonical_endpoint_name": endpoint,
                "endpoint_name": endpoint,
                "canonical_measurement_text": outcomes[outcome],
                "canonical_unit_text": "binary_outcome",
                "direct_vote_label": outcome,
                "direct_vote_unit_id": f"conditioned_benchmark:{benchmark_id}",
                "condition_group": label.get("condition_group"),
                "condition_scope": label.get("condition_scope"),
                "condition_key_status": "gold_training_label",
                "condition_atoms": condition_atoms,
                "condition_atoms_json": json.dumps(condition_atoms, separators=(",", ":")),
                "pair_bucket_key": None,
                "bucket_eligible": False,
                "bucket_exclusion_reason": "frozen_direct_gold_label",
                "assay_transfer_eligible": False,
                "assay_transfer_ineligibility_reason": "frozen_direct_gold_label",
                "aggregation_method": "direct_binary_vote",
                "aggregation_status": "consensus",
                "aggregate_counts_json": json.dumps({str(outcome): 1}),
                "display_measurement_text": outcomes[outcome],
                "display_unit_text": "binary_outcome",
                "source_record_count": int(label.get("source_record_count") or 1),
                "deduplicated_source_record_count": 1,
            }
        )
    output = retained + synthetic
    return output, {
        "version": "gold_label_swap_for_direct_labels.v1",
        "train_labels": {
            "path": str(train_labels_path),
            "sha256": file_sha256(train_labels_path),
        },
        "direct_group_id": direct_group,
        "removed_reconstructed_direct_vote_records": len(direct_votes),
        "inserted_gold_training_records": len(synthetic),
        "preserved_direct_residual_records": preserved_counts["direct_residual"],
        "preserved_indirect_records": preserved_counts["indirect"],
        "output_records": len(output),
    }


def _filter_records(
    records: list[dict[str, Any]],
    *,
    heldout_keys: set[str],
    heldout_scaffolds: set[str],
    view_predicate: Callable[[Mapping[str, Any]], bool],
    heldout_filter_mode: str = ALL_PARENT_FILTER,
    downstream_spec: SplitDownstreamSpec | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    identity_cache: dict[str, tuple[str, str]] = {}
    excluded_heldout_records = 0
    excluded_scaffold_records = 0
    excluded_unresolved_records = 0
    excluded_view_records = 0
    matched_heldout: set[str] = set()
    matched_scaffolds: set[str] = set()
    retained_heldout: set[str] = set()
    retained_heldout_records = 0
    source_input: Counter[str] = Counter()
    source_excluded: Counter[str] = Counter()
    source_kept: Counter[str] = Counter()
    for record in records:
        if not record.get("retrieval_eligible"):
            excluded_view_records += 1
            continue
        if not view_predicate(record):
            excluded_view_records += 1
            continue
        source_id = str(record.get("source_id") or "")
        source_input[source_id] += 1
        smiles = str(record.get("canonical_smiles") or record.get("smiles") or "")
        if smiles not in identity_cache:
            identity = normalize_molecule_identity(smiles)
            key = identity.parent_inchi_key or identity.parent_smiles
            scaffold = bemis_murcko_scaffold(identity.parent_smiles) if key else ""
            identity_cache[smiles] = (key, scaffold)
        key, scaffold = identity_cache[smiles]
        if not key:
            excluded_unresolved_records += 1
            continue
        in_filter_scope = _in_heldout_filter_scope(
            record,
            heldout_filter_mode=heldout_filter_mode,
            downstream_spec=downstream_spec,
        )
        scaffold_match = (
            heldout_filter_mode == ALL_SCAFFOLD_FILTER
            and bool(scaffold)
            and scaffold in heldout_scaffolds
        )
        if in_filter_scope and (key in heldout_keys or scaffold_match):
            excluded_heldout_records += 1
            excluded_scaffold_records += int(scaffold_match)
            source_excluded[source_id] += 1
            if key in heldout_keys:
                matched_heldout.add(key)
            if scaffold_match:
                matched_scaffolds.add(scaffold)
            continue
        if key in heldout_keys:
            retained_heldout_records += 1
            retained_heldout.add(key)
        kept.append(record)
        source_kept[source_id] += 1

    residual = {
        identity_cache[str(record.get("canonical_smiles") or record.get("smiles") or "")][0]
        for record in kept
    } & heldout_keys
    residual_scaffolds = {
        identity_cache[str(record.get("canonical_smiles") or record.get("smiles") or "")][1]
        for record in kept
    } & heldout_scaffolds
    residual_filter_scope = {
        identity_cache[str(record.get("canonical_smiles") or record.get("smiles") or "")][0]
        for record in kept
        if _in_heldout_filter_scope(
            record,
            heldout_filter_mode=heldout_filter_mode,
            downstream_spec=downstream_spec,
        )
    } & heldout_keys
    if residual_filter_scope:
        raise AssertionError(
            "held-out parent leakage remains in the configured filter scope for "
            f"{len(residual_filter_scope)} identities"
        )
    if heldout_filter_mode == ALL_SCAFFOLD_FILTER and residual_scaffolds:
        raise AssertionError(
            f"held-out scaffold leakage remains for {len(residual_scaffolds)} scaffolds"
        )
    filter_source_id = downstream_spec.filter_source_id if downstream_spec else ""
    filter_scope = (
        {
            "field": downstream_spec.filter_scope_field,
            "value": downstream_spec.filter_scope_value,
        }
        if downstream_spec and downstream_spec.filter_scope_field
        else None
    )
    return kept, {
        "mode": heldout_filter_mode,
        "filter_source_id": filter_source_id,
        "filter_scope": filter_scope,
        "n_input_records": len(records),
        "n_output_records": len(kept),
        "n_excluded_records": len(records) - len(kept),
        "n_excluded_heldout_records": excluded_heldout_records,
        "n_excluded_scaffold_records": excluded_scaffold_records,
        "n_excluded_unresolved_records": excluded_unresolved_records,
        "n_excluded_by_view_records": excluded_view_records,
        "n_heldout_parent_identities": len(heldout_keys),
        "n_heldout_scaffolds": len(heldout_scaffolds),
        "n_matched_heldout_parent_identities": len(matched_heldout),
        "n_matched_filter_scope_heldout_parent_identities": len(matched_heldout),
        "n_retained_heldout_nonfilter_records": retained_heldout_records,
        "n_retained_heldout_nonfilter_parent_identities": len(retained_heldout),
        "n_residual_heldout_parent_identities": len(residual),
        "n_residual_filter_scope_heldout_parent_identities": 0,
        "n_matched_heldout_scaffolds": len(matched_scaffolds),
        "n_residual_heldout_scaffolds": len(residual_scaffolds),
        "source_counts": {
            source: {
                "input": source_input[source],
                "excluded_heldout": source_excluded[source],
                "retained": source_kept[source],
            }
            for source in sorted(source_input)
        },
        "zero_parent_overlap": not residual,
        "zero_filter_scope_parent_overlap": True,
        "zero_heldout_scaffold_overlap": not residual_scaffolds,
    }


def _load_heldout_scaffolds(path: Path) -> set[str]:
    scaffolds: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scaffold = str(row.get("bemis_murcko_scaffold") or "")
        if not scaffold:
            scaffold = bemis_murcko_scaffold(str(row.get("drug") or ""))
        if scaffold:
            scaffolds.add(scaffold)
    return scaffolds


def _in_heldout_filter_scope(
    record: Mapping[str, Any],
    *,
    heldout_filter_mode: str,
    downstream_spec: SplitDownstreamSpec | None,
) -> bool:
    if heldout_filter_mode in {ALL_PARENT_FILTER, ALL_SCAFFOLD_FILTER}:
        return True
    if downstream_spec is None:
        raise ValueError("direct_source_only requires a downstream task specification")
    if str(record.get("retrieval_source_id") or "") == "direct_vote":
        return True
    if record.get("retrieval_source_id"):
        return False
    if str(record.get("source_id") or "") != downstream_spec.filter_source_id:
        return False
    if not downstream_spec.filter_scope_field:
        return True
    return (
        str(record.get(downstream_spec.filter_scope_field) or "")
        == downstream_spec.filter_scope_value
    )


def _publish_candidate(candidate: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.backup.{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(candidate, target)
    except BaseException:
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


__all__ = [
    "AUDIT_VERSION",
    "ALL_PARENT_FILTER",
    "ALL_SCAFFOLD_FILTER",
    "DIRECT_SOURCE_ONLY_FILTER",
    "EVIDENCE_MANIFEST_VERSION",
    "FULL_VIEW",
    "HELDOUT_FILTER_MODES",
    "RECORDS_MANIFEST_VERSION",
    "VIEW_VERSION",
    "build_v7_benchmark_view",
]
