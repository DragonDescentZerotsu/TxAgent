"""Leakage-safe paper retrieval views derived from canonical v7 records.

The task v7 trees retain useful nondirect/mechanistic evidence for held-out
parents by design.  Formal paper retrieval has a stricter contract: every row
for every valid/test parent is absent before ranking.  This module performs
only that identity filter plus an optional declared view projection; it never
reparses measurements or changes canonical scientific fields.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow.parquet as pq

from tools.chembl_tool.common.molecule_identity import (
    IDENTITY_NORMALIZER_VERSION,
    normalize_molecule_identity,
)
from tools.chembl_tool.common.starling.compact_artifacts import (
    build_relational_evidence_catalog,
    validate_compact_neighbor_index,
    write_compact_neighbor_index,
)
from tools.chembl_tool.common.starling.heldout_index import (
    load_heldout_identity_keys,
)
from tools.chembl_tool.common.starling.normalization.audit import (
    read_parquet_records,
    write_parquet,
)
from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.normalization.task_policy import (
    StarlingTaskPolicy,
)
from tools.chembl_tool.common.starling.split_downstream import (
    evidence_catalog_projection_columns,
)


VIEW_VERSION = "starling_v7_benchmark_view.v1"
FULL_VIEW = "full"
DIRECT_NUMERIC_VIEW = "direct_numeric"
DIRECT_BIOAVAILABILITY_GROUP = "Observed.direct_oral_bioavailability"


def build_v7_benchmark_view(
    *,
    policy: StarlingTaskPolicy,
    normalized_root: str | Path,
    heldout_labels_jsonl: str | Path,
    out_dir: str | Path,
    benchmark_split: str,
    view: str = FULL_VIEW,
    workers: int = 1,
    progress_every: int = 10_000,
    max_record_examples: int = 6,
) -> dict[str, Any]:
    """Build one self-contained compact index from canonical v7 records."""
    source_root = Path(normalized_root)
    records_path = source_root / "03_records" / "records.parquet"
    heldout_path = Path(heldout_labels_jsonl)
    target = Path(out_dir)
    if view not in {FULL_VIEW, DIRECT_NUMERIC_VIEW}:
        raise ValueError(f"unsupported v7 benchmark view: {view}")
    schema = set(pq.read_schema(records_path).names)
    if "canonical_record_id" not in schema:
        raise ValueError(f"paper view requires v7 canonical records: {records_path}")
    if view == DIRECT_NUMERIC_VIEW and policy.task_id != "bioavailability_ma":
        raise ValueError("direct_numeric is defined only for Bioavailability_Ma")

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
    heldout_keys = load_heldout_identity_keys(heldout_path)
    filtered, filter_stats = _filter_records(
        records,
        heldout_keys=heldout_keys,
        view_predicate=_view_predicate(view),
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
    records_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    write_parquet(records_dir / "records.parquet", filtered)
    write_parquet(evidence_dir / "molecule_families.parquet", families)
    write_parquet(evidence_dir / "molecule_family_records.parquet", bridge)

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
            "records_path": "../06_records/records.parquet",
            "benchmark_view": view,
            "benchmark_split": benchmark_split,
            "heldout_filter": filter_stats,
            "source_v7_records": {
                "path": str(records_path),
                "sha256": file_sha256(records_path),
            },
            "heldout_labels": {
                "path": str(heldout_path),
                "sha256": file_sha256(heldout_path),
            },
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
    manifest = {
        "version": VIEW_VERSION,
        "task_id": policy.task_id,
        "benchmark_split": benchmark_split,
        "view": view,
        "source_v7_records": index_manifest["source_v7_records"],
        "heldout_labels": index_manifest["heldout_labels"],
        "heldout_filter": filter_stats,
        "records": len(filtered),
        "families": len(families),
        "record_references": len(bridge),
        "index": index_manifest,
        "validation": validation,
        "files": {
            "records": file_sha256(records_dir / "records.parquet"),
            "families": file_sha256(evidence_dir / "molecule_families.parquet"),
            "bridge": file_sha256(evidence_dir / "molecule_family_records.parquet"),
        },
    }
    (candidate / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _publish_candidate(candidate, target)
    return manifest


def _view_predicate(view: str) -> Callable[[Mapping[str, Any]], bool]:
    if view == FULL_VIEW:
        return lambda record: True
    return lambda record: (
        str(record.get("group_id") or "") == DIRECT_BIOAVAILABILITY_GROUP
        and record.get("finite_scalar_value") is not None
    )


def _filter_records(
    records: list[dict[str, Any]],
    *,
    heldout_keys: set[str],
    view_predicate: Callable[[Mapping[str, Any]], bool],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    identity_cache: dict[str, str] = {}
    excluded_heldout_records = 0
    excluded_unresolved_records = 0
    excluded_view_records = 0
    matched_heldout: set[str] = set()
    for record in records:
        if not record.get("retrieval_eligible"):
            excluded_view_records += 1
            continue
        smiles = str(record.get("canonical_smiles") or "")
        if smiles not in identity_cache:
            identity = normalize_molecule_identity(smiles)
            identity_cache[smiles] = identity.parent_inchi_key or identity.parent_smiles
        key = identity_cache[smiles]
        if not key:
            excluded_unresolved_records += 1
            continue
        if key in heldout_keys:
            excluded_heldout_records += 1
            matched_heldout.add(key)
            continue
        if not view_predicate(record):
            excluded_view_records += 1
            continue
        kept.append(record)

    residual = {
        identity_cache[str(record.get("canonical_smiles") or "")]
        for record in kept
    } & heldout_keys
    if residual:
        raise AssertionError(
            f"held-out parent leakage remains for {len(residual)} identities"
        )
    return kept, {
        "n_input_records": len(records),
        "n_output_records": len(kept),
        "n_excluded_records": len(records) - len(kept),
        "n_excluded_heldout_records": excluded_heldout_records,
        "n_excluded_unresolved_records": excluded_unresolved_records,
        "n_excluded_by_view_records": excluded_view_records,
        "n_heldout_parent_identities": len(heldout_keys),
        "n_matched_heldout_parent_identities": len(matched_heldout),
        "n_residual_heldout_parent_identities": 0,
        "zero_parent_overlap": True,
    }


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
    "DIRECT_NUMERIC_VIEW",
    "FULL_VIEW",
    "VIEW_VERSION",
    "build_v7_benchmark_view",
]
