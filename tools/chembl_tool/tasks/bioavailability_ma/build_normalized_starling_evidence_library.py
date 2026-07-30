"""Build the policy-decoupled layered v5 Bioavailability evidence library."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import time
from typing import Any

import pandas as pd

from tools.chembl_tool.common.export import ensure_dir
from tools.chembl_tool.common.starling.evidence_library import write_jsonl
from tools.chembl_tool.common.starling.normalization.audit import (
    read_parquet_records,
    scalar_distribution_audit,
    stage_manifest,
    validate_cleaned_normalized_identity,
    validate_measurement_pairs,
    validate_stage_schema,
    write_parquet,
)
from tools.chembl_tool.common.starling.normalization.cleaning import (
    clean_source_rows,
    file_sha256,
    normalize_endpoint_name,
)
from tools.chembl_tool.common.starling.normalization.contracts import (
    CLEANING_STAGE_VERSION,
    NORMALIZATION_STAGE_VERSION,
    ORGANIZATION_STAGE_VERSION,
    NORMALIZED_ARTIFACT_VERSION,
    NORMALIZED_RECORD_VERSION,
    SCALAR_PARSER_VERSION,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_cleaned_records,
)
from tools.chembl_tool.common.starling.normalization.organization import (
    aggregate_molecule_family_records,
    organize_normalized_records,
)
from tools.chembl_tool.common.task_workflows.evidence_library import (
    build_neighbor_index,
    fingerprint_metadata,
)
from tools.chembl_tool.common.units import UNIT_NORMALIZER_VERSION
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_evidence_library import (
    DEFAULT_DROPPED_JSONL,
    DEFAULT_SOURCE_JSONL,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_normalization_policy import (
    ENDPOINT_POLICY_VERSION,
    endpoint_specific_standardization_of_unit,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    SPACING_AND_SPELLING_VERSION,
    family_assignment,
    spacing_and_spelling_decision,
    validate_endpoint_inventory,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_normalization_sources import (
    EXPECTED_SOURCE_ROWS,
    direct_hf_profile,
    load_direct_hf_rows,
    source_profiles,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_record_canonicalization import (
    CONTEXT_CANONICALIZATION_VERSION,
    NORMALIZATION_DOMAIN_RULES_VERSION,
    canonicalization_policy_manifest,
    canonicalize_bioavailability_record,
)


DEFAULT_STARLING_DATA_DIR = "data/starling_data/bioavailability_ma"
DEFAULT_SMILES_MAPPING = "data/starling_data/_shared/final_smiles_mapping_v2.parquet"
EXPECTED_SMILES_MAPPING_SHA256 = "98ae43b6d9c61b77f0a95e4dce681010694b163a02e7a313667592d985496a0b"
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v5"
)
DEFAULT_V65_ELIGIBLE_RECORDS = (
    "/data1/joseph/starling_assay_transfer/datasets/eligible/"
    "assay_transfer_soft_evidence_v6_5/records.parquet"
)

CLEANED_FILENAME = "01_cleaned_records.parquet"
NORMALIZED_RECORDS_FILENAME = "02_normalized_records.parquet"
RECORDS_FILENAME = "records.parquet"
REJECTIONS_FILENAME = "organization_exclusions.parquet"
DUPLICATES_FILENAME = "duplicates.parquet"
DISTRIBUTION_AUDIT_FILENAME = "scalar_distribution_audit.parquet"
V65_RECONCILIATION_FILENAME = "v65_reconciliation.parquet"
V65_RECONCILIATION_SUMMARY_FILENAME = "v65_reconciliation_summary.json"
ENDPOINT_REGISTRY_FILENAME = "endpoint_metric_registry.json"
ENDPOINT_INVENTORY_FILENAME = "endpoint_inventory.json"
SOURCE_INVENTORY_FILENAME = "source_inventory.json"
MANIFEST_FILENAME = "manifest.json"
EVIDENCE_FILENAME = "molecule_family_evidence.jsonl"
INDEX_FILENAME = "neighbor_index.pkl"
INDEX_META_FILENAME = "neighbor_index.meta.json"
CANONICALIZATION_POLICY_FILENAME = "context_canonicalization_policy.json"
INDEX_VERSION = "bioavailability_ma_starling_normalized_neighbor_index.v5"

STAGES = ("clean", "normalize", "organize", "index")
STAGE_ARTIFACTS = {
    "clean": (CLEANED_FILENAME, "01_cleaning.manifest.json", CLEANING_STAGE_VERSION),
    "normalize": (
        NORMALIZED_RECORDS_FILENAME,
        "02_normalization.manifest.json",
        NORMALIZATION_STAGE_VERSION,
    ),
    "organize": (RECORDS_FILENAME, "03_organization.manifest.json", ORGANIZATION_STAGE_VERSION),
}
STAGE_OUTPUT_FILENAMES = {
    "clean": (
        CLEANED_FILENAME,
        "01_cleaning.manifest.json",
        SOURCE_INVENTORY_FILENAME,
        ENDPOINT_INVENTORY_FILENAME,
        ENDPOINT_REGISTRY_FILENAME,
    ),
    "normalize": (
        NORMALIZED_RECORDS_FILENAME,
        "02_normalization.manifest.json",
        CANONICALIZATION_POLICY_FILENAME,
    ),
    "organize": (
        RECORDS_FILENAME,
        "03_organization.manifest.json",
        DUPLICATES_FILENAME,
        REJECTIONS_FILENAME,
        DISTRIBUTION_AUDIT_FILENAME,
    ),
    "index": (
        EVIDENCE_FILENAME,
        INDEX_FILENAME,
        INDEX_META_FILENAME,
        MANIFEST_FILENAME,
    ),
}
STAGE_UPSTREAM_INPUTS = {
    "normalize": ("cleaned_records", CLEANED_FILENAME),
    "organize": ("normalized_records", NORMALIZED_RECORDS_FILENAME),
}
RECORD_DEPENDENT_DIRECTORIES = ("pair_buckets", "endpoint_policies", "analysis")
RECORD_DEPENDENT_FILES = (
    V65_RECONCILIATION_FILENAME,
    V65_RECONCILIATION_SUMMARY_FILENAME,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.monotonic()
    out_dir = Path(ensure_dir(args.out_dir))
    first_stage = STAGES.index(args.from_stage)
    final_stage = STAGES.index(args.through_stage)
    invalidated_artifacts: list[str] = []

    source_inventory = _read_optional_json(out_dir / SOURCE_INVENTORY_FILENAME)
    endpoint_inventories = _read_optional_json(out_dir / ENDPOINT_INVENTORY_FILENAME)

    if first_stage == 0:
        (
            cleaned,
            source_inventory,
            endpoint_inventories,
            cleaning_inputs,
        ) = _build_cleaned_records(args)
        _require_valid_schema(cleaned, "clean")
        with _temporary_stage_directory(out_dir, "clean") as stage_dir:
            cleaned_path = stage_dir / CLEANED_FILENAME
            write_parquet(cleaned_path, cleaned)
            _write_json(stage_dir / SOURCE_INVENTORY_FILENAME, source_inventory)
            _write_json(stage_dir / ENDPOINT_INVENTORY_FILENAME, endpoint_inventories)
            _write_json(
                stage_dir / ENDPOINT_REGISTRY_FILENAME,
                {
                    source_id: inventory["endpoints"]
                    for source_id, inventory in endpoint_inventories.items()
                },
            )
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "clean",
                inputs=cleaning_inputs,
                output_filename=CLEANED_FILENAME,
                row_counts={"cleaned_records": len(cleaned)},
                validations={"one_cleaned_record_per_source_row": True},
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "clean", stage_dir)
            )
        _log(f"clean: records={len(cleaned):,}")
    else:
        cleaned = (
            _load_verified_stage(out_dir, "clean") if first_stage == 1 else []
        )

    if final_stage == 0:
        return _finish_partial(
            out_dir,
            args,
            source_inventory,
            endpoint_inventories,
            started,
            invalidated_artifacts,
        )

    if first_stage <= 1:
        if not cleaned:
            cleaned = _load_verified_stage(out_dir, "clean")
        normalized = normalize_cleaned_records(
            cleaned,
            endpoint_normalizer=spacing_and_spelling_decision,
            endpoint_standardizer=endpoint_specific_standardization_of_unit,
            family_resolver=family_assignment,
            record_enricher=canonicalize_bioavailability_record,
        )
        identity_errors = validate_cleaned_normalized_identity(cleaned, normalized)
        if identity_errors:
            raise ValueError(
                f"{len(identity_errors)} cleaned/normalized identity failure(s); "
                f"first={identity_errors[0]}"
            )
        pair_errors = validate_measurement_pairs(
            normalized,
            endpoint_specific_standardization_of_unit,
        )
        if pair_errors:
            raise ValueError(
                f"{len(pair_errors)} measurement/unit pair invariant failure(s); "
                f"first={pair_errors[0]}"
            )
        _require_valid_schema(normalized, "normalize")
        with _temporary_stage_directory(out_dir, "normalize") as stage_dir:
            normalized_path = stage_dir / NORMALIZED_RECORDS_FILENAME
            write_parquet(normalized_path, normalized)
            _write_json(
                stage_dir / CANONICALIZATION_POLICY_FILENAME,
                canonicalization_policy_manifest(),
            )
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "normalize",
                inputs={"cleaned_records": out_dir / CLEANED_FILENAME},
                output_filename=NORMALIZED_RECORDS_FILENAME,
                row_counts={
                    "cleaned_records": len(cleaned),
                    "normalized_records": len(normalized),
                    "finite_scalars": sum(
                        row.get("finite_scalar_value") is not None
                        for row in normalized
                    ),
                    "absolute_and_continuous": sum(
                        bool(row.get("is_absolute_and_continuous"))
                        for row in normalized
                    ),
                    "normalization_valid": sum(
                        row.get("normalization_validity_status") == "valid"
                        for row in normalized
                    ),
                },
                validations={
                    "one_to_one_cleaned_to_normalized_ids": True,
                    "measurement_unit_pair_errors": 0,
                    "endpoint_orthography_provenance": True,
                    "canonical_endpoint_present": True,
                    "policy_independent_validity_present": True,
                },
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "normalize", stage_dir)
            )
        _log(
            "normalize: "
            f"records={len(normalized):,} "
            f"scalars={sum(row.get('finite_scalar_value') is not None for row in normalized):,}"
        )
    else:
        normalized = (
            _load_verified_stage(out_dir, "normalize") if first_stage == 2 else []
        )

    if final_stage == 1:
        return _finish_partial(
            out_dir,
            args,
            source_inventory,
            endpoint_inventories,
            started,
            invalidated_artifacts,
        )

    if first_stage <= 2:
        if not normalized:
            normalized = _load_verified_stage(out_dir, "normalize")
        records, duplicates, exclusions, organization_stats = organize_normalized_records(
            normalized
        )
        _require_valid_schema(records, "organize")
        distribution_rows = scalar_distribution_audit(records)
        with _temporary_stage_directory(out_dir, "organize") as stage_dir:
            records_path = stage_dir / RECORDS_FILENAME
            write_parquet(records_path, records)
            write_parquet(stage_dir / DUPLICATES_FILENAME, duplicates)
            write_parquet(stage_dir / REJECTIONS_FILENAME, exclusions)
            write_parquet(
                stage_dir / DISTRIBUTION_AUDIT_FILENAME, distribution_rows
            )
            _write_staged_stage_manifest(
                stage_dir,
                out_dir,
                "organize",
                inputs={
                    "normalized_records": out_dir / NORMALIZED_RECORDS_FILENAME
                },
                output_filename=RECORDS_FILENAME,
                row_counts=organization_stats,
                validations={
                    "cross_source_deduplication": False,
                    "unusable_structures_retained_in_records": True,
                },
            )
            invalidated_artifacts.extend(
                _commit_stage_outputs(out_dir, "organize", stage_dir)
            )
        _log(
            f"organize: records={len(records):,} "
            f"retrieval_eligible={organization_stats['n_retrieval_eligible']:,}"
        )
    else:
        records = _load_verified_stage(out_dir, "organize")
        duplicates = read_parquet_records(out_dir / DUPLICATES_FILENAME)
        organization_stats = {
            "n_normalized_records_before_deduplication": len(records) + len(duplicates),
            "n_records": len(records),
            "n_duplicates_removed": len(duplicates),
            "n_retrieval_eligible": sum(
                bool(row.get("retrieval_eligible")) for row in records
            ),
            "n_organization_exclusions": sum(
                not bool(row.get("retrieval_eligible")) for row in records
            ),
            "n_absolute_and_continuous": sum(
                bool(row.get("is_absolute_and_continuous")) for row in records
            ),
        }

    frozen_census = {
        "n_cleaned_records": sum(
            int(value)
            for value in (source_inventory.get("source_row_counts") or {}).values()
        ),
        **organization_stats,
            "n_fg_finite_scalars": sum(
            row.get("source_id") == "fg"
            and row.get("finite_scalar_value") is not None
            for row in records
        ),
        "normalization_validity_status_counts": _count_by(
            records, "normalization_validity_status"
        ),
    }
    reconciliation_summary = {
        "status": "not_performed",
        "matching_performed": False,
    }

    if final_stage == 2:
        return _finish_partial(
            out_dir,
            args,
            source_inventory,
            endpoint_inventories,
            started,
            invalidated_artifacts,
        )

    evidence_rows = aggregate_molecule_family_records(
        records,
        max_record_examples=args.max_record_examples,
    )
    index = build_neighbor_index(
        evidence_rows,
        index_version=INDEX_VERSION,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    index["source"] = {
        "type": "starling_normalized_records",
        "dataset": "starling-labs/Bioavailability_Ma",
        "record_artifact": str(out_dir / RECORDS_FILENAME),
        "groups": sorted(index.get("group_to_molecule_indices", {})),
        "exact_query_exclusion": True,
    }
    with _temporary_stage_directory(out_dir, "index") as stage_dir:
        write_jsonl(stage_dir / EVIDENCE_FILENAME, evidence_rows)
        with (stage_dir / INDEX_FILENAME).open("wb") as handle:
            pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        artifact_filenames = {
            "cleaned_records": CLEANED_FILENAME,
            "normalized_records": NORMALIZED_RECORDS_FILENAME,
            "records": RECORDS_FILENAME,
            "duplicates": DUPLICATES_FILENAME,
            "organization_exclusions": REJECTIONS_FILENAME,
            "molecule_family_evidence": EVIDENCE_FILENAME,
            "scalar_distribution_audit": DISTRIBUTION_AUDIT_FILENAME,
            "context_canonicalization_policy": CANONICALIZATION_POLICY_FILENAME,
            "neighbor_index": INDEX_FILENAME,
        }
        artifact_hashes = {
            name: file_sha256(
                stage_dir / filename
                if (stage_dir / filename).exists()
                else out_dir / filename
            )
            for name, filename in artifact_filenames.items()
        }
        manifest = {
            "artifact_version": NORMALIZED_ARTIFACT_VERSION,
            "record_version": NORMALIZED_RECORD_VERSION,
            "scalar_parser_version": SCALAR_PARSER_VERSION,
            "unit_normalizer_version": UNIT_NORMALIZER_VERSION,
            "spacing_and_spelling_version": SPACING_AND_SPELLING_VERSION,
            "endpoint_policy_version": ENDPOINT_POLICY_VERSION,
            "context_canonicalization_version": CONTEXT_CANONICALIZATION_VERSION,
            "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
            "v65_reconciliation_version": None,
            "index_version": INDEX_VERSION,
            "completed_stages": list(STAGES),
            "rebuild_request": {
                "from_stage": args.from_stage,
                "through_stage": args.through_stage,
            },
            "invalidated_artifacts": sorted(set(invalidated_artifacts)),
            "source_inventory": source_inventory,
            "endpoint_inventory": endpoint_inventories,
            "stats": {
                **frozen_census,
                "n_input_rows": sum(
                    int(value)
                    for value in (
                        source_inventory.get("source_row_counts") or {}
                    ).values()
                ),
                "n_molecule_family_evidence_rows": len(evidence_rows),
                "n_index_molecules": len(index["molecules"]),
            },
            "groups": sorted(index["group_to_molecule_indices"]),
            "fingerprint": fingerprint_metadata(),
            "artifact_hashes": artifact_hashes,
            "v65_reconciliation": reconciliation_summary,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        _write_json(stage_dir / MANIFEST_FILENAME, manifest)
        _write_json(
            stage_dir / INDEX_META_FILENAME,
            {
                "index_version": INDEX_VERSION,
                "n_evidence_rows": len(evidence_rows),
                "n_index_molecules": len(index["molecules"]),
                "groups": sorted(index["group_to_molecule_indices"]),
                "record_artifact": str(out_dir / RECORDS_FILENAME),
                "record_artifact_sha256": artifact_hashes["records"],
                "fingerprint": fingerprint_metadata(),
            },
        )
        invalidated_artifacts.extend(
            _commit_stage_outputs(out_dir, "index", stage_dir)
        )
    _log(
        f"complete: records={len(records):,} "
        f"molecule-family rows={len(evidence_rows):,} "
        f"index molecules={len(index['molecules']):,} out={out_dir}"
    )
    return 0


def _build_cleaned_records(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Path]]:
    data_dir = Path(args.starling_data_dir)
    mapping_path = Path(args.smiles_mapping)
    if not mapping_path.exists():
        raise FileNotFoundError(f"authoritative SMILES mapping not found: {mapping_path}")
    mapping_sha = file_sha256(mapping_path)
    if not args.allow_unpinned_smiles_mapping and mapping_sha != EXPECTED_SMILES_MAPPING_SHA256:
        raise ValueError(
            "SMILES mapping hash mismatch: "
            f"expected {EXPECTED_SMILES_MAPPING_SHA256}, found {mapping_sha}"
        )

    profiles = source_profiles(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    source_hashes: dict[str, str] = {}
    endpoint_inventories: dict[str, Any] = {}
    needed_identifiers: set[str] = set()
    inputs: dict[str, Path] = {"smiles_mapping": mapping_path}
    for profile in profiles:
        path = Path(profile.source_path)
        inputs[profile.source_id] = path
        source_hashes[profile.source_id] = file_sha256(path)
        frame = pd.read_parquet(path)
        if not args.max_rows_per_source and len(frame) != EXPECTED_SOURCE_ROWS[profile.source_id]:
            raise ValueError(
                f"source row-count drift for {profile.source_id}: "
                f"expected {EXPECTED_SOURCE_ROWS[profile.source_id]:,}, found {len(frame):,}"
            )
        if args.max_rows_per_source:
            frame = frame.head(args.max_rows_per_source)
        frames[profile.source_id] = frame
        needed_identifiers.update(
            str(value)
            for value in frame.get("global_identifier", [])
            if value is not None and str(value).strip()
        )
        endpoints = [
            normalize_endpoint_name(value) or ""
            for value in frame[profile.endpoint_field].tolist()
        ]
        endpoint_inventories[profile.source_id] = (
            validate_endpoint_inventory(profile.source_id, endpoints)
            if args.strict_endpoint_inventory
            else _unfrozen_endpoint_inventory(profile.source_id, endpoints)
        )

    smiles_mapping = load_smiles_mapping(mapping_path, needed_identifiers)
    cleaned: list[dict[str, Any]] = []
    for profile in profiles:
        cleaned.extend(
            clean_source_rows(
                frames[profile.source_id].to_dict(orient="records"),
                profile,
                smiles_mapping=smiles_mapping,
                source_sha256=source_hashes[profile.source_id],
            )
        )

    direct_info: dict[str, Any] | None = None
    if args.include_direct_hf:
        records_path = Path(args.direct_source_jsonl)
        dropped_path = Path(args.direct_dropped_jsonl)
        inputs["direct_records"] = records_path
        inputs["direct_dropped"] = dropped_path
        direct_info = {
            "records_jsonl": str(records_path),
            "records_sha256": file_sha256(records_path),
            "dropped_jsonl": str(dropped_path),
            "dropped_sha256": file_sha256(dropped_path),
        }
        direct_hash = hashlib.sha256(
            json.dumps(direct_info, sort_keys=True).encode("utf-8")
        ).hexdigest()
        direct_rows = load_direct_hf_rows(
            records_path,
            dropped_path,
            max_rows=args.max_direct_rows,
        )
        if not args.max_direct_rows and len(direct_rows) != EXPECTED_SOURCE_ROWS["direct_hf"]:
            raise ValueError(
                "direct HF row-count drift: "
                f"expected {EXPECTED_SOURCE_ROWS['direct_hf']:,}, found {len(direct_rows):,}"
            )
        endpoint_inventories["direct_hf"] = (
            validate_endpoint_inventory(
                "direct_hf", ["oral_bioavailability"] if direct_rows else []
            )
            if args.strict_endpoint_inventory
            else _unfrozen_endpoint_inventory(
                "direct_hf", ["oral_bioavailability"] if direct_rows else []
            )
        )
        cleaned.extend(
            clean_source_rows(
                direct_rows,
                direct_hf_profile(records_path, dropped_path),
                smiles_mapping=None,
                source_sha256=direct_hash,
            )
        )
        source_hashes["direct_hf"] = direct_hash

    source_inventory = {
        "dataset": "starling-labs/Bioavailability_Ma",
        "source_hashes": source_hashes,
        "source_row_counts": _count_by(cleaned, "source_id"),
        "direct_hf": direct_info,
        "smiles_mapping": {
            "path": str(mapping_path),
            "sha256": mapping_sha,
            "expected_sha256": EXPECTED_SMILES_MAPPING_SHA256,
            "n_loaded_identifiers": len(smiles_mapping),
        },
    }
    return cleaned, source_inventory, endpoint_inventories, inputs


def load_smiles_mapping(path: Path, needed_identifiers: set[str]) -> dict[str, str]:
    frame = pd.read_parquet(path, columns=["global_identifier", "smiles"])
    if needed_identifiers:
        frame = frame[frame["global_identifier"].isin(needed_identifiers)]
    mapping: dict[str, str] = {}
    conflicts: set[str] = set()
    for identifier, smiles in frame[["global_identifier", "smiles"]].itertuples(index=False):
        key, value = str(identifier), str(smiles)
        existing = mapping.get(key)
        if existing is not None and existing != value:
            conflicts.add(key)
        else:
            mapping[key] = value
    if conflicts:
        raise ValueError(
            f"authoritative SMILES mapping has {len(conflicts)} conflicting identifier(s)"
        )
    return mapping


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _unfrozen_endpoint_inventory(source_id: str, endpoints: list[str]) -> dict[str, Any]:
    unique = sorted(set(endpoints))
    registry = [
        spacing_and_spelling_decision(source_id, endpoint).to_dict()
        for endpoint in unique
    ]
    return {
        "source_id": source_id,
        "count": len(unique),
        "n_reviewed_corrections": sum(
            item["status"] == "reviewed_correction" for item in registry
        ),
        "coverage": 1.0,
        "strict_frozen_validation": False,
        "endpoints": registry,
    }


def _load_verified_stage(out_dir: Path, stage: str) -> list[dict[str, Any]]:
    filename, manifest_filename, version = STAGE_ARTIFACTS[stage]
    artifact = out_dir / filename
    manifest_path = out_dir / manifest_filename
    if not artifact.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"cannot restart from {stage}: missing {artifact} or {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != version:
        raise ValueError(
            f"{stage} stage version mismatch: expected {version}, "
            f"found {manifest.get('version')}"
        )
    expected_hash = manifest.get("output", {}).get("sha256")
    actual_hash = file_sha256(artifact)
    if expected_hash != actual_hash:
        raise ValueError(
            f"{stage} stage artifact hash mismatch: expected {expected_hash}, found {actual_hash}"
        )
    upstream = STAGE_UPSTREAM_INPUTS.get(stage)
    if upstream is not None:
        input_name, upstream_filename = upstream
        upstream_artifact = out_dir / upstream_filename
        expected_upstream_hash = (
            (manifest.get("inputs") or {}).get(input_name, {}).get("sha256")
        )
        if not upstream_artifact.exists() or not expected_upstream_hash:
            raise ValueError(
                f"{stage} stage manifest is missing a verifiable upstream "
                f"input: {input_name}"
            )
        actual_upstream_hash = file_sha256(upstream_artifact)
        if expected_upstream_hash != actual_upstream_hash:
            raise ValueError(
                f"{stage} stage upstream input hash mismatch for {input_name}: "
                f"expected {expected_upstream_hash}, found {actual_upstream_hash}"
            )
    return read_parquet_records(artifact)


def _write_staged_stage_manifest(
    stage_dir: Path,
    out_dir: Path,
    stage: str,
    *,
    inputs: dict[str, Path],
    output_filename: str,
    row_counts: dict[str, int],
    validations: dict[str, Any],
) -> None:
    _, manifest_filename, version = STAGE_ARTIFACTS[stage]
    payload = stage_manifest(
        stage=stage,
        version=version,
        inputs=inputs,
        output=stage_dir / output_filename,
        row_counts=row_counts,
        validations=validations,
    )
    payload["output"]["path"] = str(out_dir / output_filename)
    _write_json(
        stage_dir / manifest_filename,
        payload,
    )


@contextmanager
def _temporary_stage_directory(out_dir: Path, stage: str):
    with tempfile.TemporaryDirectory(
        dir=out_dir, prefix=f".{stage}-stage-"
    ) as directory:
        yield Path(directory)


def _commit_stage_outputs(
    out_dir: Path,
    stage: str,
    stage_dir: Path,
) -> list[str]:
    expected_outputs = STAGE_OUTPUT_FILENAMES[stage]
    missing = [
        filename
        for filename in expected_outputs
        if not (stage_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"cannot publish {stage} stage; missing staged outputs: {missing}"
        )

    invalidated = _invalidate_downstream_artifacts(out_dir, stage)
    if stage != "index":
        (out_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
    for filename in expected_outputs:
        os.replace(stage_dir / filename, out_dir / filename)
    return invalidated


def _invalidate_downstream_artifacts(out_dir: Path, stage: str) -> list[str]:
    stage_index = STAGES.index(stage)
    targets: list[Path] = []
    for downstream_stage in STAGES[stage_index + 1 :]:
        targets.extend(
            out_dir / filename
            for filename in STAGE_OUTPUT_FILENAMES[downstream_stage]
            if filename != MANIFEST_FILENAME
        )
    if stage_index <= STAGES.index("organize"):
        targets.extend(out_dir / filename for filename in RECORD_DEPENDENT_FILES)
        targets.extend(out_dir / directory for directory in RECORD_DEPENDENT_DIRECTORIES)

    invalidated: list[str] = []
    for target in targets:
        if not target.exists():
            continue
        relative = str(target.relative_to(out_dir))
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        invalidated.append(relative)
    if invalidated:
        _log(
            f"invalidate after {stage}: "
            + ", ".join(sorted(invalidated))
        )
    return sorted(invalidated)


def _require_valid_schema(rows: list[dict[str, Any]], stage: str) -> None:
    errors = validate_stage_schema(rows, stage)
    if errors:
        raise ValueError(errors[0])


def _finish_partial(
    out_dir: Path,
    args: argparse.Namespace,
    source_inventory: dict[str, Any],
    endpoint_inventories: dict[str, Any],
    started: float,
    invalidated_artifacts: list[str],
) -> int:
    completed = list(STAGES[: STAGES.index(args.through_stage) + 1])
    payload = {
        "artifact_version": NORMALIZED_ARTIFACT_VERSION,
        "record_version": NORMALIZED_RECORD_VERSION,
        "spacing_and_spelling_version": SPACING_AND_SPELLING_VERSION,
        "endpoint_policy_version": ENDPOINT_POLICY_VERSION,
        "context_canonicalization_version": CONTEXT_CANONICALIZATION_VERSION,
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "v65_reconciliation": {
            "status": "not_performed",
            "matching_performed": False,
        },
        "completed_stages": completed,
        "rebuild_request": {
            "from_stage": args.from_stage,
            "through_stage": args.through_stage,
        },
        "invalidated_artifacts": sorted(set(invalidated_artifacts)),
        "source_inventory": source_inventory,
        "endpoint_inventory": endpoint_inventories,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    with _temporary_stage_directory(out_dir, "manifest") as stage_dir:
        _write_json(stage_dir / MANIFEST_FILENAME, payload)
        os.replace(stage_dir / MANIFEST_FILENAME, out_dir / MANIFEST_FILENAME)
    _log(f"partial build complete through {args.through_stage}: out={out_dir}")
    return 0


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _log(message: str) -> None:
    print(f"[build_normalized_starling_evidence_library] {message}", flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starling-data-dir", default=DEFAULT_STARLING_DATA_DIR)
    parser.add_argument("--smiles-mapping", default=DEFAULT_SMILES_MAPPING)
    parser.add_argument("--allow-unpinned-smiles-mapping", action="store_true")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--direct-source-jsonl", default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--direct-dropped-jsonl", default=DEFAULT_DROPPED_JSONL)
    parser.add_argument(
        "--include-direct-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict-endpoint-inventory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--from-stage", choices=STAGES, default="clean")
    parser.add_argument("--through-stage", choices=STAGES, default="index")
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--max-direct-rows", type=int, default=0)
    parser.add_argument("--max-record-examples", type=int, default=6)
    parser.add_argument(
        "--v65-reconciliation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--v65-eligible-records",
        default=DEFAULT_V65_ELIGIBLE_RECORDS,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.v65_reconciliation:
        parser.error(
            "v6.5 reconciliation is outside the source-aware pair-bucket sidecar contract"
        )
    if STAGES.index(args.from_stage) > STAGES.index(args.through_stage):
        parser.error("--from-stage cannot be later than --through-stage")
    if (args.max_rows_per_source or args.max_direct_rows) and args.strict_endpoint_inventory:
        parser.error("bounded source runs require --no-strict-endpoint-inventory")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
