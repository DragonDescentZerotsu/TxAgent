"""DILI V10 policy for source-faithful Stage 0-1 construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.compact_artifacts import CompactArtifactProfile
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_literal_text,
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.contracts import (
    NormalizedSourceProfile,
)
from data.processing.evidence_library.shared.v2.normalization.source_value_cleaning import (
    SourceValueCleaningResult,
)
from data.processing.evidence_library.shared.v2.normalization.task_policy import (
    StarlingTaskPolicy,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
    attach_stage1_routes,
)
from data.processing.evidence_library.versions.v10.tasks.dili.starling_schema import (
    BASE_SOURCE_GROUP,
    RAW_SOURCE_COLUMNS,
    RECORD_CONTRACT,
    ROLE_FIELDS,
    SOURCE_COLUMNS,
    TASK_ID,
)
from data.processing.evidence_library.versions.v10.tasks.dili.starling_measurement_resolution import (
    DILI_MEASUREMENT_ROUTING_VERSION,
)
from data.processing.gold_labels.level_mappings import level_mapping_path
from data.processing.paths import (
    evidence_library_root,
    raw_starling_task_root,
)


TASK_ROOT = Path(__file__).resolve().parent
V10_ROOT = TASK_ROOT.parents[1]
SOURCE_MANIFEST_PATH = TASK_ROOT / "source_manifest.json"
SOURCE_MANIFEST = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
SOURCE_SPECS = SOURCE_MANIFEST["sources"]

DATASET_NAME = "Starling DILI six-layer evidence"
DEFAULT_DATA_DIR = str(raw_starling_task_root(TASK_ID))
DEFAULT_OUT_DIR = str(evidence_library_root(TASK_ID, "v10"))
EXPECTED_SOURCE_ROWS = {
    source_id: int(spec["rows"]) for source_id, spec in SOURCE_SPECS.items()
}


def _source_path(data_dir: Path, source_id: str, filename: str) -> Path:
    return data_dir / SOURCE_SPECS[source_id]["directory"] / filename


def source_profiles(data_dir: Path) -> list[NormalizedSourceProfile]:
    profiles = []
    for source_id in RAW_SOURCE_COLUMNS:
        endpoint, measurement, unit = ROLE_FIELDS[source_id]
        literal_fields = tuple(
            field
            for field in (
                endpoint,
                "endpoint_name" if source_id == "dili_v5" else "",
            )
            if field
        )
        profiles.append(
            NormalizedSourceProfile(
                source_id=source_id,
                source_name=f"starling-labs/dili/{SOURCE_SPECS[source_id]['directory']}",
                source_path=str(_source_path(data_dir, source_id, "extractions.parquet")),
                endpoint_field=endpoint,
                endpoint_constant=BASE_SOURCE_GROUP if source_id == "dili_base" else "",
                measurement_field=measurement,
                unit_field=unit,
                embedded_unit=source_id == "dili_v2",
                smiles_field="SMILES",
                structure_mode="direct",
                record_id_field="extraction_id",
                name_fields=("molecule_name",)
                if "molecule_name" in RAW_SOURCE_COLUMNS[source_id]
                else (),
                context_fields=(
                    ("quantitative_measure_type", "assay_and_platform", "effect_direction")
                    if source_id == "dili_v5"
                    else ()
                ),
                literal_text_fields=literal_fields,
            )
        )
    return profiles


def validate_source_digest(source_id: str, source_path: Path) -> None:
    spec = SOURCE_SPECS[source_id]
    actual_hash = file_sha256(source_path)
    if actual_hash != spec["parquet_sha256"]:
        raise ValueError(
            f"source digest drift for {source_id}: expected {spec['parquet_sha256']}, "
            f"found {actual_hash}"
        )
    columns = tuple(pq.read_schema(source_path).names)
    expected_columns = (*RAW_SOURCE_COLUMNS[source_id], "source_row_uid")
    if columns != expected_columns:
        raise ValueError(
            f"source schema drift for {source_id}: expected={expected_columns!r}, "
            f"found={columns!r}"
        )
    guidance_path = source_path.with_name("extraction_guidance.json")
    guidance_hash = file_sha256(guidance_path)
    if guidance_hash != spec["guidance_sha256"]:
        raise ValueError(
            f"guidance digest drift for {source_id}: "
            f"expected {spec['guidance_sha256']}, found {guidance_hash}"
        )


def endpoint_inventory(
    source_id: str, endpoints: list[str], *, strict: bool
) -> dict[str, Any]:
    values = sorted({clean_literal_text(value) or "" for value in endpoints})
    return {
        "source_id": source_id,
        "count": len(values),
        "coverage": 1.0,
        "mapping_coverage": 1.0,
        "strict_source_snapshot": strict,
        "endpoint_mapping_applied": False,
        "off_schema_values": [],
        "excluded_values": [],
        "canonical_endpoints": [],
        "endpoints": values,
    }


def clean_source_values(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> SourceValueCleaningResult:
    for record in records:
        source_id = str(record["source_id"])
        endpoint_field = ROLE_FIELDS[source_id][0]
        if endpoint_field:
            payload = json.loads(str(record.get("source_payload_json") or "{}"))
            if not isinstance(payload, dict):
                raise ValueError("source_payload_json must contain an object")
            record["endpoint_name"] = clean_literal_text(payload.get(endpoint_field))
            if source_id == "dili_v5":
                record["specific_endpoint_name"] = clean_literal_text(
                    payload.get("endpoint_name")
                )
    routed = attach_stage1_routes(records, task=TASK_ID)
    data_dir = Path(args.starling_data_dir)
    guidance_paths = tuple(
        _source_path(data_dir, source_id, "extraction_guidance.json")
        for source_id in RAW_SOURCE_COLUMNS
    )
    return SourceValueCleaningResult(
        records=routed,
        audit_rows=[],
        manifest={
            "version": "dili_source_preservation.v1",
            "n_input_records": len(records),
            "n_output_records": len(routed),
            "n_repaired_records": 0,
            "n_dropped_records": 0,
            "audit_counts": {},
            "scientific_fields_changed": False,
            "endpoint_mapping_applied": False,
            "raw_endpoint_values_preserved": True,
        },
        input_paths=(SOURCE_MANIFEST_PATH, *guidance_paths),
    )


def attach_source_columns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return records


def _stage2_unavailable(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise RuntimeError("DILI V10 currently implements only source and clean stages")


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.through_stage not in {"source", "clean"}:
        parser.error("DILI V10 currently implements only source and clean stages")


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    del complete
    return {
        "evidence_library_version": "dili_normalized_v10",
        "dili_source_snapshot_version": SOURCE_MANIFEST["version"],
        "dili_base_source_group": BASE_SOURCE_GROUP,
        "dili_endpoint_policy": "raw_identity_or_missing_batch_key.v1",
        "measurement_routing_version": MEASUREMENT_ROUTING_VERSION,
        "dili_measurement_routing_version": DILI_MEASUREMENT_ROUTING_VERSION,
        "stage2_status": "not_implemented",
    }


COMPACT_PROFILE = CompactArtifactProfile(
    task_id=TASK_ID,
    artifact_version="dili.compact_stage1.v1",
    index_version="dili.index_not_built.v1",
    evidence_source_label="Starling DILI evidence",
    source_columns=SOURCE_COLUMNS,
    llm_source_projection=RECORD_CONTRACT.source_projection,
)

DEFAULT_GUIDANCE_PATHS = tuple(
    _source_path(Path(DEFAULT_DATA_DIR), source_id, "extraction_guidance.json")
    for source_id in RAW_SOURCE_COLUMNS
)
TASK_IMPLEMENTATION_PATHS = (
    TASK_ROOT / "build_normalized_starling_evidence_library.py",
    TASK_ROOT / "starling_policy.py",
    TASK_ROOT / "starling_schema.py",
    TASK_ROOT / "starling_measurement_resolution.py",
    V10_ROOT / "build_normalized_evidence_library.py",
    V10_ROOT / "measurement_routing.py",
)

POLICY = StarlingTaskPolicy(
    task_id=TASK_ID,
    dataset_name=DATASET_NAME,
    default_data_dir=DEFAULT_DATA_DIR,
    default_out_dir=DEFAULT_OUT_DIR,
    compact=COMPACT_PROFILE,
    expected_source_rows=EXPECTED_SOURCE_ROWS,
    source_profiles=source_profiles,
    endpoint_inventory=endpoint_inventory,
    family_resolver=_stage2_unavailable,
    build_hooks=_stage2_unavailable,
    attach_source_columns=attach_source_columns,
    stage_documents=_stage2_unavailable,
    manifest_versions=manifest_versions,
    record_contract=RECORD_CONTRACT,
    source_value_cleaner=clean_source_values,
    validate_arguments=validate_arguments,
    verify_source_digest=validate_source_digest,
    scientific_assets=(
        *TASK_IMPLEMENTATION_PATHS,
        SOURCE_MANIFEST_PATH,
        *DEFAULT_GUIDANCE_PATHS,
    ),
    source_universe_mapping=level_mapping_path("dili", "v1"),
    stage1_measurement_routing_enabled=True,
)


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_OUT_DIR",
    "EXPECTED_SOURCE_ROWS",
    "POLICY",
    "SOURCE_MANIFEST",
    "clean_source_values",
    "endpoint_inventory",
    "source_profiles",
    "validate_source_digest",
]
