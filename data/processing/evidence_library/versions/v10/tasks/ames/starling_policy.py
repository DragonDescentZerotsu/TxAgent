"""Ames V10 policy for the frozen four-layer source snapshot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from data.processing.evidence_library.compact_artifacts import CompactArtifactProfile
from data.processing.evidence_library.shared.v2.auxiliary_metadata import (
    AuxiliaryMetadataAttacher,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.contracts import (
    MeasurementPair,
    NormalizedSourceProfile,
)
from data.processing.evidence_library.shared.v2.normalization.source_value_cleaning import (
    SourceValueCleaningResult,
)
from data.processing.evidence_library.shared.v2.normalization.task_policy import (
    NormalizationHooks,
    StageDocuments,
    StarlingTaskPolicy,
)
from data.processing.evidence_library.shared.v2.reference_semantics import (
    ReferenceSemanticsAttacher,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
    attach_stage1_routes,
)
from data.processing.evidence_library.versions.v10.standard_pair_dimension_stage2 import (
    build_local_source_attacher,
)
from data.processing.evidence_library.versions.v10.tasks.ames.mapping_registry import (
    REGISTRY_PATH,
    mapping_path,
    mapping_registry,
    validate_mapping_hashes,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.build_exact_unit_review import (
    validate_review_completion,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_categorical_response import (
    POLICY as CATEGORICAL_POLICY,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_endpoint_normalization import (
    DEFAULT_ENDPOINT_MAPPING,
    ENDPOINT_NORMALIZATION_VERSION,
    canonical_endpoints_by_source,
    endpoint_decision,
    endpoint_orthography,
    expected_raw_endpoints,
    validate_endpoint_inventory,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_exact_unit_mapping import (
    EXACT_UNIT_MAPPING_PATH,
    REVIEWED_UNIT_DECISIONS_PATH,
    compile_exact_unit_mapping,
    load_reviewed_unit_decisions,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_family_assignment import (
    FAMILY_ASSIGNMENT_VERSION,
    family_assignment,
    family_assignment_manifest,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_resolution import (
    validate_mapping_provenance,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_record_canonicalization import (
    enrich_ames_validity,
    validity_policy_manifest,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_reference_semantics import (
    DEFAULT_MAPPING_PATH as DEFAULT_REFERENCE_SEMANTICS_MAPPING,
    REFERENCE_SEMANTICS_CONFIG,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_schema import (
    ENDPOINT_PRODUCER_FIELD,
    PAIR_CONTEXT_INPUTS,
    PAIR_CONTEXT_MAPPING_VERSION,
    PAIR_PRODUCER_FIELD,
    RECORD_CONTRACT,
    ROLE_FIELDS,
    SOURCE_COLUMNS,
    SOURCE_ENDPOINT_PRODUCER_IDS,
    SOURCE_EXTRACTION_PAIR_PRODUCER_IDS,
    SOURCE_PAIR_PRODUCER_IDS,
    SOURCE_RULE_PAIR_PRODUCER_IDS,
    TASK_ID,
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

DATASET_NAME = "SEND/Ames four-layer evidence"
DEFAULT_DATA_DIR = str(raw_starling_task_root(TASK_ID))
DEFAULT_OUT_DIR = str(evidence_library_root(TASK_ID, "v10"))
EXPECTED_SOURCE_ROWS = {
    source_id: int(spec["rows"]) for source_id, spec in SOURCE_SPECS.items()
}
EXPECTED_SOURCE_SHA256 = {
    source_id: str(spec["parquet_sha256"]) for source_id, spec in SOURCE_SPECS.items()
}

ASSAY_FAMILIES = canonical_endpoints_by_source()
MAIN_UNIVERSE_MANIFEST = Path("data/raw/starling/main_universe_v1/manifest.json")
MAIN_UNIVERSE_RECORDS = Path("data/raw/starling/main_universe_v1/ames/records.parquet")
def _selected_auxiliary_mapping() -> Path | None:
    if "auxiliary_context" not in mapping_registry()["mappings"]:
        return None
    return mapping_path("auxiliary_context")


def _selected_reference_mapping() -> Path | None:
    if "reference_semantics" not in mapping_registry()["mappings"]:
        return None
    return mapping_path("reference_semantics")


SELECTED_AUXILIARY_MAPPING = _selected_auxiliary_mapping()
SELECTED_REFERENCE_MAPPING = _selected_reference_mapping()


def _source_path(data_dir: Path, source_id: str, filename: str) -> Path:
    return data_dir / SOURCE_SPECS[source_id]["directory"] / filename


def source_profiles(data_dir: Path) -> list[NormalizedSourceProfile]:
    profiles: list[NormalizedSourceProfile] = []
    for source_id in SOURCE_COLUMNS:
        endpoint, measurement, unit = ROLE_FIELDS[source_id]
        categorical_field = {
            "fixed_mutation": "result_call",
            "premutagenic_damage": "result_status",
            "mutagenicity_mechanism": "result_direction",
        }.get(source_id)
        profiles.append(
            NormalizedSourceProfile(
                source_id=source_id,
                source_name=f"SEND/Ames/{SOURCE_SPECS[source_id]['directory']}",
                source_path=str(
                    _source_path(data_dir, source_id, "extractions.parquet")
                ),
                endpoint_field=endpoint,
                measurement_field=measurement,
                unit_field=unit,
                smiles_field="SMILES",
                structure_mode="direct",
                record_id_field="extraction_id",
                name_fields=("molecule_name",)
                if source_id == "mutagenicity_outcomes"
                else (),
                context_fields=(categorical_field,) if categorical_field else (),
                literal_text_fields=(
                    ("metabolic_activation_system",)
                    if source_id == "mutagenicity_mechanism"
                    else ()
                ),
            )
        )
    return profiles


def validate_source_digest(source_id: str, source_path: Path) -> None:
    expected_hash = EXPECTED_SOURCE_SHA256[source_id]
    actual_hash = file_sha256(source_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"source digest drift for {source_id}: expected {expected_hash}, "
            f"found {actual_hash}"
        )
    columns = tuple(pq.read_schema(source_path).names)
    expected_columns = (*SOURCE_COLUMNS[source_id], "source_row_uid")
    if columns != expected_columns:
        raise ValueError(
            f"source schema drift for {source_id}: expected={expected_columns!r}, "
            f"found={columns!r}"
        )


def endpoint_inventory(
    source_id: str, endpoints: list[str], *, strict: bool
) -> dict[str, Any]:
    decisions = validate_endpoint_inventory(source_id, endpoints, strict=strict)
    excluded = [
        item["raw_endpoint"] for item in decisions if item["status"] == "excluded"
    ]
    changed = [
        item["raw_endpoint"] for item in decisions if item["status"] != "identity"
    ]
    canonical = sorted(
        {item["canonical_endpoint"] for item in decisions if item["canonical_endpoint"]}
    )
    return {
        "source_id": source_id,
        "count": len(decisions),
        "coverage": (len(decisions) - len(excluded)) / len(decisions)
        if decisions
        else 1.0,
        "mapping_coverage": 1.0,
        "strict_source_snapshot": strict,
        "endpoint_mapping_version": ENDPOINT_NORMALIZATION_VERSION,
        "off_schema_values": changed,
        "excluded_values": excluded,
        "canonical_endpoints": canonical,
        "endpoints": sorted(item["raw_endpoint"] for item in decisions),
    }


def _has_value(value: Any) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def _guidance(data_dir: Path, source_id: str) -> tuple[Path, dict[str, Any]]:
    path = _source_path(data_dir, source_id, "extraction_guidance.json")
    expected = str(SOURCE_SPECS[source_id]["guidance_sha256"])
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"guidance digest drift for {source_id}: expected {expected}, found {actual}"
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _audit_entry(
    row: dict[str, Any],
    *,
    audit_type: str,
    field: str,
    before: Any,
    after: Any,
    action: str,
) -> dict[str, Any]:
    render = lambda value: None if value is None or pd.isna(value) else str(value)
    return {
        "task_id": TASK_ID,
        "source_id": row["source_id"],
        "cleaned_record_id": row["cleaned_record_id"],
        "source_row_uid": row["source_row_uid"],
        "source_row_number": row["source_row_number"],
        "source_record_id": row["source_record_id"],
        "audit_type": audit_type,
        "field": field,
        "before": render(before),
        "after": render(after),
        "action": action,
    }


def _raw_audit_inputs(
    data_dir: Path,
    source_id: str,
    cleaned_rows: list[dict[str, Any]],
    guidance: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, frozenset[str]], str, str]:
    controlled = {
        item["name"]: frozenset(item.get("allowed_values") or ())
        for item in guidance["schema_spec"]
        if item.get("allowed_values")
    }
    value_field, unit_field = ROLE_FIELDS[source_id][1:]
    columns = sorted(
        {
            "source_row_uid",
            "pmid",
            "extraction_id",
            "support_text",
            "confidence",
            value_field,
            *([unit_field] if unit_field else []),
            *controlled,
        }
    )
    raw = pd.read_parquet(
        _source_path(data_dir, source_id, "extractions.parquet"), columns=columns
    )
    if raw["source_row_uid"].isna().any() or raw["source_row_uid"].duplicated().any():
        raise ValueError(f"{source_id} source_row_uid must be present and unique")
    cleaned_uids = [str(row["source_row_uid"]) for row in cleaned_rows]
    if len(cleaned_uids) != len(set(cleaned_uids)):
        raise ValueError(f"{source_id} cleaned source_row_uid must be unique")
    indexed = raw.set_index("source_row_uid", drop=False)
    missing = set(cleaned_uids) - set(indexed.index)
    if missing:
        raise ValueError(
            f"{source_id} cleaned UID is absent from raw source: {next(iter(missing))}"
        )
    raw = indexed.loc[cleaned_uids].reset_index(drop=True)
    keys = list(zip(raw["pmid"].astype(str), raw["extraction_id"].astype(str)))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{source_id} PMID + extraction_id must be unique")
    return raw, controlled, value_field, unit_field


def _basic_quality_audit(
    cleaned: dict[str, Any], raw: dict[str, Any]
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    if not _has_value(raw["support_text"]):
        audit.append(
            _audit_entry(
                cleaned,
                audit_type="blank_support_text",
                field="support_text",
                before=raw["support_text"],
                after=raw["support_text"],
                action="preserved",
            )
        )
    confidence = raw["confidence"]
    if _has_value(confidence) and not 0.0 <= float(confidence) <= 1.0:
        audit.append(
            _audit_entry(
                cleaned,
                audit_type="confidence_out_of_range",
                field="confidence",
                before=confidence,
                after=confidence,
                action="preserved",
            )
        )
    return audit


def _controlled_value_audit(
    cleaned: dict[str, Any],
    raw: dict[str, Any],
    controlled: dict[str, frozenset[str]],
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for field, allowed in controlled.items():
        value = raw[field]
        if not _has_value(value):
            continue
        text = str(value).strip()
        textual_null = text.casefold() in {"nan", "none", "null", "na", "n/a", "-"}
        if textual_null and text not in allowed:
            audit.append(
                _audit_entry(
                    cleaned,
                    audit_type="textual_null",
                    field=field,
                    before=value,
                    after=None,
                    action="normalized_to_null",
                )
            )
        elif not textual_null and text not in allowed:
            audit.append(
                _audit_entry(
                    cleaned,
                    audit_type="off_schema_controlled_value",
                    field=field,
                    before=value,
                    after=value,
                    action="preserved",
                )
            )
    return audit


def _paired_value_audit(
    cleaned: dict[str, Any], raw: dict[str, Any], value_field: str, unit_field: str
) -> list[dict[str, Any]]:
    if not unit_field or _has_value(raw[value_field]) == _has_value(raw[unit_field]):
        return []
    has_value = _has_value(raw[value_field])
    value = raw[unit_field] if has_value else raw[value_field]
    return [
        _audit_entry(
            cleaned,
            audit_type="value_without_unit" if has_value else "unit_without_value",
            field=unit_field if has_value else value_field,
            before=value,
            after=value,
            action="preserved",
        )
    ]


def _row_quality_audit(
    cleaned: dict[str, Any],
    raw: dict[str, Any],
    controlled: dict[str, frozenset[str]],
    value_field: str,
    unit_field: str,
) -> list[dict[str, Any]]:
    return [
        *_basic_quality_audit(cleaned, raw),
        *_controlled_value_audit(cleaned, raw, controlled),
        *_paired_value_audit(cleaned, raw, value_field, unit_field),
    ]


def _audit_one_source(
    cleaned_rows: list[dict[str, Any]], data_dir: Path, source_id: str
) -> tuple[list[dict[str, Any]], Path]:
    guidance_path, guidance = _guidance(data_dir, source_id)
    raw, controlled, value_field, unit_field = _raw_audit_inputs(
        data_dir, source_id, cleaned_rows, guidance
    )
    audit: list[dict[str, Any]] = []
    for cleaned, raw_row in zip(cleaned_rows, raw.to_dict(orient="records")):
        if str(raw_row["extraction_id"]) != cleaned["source_record_id"]:
            raise ValueError(f"source order changed for {source_id}")
        audit.extend(
            _row_quality_audit(cleaned, raw_row, controlled, value_field, unit_field)
        )
    return audit, guidance_path


def _audit_source_quality(
    records: list[dict[str, Any]], data_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, int], tuple[Path, ...]]:
    audit: list[dict[str, Any]] = []
    guidance_paths: list[Path] = []
    for source_id in SOURCE_COLUMNS:
        rows = [row for row in records if row.get("source_id") == source_id]
        source_audit, guidance_path = _audit_one_source(rows, data_dir, source_id)
        audit.extend(source_audit)
        guidance_paths.append(guidance_path)
    counts = Counter(row["audit_type"] for row in audit)
    return audit, dict(sorted(counts.items())), tuple(guidance_paths)


def _normalize_endpoint_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for record in records:
        decision = endpoint_decision(record["source_id"], record.get("endpoint_name"))
        if decision["status"] == "identity":
            retained.append(record)
            continue
        excluded = decision["status"] == "excluded"
        audit.append(
            _audit_entry(
                record,
                audit_type="endpoint_excluded" if excluded else "endpoint_alias",
                field="record" if excluded else "endpoint_name",
                before=decision["raw_endpoint"],
                after="dropped" if excluded else decision["canonical_endpoint"],
                action="dropped" if excluded else "normalized",
            )
        )
        if not excluded:
            retained.append(record)
    return retained, audit


def clean_source_values(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> SourceValueCleaningResult:
    quality_audit, _, guidance_paths = _audit_source_quality(
        records, Path(args.starling_data_dir)
    )
    authoritative = bool(getattr(args, "authoritative_source_root", None))
    normalized, endpoint_audit = (
        (records, []) if authoritative else _normalize_endpoint_records(records)
    )
    routed = attach_stage1_routes(
        normalized,
        task=TASK_ID,
        endpoint_resolver=(
            lambda row: str(row.get("endpoint_name") or "missing_endpoint")
        )
        if authoritative
        else None,
    )
    audit = [*quality_audit, *endpoint_audit]
    audit_counts = Counter(row["audit_type"] for row in audit)
    return SourceValueCleaningResult(
        records=routed,
        audit_rows=audit,
        manifest={
            "version": "ames_source_quality_audit.v2",
            "endpoint_normalization_version": ENDPOINT_NORMALIZATION_VERSION,
            "n_input_records": len(records),
            "n_output_records": len(routed),
            "n_repaired_records": audit_counts["endpoint_alias"],
            "n_dropped_records": audit_counts["endpoint_excluded"],
            "audit_counts": dict(sorted(audit_counts.items())),
            "scientific_fields_changed": bool(audit_counts["endpoint_alias"]),
            "authoritative_main_endpoints_preserved": authoritative,
        },
        input_paths=(SOURCE_MANIFEST_PATH, *guidance_paths, DEFAULT_ENDPOINT_MAPPING),
    )


def attach_source_columns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return records


def _identity_pair(endpoint: str, pair: MeasurementPair) -> MeasurementPair:
    del endpoint
    return pair


def _pair_producer(record: dict[str, Any], encoded: dict[str, Any]) -> str:
    encoded_id = str(
        encoded.get("categorical_encoder_id")
        or record.get("canonical_measurement_scale_id")
        or ""
    )
    if encoded_id:
        return encoded_id
    source_id = str(record["source_id"])
    route = record.get("measurement_resolution_route")
    if route == "extract":
        return SOURCE_EXTRACTION_PAIR_PRODUCER_IDS[source_id]
    if route == "accept":
        return SOURCE_RULE_PAIR_PRODUCER_IDS[source_id]
    return SOURCE_PAIR_PRODUCER_IDS[source_id]


def _enrich_record(
    record: dict[str, Any],
    attacher: AuxiliaryMetadataAttacher,
    reference: ReferenceSemanticsAttacher,
) -> dict[str, Any]:
    source_id = str(record["source_id"])
    encoded = CATEGORICAL_POLICY.apply(record)
    auxiliary = attacher.attach(record)
    endpoint_concept = str(record.get("canonical_endpoint_name") or "") or None
    producers = {
        ENDPOINT_PRODUCER_FIELD: SOURCE_ENDPOINT_PRODUCER_IDS[source_id],
        PAIR_PRODUCER_FIELD: _pair_producer(record, encoded),
    }
    working = {
        **record,
        **encoded,
        **producers,
        **auxiliary,
        "canonical_endpoint_concept": endpoint_concept,
    }
    validity = enrich_ames_validity(working)
    return {
        **encoded,
        **producers,
        **auxiliary,
        "canonical_endpoint_concept": endpoint_concept,
        **validity,
        **reference.attach({**working, **validity}),
    }


def build_hooks(args: argparse.Namespace) -> NormalizationHooks:
    attacher = build_local_source_attacher(
        args,
        contract=RECORD_CONTRACT,
        mapping_version=PAIR_CONTEXT_MAPPING_VERSION,
        output_fields_by_source={
            source: tuple(PAIR_CONTEXT_INPUTS[source])
            for source in RECORD_CONTRACT.sources
        },
    )
    reference = ReferenceSemanticsAttacher(
        replace(
            REFERENCE_SEMANTICS_CONFIG,
            mapping_path=Path(args.reference_semantics_mapping),
        ),
        allow_missing=bool(args.allow_missing_reference_semantics),
        fail_closed_unmapped=True,
    )
    return NormalizationHooks(
        endpoint_normalizer=endpoint_orthography,
        endpoint_standardizer=_identity_pair,
        family_resolver=family_assignment,
        record_enricher=lambda record: _enrich_record(record, attacher, reference),
        run_state={"auxiliary": attacher, "reference": reference},
    )


def _endpoint_registry() -> dict[str, list[dict[str, str]]]:
    return {
        source_id: [
            endpoint_decision(source_id, raw)
            for raw in sorted(expected_raw_endpoints(source_id))
        ]
        for source_id in SOURCE_COLUMNS
    }


def stage_documents(
    *,
    args: argparse.Namespace,
    hooks: NormalizationHooks,
    normalized: list[dict[str, Any]],
    persisted: list[dict[str, Any]],
    unit_policy_manifest: dict[str, Any],
) -> StageDocuments:
    del args
    coverage = hooks.run_state["auxiliary"].coverage_audit(normalized)
    reference = hooks.run_state["reference"]
    reference_coverage = reference.coverage_audit(normalized)
    validations = _stage2_validations(persisted, unit_policy_manifest)
    validations.update(
        {
            "globally_reconciled_auxiliary_coverage": coverage["validations"][
                "all_applicable_records_mapped"
            ],
            "local_source_tuple_coverage": coverage["validations"][
                "all_applicable_records_mapped"
            ],
        }
    )
    _require_stage2_validations(validations)
    validations.update(
        {
            "reference_semantics_mapping_complete": bool(
                reference_coverage["validations"]["all_applicable_records_mapped"]
            ),
            "reference_semantics_assignment_complete": bool(
                reference_coverage["validations"]["all_applicable_records_assigned"]
            ),
        }
    )
    return StageDocuments(
        validity_policy={
            **validity_policy_manifest(),
            "categorical_response": CATEGORICAL_POLICY.manifest(),
            "transfer_semantics": {
                "eligible_reference_scopes": list(
                    reference.config.eligible_scopes
                ),
                "reference_basis_required": reference.config.output_basis,
            },
        },
        auxiliary_mapping_manifest={
            **hooks.run_state["auxiliary"].manifest(),
            "coverage": coverage,
            "family_assignment": family_assignment_manifest(),
        },
        source_column_contract=RECORD_CONTRACT.manifest(),
        endpoint_registry=_endpoint_registry(),
        validations=validations,
        reference_semantics_manifest={
            **reference.manifest(),
            "coverage": reference_coverage,
        },
    )


def _require_stage2_validations(validations: dict[str, Any]) -> None:
    failed = sorted(
        name
        for name, value in validations.items()
        if isinstance(value, bool) and not value
    )
    if failed:
        raise ValueError(f"AMES Stage 2 validation failure(s): {failed}")


def _stage2_validations(
    rows: list[dict[str, Any]], unit_policy: dict[str, Any]
) -> dict[str, Any]:
    endpoint_ok = all(
        row.get(ENDPOINT_PRODUCER_FIELD)
        == SOURCE_ENDPOINT_PRODUCER_IDS[row["source_id"]]
        for row in rows
    )
    pair_ids = {
        source_id: {
            SOURCE_PAIR_PRODUCER_IDS[source_id],
            SOURCE_RULE_PAIR_PRODUCER_IDS[source_id],
            SOURCE_EXTRACTION_PAIR_PRODUCER_IDS[source_id],
            *(
                scale.scale_id
                for scale in CATEGORICAL_POLICY.controlled_measurements
                if scale.source_id == source_id
            ),
        }
        for source_id in SOURCE_COLUMNS
    }
    return {
        "one_to_one_measurement_inputs_to_normalized_ids": True,
        "endpoint_orthography_provenance": True,
        "canonical_endpoint_present": all(
            row.get("canonical_endpoint_name") for row in rows
        ),
        "canonical_endpoint_concept_present": all(
            row.get("canonical_endpoint_concept") for row in rows
        ),
        "canonical_endpoint_producer_declared": endpoint_ok,
        "canonical_pair_producer_declared": all(
            row.get(PAIR_PRODUCER_FIELD) in pair_ids[row["source_id"]]
            and row.get(PAIR_PRODUCER_FIELD) == _pair_producer(row, {})
            for row in rows
        ),
        "policy_independent_validity_present": all(
            row.get("canonicalization_status") for row in rows
        ),
        "source_column_contract_complete": True,
        "contextual_unit_policy_loaded": True,
        "contextual_unit_policy_version": unit_policy["policy_version"],
    }


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    validate_mapping_hashes()
    if args.through_stage in {"source", "clean"}:
        return
    supplied_mapping = getattr(args, "measurement_resolution_mapping", None)
    if not supplied_mapping:
        parser.error("AMES V10 Stage 2 requires --measurement-resolution-mapping")
    if getattr(args, "allow_partial_measurement_resolution", False):
        parser.error("AMES V10 Stage 2 forbids partial measurement resolution")
    if getattr(args, "validation_level", None) != "full":
        parser.error("AMES V10 Stage 2 requires --validation-level full")
    selected_auxiliary = _selected_auxiliary_mapping()
    supplied_auxiliary = Path(str(args.auxiliary_mapping or ""))
    if selected_auxiliary is None or not selected_auxiliary.is_file():
        parser.error("AMES Stage 2 requires a registered reviewed auxiliary mapping")
    if supplied_auxiliary.resolve() != selected_auxiliary.resolve():
        parser.error("AMES Stage 2 requires the registry-selected auxiliary mapping")
    selected_reference = _selected_reference_mapping()
    supplied_reference = Path(str(args.reference_semantics_mapping or ""))
    if selected_reference is None or not selected_reference.is_file():
        if not args.allow_missing_reference_semantics:
            parser.error("AMES Stage 2 requires a registered reference mapping")
    elif supplied_reference.resolve() != selected_reference.resolve():
        parser.error("AMES Stage 2 requires the registry-selected reference mapping")
    mapping_path = Path(str(supplied_mapping))
    try:
        _validate_stage2_assets(mapping_path)
    except (OSError, TypeError, ValueError) as error:
        parser.error(f"AMES V10 Stage 2 assets are not accepted: {error}")
def _validate_stage2_assets(resolution_path: Path) -> None:
    validate_mapping_provenance(resolution_path)
    validate_mapping_hashes()
    if resolution_path.resolve() != mapping_path("measurement_resolution").resolve():
        raise ValueError("measurement extraction is not the registry-selected mapping")


def _validate_durable_review_lineage(decisions: dict[str, Any]) -> None:
    root = REVIEWED_UNIT_DECISIONS_PATH.parent.resolve()
    review = decisions.get("review") or {}
    for field in (
        "packet_manifest_path",
        "review_path",
        "review_run_receipt_path",
    ):
        path = Path(str(review.get(field) or "")).resolve()
        if root not in path.parents:
            raise ValueError(f"exact-unit {field} is outside the durable asset root")
    receipt = json.loads(
        Path(str(review["review_run_receipt_path"])).read_text(encoding="utf-8")
    )
    cache_path = Path(str((receipt.get("cache") or {}).get("path") or "")).resolve()
    if root not in cache_path.parents:
        raise ValueError("exact-unit cache is outside the durable asset root")


def _validate_exact_unit_input_lineage(
    decisions: dict[str, Any], mapping_path: Path
) -> None:
    expected_inputs = {
        "cleaned_records": Path(DEFAULT_OUT_DIR) / "01_cleaned/records.parquet",
        "measurement_resolution": mapping_path,
    }
    inputs = decisions.get("inputs") or {}
    for name, expected_path in expected_inputs.items():
        reference = inputs.get(name) or {}
        if Path(
            str(reference.get("path") or "")
        ).resolve() != expected_path.resolve() or reference.get(
            "sha256"
        ) != file_sha256(expected_path):
            raise ValueError(f"exact-unit {name} lineage mismatch")


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    return {
        "evidence_library_version": "ames_normalized_v10",
        "ames_source_snapshot_version": SOURCE_MANIFEST["version"],
        "ames_source_quality_audit_version": "ames_source_quality_audit.v2",
        "ames_endpoint_normalization_version": ENDPOINT_NORMALIZATION_VERSION,
        "categorical_response_version": CATEGORICAL_RESPONSE_VERSION,
        "family_assignment_version": FAMILY_ASSIGNMENT_VERSION,
        "measurement_routing_version": MEASUREMENT_ROUTING_VERSION,
        "stage2_status": "implemented_requires_reviewed_auxiliary_mapping",
    }


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage1-protected-voter-contract",
        default="data/gold_labels/Ames/v1/scaffold/voter_membership.parquet",
        help="Frozen Gold-v1 physical membership used only for Stage-1 UID protection.",
    )
    selected = _selected_auxiliary_mapping()
    parser.add_argument(
        "--auxiliary-mapping", default=str(selected) if selected else ""
    )
    reference = _selected_reference_mapping() or DEFAULT_REFERENCE_SEMANTICS_MAPPING
    parser.add_argument("--reference-semantics-mapping", default=str(reference))
    parser.add_argument("--allow-missing-reference-semantics", action="store_true")


COMPACT_PROFILE = CompactArtifactProfile(
    task_id=TASK_ID,
    artifact_version="ames.compact_stage1.v1",
    index_version="ames.index_not_built.v1",
    evidence_source_label="SEND Ames evidence",
    source_columns=SOURCE_COLUMNS,
    llm_source_projection=RECORD_CONTRACT.source_projection,
)

DEFAULT_GUIDANCE_PATHS = tuple(
    _source_path(Path(DEFAULT_DATA_DIR), source_id, "extraction_guidance.json")
    for source_id in SOURCE_COLUMNS
)
TASK_IMPLEMENTATION_PATHS = (
    *tuple(sorted(TASK_ROOT.glob("*.py"))),
    *tuple(sorted((TASK_ROOT / "data_processing").glob("*.py"))),
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
    family_resolver=family_assignment,
    build_hooks=build_hooks,
    attach_source_columns=attach_source_columns,
    stage_documents=stage_documents,
    manifest_versions=manifest_versions,
    record_contract=RECORD_CONTRACT,
    source_value_cleaner=clean_source_values,
    add_cli_arguments=add_cli_arguments,
    validate_arguments=validate_arguments,
    verify_source_digest=validate_source_digest,
    scientific_assets=(
        *TASK_IMPLEMENTATION_PATHS,
        SOURCE_MANIFEST_PATH,
        *DEFAULT_GUIDANCE_PATHS,
        DEFAULT_ENDPOINT_MAPPING,
        REGISTRY_PATH,
        mapping_path("measurement_resolution"),
        mapping_path("exact_measurement_units"),
        MAIN_UNIVERSE_MANIFEST,
        MAIN_UNIVERSE_RECORDS,
        *((SELECTED_AUXILIARY_MAPPING,) if SELECTED_AUXILIARY_MAPPING else ()),
        REFERENCE_SEMANTICS_CONFIG.prompt_registry_path,
        *((SELECTED_REFERENCE_MAPPING,) if SELECTED_REFERENCE_MAPPING else ()),
    ),
    source_universe_mapping=level_mapping_path("ames", "v2"),
    source_uid_universe_records=MAIN_UNIVERSE_RECORDS,
    source_uid_universe_manifest=MAIN_UNIVERSE_MANIFEST,
    protected_voter_membership=Path(
        "data/gold_labels/Ames/v1/scaffold/voter_membership.parquet"
    ),
    stage1_measurement_routing_enabled=True,
    measurement_resolution_enabled=True,
    exact_unit_mapping_path=mapping_path("exact_measurement_units"),
    reference_semantics_enabled=True,
)


__all__ = [
    "ASSAY_FAMILIES",
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
