"""Bioavailability_Ma plug-in for the shared normalized-Starling builder.

This module only wires this task's existing policy modules into the
:class:`StarlingTaskPolicy` contract.  It contains no normalization logic of
its own; every rule still lives in the module that owns it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_measurement_and_unit,
    parse_point_measurement,
)
from tools.chembl_tool.common.starling.normalization.task_policy import (
    ExtraSourceBatch,
    NormalizationHooks,
    SmilesMappingSpec,
    StageDocuments,
    StarlingTaskPolicy,
)
from tools.chembl_tool.common.starling.normalization.source_value_cleaning import (
    clean_source_values,
)
from tools.chembl_tool.common.starling.reference_semantics import (
    ReferenceSemanticsAttacher,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    FG_TARGET_ALIAS_VERSION,
    MEASUREMENT_SCALES,
    POLICY as CATEGORICAL_RESPONSE_POLICY,
    canonical_fg_target_id,
    encoding_policy_manifest,
)
from tools.chembl_tool.tasks.bioavailability_ma.canonical_source import (
    DIRECT_MEASUREMENT_EXTRACTION_VERSION,
    DIRECT_REPORT_TYPES,
    NONDIRECT_MEASUREMENT_EXTRACTION_VERSION,
    direct_measurement_fields,
    nondirect_measurement_fields,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_schema import (
    RECORD_CONTRACT,
    SOURCE_ENDPOINT_PRODUCER_IDS,
    SOURCE_PAIR_PRODUCER_IDS,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    DEFAULT_MAPPING_PATH,
    AuxiliaryMetadataAttacher,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_compact_artifacts import (
    COMPACT_PROFILE,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_contextual_unit_reconciliation import (
    contextual_canonical_record_fields,
    contextual_standardization_of_unit,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_fg_scalar_rules import (
    FG_SCALAR_RULE_VERSION,
    fg_scalar_rule_provenance,
    resolve_fg_measurement_pair,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_normalization_policy import (
    ENDPOINT_POLICY_VERSION,
    endpoint_specific_standardization_of_unit,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_normalization_sources import (
    DEFAULT_HF_BIOAVAILABILITY_PARQUET,
    EXPECTED_RAW_HF_BIOAVAILABILITY_ROWS,
    EXPECTED_SOURCE_ROWS,
    hf_bioavailability_profile,
    load_hf_bioavailability_rows,
    source_profiles,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_record_canonicalization import (
    DIRECT_EVIDENCE_SCOPE,
    EVIDENCE_SCOPE_VERSION,
    NORMALIZATION_DOMAIN_RULES_VERSION,
    NONDIRECT_EVIDENCE_SCOPE,
    REPORT_TYPE_NORMALIZATION_VERSION,
    bioavailability_evidence_scope,
    enrich_bioavailability_validity,
    validity_policy_manifest,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_reference_semantics import (
    DEFAULT_MAPPING_PATH as DEFAULT_REFERENCE_SEMANTICS_MAPPING,
    REFERENCE_SEMANTICS_CONFIG,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_source_column_contracts import (
    SOURCE_COLUMN_CONTRACT_VERSION,
    llm_source_projection,
    source_column_contract_manifest,
    source_fields_from_record,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    SPACING_AND_SPELLING_VERSION,
    family_assignment,
    spacing_and_spelling_decision,
    validate_endpoint_inventory,
)


TASK_ID = "bioavailability_ma"
DATASET_NAME = "starling-labs/Bioavailability_Ma"
DEFAULT_STARLING_DATA_DIR = "data/starling_data/bioavailability_ma"
DEFAULT_SMILES_MAPPING = "data/starling_data/_shared/final_smiles_mapping_v2.parquet"
EXPECTED_SMILES_MAPPING_SHA256 = (
    "98ae43b6d9c61b77f0a95e4dce681010694b163a02e7a313667592d985496a0b"
)
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v7"
)
DEFAULT_V65_ELIGIBLE_RECORDS = (
    "/data1/joseph/starling_assay_transfer/datasets/eligible/"
    "assay_transfer_soft_evidence_v6_5/records.parquet"
)
DEFAULT_BENCHMARK_SPLIT_ROOT = "data/processed_starling/Bioavailability_Ma"
HF_SOURCE_CLASSIFICATION_VERSION = "bioavailability_hf_evidence_scope.v1"
HF_DIRECT_EXPLICIT_UNIT_VERSION = "hf_direct_explicit_unit.v3"
DEFAULT_SOURCE_VALUE_REPAIRS = (
    Path(__file__).resolve().parent
    / "data_processing/source_value_cleaning_v1/reviewed_repairs.jsonl"
)


def _clean_source_values(records: list[dict[str, Any]], args: argparse.Namespace):
    require_all = (
        _include_hf_bioavailability(args)
        and not _max_hf_rows(args)
        and not int(getattr(args, "max_rows_per_source", 0) or 0)
    )
    return clean_source_values(
        records,
        task_id=TASK_ID,
        reviewed_repairs_path=DEFAULT_SOURCE_VALUE_REPAIRS,
        require_all_reviewed_repairs=require_all,
    )


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smiles-mapping", default=DEFAULT_SMILES_MAPPING)
    parser.add_argument("--auxiliary-mapping", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--allow-unpinned-smiles-mapping", action="store_true")
    parser.add_argument(
        "--hf-source-parquet",
        "--direct-source-parquet",
        dest="hf_source_parquet",
        default=str(DEFAULT_HF_BIOAVAILABILITY_PARQUET),
    )
    parser.add_argument(
        "--include-hf-bioavailability",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-direct-hf",
        dest="include_hf_bioavailability",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-include-direct-hf",
        dest="include_hf_bioavailability",
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-hf-rows", "--max-direct-rows", dest="max_hf_rows", type=int, default=0
    )
    parser.add_argument(
        "--v65-reconciliation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--v65-eligible-records", default=DEFAULT_V65_ELIGIBLE_RECORDS)
    parser.add_argument(
        "--benchmark-split-root",
        default=DEFAULT_BENCHMARK_SPLIT_ROOT,
        help=(
            "Directory containing random/scaffold train and heldout molecule "
            "label mappings used by the post-record filtering stage."
        ),
    )
    parser.add_argument(
        "--reference-semantics-mapping",
        default=str(DEFAULT_REFERENCE_SEMANTICS_MAPPING),
    )
    parser.add_argument(
        "--allow-missing-reference-semantics",
        action="store_true",
        help="Allow an explicitly incomplete pre-generation smoke build.",
    )


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.v65_reconciliation:
        parser.error(
            "v6.5 reconciliation is outside the source-aware pair-bucket sidecar contract"
        )
    if _max_hf_rows(args) and args.strict_endpoint_inventory:
        parser.error("bounded source runs require --no-strict-endpoint-inventory")
    reference_mapping = Path(args.reference_semantics_mapping)
    if (
        args.through_stage != "clean"
        and not reference_mapping.exists()
        and not args.allow_missing_reference_semantics
    ):
        parser.error(
            f"reference-semantics mapping not found: {reference_mapping}; build it "
            "with common.starling.build_reference_semantics_mapping"
        )


def _include_hf_bioavailability(args: argparse.Namespace) -> bool:
    return bool(
        getattr(
            args,
            "include_hf_bioavailability",
            getattr(args, "include_direct_hf", True),
        )
    )


def _max_hf_rows(args: argparse.Namespace) -> int:
    return int(
        getattr(args, "max_hf_rows", getattr(args, "max_direct_rows", 0)) or 0
    )


def _hf_source_parquet(args: argparse.Namespace) -> Path:
    return Path(
        getattr(
            args,
            "hf_source_parquet",
            getattr(args, "direct_source_parquet", DEFAULT_HF_BIOAVAILABILITY_PARQUET),
        )
    )


def smiles_mapping(args: argparse.Namespace) -> SmilesMappingSpec:
    return SmilesMappingSpec(
        path=Path(args.smiles_mapping),
        expected_sha256=EXPECTED_SMILES_MAPPING_SHA256,
        allow_unpinned=args.allow_unpinned_smiles_mapping,
    )


def endpoint_inventory(
    source_id: str, endpoints: list[str], *, strict: bool
) -> dict[str, Any]:
    if strict:
        return validate_endpoint_inventory(source_id, endpoints)
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


def load_extra_source(
    args: argparse.Namespace,
) -> ExtraSourceBatch | None:
    """Load the complete HF snapshot as one physical evidence source."""
    if not _include_hf_bioavailability(args):
        return None
    records_path = _hf_source_parquet(args)
    source_hash = file_sha256(records_path)
    rows = load_hf_bioavailability_rows(
        records_path, max_rows=_max_hf_rows(args)
    )
    scope_counts = {DIRECT_EVIDENCE_SCOPE: 0, NONDIRECT_EVIDENCE_SCOPE: 0}
    for row in rows:
        scope_counts[
            bioavailability_evidence_scope(
                row.get("bioavailability_report_type")
            )
        ] += 1
    if not _max_hf_rows(args) and len(rows) != EXPECTED_RAW_HF_BIOAVAILABILITY_ROWS:
        raise ValueError(
            "raw HF evidence source drift: "
            f"rows={len(rows):,}/{EXPECTED_RAW_HF_BIOAVAILABILITY_ROWS:,}"
        )
    if not _max_hf_rows(args) and scope_counts != {
        DIRECT_EVIDENCE_SCOPE: 112_245,
        NONDIRECT_EVIDENCE_SCOPE: 51_570,
    }:
        raise ValueError(f"raw HF evidence-scope drift: {scope_counts}")
    classification = {
        "version": HF_SOURCE_CLASSIFICATION_VERSION,
        "raw_source_rows": len(rows),
        "scope_counts": dict(sorted(scope_counts.items())),
        "reconciles": sum(scope_counts.values()) == len(rows),
        "direct_report_types": sorted(DIRECT_REPORT_TYPES),
    }
    return ExtraSourceBatch(
        source_id="hf_bioavailability",
        profile=hf_bioavailability_profile(records_path),
        rows=rows,
        source_path=records_path,
        source_sha256=source_hash,
        endpoint_names=["oral_bioavailability"] if rows else [],
        inventory_key="hf_bioavailability",
        inventory_entry={
            "records_parquet": str(records_path),
            "records_sha256": source_hash,
            "source_rows": len(rows),
            "historical_partition_dependency": False,
            "row_classification": classification,
        },
    )


def _resolve_source_measurement_pair(
    record: dict[str, Any],
    canonical_endpoint: str,
    baseline_pair: Any,
):
    source_id = str(record.get("source_id") or "")
    if source_id == "hf_bioavailability":
        measurement_text = str(record.get("measurement_text") or "").strip()
        scope = bioavailability_evidence_scope(
            record.get("bioavailability_report_type")
        )
        if scope == NONDIRECT_EVIDENCE_SCOPE:
            extracted = nondirect_measurement_fields(measurement_text)
            pair = normalize_measurement_and_unit(
                extracted["measurement_text"],
                extracted["value_units"],
                task=TASK_ID,
            )
            return endpoint_specific_standardization_of_unit(
                canonical_endpoint, pair
            )
        extracted = direct_measurement_fields(measurement_text)
        pair = normalize_measurement_and_unit(
            extracted["measurement_text"],
            extracted["value_units"],
            task=TASK_ID,
        )
        return endpoint_specific_standardization_of_unit(
            canonical_endpoint, pair
        )
    return resolve_fg_measurement_pair(record, canonical_endpoint, baseline_pair)


def _is_hf_direct_scope_unitless_numeric_abstention(record: dict[str, Any]) -> bool:
    if (
        str(record.get("source_id") or "") != "hf_bioavailability"
        or bioavailability_evidence_scope(
            record.get("bioavailability_report_type")
        )
        != DIRECT_EVIDENCE_SCOPE
        or record.get("direct_measurement_unit_extraction_status")
        != "no_explicit_unit"
    ):
        return False
    source_point = parse_point_measurement(record.get("measurement_text"))
    return source_point.value is not None


def build_hooks(args: argparse.Namespace) -> NormalizationHooks:
    attacher = AuxiliaryMetadataAttacher(args.auxiliary_mapping)
    reference_attacher = ReferenceSemanticsAttacher(
        replace(
            REFERENCE_SEMANTICS_CONFIG,
            mapping_path=Path(args.reference_semantics_mapping),
        ),
        allow_missing=args.allow_missing_reference_semantics,
    )
    return NormalizationHooks(
        endpoint_normalizer=spacing_and_spelling_decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        source_measurement_resolver=_resolve_source_measurement_pair,
        family_resolver=family_assignment,
        contextual_standardizer=contextual_standardization_of_unit,
        record_enricher=lambda record: _enrich_record(
            record, attacher, reference_attacher
        ),
        # stage_documents needs the same instance to write its coverage audit.
        run_state={"auxiliary": attacher, "reference": reference_attacher},
    )


def attach_source_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Columnize the raw source contract before removing duplicate JSON."""
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        for field, value in source_fields_from_record(row).items():
            enriched.setdefault(field, value)
        output.append(enriched)
    return output


def stage_documents(
    *,
    args: argparse.Namespace,
    hooks: NormalizationHooks,
    normalized: list[dict[str, Any]],
    persisted: list[dict[str, Any]],
    unit_policy_manifest: dict[str, Any],
) -> StageDocuments:
    del args
    attacher: AuxiliaryMetadataAttacher = hooks.run_state["auxiliary"]
    reference_attacher: ReferenceSemanticsAttacher = hooks.run_state["reference"]
    reference_coverage = reference_attacher.coverage_audit(normalized)
    return StageDocuments(
        validity_policy={
            **validity_policy_manifest(),
            "categorical_response": encoding_policy_manifest(),
        },
        auxiliary_mapping_manifest={
            **attacher.manifest(),
            "coverage": attacher.coverage_audit(normalized),
        },
        source_column_contract=source_column_contract_manifest(
            sorted({column for row in persisted for column in row})
        ),
        validations={
            "one_to_one_cleaned_to_normalized_ids": True,
            "measurement_unit_pair_errors": 0,
            "endpoint_orthography_provenance": True,
            "canonical_endpoint_present": True,
            "policy_independent_validity_present": True,
            "globally_reconciled_auxiliary_coverage": True,
            "contextual_unit_policy_loaded": True,
            "contextual_unit_policy_version": unit_policy_manifest["policy_version"],
            "categorical_response_loaded": True,
            "categorical_encoders_declared": all(
                str(row.get("categorical_encoder_id") or "")
                in MEASUREMENT_SCALES
                for row in normalized
                if row.get("categorical_encoder_id")
            ),
            "heuristic_auxiliary_fields_absent": all(
                not any(
                    field in row
                    for field in (
                        "canonical_dose_key",
                        "canonical_assay_system",
                        "canonical_species",
                    )
                )
                for row in normalized
            ),
            "source_column_contract_complete": True,
            "llm_source_projection_fail_closed": True,
            "reference_semantics_mapping_complete": bool(
                reference_coverage["validations"]["all_applicable_records_mapped"]
            ),
        },
        reference_semantics_manifest={
            **reference_attacher.manifest(),
            "coverage": reference_coverage,
        },
    )


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "fg_scalar_rule_version": FG_SCALAR_RULE_VERSION,
        "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
        "spacing_and_spelling_version": SPACING_AND_SPELLING_VERSION,
        "endpoint_policy_version": ENDPOINT_POLICY_VERSION,
        "report_type_normalization_version": REPORT_TYPE_NORMALIZATION_VERSION,
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "categorical_response_version": CATEGORICAL_RESPONSE_VERSION,
        "fg_target_alias_version": FG_TARGET_ALIAS_VERSION,
        "hf_source_classification_version": HF_SOURCE_CLASSIFICATION_VERSION,
        "bioavailability_evidence_scope_version": EVIDENCE_SCOPE_VERSION,
        "nondirect_measurement_extraction_version": (
            NONDIRECT_MEASUREMENT_EXTRACTION_VERSION
        ),
        "direct_measurement_extraction_version": (
            DIRECT_MEASUREMENT_EXTRACTION_VERSION
        ),
        "hf_direct_explicit_unit_version": HF_DIRECT_EXPLICIT_UNIT_VERSION,
        "v65_reconciliation": {"status": "not_performed", "matching_performed": False},
    }
    if complete:
        versions["source_column_contract_version"] = SOURCE_COLUMN_CONTRACT_VERSION
        versions["v65_reconciliation_version"] = None
    return versions


def census_extras(records: list[dict[str, Any]]) -> dict[str, Any]:
    encoded: dict[str, int] = {}
    finite_by_source = {source: 0 for source in EXPECTED_SOURCE_ROWS}
    for record in records:
        if record.get("finite_scalar_value") is None:
            continue
        source_id = str(record.get("source_id") or "")
        finite_by_source[source_id] = finite_by_source.get(source_id, 0) + 1
        encoder_id = str(record.get("categorical_encoder_id") or "")
        if encoder_id:
            encoded[encoder_id] = encoded.get(encoder_id, 0) + 1
    return {
        "n_fg_finite_scalars": sum(
            record.get("source_id") == "fg"
            and record.get("finite_scalar_value") is not None
            and not record.get("categorical_encoder_id")
            for record in records
        ),
        "n_finite_scalars_by_source": dict(sorted(finite_by_source.items())),
        "n_categorically_encoded_by_encoder": dict(sorted(encoded.items())),
        "n_hf_direct_unitless_numeric_abstentions": sum(
            _is_hf_direct_scope_unitless_numeric_abstention(record)
            for record in records
        ),
    }


def _enrich_record(
    record: dict[str, Any],
    auxiliary_attacher: AuxiliaryMetadataAttacher,
    reference_attacher: ReferenceSemanticsAttacher,
) -> dict[str, Any]:
    provenance = fg_scalar_rule_provenance(record)
    measurement_extraction: dict[str, Any] = {}
    if str(record.get("source_id") or "") == "hf_bioavailability":
        scope = bioavailability_evidence_scope(
            record.get("bioavailability_report_type")
        )
        if scope == NONDIRECT_EVIDENCE_SCOPE:
            extracted = nondirect_measurement_fields(record.get("measurement_text"))
            measurement_extraction = {
                "nondirect_measurement_extraction_version": (
                    NONDIRECT_MEASUREMENT_EXTRACTION_VERSION
                ),
                "nondirect_measurement_unit_extraction_status": extracted[
                    "measurement_unit_extraction_status"
                ],
            }
        else:
            extracted = direct_measurement_fields(record.get("measurement_text"))
            measurement_extraction = {
                "direct_measurement_extraction_version": (
                    DIRECT_MEASUREMENT_EXTRACTION_VERSION
                ),
                "direct_measurement_unit_extraction_status": extracted[
                    "measurement_unit_extraction_status"
                ],
            }
    enriched = {**record, **provenance, **measurement_extraction}
    source_projection = llm_source_projection(enriched)
    auxiliary = auxiliary_attacher.attach(enriched)
    with_auxiliary = {**enriched, **auxiliary}
    contextual_fields = contextual_canonical_record_fields(with_auxiliary)
    canonical = {**with_auxiliary, **contextual_fields}
    # Preserve a parseable source number even when its unit is unresolved.  A
    # categorical anchor may fill only a genuinely nonnumeric outcome.
    parsed_source = parse_point_measurement(canonical.get("canonical_measurement"))
    encoded = (
        {}
        if parsed_source.value is not None
        else CATEGORICAL_RESPONSE_POLICY.apply(canonical)
    )
    if encoded:
        encoder_id = str(encoded["categorical_encoder_id"])
        if encoder_id == "direct_oral_bioavailability_binary.v1":
            semantic_endpoint = "oral_bioavailability_outcome"
        elif encoder_id == "fg_substrate_status_binary.v1":
            target_id = canonical_fg_target_id(canonical.get("transporter_or_enzyme"))
            if target_id is None:
                raise ValueError("encoded Fg substrate status lacks a canonical target")
            semantic_endpoint = f"fg_substrate_outcome:{target_id}"
        else:
            raise ValueError(f"unknown Bioavailability categorical encoder: {encoder_id}")
        encoded = {**encoded, "canonical_endpoint": semantic_endpoint}
    source_id = str(record.get("source_id") or "")
    producer_id = str(encoded.get("categorical_encoder_id") or "")
    producer_fields = {
        "canonical_endpoint_producer_id": (
            producer_id or SOURCE_ENDPOINT_PRODUCER_IDS[source_id]
        ),
        "canonical_pair_producer_id": (
            producer_id or SOURCE_PAIR_PRODUCER_IDS[source_id]
        ),
    }
    validity = enrich_bioavailability_validity(
        {**canonical, **encoded, **producer_fields}
    )
    reference = reference_attacher.attach(
        {**canonical, **encoded, **producer_fields, **validity}
    )
    return {
        **provenance,
        **measurement_extraction,
        "source_column_contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
        "llm_source_contract_json": json.dumps(
            {
                key: value
                for key, value in source_projection.items()
                if key != "source_fields"
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "llm_source_fields_json": json.dumps(
            source_projection["source_fields"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        **auxiliary,
        **contextual_fields,
        **encoded,
        **producer_fields,
        **validity,
        **reference,
    }


POLICY = StarlingTaskPolicy(
    task_id=TASK_ID,
    dataset_name=DATASET_NAME,
    default_data_dir=DEFAULT_STARLING_DATA_DIR,
    default_out_dir=DEFAULT_OUT_DIR,
    compact=COMPACT_PROFILE,
    expected_source_rows=EXPECTED_SOURCE_ROWS,
    record_contract=RECORD_CONTRACT,
    source_value_cleaner=_clean_source_values,
    source_profiles=source_profiles,
    endpoint_inventory=endpoint_inventory,
    family_resolver=family_assignment,
    build_hooks=build_hooks,
    attach_source_columns=attach_source_columns,
    stage_documents=stage_documents,
    manifest_versions=manifest_versions,
    add_cli_arguments=add_cli_arguments,
    validate_arguments=validate_arguments,
    load_extra_source=load_extra_source,
    census_extras=census_extras,
    smiles_mapping=smiles_mapping,
    scientific_assets=(
        DEFAULT_SOURCE_VALUE_REPAIRS,
        REFERENCE_SEMANTICS_CONFIG.prompt_registry_path,
    ),
    reference_semantics_enabled=True,
    family_resolver_input_fields=(
        "canonical_bioavailability_evidence_scope",
        "bioavailability_report_type",
    ),
)


__all__ = ["POLICY", "TASK_ID"]
