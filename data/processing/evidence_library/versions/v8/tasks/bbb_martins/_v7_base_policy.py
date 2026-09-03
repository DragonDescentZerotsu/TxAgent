"""BBB Martins plug-in for the shared normalized-Starling v6 builder."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from data.processing.paths import evidence_library_root, raw_starling_task_root
from data.processing.evidence_library.shared.v1.normalization.task_policy import (
    NormalizationHooks,
    StageDocuments,
    StarlingTaskPolicy,
)
from data.processing.evidence_library.shared.v1.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
    RESOLUTION_APPLY_VERSION,
)
from data.processing.evidence_library.shared.v1.normalization.source_value_cleaning import (
    clean_source_values,
)
from data.processing.evidence_library.versions.v8.measurement_routing import (
    attach_stage1_routes,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    DEFAULT_MAPPING_PATH,
    AuxiliaryMetadataAttacher,
    PendingAuxiliaryAttacher,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    POLICY as CATEGORICAL_RESPONSE_POLICY,
    encoding_policy_manifest,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_compact_artifacts import (
    COMPACT_PROFILE,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_endpoint_normalization import (
    DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING,
    ENDPOINT_NORMALIZATION_VERSION,
    EndpointNormalizer,
    alternate_endpoint_fields,
    context_fields as endpoint_context_fields,
    validate_missing_endpoint_mapping,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_normalization_policy import (
    ENDPOINT_POLICY_VERSION,
    SOURCE_MEASUREMENT_RESOLVER_VERSION,
    endpoint_specific_standardization_of_unit,
    policy_manifest as normalization_policy_manifest,
    source_measurement_resolver,
    unit_provenance_fields,
)
from data.processing.evidence_library.shared.v1.normalization.measurements import (
    canonicalize_endpoint,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_normalization_sources import (
    EXPECTED_SOURCE_ROWS,
    source_profiles,
    validate_source_digest,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_record_canonicalization import (
    NORMALIZATION_DOMAIN_RULES_VERSION,
    enrich_bbb_validity,
    validity_policy_manifest,
)
from data.processing.evidence_library.shared.v1.reference_semantics import (
    ReferenceSemanticsAttacher,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_reference_semantics import (
    DEFAULT_MAPPING_PATH as DEFAULT_REFERENCE_SEMANTICS_MAPPING,
    REFERENCE_SEMANTICS_CONFIG,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_measurement_semantics import (
    DEFAULT_SEMANTICS_PATH,
    load_measurement_semantics_policy,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_source_column_contracts import (
    SOURCE_COLUMN_CONTRACT_VERSION,
    SOURCE_COLUMN_POLICY_VERSION,
    llm_source_projection,
    source_column_contract_manifest,
    source_fields_from_record,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_schema import (
    RECORD_CONTRACT,
    SOURCE_ENDPOINT_PRODUCER_IDS,
    SOURCE_EXTRACTION_PAIR_PRODUCER_IDS,
    SOURCE_PAIR_PRODUCER_IDS,
    SOURCE_RULE_PAIR_PRODUCER_IDS,
)
from data.processing.evidence_library.versions.v8.tasks.bbb_martins.starling_spacing_and_spelling import (
    ENDPOINT_CONCEPT_PATHS,
    ENDPOINT_CONCEPT_VERSION,
    SPACING_AND_SPELLING_VERSION,
    endpoint_concept,
    family_assignment,
    spacing_and_spelling_decision,
    validate_endpoint_inventory,
)


TASK_ID = "bbb_martins"
DATASET_NAME = "starling-labs/BBB plus BBB mechanism-family acquisitions"
DEFAULT_STARLING_DATA_DIR = str(raw_starling_task_root(TASK_ID))
DEFAULT_OUT_DIR = str(evidence_library_root(TASK_ID, "v7"))
DEFAULT_SOURCE_VALUE_REPAIRS = (
    Path(__file__).parent
    / "data_processing/source_value_cleaning_v1/reviewed_repairs.jsonl"
)
DEFAULT_REVIEWED_SOURCE_DROPS = (
    Path(__file__).parent
    / "data_processing/source_value_cleaning_v1/reviewed_drops.jsonl"
)
DEFAULT_SMILES_IDENTITY_AUDIT = (
    Path(__file__).parent
    / "data_processing/source_value_cleaning_v1/smiles_identity_audit.jsonl"
)
DEFAULT_SMILES_SAMPLE_AUDIT = (
    Path(__file__).parent
    / "data_processing/source_value_cleaning_v1/smiles_identity_sample_audit_20260829.json"
)
DEFAULT_REVIEWED_NAME_SMILES_CONFLICTS = (
    Path(__file__).parent
    / "data_processing/source_value_cleaning_v1/reviewed_name_smiles_conflicts.v2.parquet"
)
_KINETIC_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9])(?P<symbol>K_?IN|kout|k1k2|k13|k10|k01|"
    r"k[123](?:\*)?|ki|k\*)"
    r"(?![A-Za-z0-9])[^;\n]{0,80}?(?:=|\bis\b)\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    flags=re.IGNORECASE,
)
_KINETIC_SYMBOL = re.compile(
    r"(?<![A-Za-z0-9])(?:K_?IN|kout|k1k2|k13|k10|k01|"
    r"k[123](?:\*)?|ki|k\*)(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


def _clean_source_values(records: list[dict[str, Any]], args: argparse.Namespace):
    result = clean_source_values(
        records,
        task_id=TASK_ID,
        reviewed_repairs_path=DEFAULT_SOURCE_VALUE_REPAIRS,
        reviewed_drops_path=DEFAULT_REVIEWED_SOURCE_DROPS,
        smiles_identity_audit_path=DEFAULT_SMILES_IDENTITY_AUDIT,
        reviewed_smiles_conflicts_path=DEFAULT_REVIEWED_NAME_SMILES_CONFLICTS,
        require_all_reviewed_smiles_overrides=not int(
            getattr(args, "max_rows_per_source", 0) or 0
        ),
        require_scientific_scale_review=True,
    )
    endpoint_mapping = Path(args.direct_endpoint_mapping)
    endpoint_normalizer = EndpointNormalizer(endpoint_mapping)

    def resolve_endpoint(record: dict[str, Any]) -> str:
        decision = endpoint_normalizer.decision(
            str(record.get("source_id") or ""),
            str(record.get("endpoint_name") or ""),
        )
        return canonicalize_endpoint(decision.spacing_and_spelling_endpoint)

    return replace(
        result,
        records=attach_stage1_routes(
            result.records,
            task=TASK_ID,
            endpoint_resolver=resolve_endpoint,
        ),
        input_paths=(
            *result.input_paths,
            *((endpoint_mapping,) if endpoint_mapping.is_file() else ()),
        ),
    )


DEFAULT_BENCHMARK_SPLIT_ROOT = "data/gold_labels/legacy/processed_starling/BBB_Martins"


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--auxiliary-mapping", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument(
        "--allow-missing-auxiliary-mapping",
        action="store_true",
        help="Allow an explicitly incomplete cleaning/canonicalization smoke build.",
    )
    parser.add_argument(
        "--direct-endpoint-mapping",
        default=str(DEFAULT_APPROVED_DIRECT_ENDPOINT_MAPPING),
    )
    parser.add_argument(
        "--allow-missing-direct-endpoint-mapping",
        action="store_true",
        help="Allow an explicitly incomplete pre-approval smoke build.",
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


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    needs_normalization = args.through_stage not in {"source", "clean"}
    if (
        needs_normalization
        and not Path(args.auxiliary_mapping).exists()
        and not args.allow_missing_auxiliary_mapping
    ):
        parser.error(
            f"reconciled auxiliary mapping not found: {args.auxiliary_mapping}; "
            "build it with bbb_martins.data_processing.build_embedding_bucket_mapping"
        )
    direct_mapping = Path(args.direct_endpoint_mapping)
    if (
        args.through_stage != "source"
        and not direct_mapping.exists()
        and not args.allow_missing_direct_endpoint_mapping
    ):
        parser.error(
            f"human-approved Direct endpoint mapping not found: {direct_mapping}; "
            "build and review the proposal with "
            "bbb_martins.data_processing.build_direct_endpoint_mapping"
        )
    reference_mapping = Path(args.reference_semantics_mapping)
    if (
        needs_normalization
        and not reference_mapping.exists()
        and not args.allow_missing_reference_semantics
    ):
        parser.error(
            f"reference-semantics mapping not found: {reference_mapping}; build it "
            "with common.starling.build_reference_semantics_mapping"
        )


def endpoint_inventory(source_id: str, endpoints: list[str], *, strict: bool) -> dict[str, Any]:
    if strict:
        inventory = validate_endpoint_inventory(source_id, endpoints)
        return {
            **inventory,
            "inventory_semantics": "raw_source_inventory_before_run_specific_endpoint_mapping",
            "decision_semantics": "default_policy_preview_not_runtime_registry",
            "runtime_decision_registry": "02_canonicalized/endpoint_registry.json",
        }
    unique = sorted(set(endpoints))
    return {
        "source_id": source_id,
        "inventory_semantics": "raw_source_inventory_before_run_specific_endpoint_mapping",
        "decision_semantics": "default_policy_preview_not_runtime_registry",
        "runtime_decision_registry": "02_canonicalized/endpoint_registry.json",
        "count": len(unique),
        "n_reviewed_corrections": 0,
        "coverage": 1.0,
        "strict_frozen_validation": False,
        "endpoints": [
            spacing_and_spelling_decision(source_id, value).to_dict()
            for value in unique
        ],
    }


def build_hooks(args: argparse.Namespace) -> NormalizationHooks:
    mapping_path = Path(args.auxiliary_mapping)
    attacher = (
        AuxiliaryMetadataAttacher(mapping_path)
        if mapping_path.exists()
        else PendingAuxiliaryAttacher(mapping_path)
    )
    endpoint_normalizer = EndpointNormalizer(args.direct_endpoint_mapping)
    reference_attacher = ReferenceSemanticsAttacher(
        replace(
            REFERENCE_SEMANTICS_CONFIG,
            mapping_path=Path(args.reference_semantics_mapping),
        ),
        allow_missing=args.allow_missing_reference_semantics,
        fail_closed_unmapped=True,
    )
    return NormalizationHooks(
        endpoint_normalizer=endpoint_normalizer.decision,
        endpoint_standardizer=endpoint_specific_standardization_of_unit,
        source_measurement_resolver=source_measurement_resolver,
        family_resolver=family_assignment,
        record_enricher=lambda record: _enrich_record(
            record, attacher, reference_attacher
        ),
        assay_transfer_revalidator=lambda record: _revalidate_assay_transfer_record(
            record, reference_attacher
        ),
        run_state={
            "auxiliary": attacher,
            "endpoint": endpoint_normalizer,
            "reference": reference_attacher,
        },
    )


def _revalidate_assay_transfer_record(
    record: dict[str, Any], reference_attacher: ReferenceSemanticsAttacher
) -> dict[str, Any]:
    validity = enrich_bbb_validity(record)
    reference = reference_attacher.attach_post_scale_fail_closed(
        {**record, **validity}
    )
    return {**validity, **reference}


def _kinetic_symbol(record: dict[str, Any]) -> str | None:
    if (
        record.get("source_id") != "influx_transport"
        or record.get("canonical_endpoint") != "blood_to_brain_transport"
        or record.get("canonical_unit") != "/min"
    ):
        return None
    try:
        value = Decimal(str(record["measurement_resolution_input_measurement"]))
    except (InvalidOperation, KeyError, TypeError):
        return None
    text = str(record.get("measurement_text") or "")
    for match in _KINETIC_ASSIGNMENT.finditer(text):
        if Decimal(match.group("value")) == value:
            return match.group("symbol").casefold().replace("_", "")
    symbols = {
        match.group(0).casefold().replace("_", "")
        for match in _KINETIC_SYMBOL.finditer(text)
    }
    return next(iter(symbols)) if len(symbols) == 1 else None


def attach_source_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        for field, value in source_fields_from_record(row).items():
            row.setdefault(field, value)
    return rows


def stage_documents(
    *,
    args: argparse.Namespace,
    hooks: NormalizationHooks,
    normalized: list[dict[str, Any]],
    persisted: list[dict[str, Any]],
    unit_policy_manifest: dict[str, Any],
) -> StageDocuments:
    del args
    attacher = hooks.run_state["auxiliary"]
    endpoint_normalizer = hooks.run_state["endpoint"]
    reference_attacher = hooks.run_state["reference"]
    coverage = attacher.coverage_audit(normalized)
    reference_coverage = reference_attacher.coverage_audit(normalized)
    missing_endpoint_recovery = validate_missing_endpoint_mapping(normalized)
    semantics_policy = load_measurement_semantics_policy()
    return StageDocuments(
        validity_policy={
            **validity_policy_manifest(),
            "normalization": normalization_policy_manifest(),
            "categorical_response": encoding_policy_manifest(),
            "endpoint_normalization": endpoint_normalizer.manifest(),
            "missing_endpoint_recovery": missing_endpoint_recovery,
        },
        auxiliary_mapping_manifest={**attacher.manifest(), "coverage": coverage},
        source_column_contract=source_column_contract_manifest(
            sorted({column for row in persisted for column in row})
        ),
        validations={
            "one_to_one_measurement_inputs_to_normalized_ids": True,
            "endpoint_orthography_provenance": True,
            "canonical_endpoint_present": True,
            "reviewed_missing_endpoint_mapping_complete": all(
                missing_endpoint_recovery["validations"].values()
            ),
            "direct_endpoint_mapping_human_approved": endpoint_normalizer.direct_ready,
            "policy_independent_validity_present": True,
            "globally_reconciled_auxiliary_coverage": bool(
                coverage["validations"]["all_applicable_records_mapped"]
            ),
            "contextual_unit_policy_loaded": True,
            "contextual_unit_policy_version": unit_policy_manifest["policy_version"],
            "source_column_contract_complete": True,
            "llm_source_projection_fail_closed": True,
            "measurement_semantics_policy_loaded": True,
            "measurement_semantics_policy_version": semantics_policy.policy_version,
            "all_valid_scalars_have_approved_semantics": all(
                row.get("canonical_semantics_status") == "approved"
                for row in normalized
                if row.get("normalization_validity_status") == "valid"
            ),
            "reference_semantics_mapping_complete": bool(
                reference_coverage["validations"]["all_applicable_records_mapped"]
            ),
            "reference_semantics_assignment_complete": bool(
                reference_coverage["validations"]["all_applicable_records_assigned"]
            ),
        },
        endpoint_registry=_endpoint_registry(normalized),
        reference_semantics_manifest={
            **reference_attacher.manifest(),
            "coverage": reference_coverage,
        },
    )


def _endpoint_registry(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist the decisions actually applied by this run's endpoint hook."""
    registry: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        source_id = str(record.get("source_id") or "")
        endpoint_name = str(record.get("endpoint_name") or "missing_endpoint")
        producer_id = str(record.get("canonical_endpoint_producer_id") or "")
        source_field = str(record.get("canonical_endpoint_source_field") or "")
        source_value = record.get(source_field) if source_field else endpoint_name
        decision = {
            "endpoint_name": endpoint_name,
            "spacing_and_spelling_endpoint": record.get("spacing_and_spelling_endpoint"),
            "canonical_endpoint": (
                record.get("canonical_endpoint")
                or record.get("canonical_endpoint_name")
            ),
            "canonical_endpoint_producer_id": producer_id,
            "canonical_endpoint_source_field": source_field or None,
            "status": record.get("spacing_and_spelling_status"),
            "reason": record.get("spacing_and_spelling_reason"),
            "version": record.get("spacing_and_spelling_version"),
        }
        registry_key = json.dumps(
            [endpoint_name, producer_id, source_value],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        previous = registry.setdefault(source_id, {}).setdefault(registry_key, decision)
        if previous != decision:
            raise ValueError(
                f"endpoint normalization is not deterministic for {source_id}/{endpoint_name}"
            )
    return {
        source_id: [mapping[key] for key in sorted(mapping)]
        for source_id, mapping in sorted(registry.items())
    }


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    semantics = load_measurement_semantics_policy()
    versions: dict[str, Any] = {
        "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
        "spacing_and_spelling_version": SPACING_AND_SPELLING_VERSION,
        "endpoint_concept_version": ENDPOINT_CONCEPT_VERSION,
        "endpoint_policy_version": ENDPOINT_POLICY_VERSION,
        "source_measurement_resolver_version": SOURCE_MEASUREMENT_RESOLVER_VERSION,
        "normalization_domain_rules_version": NORMALIZATION_DOMAIN_RULES_VERSION,
        "categorical_response_version": CATEGORICAL_RESPONSE_VERSION,
        "endpoint_normalization_version": ENDPOINT_NORMALIZATION_VERSION,
        "measurement_semantics_policy_version": semantics.policy_version,
        "measurement_semantics_policy_sha256": semantics.sha256,
    }
    if complete:
        versions["source_column_contract_version"] = SOURCE_COLUMN_CONTRACT_VERSION
        versions["source_column_policy_version"] = SOURCE_COLUMN_POLICY_VERSION
    return versions


def census_extras(records: list[dict[str, Any]]) -> dict[str, Any]:
    scalars = {source: 0 for source in EXPECTED_SOURCE_ROWS}
    encoded: dict[str, int] = {}
    for record in records:
        if record.get("finite_scalar_value") is None:
            continue
        source = str(record.get("source_id") or "")
        scalars[source] = scalars.get(source, 0) + 1
        encoder = str(record.get("categorical_encoder_id") or "")
        if encoder:
            encoded[encoder] = encoded.get(encoder, 0) + 1
    return {
        "n_finite_scalars_by_source": dict(sorted(scalars.items())),
        "n_categorically_encoded_by_encoder": dict(sorted(encoded.items())),
    }


def _enrich_record(
    record: dict[str, Any], attacher: Any, reference_attacher: Any
) -> dict[str, Any]:
    source_projection = llm_source_projection(record)
    auxiliary = attacher.attach(record)
    exact = str(record.get("measurement_resolution_status") or "") in {
        "ok",
        "relative",
        "unsure",
        "unavailable",
    }
    resolution_route = str(record.get("measurement_resolution_route") or "")
    routed = exact or (
        bool(record.get("measurement_resolution_active")) and bool(resolution_route)
    )
    mapped = exact and record.get("measurement_unit_mapping_status") == "mapped"
    encoded = CATEGORICAL_RESPONSE_POLICY.apply({**record, **auxiliary})
    source_id = str(record.get("source_id") or "")
    producer_id = str(encoded.get("categorical_encoder_id") or "")
    endpoint_fallback = alternate_endpoint_fields({**record, **encoded})
    producer_fields = {
        "canonical_endpoint_producer_id": endpoint_fallback.get(
            "canonical_endpoint_producer_id",
            SOURCE_ENDPOINT_PRODUCER_IDS[source_id],
        ),
        "canonical_pair_producer_id": (
            producer_id
            or (
                SOURCE_EXTRACTION_PAIR_PRODUCER_IDS[source_id]
                if routed and resolution_route == "extract"
                else SOURCE_RULE_PAIR_PRODUCER_IDS[source_id]
                if routed and resolution_route == "accept"
                else SOURCE_PAIR_PRODUCER_IDS[source_id]
            )
        ),
    }
    endpoint_fields = {
        **endpoint_context_fields({**record, **encoded}),
        **endpoint_fallback,
    }
    reviewed_endpoint_concept = endpoint_concept(
        source_id,
        str(record.get("endpoint_name") or ""),
        str(
            endpoint_fields.get("canonical_endpoint")
            or record.get("canonical_endpoint")
            or ""
        ),
    )
    mapping_status = str(record.get("measurement_unit_mapping_status") or "")
    unit_decided = exact and mapping_status == "mapped"
    if encoded:
        unit_fields = {
            "canonical_measurement_source": "categorical_encoder",
            "canonical_unit_resolution_source": "categorical_encoder",
            "canonical_unit_rule_id": str(encoded["categorical_encoder_id"]),
            "canonical_unit_policy_version": CATEGORICAL_RESPONSE_VERSION,
        }
    elif exact:
        unit_fields = {
            "canonical_measurement_source": (
                "source_exact"
                if record.get("measurement_resolution_origin") == "source_exact"
                else "frozen_measurement_resolution"
            ),
            "canonical_unit_resolution_source": (
                "exact_measurement_unit_map" if unit_decided else "none"
            ),
            "canonical_unit_rule_id": (
                EXACT_UNIT_MAPPING_VERSION if unit_decided else None
            ),
            "canonical_unit_policy_version": (
                EXACT_UNIT_MAPPING_VERSION
                if unit_decided
                else RESOLUTION_APPLY_VERSION
            ),
        }
    elif routed:
        unit_fields = {
            "canonical_measurement_source": (
                "categorical_encoder" if encoded else "none"
            ),
            "canonical_unit_resolution_source": (
                "categorical_encoder" if encoded else "none"
            ),
            "canonical_unit_rule_id": (
                str(encoded.get("categorical_encoder_id") or "") or None
            ),
            "canonical_unit_policy_version": CATEGORICAL_RESPONSE_VERSION,
        }
    else:
        unit_fields = unit_provenance_fields(record, encoded=encoded)
    enriched = {
        **record,
        **auxiliary,
        **encoded,
        **endpoint_fields,
        "canonical_endpoint_concept": reviewed_endpoint_concept,
        **unit_fields,
        **producer_fields,
    }
    if encoded:
        structurally_valid = (
            str(record.get("structure_status") or "") == "resolved"
            and bool(record.get("canonical_smiles"))
        )
        validity = {
            "normalization_validity_status": (
                "valid" if structurally_valid else "unresolved_structure"
            ),
            "canonical_semantics_status": "approved",
            "canonical_quantity_kind": "controlled_categorical",
            "canonical_numeric_domain": "controlled",
            "canonical_semantics_rule_id": str(encoded["categorical_encoder_id"]),
            "canonical_semantics_policy_version": CATEGORICAL_RESPONSE_VERSION,
        }
    elif exact:
        validity = {
            "normalization_validity_status": (
                "unresolved_structure"
                if str(record.get("structure_status") or "") != "resolved"
                or not record.get("canonical_smiles")
                else "valid"
                if mapped
                else "exact_measurement_excluded"
                if unit_decided
                else "non_scalar_measurement"
            ),
            "canonical_semantics_status": "approved" if mapped else "evidence_only",
            "canonical_quantity_kind": (
                "exact_mapped_continuous"
                if mapped
                else "excluded_measurement"
                if unit_decided
                else "unresolved_measurement"
            ),
            "canonical_numeric_domain": (
                str(record.get("measurement_numeric_domain") or "unreviewed")
                if mapped
                else "none"
            ),
            "canonical_semantics_rule_id": (
                EXACT_UNIT_MAPPING_VERSION
                if unit_decided
                else RESOLUTION_APPLY_VERSION
            ),
            "canonical_semantics_policy_version": (
                EXACT_UNIT_MAPPING_VERSION
                if unit_decided
                else RESOLUTION_APPLY_VERSION
            ),
        }
    elif routed:
        structurally_valid = (
            str(record.get("structure_status") or "") == "resolved"
            and bool(record.get("canonical_smiles"))
        )
        validity = {
            "normalization_validity_status": (
                "unresolved_structure"
                if not structurally_valid
                else "valid" if encoded else "non_scalar_measurement"
            ),
            "canonical_semantics_status": (
                "approved" if encoded else "evidence_only"
            ),
            "canonical_quantity_kind": (
                "controlled_categorical" if encoded else "non_scalar"
            ),
            "canonical_numeric_domain": "controlled" if encoded else "none",
            "canonical_semantics_rule_id": (
                str(encoded.get("categorical_encoder_id") or "")
                or "measurement_resolution_route.v1"
            ),
            "canonical_semantics_policy_version": (
                CATEGORICAL_RESPONSE_VERSION
            ),
        }
    else:
        validity = enrich_bbb_validity(enriched)
    reference = reference_attacher.attach(
        {
            **record,
            **auxiliary,
            **encoded,
            **endpoint_fields,
            **unit_fields,
            **producer_fields,
            **validity,
        }
    )
    return {
        "source_column_contract_version": SOURCE_COLUMN_CONTRACT_VERSION,
        "source_column_policy_version": SOURCE_COLUMN_POLICY_VERSION,
        **encoded,
        "llm_source_contract_json": json.dumps(
            {key: value for key, value in source_projection.items() if key != "source_fields"},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "llm_source_fields_json": json.dumps(
            source_projection["source_fields"], ensure_ascii=False, sort_keys=True
        ),
        **auxiliary,
        **endpoint_fields,
        "canonical_endpoint_concept": reviewed_endpoint_concept,
        **unit_fields,
        **producer_fields,
        **validity,
        **reference,
        "canonical_kinetic_symbol": _kinetic_symbol(enriched),
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
    census_extras=census_extras,
    verify_source_digest=lambda source_id, path: validate_source_digest(source_id, path),
    scientific_assets=(
        DEFAULT_SOURCE_VALUE_REPAIRS,
        DEFAULT_REVIEWED_SOURCE_DROPS,
        DEFAULT_SMILES_IDENTITY_AUDIT,
        DEFAULT_REVIEWED_NAME_SMILES_CONFLICTS,
        DEFAULT_SMILES_SAMPLE_AUDIT,
        DEFAULT_SEMANTICS_PATH,
        *ENDPOINT_CONCEPT_PATHS,
        REFERENCE_SEMANTICS_CONFIG.prompt_registry_path,
        Path(__file__).parent
        / "data_processing/assay_transfer_measurements_v2/policy.json",
    ),
    assay_transfer_measurement_policy=(
        Path(__file__).parent
        / "data_processing/assay_transfer_measurements_v2/policy.json"
    ),
    measurement_resolution_enabled=True,
    exact_unit_mapping_path=(
        Path(__file__).parent
        / "data_processing/canonicalization_v7/exact_measurement_unit_map.v2.json"
    ),
    reference_semantics_enabled=True,
    endpoint_identity_required_sources=(
        "passive_permeability",
        "efflux_transport",
        "influx_transport",
    ),
)


__all__ = [
    "DEFAULT_BENCHMARK_SPLIT_ROOT",
    "DEFAULT_OUT_DIR",
    "POLICY",
    "TASK_ID",
]
