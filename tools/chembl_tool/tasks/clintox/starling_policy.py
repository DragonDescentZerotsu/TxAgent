"""ClinTox plug-in for the shared normalized Starling v7 builder."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.normalization.contracts import (
    FamilyAssignment,
    MeasurementPair,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    EndpointOrthography,
)
from tools.chembl_tool.common.starling.normalization.source_value_cleaning import (
    clean_source_values,
)
from tools.chembl_tool.common.starling.normalization.task_policy import (
    NormalizationHooks,
    StageDocuments,
    StarlingTaskPolicy,
)
from tools.chembl_tool.common.starling.reference_semantics import (
    REFERENCE_SCOPE_ABSOLUTE,
    REFERENCE_SCOPE_UNKNOWN,
)
from tools.chembl_tool.tasks.clintox.clintox_base_benchmark import (
    ADAPTER_VERSION as DIRECT_ADAPTER_VERSION,
)
from tools.chembl_tool.tasks.clintox.clintox_base_benchmark import (
    is_direct_gold_scope,
)
from tools.chembl_tool.tasks.clintox.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
    CONTROLLED_VOCABULARIES,
    controlled_vocabulary_violations,
)
from tools.chembl_tool.tasks.clintox.starling_categorical_response import (
    POLICY as CATEGORICAL_RESPONSE_POLICY,
)
from tools.chembl_tool.tasks.clintox.starling_compact_artifacts import (
    COMPACT_PROFILE,
)
from tools.chembl_tool.tasks.clintox.starling_measurement_semantics import (
    DEFAULT_SEMANTICS_PATH,
    MEASUREMENT_SEMANTICS_VERSION,
    MeasurementSemantics,
    censored_support_error,
    load_measurement_semantics_policy,
    measurement_semantics_audit,
    numeric_domain_error,
    resolve_measurement_pair,
    resolve_measurement_semantics,
)
from tools.chembl_tool.tasks.clintox.starling_normalization_sources import (
    EXPECTED_SOURCE_ROWS,
    llm_source_projection,
    source_column_contract_manifest,
    source_fields_from_record,
    source_profiles,
    validate_source_digest,
)
from tools.chembl_tool.tasks.clintox.starling_schema import RECORD_CONTRACT
from tools.chembl_tool.tasks.clintox.starling_reference_semantics import (
    REFERENCE_SEMANTICS_VERSION,
    deterministic_reference_assignment,
    reference_semantics_manifest,
)
from tools.chembl_tool.tasks.clintox.starling_source import (
    DEFAULT_DATA_ROOT,
    DIRECT_SOURCE_ID,
    SOURCE_RELEASE,
)

TASK_ID = "clintox"
DATASET_NAME = "ClinTox send_v2 seven-source delivery"
DEFAULT_STARLING_DATA_DIR = str(DEFAULT_DATA_ROOT)
DEFAULT_OUT_DIR = (
    "outputs/chembl_tool/tasks/clintox/evidence_library/starling_normalized_v7"
)
DEFAULT_BENCHMARK_SPLIT_ROOT = (
    "data/processed_starling_record_supported_v2/ClinTox_Human_Toxicity"
)

SPACING_AND_SPELLING_VERSION = "clintox_endpoint_orthography.v1"
EXACT_CONTEXT_MAPPING_VERSION = "clintox_exact_context.v1"
EXACT_CONTEXT_ATTACHMENT_VERSION = "clintox_exact_context_attachment.v1"
VALIDITY_POLICY_VERSION = "clintox_normalization_validity.v1"
ASSAY_TRANSFER_MEASUREMENT_POLICY = (
    Path(__file__).resolve().parent
    / "data_processing/assay_transfer_measurements_v2/policy.json"
)

EXPECTED_ENDPOINT_INVENTORIES = {
    DIRECT_SOURCE_ID: {
        "count": 1,
        "sha256": "de0ed95fe5fd0b8b7c5989fb1cbad43db97a72d274d62f7d07bfdd5785484a54",
    },
    "nonclinical_in_vivo_toxicity": {
        "count": 175,
        "sha256": "caa0cc8ed278e6f276a285c4509bd02375e5ea79813224742e47174e48750f5b",
    },
    "organ_specific_toxicity": {
        "count": 248_924,
        "sha256": "e6fafbdff13ca794404c2fb031862c9615294983680e3f7d9640ad91cc0ea01c",
    },
    "genotoxicity_carcinogenicity": {
        "count": 138_825,
        "sha256": "37f96ec54463a2d2694a019f18beafaf58d6016ccc7fed4e84f8bdc3414aa7f9",
    },
    "cellular_stress": {
        "count": 201,
        "sha256": "35c04e724f853599eba94468544d6ad1dffae665f1f877ea20c89344d7c477a7",
    },
    "general_cytotoxicity": {
        "count": 158,
        "sha256": "30bbfac0302218f2bb6ef580bde3019d9f4f398fbb7ce32e3fdd73cc8b487653",
    },
    "off_target_ddi_exposure": {
        "count": 109_612,
        "sha256": "0bda2aea1bd98df54d101542096160ec151fbd7a0ea7ba6a3b3d8a1918c658e6",
    },
}

_CONTEXT_OUTPUTS = {
    DIRECT_SOURCE_ID: {"canonical_clinical_context": "clinical_context"},
    "nonclinical_in_vivo_toxicity": {
        "canonical_animal_context": "animal_context",
        "canonical_exposure_context": "exposure_context",
    },
    "organ_specific_toxicity": {
        "canonical_organ_system": "organ_system",
        "canonical_evidence_context": "evidence_context",
        "canonical_biological_system": "biological_system",
    },
    "genotoxicity_carcinogenicity": {
        "canonical_evidence_category": "evidence_category",
        "canonical_assay_type": "assay_type",
        "canonical_study_context": "study_context",
        "canonical_biological_system": "biological_system",
    },
    "cellular_stress": {
        "canonical_evidence_basis": "evidence_basis",
        "canonical_biological_model": "biological_model",
    },
    "general_cytotoxicity": {
        "canonical_cell_model": "cell_model",
        "canonical_assay_method": "assay_method",
        "canonical_exposure_time": "exposure_time_h",
    },
    "off_target_ddi_exposure": {
        "canonical_evidence_type": "evidence_type",
        "canonical_result_metric": "result_metric",
        "canonical_assay_context": "assay_context",
    },
}


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark-split-root",
        default=DEFAULT_BENCHMARK_SPLIT_ROOT,
        help="ClinTox record_supported_v2 task root used only by legacy Stage 06-09 builds.",
    )


def endpoint_decision(source_id: str, endpoint_name: str) -> EndpointOrthography:
    del source_id
    endpoint = endpoint_name.strip() or "missing_endpoint"
    return EndpointOrthography(
        endpoint_name=endpoint,
        spacing_and_spelling_endpoint=endpoint,
        status="unchanged",
        reason="source_endpoint_preserved",
        spacing_and_spelling_version=SPACING_AND_SPELLING_VERSION,
    )


def endpoint_inventory(
    source_id: str, endpoints: list[str], *, strict: bool
) -> dict[str, Any]:
    from tools.chembl_tool.common.starling.normalization.cleaning import (
        endpoint_inventory_hash,
    )

    values = sorted(
        set(["human_clinical_toxicity"] if source_id == DIRECT_SOURCE_ID else endpoints)
    )
    actual = {"count": len(values), "sha256": endpoint_inventory_hash(values)}
    expected = EXPECTED_ENDPOINT_INVENTORIES[source_id]
    if strict and actual != expected:
        raise ValueError(
            f"endpoint inventory drift for {source_id}: expected {expected}, found {actual}"
        )
    return {
        "source_id": source_id,
        **actual,
        "coverage": 1.0,
        "strict_frozen_validation": strict,
        "semantic_merges": False,
        "endpoints": [endpoint_decision(source_id, value).to_dict() for value in values],
    }


def family_assignment(
    source_id: str,
    endpoint_name: str,
    record: Mapping[str, Any] | None = None,
) -> FamilyAssignment | None:
    del endpoint_name
    row = record or {}
    if source_id == DIRECT_SOURCE_ID:
        if not _is_direct_gold_record(row):
            return None
        return FamilyAssignment(
            "Direct.human_clinical_toxicity",
            "Direct",
            "clinical_human_safety",
            "direct_outcome",
            "human clinical toxicity",
        )
    families = {
        "nonclinical_in_vivo_toxicity": "in_vivo_toxicology",
        "organ_specific_toxicity": "organ_specific_toxicity",
        "genotoxicity_carcinogenicity": "genotoxicity_carcinogenicity",
        "cellular_stress": "cellular_stress_pathways",
        "general_cytotoxicity": "general_cytotoxicity",
        "off_target_ddi_exposure": "off_target_ddi_exposure",
    }
    family = families.get(source_id)
    if family is None:
        return None
    return FamilyAssignment(
        f"Mechanism.{family}",
        "Mechanism",
        family,
        "mechanistic_factor",
        family.replace("_", " "),
    )


def _is_direct_gold_record(record: Mapping[str, Any]) -> bool:
    return is_direct_gold_scope(
        {
            "toxicity_category": record.get("measurement_text"),
            "toxicity_outcome": record.get("toxicity_outcome"),
            "needs_more_context": record.get("needs_more_context"),
            "SMILES": record.get("source_smiles") or record.get("smiles"),
            "support_text": record.get("support_text"),
            "pmid": record.get("pmid"),
            "extraction_id": record.get("source_record_id"),
        }
    )


def _identity_pair(endpoint: str, pair: MeasurementPair) -> MeasurementPair:
    del endpoint
    return pair


def build_hooks(args: argparse.Namespace) -> NormalizationHooks:
    del args
    return NormalizationHooks(
        endpoint_normalizer=endpoint_decision,
        endpoint_standardizer=_identity_pair,
        source_measurement_resolver=resolve_measurement_pair,
        family_resolver=family_assignment,
        record_enricher=_enrich_record,
        assay_transfer_revalidator=_enrich_record,
    )


def _enrich_record(record: dict[str, Any]) -> dict[str, Any]:
    encoded = CATEGORICAL_RESPONSE_POLICY.apply(record)
    working = {**record, **encoded}
    violations = controlled_vocabulary_violations(working)
    if encoded:
        semantics = MeasurementSemantics(
            rule_id=str(encoded.get("categorical_encoder_id") or "controlled_scale"),
            policy_version=MEASUREMENT_SEMANTICS_VERSION,
            status="approved",
            quantity_kind="controlled_categorical_outcome",
            numeric_domain="declared_categorical",
            semantic_unit_basis="controlled_category",
            reference_scope="not_applicable",
            reference_basis="none",
        )
    else:
        semantics = resolve_measurement_semantics(
            working,
            str(working.get("canonical_endpoint") or ""),
            working.get("canonical_unit"),
        )
    semantic_fields = semantics.fields()
    reference = deterministic_reference_assignment({**working, **semantic_fields})
    domain_error = (
        numeric_domain_error(semantics, working.get("finite_scalar_value"))
        if working.get("finite_scalar_value") is not None
        else None
    )
    support_error = censored_support_error(working, semantics)
    if violations:
        validity = "off_schema_controlled_value"
    elif encoded:
        validity = "valid"
    elif support_error:
        validity = support_error
    elif working.get("finite_scalar_value") is None:
        validity = "non_scalar_measurement"
    elif semantics.status != "approved":
        validity = "unreviewed_measurement_semantics"
    elif domain_error:
        validity = domain_error
    elif reference.scope == REFERENCE_SCOPE_UNKNOWN:
        validity = "unknown_reference_semantics"
    else:
        validity = "valid"

    source_projection = llm_source_projection(record)
    context = {
        canonical: _exact_context(record.get(source_field))
        for canonical, source_field in _CONTEXT_OUTPUTS[
            str(record.get("source_id") or "")
        ].items()
    }
    categorical = bool(encoded)
    return {
        **encoded,
        **semantic_fields,
        **context,
        "controlled_vocabulary_status": "off_schema" if violations else "valid",
        "controlled_vocabulary_violation_fields": ",".join(violations) or None,
        "normalization_validity_status": validity,
        "canonical_reference_scope": reference.scope,
        "canonical_reference_basis": reference.basis,
        "reference_semantics_assignment_method": reference.method,
        "is_absolute_and_continuous": bool(
            not categorical
            and working.get("finite_scalar_value") is not None
            and reference.scope == REFERENCE_SCOPE_ABSOLUTE
            and validity == "valid"
        ),
        "absolute_and_continuous_value": (
            working.get("finite_scalar_value")
            if not categorical
            and reference.scope == REFERENCE_SCOPE_ABSOLUTE
            and validity == "valid"
            else None
        ),
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
            source_projection["source_fields"], ensure_ascii=False, sort_keys=True
        ),
    }


def _exact_context(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value).strip()


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
    del args, hooks
    vocabulary = _vocabulary_audit(normalized)
    direct_source_audit = _direct_source_audit(normalized)
    semantics_audit = measurement_semantics_audit(normalized)
    reference_manifest = reference_semantics_manifest(normalized)
    output_fields = sorted(
        {field for fields in _CONTEXT_OUTPUTS.values() for field in fields}
    )
    persisted_columns = sorted({field for row in persisted for field in row})
    return StageDocuments(
        validity_policy={
            "version": VALIDITY_POLICY_VERSION,
            "benchmark_status": "candidate_pending_qa",
            "promotion_gate": "frozen 360-row category-stratified manual QA",
            "qualifying_conditions": {
                "status": "unavailable_in_source_schema",
                "effect": "no qualifying-condition exclusion can be applied",
            },
            "direct_source_scope_adapter": DIRECT_ADAPTER_VERSION,
            "direct_source_non_gold_rows": "retained but not retrieval eligible",
            "controlled_categorical_response": CATEGORICAL_RESPONSE_POLICY.manifest(),
            "controlled_vocabulary_audit": vocabulary,
            "direct_source_audit": direct_source_audit,
            "reference_semantics": {
                "version": REFERENCE_SEMANTICS_VERSION,
                "ambiguous_values": "unknown and excluded from pair buckets",
                "runtime_llm_calls": False,
            },
            "measurement_semantics": load_measurement_semantics_policy().manifest(),
            "unit_policy_version": unit_policy_manifest["policy_version"],
        },
        auxiliary_mapping_manifest={
            "mapping_version": EXACT_CONTEXT_MAPPING_VERSION,
            "attachment_version": EXACT_CONTEXT_ATTACHMENT_VERSION,
            "method": "exact simply-cleaned source context",
            "output_fields": output_fields,
            "coverage": {
                "records": len(normalized),
                "missing_values_use_explicit_unknown": True,
            },
        },
        source_column_contract=source_column_contract_manifest(persisted_columns),
        validations={
            "one_to_one_cleaned_to_canonical_ids": True,
            "global_identifier_not_persisted": "global_identifier" not in persisted_columns,
            "off_schema_values_preserved_not_rewritten": True,
            "off_schema_values_excluded_from_pair_buckets": True,
            "direct_retrieval_uses_gold_adapter_scope": True,
            "direct_source_counts_match_frozen_base": direct_source_audit[
                "frozen_count_validation"
            ],
            "reference_semantics_present": all(
                row.get("canonical_reference_scope")
                and row.get("canonical_reference_basis")
                for row in normalized
            ),
            "measurement_semantics_policy_loaded": True,
            "all_valid_scalars_have_approved_semantics": all(
                row.get("canonical_semantics_status") == "approved"
                for row in normalized
                if row.get("normalization_validity_status") == "valid"
            ),
            "categorical_scales_declared": all(
                not row.get("categorical_encoder_id")
                or row.get("categorical_encoder_id")
                in CATEGORICAL_RESPONSE_POLICY.measurement_scales
                for row in normalized
            ),
        },
        endpoint_registry=_endpoint_registry(normalized),
        reference_semantics_manifest={
            **reference_manifest,
            "measurement_semantics": semantics_audit,
        },
    )


def _vocabulary_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    status: dict[str, Counter[str]] = defaultdict(Counter)
    fields: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        source_id = str(row.get("source_id") or "")
        row_status = str(row.get("controlled_vocabulary_status") or "")
        status[source_id][row_status] += 1
        for field in str(row.get("controlled_vocabulary_violation_fields") or "").split(","):
            if not field:
                continue
            fields[source_id][field] += 1
            if len(examples[source_id]) < 20:
                examples[source_id].append(
                    {
                        "source_row_number": row.get("source_row_number"),
                        "field": field,
                        "value": row.get(field),
                    }
                )
    return {
        "declared_fields": {
            source: {field: sorted(values) for field, values in field_map.items()}
            for source, field_map in sorted(CONTROLLED_VOCABULARIES.items())
        },
        "status_counts": {
            source: dict(sorted(counts.items())) for source, counts in sorted(status.items())
        },
        "violation_field_counts": {
            source: dict(sorted(counts.items())) for source, counts in sorted(fields.items())
        },
        "violation_examples": dict(sorted(examples.items())),
        "off_schema_policy": "retain source evidence; do not encode or bucket",
    }


def _direct_source_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    base = [row for row in records if row.get("source_id") == DIRECT_SOURCE_ID]
    scoped = [
        row
        for row in base
        if row.get("group_id") == "Direct.human_clinical_toxicity"
    ]
    resolved = [row for row in scoped if row.get("canonical_smiles")]
    full_source = len(base) == EXPECTED_SOURCE_ROWS[DIRECT_SOURCE_ID]
    matches = (
        len(scoped) == 511_805 and len(resolved) == 510_620
        if full_source
        else True
    )
    return {
        "source_rows": len(base),
        "gold_adapter_scope_rows": len(scoped),
        "gold_adapter_scope_resolved_structure_rows": len(resolved),
        "expected_full_source_rows": EXPECTED_SOURCE_ROWS[DIRECT_SOURCE_ID],
        "expected_full_gold_adapter_scope_rows": 511_805,
        "expected_full_resolved_structure_rows": 510_620,
        "bounded_build": not full_source,
        "frozen_count_validation": matches,
    }


def _endpoint_registry(records: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints: dict[str, set[str]] = defaultdict(set)
    for row in records:
        endpoints[str(row.get("source_id") or "")].add(
            str(row.get("canonical_endpoint") or "")
        )
    return {
        "policy_version": SPACING_AND_SPELLING_VERSION,
        "semantic_merges": False,
        "sources": {
            source: {"canonical_endpoint_count": len(values)}
            for source, values in sorted(endpoints.items())
        },
        "complete_value_registry": "../01_cleaned/endpoint_inventory.json",
    }


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    versions = {
        "benchmark_candidate_status": "candidate_pending_qa",
        "source_release": SOURCE_RELEASE,
        "categorical_response_version": CATEGORICAL_RESPONSE_VERSION,
        "context_mapping_version": EXACT_CONTEXT_MAPPING_VERSION,
        "direct_scope_adapter_version": DIRECT_ADAPTER_VERSION,
        "reference_semantics_version": REFERENCE_SEMANTICS_VERSION,
        "measurement_semantics_version": MEASUREMENT_SEMANTICS_VERSION,
        "spacing_and_spelling_version": SPACING_AND_SPELLING_VERSION,
        "validity_policy_version": VALIDITY_POLICY_VERSION,
        "qualifying_conditions_status": "unavailable_in_source_schema",
    }
    if complete:
        versions["source_column_contract_version"] = (
            "clintox_source_column_contract.v2"
        )
    return versions


def census_extras(records: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    vocab_counts: Counter[str] = Counter()
    for row in records:
        source_id = str(row.get("source_id") or "")
        if row.get("finite_scalar_value") is not None:
            scalar_counts[source_id] += 1
        group_counts[str(row.get("group_id") or "unassigned")] += 1
        vocab_counts[
            f"{source_id}:{row.get('controlled_vocabulary_status') or 'missing'}"
        ] += 1
    return {
        "n_finite_scalars_by_source": dict(sorted(scalar_counts.items())),
        "group_record_counts": dict(sorted(group_counts.items())),
        "controlled_vocabulary_status_counts": dict(sorted(vocab_counts.items())),
    }


def _clean_source_values(records: list[dict[str, Any]], args: argparse.Namespace):
    del args
    return clean_source_values(records, task_id=TASK_ID)


_DATA_ROOT = Path(DEFAULT_STARLING_DATA_DIR)
SCIENTIFIC_ASSETS = (
    _DATA_ROOT / "SOURCE_MANIFEST.json",
    *tuple(
        _DATA_ROOT / source / "extraction_guidance.json"
        for source in EXPECTED_SOURCE_ROWS
    ),
    DEFAULT_SEMANTICS_PATH,
    ASSAY_TRANSFER_MEASUREMENT_POLICY,
)

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
    census_extras=census_extras,
    verify_source_digest=validate_source_digest,
    scientific_assets=SCIENTIFIC_ASSETS,
    assay_transfer_measurement_policy=ASSAY_TRANSFER_MEASUREMENT_POLICY,
    reference_semantics_enabled=True,
    family_resolver_input_fields=(
        "toxicity_outcome",
        "needs_more_context",
        "pmid",
    ),
)


__all__ = [
    "DEFAULT_BENCHMARK_SPLIT_ROOT",
    "DEFAULT_OUT_DIR",
    "DEFAULT_STARLING_DATA_DIR",
    "POLICY",
    "TASK_ID",
    "endpoint_inventory",
    "family_assignment",
]
