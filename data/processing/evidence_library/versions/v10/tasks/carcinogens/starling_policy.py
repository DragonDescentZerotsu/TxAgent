"""Source-faithful Carcinogens v10 Stage 0-1 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.compact_artifacts import CompactArtifactProfile
from data.processing.evidence_library.shared.v2.canonicalization_v7 import (
    CanonicalDimensionSpec,
    PairBucketSpec,
    SourceProfile,
    StarlingRecordContract,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_literal_text,
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.contracts import NormalizedSourceProfile
from data.processing.evidence_library.shared.v2.normalization.source_value_cleaning import (
    SourceValueCleaningResult,
)
from data.processing.evidence_library.shared.v2.normalization.task_policy import (
    NormalizationHooks,
    StageDocuments,
    StarlingTaskPolicy,
)
from data.processing.evidence_library.shared.v2.reference_semantics import (
    REFERENCE_SEMANTICS_VERSION,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
    attach_stage1_routes,
)
from data.processing.evidence_library.versions.v10.standard_pair_dimension_stage2 import (
    build_standard_hooks,
    build_standard_stage_documents,
    no_family,
    validate_registered_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.carcinogens.mapping_registry import (
    REGISTRY_PATH,
    mapping_path,
    mapping_registry,
    validate_mapping_hashes,
)
from data.processing.evidence_library.versions.v10.tasks.carcinogens.starling_reference_semantics import (
    DEFAULT_MAPPING_PATH as DEFAULT_REFERENCE_SEMANTICS_MAPPING,
    REFERENCE_SEMANTICS_CONFIG,
)
from data.processing.gold_labels.level_mappings import level_mapping_path
from data.processing.paths import evidence_library_root, raw_starling_task_root

TASK_ID = "carcinogens"
PAIR_MAPPING_VERSION = "carcinogens_pair_dimensions.v1"
TASK_ROOT = Path(__file__).resolve().parent
V10_ROOT = TASK_ROOT.parents[1]
SOURCE_MANIFEST_PATH = TASK_ROOT / "source_manifest.json"
SOURCE_MANIFEST = json.loads(SOURCE_MANIFEST_PATH.read_text())
SPECS = SOURCE_MANIFEST["sources"]
ROLES = {
    "carcinogens_base": ("cancer_or_tumor", "carcinogenicity_conclusion", ""),
    "carcinogens_v1": ("assay_family", "result_value", "measurement_unit"),
    "carcinogens_v2": ("assay_family", "response_value", "response_unit"),
    "carcinogens_v3": ("endpoint_category", "reported_result", "measurement_unit"),
    "carcinogens_v4": ("assay_domain", "endpoint_result", "measurement_unit"),
    "carcinogens_v5": ("phenotype_domain", "endpoint_result", "endpoint_unit"),
}
UNIT_EXCEPTIONS = {
    "carcinogens_base": {
        "mode": "unitless_categorical",
        "reason": "carcinogenicity_conclusion is a qualitative source outcome",
    }
}
RAW_COLUMNS = {
    "carcinogens_base": (
        "paragraph_idx", "support_text", "agent_name", "agent_category",
        "carcinogenicity_conclusion", "evidence_basis", "evidence_scope",
        "evidence_population_or_model", "cancer_or_tumor", "cancer_identifier",
        "carcinogenic_role", "classification_authority", "classification_label",
        "exposure_route", "qualifying_conditions", "extra_details", "confidence",
        "needs_more_context", "pmid", "extraction_id", "SMILES",
    ),
    "carcinogens_v1": (
        "paragraph_idx", "support_text", "assay_family", "assay_method",
        "biological_test_system", "endpoint_measure", "effect_direction",
        "result_value", "measurement_unit", "molecule_concentration_or_dose",
        "exposure_regimen", "assay_interpretation_conditions", "extra_details",
        "confidence", "needs_more_context", "pmid", "extraction_id", "SMILES",
    ),
    "carcinogens_v2": (
        "paragraph_idx", "support_text", "entity_name", "assay_family",
        "test_system", "endpoint_detail", "metabolic_activation",
        "dose_or_concentration", "dose_unit", "exposure_timing",
        "result_interpretation", "response_value", "response_unit", "extra_details",
        "confidence", "needs_more_context", "pmid", "extraction_id", "SMILES",
    ),
    "carcinogens_v3": (
        "paragraph_idx", "support_text", "endpoint_category",
        "assay_and_detection_method", "endpoint", "effect_direction",
        "reported_result", "measurement_unit", "molecule_concentration_or_dose",
        "exposure_and_readout_timing", "biological_system", "species",
        "metabolic_activation_system", "extra_details", "confidence",
        "needs_more_context", "pmid", "extraction_id", "SMILES",
    ),
    "carcinogens_v4": (
        "paragraph_idx", "support_text", "assay_domain", "assay_type",
        "biological_test_system", "endpoint", "endpoint_result", "measurement_unit",
        "effect_direction", "concentration_or_dose", "exposure_duration",
        "post_exposure_followup", "persistence_outcome", "extra_details", "confidence",
        "needs_more_context", "pmid", "extraction_id", "SMILES",
    ),
    "carcinogens_v5": (
        "paragraph_idx", "support_text", "phenotype_domain", "assay_type",
        "assay_format", "biological_model", "injury_or_growth_context",
        "exposure_details", "endpoint_and_method", "endpoint_result", "endpoint_unit",
        "effect_direction", "interpretation_conditions", "extra_details", "confidence",
        "needs_more_context", "pmid", "extraction_id", "SMILES",
    ),
}

PAIR_DIMENSION_INPUTS = {
    "carcinogens_base": {
        "canonical_assay_context": ("evidence_basis", "exposure_route"),
        "canonical_species_context": (
            "evidence_scope",
            "evidence_population_or_model",
        ),
    },
    "carcinogens_v1": {
        "canonical_assay_context": (
            "assay_method",
            "biological_test_system",
            "endpoint_measure",
        ),
    },
    "carcinogens_v2": {
        "canonical_assay_context": (
            "test_system",
            "endpoint_detail",
            "metabolic_activation",
        ),
    },
    "carcinogens_v3": {
        "canonical_assay_context": (
            "assay_and_detection_method",
            "endpoint",
            "biological_system",
            "metabolic_activation_system",
        ),
        "canonical_species_context": ("species",),
    },
    "carcinogens_v4": {
        "canonical_assay_context": (
            "assay_type",
            "biological_test_system",
            "endpoint",
        ),
    },
    "carcinogens_v5": {
        "canonical_assay_context": (
            "assay_type",
            "assay_format",
            "biological_model",
            "endpoint_and_method",
        ),
    },
}


def _canonical_dimensions(source: str) -> tuple[CanonicalDimensionSpec, ...]:
    universal = (
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "endpoint",
            ("endpoint_name",),
            "frozen_mapping",
            PAIR_MAPPING_VERSION,
        ),
        CanonicalDimensionSpec(
            "canonical_endpoint_concept",
            "endpoint_concept",
            ("endpoint_name",),
            "frozen_mapping",
            PAIR_MAPPING_VERSION,
            missing_policy="explicit_unknown",
            legacy_value_field="canonical_endpoint_concept",
        ),
        CanonicalDimensionSpec(
            "canonical_unit_text",
            "unit",
            ("endpoint_name", "measurement_text", "unit_text", "support_text"),
            "frozen_extraction",
            PAIR_MAPPING_VERSION,
        ),
        CanonicalDimensionSpec(
            "canonical_measurement_scale_id",
            "measurement_scale",
            ("measurement_text", "unit_text"),
            "controlled_encoder",
            PAIR_MAPPING_VERSION,
        ),
    )
    contexts = tuple(
        CanonicalDimensionSpec(
            output_field,
            output_field.removeprefix("canonical_"),
            input_fields,
            "frozen_mapping",
            PAIR_MAPPING_VERSION,
            missing_policy="explicit_unknown",
        )
        for output_field, input_fields in PAIR_DIMENSION_INPUTS[source].items()
    )
    reference_inputs = (
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "support_text",
        *REFERENCE_SEMANTICS_CONFIG.source_specs[source].extra_fields,
    )
    reference = (
        CanonicalDimensionSpec(
            "canonical_reference_scope",
            "measurement_reference_scope",
            tuple(dict.fromkeys(reference_inputs)),
            "frozen_mapping",
            REFERENCE_SEMANTICS_VERSION,
            missing_policy="explicit_unknown",
            atomic_group="canonical_reference_semantics_pair",
            classification_evidence=True,
            legacy_value_field="canonical_reference_scope",
        ),
        CanonicalDimensionSpec(
            "canonical_reference_basis",
            "measurement_reference_basis",
            tuple(dict.fromkeys(reference_inputs)),
            "frozen_mapping",
            REFERENCE_SEMANTICS_VERSION,
            missing_policy="explicit_unknown",
            atomic_group="canonical_reference_semantics_pair",
            classification_evidence=True,
            legacy_value_field="canonical_reference_basis",
        ),
    )
    return (*universal, *contexts, *reference)


SOURCES = {
    source: SourceProfile(
        source_id=source, source_columns=columns, endpoint_field=ROLES[source][0],
        measurement_field=ROLES[source][1], unit_field=ROLES[source][2],
        smiles_field="SMILES", structure_mode="direct",
        canonical_dimensions=_canonical_dimensions(source),
    )
    for source, columns in RAW_COLUMNS.items()
}
PAIR_BUCKETS = {
    source: PairBucketSpec(
        source,
        (
            "canonical_endpoint_name",
            "canonical_unit_text",
            "canonical_measurement_scale_id",
            *PAIR_DIMENSION_INPUTS[source],
        ),
    )
    for source in SOURCES
}
CONTRACT = StarlingRecordContract(TASK_ID, SOURCES, PAIR_BUCKETS)
DEFAULT_DATA_DIR = str(raw_starling_task_root(TASK_ID))
DEFAULT_OUT_DIR = str(evidence_library_root(TASK_ID, "v10"))
TASK_IMPLEMENTATION_PATHS = (
    TASK_ROOT / "build_normalized_starling_evidence_library.py",
    TASK_ROOT / "starling_measurement_resolution.py",
    TASK_ROOT / "starling_policy.py",
    V10_ROOT / "build_normalized_evidence_library.py",
    V10_ROOT / "measurement_routing.py",
)
MAIN_UNIVERSE_MANIFEST = Path("data/raw/starling/main_universe_v1/manifest.json")
MAIN_UNIVERSE_RECORDS = Path(
    "data/raw/starling/main_universe_v1/carcinogens/records.parquet"
)


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


def source_profiles(data_dir: Path):
    profiles = []
    for source, spec in SPECS.items():
        endpoint, measurement, unit = ROLES[source]
        profiles.append(NormalizedSourceProfile(
            source_id=source, source_name=f"starling/{spec['directory']}",
            source_path=str(data_dir / spec["directory"] / "extractions.parquet"),
            endpoint_field=endpoint, measurement_field=measurement, unit_field=unit,
            smiles_field="SMILES", structure_mode="direct", record_id_field="extraction_id",
            name_fields=tuple(name for name in ("agent_name", "entity_name") if name in RAW_COLUMNS[source]),
            literal_text_fields=(endpoint,),
        ))
    return profiles


def verify_source_digest(source: str, path: Path):
    spec = SPECS[source]
    if file_sha256(path) != spec["parquet_sha256"]:
        raise ValueError(f"source digest drift for {source}")
    if tuple(pq.read_schema(path).names) != (*RAW_COLUMNS[source], "source_row_uid"):
        raise ValueError(f"source schema drift for {source}")
    if file_sha256(path.with_name("extraction_guidance.json")) != spec["guidance_sha256"]:
        raise ValueError(f"guidance digest drift for {source}")


def endpoint_inventory(source, values, *, strict):
    endpoints = sorted({clean_literal_text(value) or "" for value in values})
    return {"source_id": source, "count": len(endpoints), "coverage": 1.0, "mapping_coverage": 1.0,
            "strict_source_snapshot": strict, "endpoint_mapping_applied": False, "off_schema_values": [],
            "excluded_values": [], "canonical_endpoints": [], "endpoints": endpoints}


def clean_source_values(records, args):
    del args
    routed = attach_stage1_routes(records, task=TASK_ID)
    return SourceValueCleaningResult(
        records=routed,
        audit_rows=[],
        manifest={
            "version": "carcinogens_source_preservation.v1",
            "n_input_records": len(records),
            "n_output_records": len(routed),
            "n_repaired_records": 0,
            "n_dropped_records": 0,
            "audit_counts": {},
            "scientific_fields_changed": False,
            "endpoint_mapping_applied": False,
            "raw_endpoint_values_preserved": True,
        },
        input_paths=(SOURCE_MANIFEST_PATH,),
    )


def build_hooks(args: argparse.Namespace) -> NormalizationHooks:
    return build_standard_hooks(
        args,
        task_id=TASK_ID,
        contract=CONTRACT,
        mapping_version=PAIR_MAPPING_VERSION,
        output_fields_by_source={
            source: ("canonical_endpoint_concept", *PAIR_DIMENSION_INPUTS[source])
            for source in CONTRACT.sources
        },
        reference_config=REFERENCE_SEMANTICS_CONFIG,
    )


def stage_documents(
    *,
    args: argparse.Namespace,
    hooks: NormalizationHooks,
    normalized: list[dict[str, Any]],
    persisted: list[dict[str, Any]],
    unit_policy_manifest: dict[str, Any],
) -> StageDocuments:
    del args
    return build_standard_stage_documents(
        task_id=TASK_ID,
        contract=CONTRACT,
        hooks=hooks,
        normalized=normalized,
        persisted=persisted,
        unit_policy_manifest=unit_policy_manifest,
    )


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage1-protected-voter-contract",
        default="data/gold_labels/Carcinogens/v1/scaffold/voter_membership.parquet",
        help="Frozen Gold-v1 physical membership used only for Stage-1 UID protection.",
    )
    selected = _selected_auxiliary_mapping()
    parser.add_argument(
        "--auxiliary-mapping", default=str(selected) if selected else ""
    )
    reference = _selected_reference_mapping() or DEFAULT_REFERENCE_SEMANTICS_MAPPING
    parser.add_argument("--reference-semantics-mapping", default=str(reference))
    parser.add_argument("--allow-missing-reference-semantics", action="store_true")


def validate_arguments(parser: argparse.ArgumentParser, args):
    validate_mapping_hashes()
    if args.through_stage in {"source", "clean"}:
        return
    validate_registered_mapping(
        parser,
        args,
        task_name="Carcinogens",
        selected=_selected_auxiliary_mapping(),
        selected_reference=_selected_reference_mapping(),
    )


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    del complete
    return {
        "evidence_library_version": "carcinogens_normalized_v10",
        "carcinogens_source_snapshot_version": SOURCE_MANIFEST["version"],
        "measurement_routing_version": MEASUREMENT_ROUTING_VERSION,
        "stage2_status": "implemented_requires_reviewed_auxiliary_mapping",
    }


POLICY = StarlingTaskPolicy(
    task_id=TASK_ID, dataset_name="Starling Carcinogens six-layer evidence",
    default_data_dir=DEFAULT_DATA_DIR, default_out_dir=DEFAULT_OUT_DIR,
    compact=CompactArtifactProfile(TASK_ID, "carcinogens.compact_stage1.v1", "carcinogens.index_not_built.v1", "Starling Carcinogens evidence", RAW_COLUMNS, CONTRACT.source_projection),
    expected_source_rows={source: spec["rows"] for source, spec in SPECS.items()},
    source_profiles=source_profiles, endpoint_inventory=endpoint_inventory,
    family_resolver=no_family, build_hooks=build_hooks,
    attach_source_columns=lambda rows: rows,
    stage_documents=stage_documents, manifest_versions=manifest_versions,
    record_contract=CONTRACT, validate_arguments=validate_arguments, verify_source_digest=verify_source_digest,
    source_value_cleaner=clean_source_values,
    add_cli_arguments=add_cli_arguments,
    stage1_measurement_routing_enabled=True,
    scientific_assets=(
        *TASK_IMPLEMENTATION_PATHS,
        SOURCE_MANIFEST_PATH,
        *tuple(
            Path(DEFAULT_DATA_DIR, spec["directory"], "extraction_guidance.json")
            for spec in SPECS.values()
        ),
        REGISTRY_PATH,
        mapping_path("measurement_resolution"),
        mapping_path("exact_measurement_units"),
        MAIN_UNIVERSE_MANIFEST,
        MAIN_UNIVERSE_RECORDS,
        *((SELECTED_AUXILIARY_MAPPING,) if SELECTED_AUXILIARY_MAPPING else ()),
        REFERENCE_SEMANTICS_CONFIG.prompt_registry_path,
        *((SELECTED_REFERENCE_MAPPING,) if SELECTED_REFERENCE_MAPPING else ()),
    ),
    source_universe_mapping=level_mapping_path("carcinogens", "v2"),
    source_uid_universe_records=MAIN_UNIVERSE_RECORDS,
    source_uid_universe_manifest=MAIN_UNIVERSE_MANIFEST,
    protected_voter_membership=Path(
        "data/gold_labels/Carcinogens/v1/scaffold/voter_membership.parquet"
    ),
    measurement_resolution_enabled=True,
    exact_unit_mapping_path=mapping_path("exact_measurement_units"),
    reference_semantics_enabled=True,
)
