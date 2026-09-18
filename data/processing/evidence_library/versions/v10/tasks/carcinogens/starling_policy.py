"""Source-faithful Carcinogens v10 Stage 0-1 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from data.processing.evidence_library.compact_artifacts import CompactArtifactProfile
from data.processing.evidence_library.shared.v2.canonicalization_v7 import PairBucketSpec, SourceProfile, StarlingRecordContract
from data.processing.evidence_library.shared.v2.normalization.cleaning import clean_literal_text, file_sha256
from data.processing.evidence_library.shared.v2.normalization.contracts import NormalizedSourceProfile
from data.processing.evidence_library.shared.v2.normalization.source_value_cleaning import (
    SourceValueCleaningResult,
)
from data.processing.evidence_library.shared.v2.normalization.task_policy import StarlingTaskPolicy
from data.processing.evidence_library.versions.v10.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
    attach_stage1_routes,
)
from data.processing.gold_labels.level_mappings import level_mapping_path
from data.processing.paths import evidence_library_root, raw_starling_task_root

TASK_ID = "carcinogens"
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
SOURCES = {
    source: SourceProfile(
        source_id=source, source_columns=columns, endpoint_field=ROLES[source][0],
        measurement_field=ROLES[source][1], unit_field=ROLES[source][2],
        smiles_field="SMILES", structure_mode="direct",
    )
    for source, columns in RAW_COLUMNS.items()
}
CONTRACT = StarlingRecordContract(TASK_ID, SOURCES, {source: PairBucketSpec(source, ()) for source in SOURCES})
DEFAULT_DATA_DIR = str(raw_starling_task_root(TASK_ID))
DEFAULT_OUT_DIR = str(evidence_library_root(TASK_ID, "v10"))
TASK_IMPLEMENTATION_PATHS = (
    TASK_ROOT / "build_normalized_starling_evidence_library.py",
    TASK_ROOT / "starling_measurement_resolution.py",
    TASK_ROOT / "starling_policy.py",
    V10_ROOT / "build_normalized_evidence_library.py",
    V10_ROOT / "measurement_routing.py",
)


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


def unavailable(*args, **kwargs):
    raise RuntimeError("Carcinogens v10 currently implements only source and clean stages")


def validate_arguments(parser: argparse.ArgumentParser, args):
    if args.through_stage not in {"source", "clean"}:
        parser.error("Carcinogens v10 currently implements only source and clean stages")


POLICY = StarlingTaskPolicy(
    task_id=TASK_ID, dataset_name="Starling Carcinogens six-layer evidence",
    default_data_dir=DEFAULT_DATA_DIR, default_out_dir=DEFAULT_OUT_DIR,
    compact=CompactArtifactProfile(TASK_ID, "carcinogens.compact_stage1.v1", "carcinogens.index_not_built.v1", "Starling Carcinogens evidence", RAW_COLUMNS, CONTRACT.source_projection),
    expected_source_rows={source: spec["rows"] for source, spec in SPECS.items()},
    source_profiles=source_profiles, endpoint_inventory=endpoint_inventory,
    family_resolver=unavailable, build_hooks=unavailable,
    attach_source_columns=lambda rows: rows,
    stage_documents=unavailable, manifest_versions=lambda **_: {
        "evidence_library_version": "carcinogens_normalized_v10",
        "carcinogens_source_snapshot_version": SOURCE_MANIFEST["version"],
        "measurement_routing_version": MEASUREMENT_ROUTING_VERSION,
        "stage2_status": "not_implemented"},
    record_contract=CONTRACT, validate_arguments=validate_arguments, verify_source_digest=verify_source_digest,
    source_value_cleaner=clean_source_values,
    stage1_measurement_routing_enabled=True,
    scientific_assets=(
        *TASK_IMPLEMENTATION_PATHS,
        SOURCE_MANIFEST_PATH,
        *tuple(
            Path(DEFAULT_DATA_DIR, spec["directory"], "extraction_guidance.json")
            for spec in SPECS.values()
        ),
    ),
    source_universe_mapping=level_mapping_path("carcinogens", "v1"),
)
