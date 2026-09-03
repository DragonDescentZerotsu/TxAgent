"""Stage 1 policy for the four-layer Ames V8 source snapshot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from data.processing.paths import evidence_library_root, raw_starling_task_root
from data.processing.evidence_library.compact_artifacts import CompactArtifactProfile
from data.processing.evidence_library.versions.v8.measurement_routing import (
    MEASUREMENT_ROUTING_VERSION,
    attach_stage1_routes,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v1.normalization.contracts import (
    NormalizedSourceProfile,
)
from data.processing.evidence_library.shared.v1.normalization.source_value_cleaning import (
    SourceValueCleaningResult,
)
from data.processing.evidence_library.shared.v1.normalization.task_policy import (
    StarlingTaskPolicy,
)
from data.processing.evidence_library.versions.v8.tasks.ames.starling_categorical_response import (
    CATEGORICAL_RESPONSE_VERSION,
)
from data.processing.evidence_library.versions.v8.tasks.ames.starling_schema import (
    RECORD_CONTRACT,
    ROLE_FIELDS,
    SOURCE_COLUMNS,
    TASK_ID,
)


TASK_ROOT = Path(__file__).resolve().parent
SOURCE_MANIFEST_PATH = TASK_ROOT / "source_manifest.json"
SOURCE_MANIFEST = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
SOURCE_SPECS = SOURCE_MANIFEST["sources"]

DATASET_NAME = "SEND/Ames four-layer evidence"
DEFAULT_DATA_DIR = str(raw_starling_task_root(TASK_ID) / "send")
DEFAULT_OUT_DIR = str(evidence_library_root(TASK_ID, "v8"))
EXPECTED_SOURCE_ROWS = {
    source_id: int(spec["rows"]) for source_id, spec in SOURCE_SPECS.items()
}
EXPECTED_SOURCE_SHA256 = {
    source_id: str(spec["parquet_sha256"])
    for source_id, spec in SOURCE_SPECS.items()
}

ASSAY_FAMILIES = {
    "mutagenicity_outcomes": frozenset(
        {
            "bacterial_reverse_mutation",
            "other_microorganism_mutation",
            "mammalian_cell_gene_mutation",
            "in_vivo_somatic_gene_mutation",
            "non_mammalian_eukaryote_mutation",
            "germ_cell_or_heritable_mutation",
            "chromosome_aberration_or_clastogenicity",
            "micronucleus",
            "aneuploidy_or_malsegregation",
            "recombination_or_gene_conversion",
            "other_mutation_assay",
            "multiple_mutation_assays",
            "unspecified_mutagenicity",
        }
    ),
    "fixed_mutation": frozenset(
        {
            "hprt_hgprt_xprt_forward_mutation",
            "tk_mouse_lymphoma",
            "pig_a_mutation",
            "transgenic_rodent_mutation",
            "other_mammalian_forward_mutation",
            "microbial_forward_mutation",
            "non_salmonella_reverse_mutation",
            "yeast_or_fungal_mutation",
            "specific_locus_or_visible_marker",
            "drosophila_mutation",
            "dominant_lethal_test",
            "heritable_translocation_test",
            "micronucleus_assay",
            "chromosome_aberration_assay",
            "sister_chromatid_exchange_assay",
            "recombination_or_gene_conversion_assay",
            "aneuploidy_or_chromosome_malsegregation",
            "polyploidy_or_chromosome_doubling",
            "fish_or_molecular_cytogenetics",
            "sequencing_based_induced_mutation",
            "other_fixed_mutation_assay",
        }
    ),
    "premutagenic_damage": frozenset(
        {
            "dna_adductomics",
            "32p_postlabelling",
            "targeted_dna_adduct_measurement",
            "oxidized_base_measurement",
            "ap_site_or_repair_intermediate_assay",
            "dna_crosslink_assay",
            "comet_assay",
            "strand_break_physical_assay",
            "repair_synthesis_assay",
            "host_cell_reactivation",
            "repair_activity_or_kinetics",
            "bacterial_ddr_reporter",
            "gadd45a_reporter",
            "gamma_h2ax_assay",
            "p53_pathway_assay",
            "multiflow",
            "toxtracker",
            "other_ddr_reporter_or_panel",
            "other_direct_damage_assay",
        }
    ),
    "mutagenicity_mechanism": frozenset(
        {
            "metabolite_formation_or_reaction_phenotyping",
            "reactive_metabolite_trapping",
            "covalent_binding_or_adduct",
            "electrophile_nucleophile_reactivity",
            "detoxification_or_conjugation_assay",
            "metabolic_enzyme_activity_or_inactivation",
            "ros_or_redox_assay",
            "antioxidant_or_thiol_assay",
            "oxidative_damage_assay",
            "dna_polymerase_or_synthesis_assay",
            "dna_fibre_or_replication_fork_assay",
            "dna_repair_activity_assay",
            "repair_deficient_or_proficient_comparison",
            "topoisomerase_cleavage_complex_assay",
            "topoisomerase_catalytic_assay",
            "tubulin_or_microtubule_assay",
            "spindle_or_centrosome_assay",
            "kinetochore_or_chromosome_segregation_assay",
            "mechanistically_informative_genotoxicity",
            "other_mechanistic_assay",
        }
    ),
}


def _source_path(data_dir: Path, source_id: str, filename: str) -> Path:
    return data_dir / SOURCE_SPECS[source_id]["directory"] / filename


def source_profiles(data_dir: Path) -> list[NormalizedSourceProfile]:
    profiles: list[NormalizedSourceProfile] = []
    for source_id, columns in SOURCE_COLUMNS.items():
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
                source_path=str(_source_path(data_dir, source_id, "extractions.parquet")),
                endpoint_field=endpoint,
                measurement_field=measurement,
                unit_field=unit,
                smiles_field="SMILES",
                structure_mode="direct",
                record_id_field="extraction_id",
                name_fields=("molecule_name",) if source_id == "mutagenicity_outcomes" else (),
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
    if columns != SOURCE_COLUMNS[source_id]:
        raise ValueError(
            f"source schema drift for {source_id}: expected={SOURCE_COLUMNS[source_id]!r}, "
            f"found={columns!r}"
        )


def endpoint_inventory(
    source_id: str, endpoints: list[str], *, strict: bool
) -> dict[str, Any]:
    observed = sorted(set(endpoints))
    allowed = ASSAY_FAMILIES[source_id]
    off_schema = sorted(value for value in observed if value and value not in allowed)
    return {
        "source_id": source_id,
        "count": len(observed),
        "coverage": (len(observed) - len(off_schema)) / len(observed) if observed else 1.0,
        "strict_source_snapshot": strict,
        "off_schema_values": off_schema,
        "endpoints": observed,
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


def _audit_source_quality(
    records: list[dict[str, Any]], data_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, int], tuple[Path, ...]]:
    by_source = {
        source_id: [row for row in records if row.get("source_id") == source_id]
        for source_id in SOURCE_COLUMNS
    }
    audit: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    guidance_paths: list[Path] = []

    def add(row: dict[str, Any], audit_type: str, field: str, before: Any) -> None:
        counts[audit_type] += 1
        rendered = None if pd.isna(before) else str(before)
        audit.append(
            {
                "task_id": TASK_ID,
                "source_id": row["source_id"],
                "cleaned_record_id": row["cleaned_record_id"],
                "source_row_number": row["source_row_number"],
                "source_record_id": row["source_record_id"],
                "audit_type": audit_type,
                "field": field,
                "before": rendered,
                "after": None if audit_type == "textual_null" else rendered,
                "action": "preserved" if audit_type != "textual_null" else "normalized_to_null",
            }
        )

    for source_id, cleaned_rows in by_source.items():
        guidance_path, guidance = _guidance(data_dir, source_id)
        guidance_paths.append(guidance_path)
        controlled = {
            item["name"]: frozenset(item.get("allowed_values") or ())
            for item in guidance["schema_spec"]
            if item.get("allowed_values")
        }
        value_field = ROLE_FIELDS[source_id][1]
        unit_field = ROLE_FIELDS[source_id][2]
        columns = sorted(
            {
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
            _source_path(data_dir, source_id, "extractions.parquet"),
            columns=columns,
        )
        # Smoke builds intentionally ingest only a source prefix. Audit that
        # same prefix so row identity and ordering remain directly comparable.
        raw = raw.head(len(cleaned_rows))
        if len(raw) != len(cleaned_rows):
            raise ValueError(f"source/cleaned row mismatch for {source_id}")
        keys = list(zip(raw["pmid"].astype(str), raw["extraction_id"].astype(str)))
        if len(keys) != len(set(keys)):
            raise ValueError(f"{source_id} PMID + extraction_id must be unique")

        for index, raw_row in enumerate(raw.to_dict(orient="records")):
            cleaned = cleaned_rows[index]
            if str(raw_row["extraction_id"]) != cleaned["source_record_id"]:
                raise ValueError(f"source order changed for {source_id}")
            if not _has_value(raw_row["support_text"]):
                add(cleaned, "blank_support_text", "support_text", raw_row["support_text"])
            confidence = raw_row["confidence"]
            if _has_value(confidence) and not 0.0 <= float(confidence) <= 1.0:
                add(cleaned, "confidence_out_of_range", "confidence", confidence)
            for field, allowed in controlled.items():
                value = raw_row[field]
                if not _has_value(value):
                    continue
                text = str(value).strip()
                if text.casefold() in {"nan", "none", "null", "na", "n/a", "-"}:
                    if text not in allowed:
                        add(cleaned, "textual_null", field, value)
                elif text not in allowed:
                    add(cleaned, "off_schema_controlled_value", field, value)
            if unit_field:
                has_value = _has_value(raw_row[value_field])
                has_unit = _has_value(raw_row[unit_field])
                if has_value != has_unit:
                    add(
                        cleaned,
                        "value_without_unit" if has_value else "unit_without_value",
                        unit_field if has_value else value_field,
                        raw_row[unit_field] if has_value else raw_row[value_field],
                    )
    return audit, dict(sorted(counts.items())), tuple(guidance_paths)


def clean_source_values(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> SourceValueCleaningResult:
    audit, audit_counts, guidance_paths = _audit_source_quality(
        records, Path(args.starling_data_dir)
    )
    routed = attach_stage1_routes(records, task=TASK_ID)
    return SourceValueCleaningResult(
        records=routed,
        audit_rows=audit,
        manifest={
            "version": "ames_source_quality_audit.v1",
            "n_input_records": len(records),
            "n_output_records": len(routed),
            "n_repaired_records": 0,
            "n_dropped_records": 0,
            "audit_counts": audit_counts,
            "scientific_fields_changed": False,
        },
        input_paths=(SOURCE_MANIFEST_PATH, *guidance_paths),
    )


def attach_source_columns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return records


def _stage2_unavailable(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("Ames Stage 2 is not implemented")


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.through_stage not in {"source", "clean"}:
        parser.error("Ames currently supports only Stage 0 and Stage 1")


def manifest_versions(*, complete: bool = True) -> dict[str, Any]:
    return {
        "ames_source_snapshot_version": SOURCE_MANIFEST["version"],
        "ames_source_quality_audit_version": "ames_source_quality_audit.v1",
        "categorical_response_version": CATEGORICAL_RESPONSE_VERSION,
        "measurement_routing_version": MEASUREMENT_ROUTING_VERSION,
        "stage2_status": "not_implemented",
    }


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
    scientific_assets=(SOURCE_MANIFEST_PATH, *DEFAULT_GUIDANCE_PATHS),
    stage1_measurement_routing_enabled=True,
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
