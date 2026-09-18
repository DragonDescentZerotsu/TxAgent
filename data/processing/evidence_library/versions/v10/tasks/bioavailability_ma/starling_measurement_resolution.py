"""Single-outcome measurement extraction across all five Oral Bio sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.evidence_library.versions.v10.measurement_routing import (
    NO_DIGIT_RULE_ID,
    RouteDecision,
    SourceRoutingRules,
    has_digit,
    source_role_contract,
)
from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.versions.v10.numeric_syntax import (
    finite_point_text,
)
from data.processing.evidence_library.versions.v10.prompts import PROMPT_ROOT
from data.processing.evidence_library.shared.v2.normalization.measurements import (
    canonicalize_endpoint,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    spacing_and_spelling_endpoint,
)
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.mapping_registry import (
    mapping_path,
)


TASK_ROOT = Path(__file__).resolve().parent

PROMPT_VERSION = "bioavailability_measurement_resolution_prompt.v7_defined_parameters"
MAPPING_VERSION = "bioavailability_ma_measurement_resolution.v5"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 10
STRATIFY_BATCHES = True
ALLOW_REBATCH_UNATTEMPTED = True
MAX_UNMETERED_ATTEMPTS_PER_ROW = 3
REQUIRE_SOURCE_ROW_UID = True

DEFAULT_CLEANED_RECORDS = Path(
    "data/evidence_libraries/bioavailability_ma/v10/01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = Path(
    "data/evidence_libraries/bioavailability_ma/v10/01_cleaned/records.parquet"
)
DEFAULT_MAPPING_PATH = mapping_path("measurement_resolution")
MAPPING_PROVENANCE_PATH = (
    TASK_ROOT / "data_processing/measurement_resolution_v5/manifest.json"
)
DEFAULT_BASE_MAPPING_PATH = None
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"
TEMPLATE_DIR = PROMPT_ROOT / "measurement_resolution"
TEMPLATE_NAME = "bioavailability_v7.jinja"

SOURCE_IDS = (
    "oral_exposure",
    "fa",
    "fg",
    "fh",
    "hf_bioavailability",
)
_EMBEDDED_UNIT_SOURCES = frozenset({"fg", "hf_bioavailability"})
UNIT_EXCEPTIONS = {
    source_id: {
        "mode": "embedded_in_measurement",
        "reason": f"{field} preserves its source unit inline",
    }
    for source_id, field in {
        "fg": "measured_value",
        "hf_bioavailability": "oral_bioavailability_value",
    }.items()
}


def source_role_manifest() -> dict[str, object]:
    from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.starling_schema import (
        RECORD_CONTRACT,
    )

    return source_role_contract(RECORD_CONTRACT.sources, UNIT_EXCEPTIONS)


def validate_mapping_provenance(mapping_path: str | Path = DEFAULT_MAPPING_PATH) -> None:
    """Validate the no-inference v4-to-v5 provenance binding."""
    mapping_path = Path(mapping_path)
    generated_manifest_path = mapping_path.with_suffix(".manifest.json")
    generated = json.loads(generated_manifest_path.read_text(encoding="utf-8"))
    if generated.get("generation_version") == "stage1_measurement_resolution_replay.v1":
        mismatches = {}
        if generated.get("task_id") != "bioavailability_ma":
            mismatches["task_id"] = generated.get("task_id")
        if generated.get("mapping_version") != MAPPING_VERSION:
            mismatches["mapping_version"] = generated.get("mapping_version")
        if generated.get("mapping_sha256") != file_sha256(mapping_path):
            mismatches["mapping_sha256"] = generated.get("mapping_sha256")
        for name, source in (generated.get("sources") or {}).items():
            source_path = Path(str(source.get("path") or ""))
            if not source_path.is_file() or source.get("sha256") != file_sha256(
                source_path
            ):
                mismatches[name] = "missing or hash-mismatched"
        if not all((generated.get("validations") or {}).values()):
            mismatches["validations"] = generated.get("validations")
        if mismatches:
            raise ValueError(
                f"Oral measurement replay provenance mismatch: {mismatches}"
            )
        return
    if generated.get("generation_version") == "measurement_resolution_generation.v8":
        source_path = Path(str(generated.get("cleaned_records_path") or ""))
        base = generated.get("base_mapping") or {}
        base_path = Path(str(base.get("path") or ""))
        delta = generated.get("delta_inference") or {}
        mismatches = {}
        if generated.get("task_id") != "bioavailability_ma":
            mismatches["task_id"] = generated.get("task_id")
        if generated.get("mapping_version") != MAPPING_VERSION:
            mismatches["mapping_version"] = generated.get("mapping_version")
        if generated.get("mapping_sha256") != file_sha256(mapping_path):
            mismatches["mapping_sha256"] = generated.get("mapping_sha256")
        if (
            not source_path.is_file()
            or generated.get("cleaned_records_sha256") != file_sha256(source_path)
        ):
            mismatches["cleaned_records"] = "missing or hash-mismatched"
        if not base_path.is_file() or base.get("sha256") != file_sha256(base_path):
            mismatches["base_mapping"] = "missing or hash-mismatched"
        if delta.get("models") != ["gpt-5.4-mini"]:
            mismatches["delta_inference"] = delta
        if (generated.get("inference") or {}).get("reasoning_mode") != "high":
            mismatches["reasoning_effort"] = generated.get("inference")
        for key in (
            "one_row_per_candidate",
            "unique_cleaned_record_ids",
            "only_ok_carries_measurements",
            "maximum_measurements_per_row",
        ):
            if (generated.get("validations") or {}).get(key) is not True:
                mismatches[key] = "validation did not pass"
        if mismatches:
            raise ValueError(f"Oral measurement successor provenance mismatch: {mismatches}")
        return
    manifest = json.loads(MAPPING_PROVENANCE_PATH.read_text(encoding="utf-8"))
    expected = {
        "version": MAPPING_VERSION,
        "migration": "validated_role_declaration_without_reinference",
        "decision_mapping_version": "bioavailability_ma_measurement_resolution.v4",
        "mapping_sha256": file_sha256(mapping_path),
        "predecessor_manifest_sha256": file_sha256(
            Path(mapping_path).with_suffix(".manifest.json")
        ),
        "source_role_contract": source_role_manifest(),
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Oral measurement provenance mismatch: {mismatches}")


def source_routing_rules() -> dict[str, SourceRoutingRules]:
    """Route only from the source measurement/unit columns.

    No qualitative context column is a safe reject rule: indirect studies often
    report a scalar and a qualitative interpretation in the same row.
    """
    return {
        source_id: SourceRoutingRules(
            source_id=source_id,
            unit_field="" if source_id in _EMBEDDED_UNIT_SOURCES else "unit_text",
            require_positive_value=False,
        )
        for source_id in SOURCE_IDS
    }


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    fields = ["endpoint_name", "measurement_text"]
    if source_id not in _EMBEDDED_UNIT_SOURCES:
        fields.append("unit_text")
    fields.append("support_text")
    context = {
        'hf_bioavailability': ('bioavailability_report_type', 'comparator', 'oral_exposure_mode', 'species_or_population'),
        'oral_exposure': ('statistic_type', 'oral_dose', 'study_context', 'comparator_exposure', 'qualifying_conditions'),
        'fa': ('assay_system', 'condition_medium', 'biological_context', 'formulation_or_solid_form', 'qualifying_conditions'),
        'fg': ('transporter_or_enzyme', 'substrate_status', 'assay_system', 'intestinal_site', 'qualifying_conditions'),
        'fh': ('assay_system', 'species', 'molecular_form', 'enzyme_or_pathway', 'qualifying_conditions'),
    }
    return (*fields, *context[source_id])


def route_measurement(record: Mapping[str, Any]) -> RouteDecision:
    """Accept a finite point with any supplied separate unit."""
    rules = source_routing_rules()[str(record.get("source_id") or "")]
    point = finite_point_text(record.get("measurement_text"))
    if point is None:
        if has_digit(record.get("measurement_text")):
            return RouteDecision("extract")
        return RouteDecision("reject", NO_DIGIT_RULE_ID)
    unit_value = record.get(rules.unit_field) if rules.unit_field else None
    unit = "" if unit_value is None else str(unit_value).strip()
    if unit:
        from data.processing.evidence_library.versions.v10.percentage_delta import review_percentage
        return review_percentage(record, RouteDecision(
            "accept", "finite_point_with_separate_unit.v1", point[0], unit
        ))
    return RouteDecision("extract")


def canonical_endpoint_name(source_id: str, endpoint_name: object) -> str:
    reviewed = spacing_and_spelling_endpoint(source_id, str(endpoint_name or ""))
    return canonicalize_endpoint(reviewed)


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_prompt(
    source_id: str,
    *,
    batch_size: int = BATCH_SIZE,
    endpoint_profiles: tuple[str, ...] = (),
) -> str:
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    return _environment().get_template(TEMPLATE_NAME).render(
        source_id=source_id,
        batch_size=batch_size,
        row_fields=list(prompt_row_fields(source_id)),
        has_unit_column=source_id not in _EMBEDDED_UNIT_SOURCES,
    )


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    template_path = TEMPLATE_DIR / TEMPLATE_NAME
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(template_path),
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "batch_size": batch_size,
        "rendered_sha256": {
            source_id: hashlib.sha256(
                render_prompt(source_id, batch_size=batch_size).encode("utf-8")
            ).hexdigest()
            for source_id in SOURCE_IDS
        },
    }


__all__ = [
    "BATCH_SIZE",
    "DEFAULT_CANONICAL_RECORDS",
    "DEFAULT_CLEANED_RECORDS",
    "DEFAULT_MAPPING_PATH",
    "DEFAULT_PROFILE_PATH",
    "MAPPING_VERSION",
    "MAPPING_PROVENANCE_PATH",
    "MAX_MEASUREMENTS_PER_ROW",
    "PROMPT_VERSION",
    "SOURCE_IDS",
    "canonical_endpoint_name",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "route_measurement",
    "source_role_manifest",
    "source_routing_rules",
    "validate_mapping_provenance",
]
