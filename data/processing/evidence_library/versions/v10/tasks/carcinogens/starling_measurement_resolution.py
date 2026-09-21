"""Carcinogens V10 deterministic routing and one-value model extraction."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.versions.v10.measurement_routing import (
    RouteDecision,
    SourceRoutingRules,
    has_digit,
    is_obvious_unit,
    source_role_contract,
)
from data.processing.evidence_library.versions.v10.numeric_syntax import finite_point_text
from data.processing.paths import REPO_ROOT, evidence_library_root
from data.processing.evidence_library.versions.v10.tasks.carcinogens.mapping_registry import (
    mapping_path,
)


TASK_ID = "carcinogens"
TASK_ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = TASK_ROOT / "prompts/measurement_resolution_v4.jinja"

PROMPT_VERSION = "carcinogens_measurement_resolution_prompt.v4"
MAPPING_VERSION = "carcinogens_measurement_resolution.v4"
RULE_POLICY_VERSION = "carcinogens_measurement_resolution_rules.v1"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 20
STRATIFY_BATCHES = True
ALLOW_REBATCH_UNATTEMPTED = True
REQUIRE_SOURCE_ROW_UID = True
AUTO_VALIDATE_GENERATED_MAPPING = True
REASONING_EFFORT = "low"
TEMPERATURE = 0.0

DEFAULT_CLEANED_RECORDS = (
    evidence_library_root(TASK_ID, "v10") / "01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = DEFAULT_CLEANED_RECORDS
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"
DEFAULT_MAPPING_PATH = mapping_path("measurement_resolution")
DEFAULT_BASE_MAPPING_PATH = None
EXACT_UNIT_MAPPING_PATH = mapping_path("exact_measurement_units")
ENFORCE_EXACT_UNITS_DURING_EXTRACTION = False
DEFAULT_GOLD_FIXTURE = (
    REPO_ROOT
    / "tests/chembl_tool/common/measurement_resolution_quality/gold/carcinogens.v10.jsonl"
)

SOURCE_IDS = (
    "carcinogens_base",
    "carcinogens_v1",
    "carcinogens_v2",
    "carcinogens_v3",
    "carcinogens_v4",
    "carcinogens_v5",
)

_CONTEXT_FIELDS = {
    "carcinogens_base": (
        "carcinogenicity_conclusion",
        "evidence_basis",
        "evidence_scope",
        "classification_label",
    ),
    "carcinogens_v1": (
        "assay_method",
        "biological_test_system",
        "endpoint_measure",
        "effect_direction",
    ),
    "carcinogens_v2": (
        "test_system",
        "endpoint_detail",
        "result_interpretation",
        "metabolic_activation",
    ),
    "carcinogens_v3": (
        "assay_and_detection_method",
        "endpoint",
        "effect_direction",
        "biological_system",
    ),
    "carcinogens_v4": (
        "assay_type",
        "endpoint",
        "effect_direction",
        "biological_test_system",
    ),
    "carcinogens_v5": (
        "assay_type",
        "endpoint_and_method",
        "effect_direction",
        "interpretation_conditions",
    ),
}
UNIT_RECONCILIATION_CONTEXT_FIELDS = _CONTEXT_FIELDS


def source_routing_rules() -> dict[str, SourceRoutingRules]:
    return {
        source_id: SourceRoutingRules(source_id=source_id, require_positive_value=False)
        for source_id in SOURCE_IDS
    }


def route_measurement(record: Mapping[str, Any]) -> RouteDecision:
    """Copy only a whole finite point with a reviewed complete physical unit."""
    measurement = record.get("measurement_text")
    if not has_digit(measurement):
        return RouteDecision("reject", "no_digit_in_measurement_column.v1")
    parsed = finite_point_text(measurement)
    unit = str(record.get("unit_text") or "").strip()
    if parsed is None or parsed[1] is not None or not is_obvious_unit(unit, task=TASK_ID):
        return RouteDecision("extract")
    value = Decimal(parsed[0])
    if value <= 0:
        return RouteDecision("extract")
    return RouteDecision(
        "accept", "finite_point_with_complete_physical_unit.v1", parsed[0], unit
    )


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    fields = ("endpoint_name", "measurement_text")
    if source_id != "carcinogens_base":
        fields += ("unit_text",)
    return (*fields, "support_text", *_CONTEXT_FIELDS[source_id])


def canonical_endpoint_name(source_id: str, endpoint_name: object) -> str:
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    return str(endpoint_name or "missing_endpoint").strip() or "missing_endpoint"


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH.parent)),
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
    del endpoint_profiles
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    return _environment().get_template(TEMPLATE_PATH.name).render(
        batch_size=batch_size,
        row_fields=list(prompt_row_fields(source_id)),
        has_unit_column=source_id != "carcinogens_base",
    )


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(TEMPLATE_PATH),
        "template_sha256": file_sha256(TEMPLATE_PATH),
        "batch_size": batch_size,
        "maximum_measurements_per_row": MAX_MEASUREMENTS_PER_ROW,
        "routing_rule_policy_version": RULE_POLICY_VERSION,
        "source_row_fields": {
            source_id: list(prompt_row_fields(source_id)) for source_id in SOURCE_IDS
        },
        "rendered_sha256": {
            source_id: hashlib.sha256(
                render_prompt(source_id, batch_size=batch_size).encode()
            ).hexdigest()
            for source_id in SOURCE_IDS
        },
    }


def source_role_manifest() -> dict[str, object]:
    from data.processing.evidence_library.versions.v10.tasks.carcinogens.starling_policy import (
        CONTRACT,
        UNIT_EXCEPTIONS,
    )

    return source_role_contract(CONTRACT.sources, UNIT_EXCEPTIONS)


def validate_mapping_provenance(
    mapping_path: str | Path,
    *,
    expected_record_ids: set[str] | None = None,
) -> None:
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        validate_full_mapping_provenance,
    )

    validate_full_mapping_provenance(
        mapping_path, task=TASK_ID, expected_record_ids=expected_record_ids
    )


__all__ = [
    "ALLOW_REBATCH_UNATTEMPTED",
    "AUTO_VALIDATE_GENERATED_MAPPING",
    "BATCH_SIZE",
    "DEFAULT_BASE_MAPPING_PATH",
    "DEFAULT_CANONICAL_RECORDS",
    "DEFAULT_CLEANED_RECORDS",
    "DEFAULT_MAPPING_PATH",
    "DEFAULT_PROFILE_PATH",
    "MAPPING_VERSION",
    "MAX_MEASUREMENTS_PER_ROW",
    "PROMPT_VERSION",
    "REASONING_EFFORT",
    "REQUIRE_SOURCE_ROW_UID",
    "SOURCE_IDS",
    "STRATIFY_BATCHES",
    "UNIT_RECONCILIATION_CONTEXT_FIELDS",
    "canonical_endpoint_name",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "route_measurement",
    "source_routing_rules",
    "source_role_manifest",
    "validate_mapping_provenance",
]
