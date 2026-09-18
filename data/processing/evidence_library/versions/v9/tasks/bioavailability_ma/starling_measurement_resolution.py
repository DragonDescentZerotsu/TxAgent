"""Single-outcome measurement extraction across all five Oral Bio sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.evidence_library.versions.v9.measurement_routing import (
    NO_DIGIT_RULE_ID,
    RouteDecision,
    SourceRoutingRules,
    has_digit,
)
from data.processing.evidence_library.versions.v9.numeric_syntax import (
    finite_point_text,
)
from data.processing.evidence_library.versions.v9.prompts import PROMPT_ROOT
from data.processing.evidence_library.shared.v2.normalization.measurements import (
    canonicalize_endpoint,
)
from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    spacing_and_spelling_endpoint,
)


TASK_ROOT = Path(__file__).resolve().parent

PROMPT_VERSION = "bioavailability_measurement_resolution_prompt.v4"
MAPPING_VERSION = "bioavailability_ma_measurement_resolution.v4"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 10
STRATIFY_BATCHES = True
ALLOW_REBATCH_UNATTEMPTED = True
MAX_UNMETERED_ATTEMPTS_PER_ROW = 3
REQUIRE_SOURCE_ROW_UID = True

DEFAULT_CLEANED_RECORDS = Path(
    "data/evidence_libraries/bioavailability_ma/v9/01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = Path(
    "data/evidence_libraries/bioavailability_ma/v9/01_cleaned/records.parquet"
)
DEFAULT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/measurement_resolution_v4/measurement_resolution.parquet"
)
DEFAULT_BASE_MAPPING_PATH = None
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"
TEMPLATE_DIR = PROMPT_ROOT / "measurement_resolution"
TEMPLATE_NAME = "bioavailability_v4.jinja"

SOURCE_IDS = (
    "oral_exposure",
    "fa",
    "fg",
    "fh",
    "hf_bioavailability",
)
_EMBEDDED_UNIT_SOURCES = frozenset({"fg", "hf_bioavailability"})


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
    return tuple(fields)


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
        return RouteDecision(
            "accept", "finite_point_with_separate_unit.v1", point[0], unit
        )
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
    "MAX_MEASUREMENTS_PER_ROW",
    "PROMPT_VERSION",
    "SOURCE_IDS",
    "canonical_endpoint_name",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "route_measurement",
    "source_routing_rules",
]
