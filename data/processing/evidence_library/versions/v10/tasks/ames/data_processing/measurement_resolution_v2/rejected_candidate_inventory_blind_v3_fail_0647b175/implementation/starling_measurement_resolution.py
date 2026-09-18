"""Frozen single-outcome measurement extraction for the four Ames sources."""

from __future__ import annotations

import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.measurement_routing import (
    SourceRoutingRules,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_endpoint_normalization import (
    canonical_endpoint_name as _canonical_endpoint_name,
)
from data.processing.paths import evidence_library_root

TASK_ROOT = Path(__file__).resolve().parent
TASK_ASSET_ROOT = TASK_ROOT / "data_processing"
TASK_PROMPT_ROOT = TASK_ROOT / "prompts"

PROMPT_VERSION = "ames_measurement_resolution_prompt.v2"
MAPPING_VERSION = "ames_measurement_resolution.v2"
MAX_MEASUREMENTS_PER_ROW = 1
BATCH_SIZE = 20
STRATIFY_BATCHES = True
ALLOW_REBATCH_UNATTEMPTED = True
REQUIRE_SOURCE_ROW_UID = True
REASONING_EFFORT = "low"

DEFAULT_CLEANED_RECORDS = (
    evidence_library_root("ames", "v10") / "01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = DEFAULT_CLEANED_RECORDS
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"
DEFAULT_MAPPING_PATH = (
    TASK_ASSET_ROOT / "measurement_resolution_v2/measurement_resolution.parquet"
)
DEFAULT_BASE_MAPPING_PATH = None
TEMPLATE_PATH = TASK_PROMPT_ROOT / "measurement_resolution_v2.jinja"

SOURCE_IDS = (
    "mutagenicity_outcomes",
    "fixed_mutation",
    "premutagenic_damage",
    "mutagenicity_mechanism",
)

_CONTEXT_FIELDS = {
    "mutagenicity_outcomes": (
        "evidence_basis",
        "test_system",
        "metabolic_activation",
    ),
    "fixed_mutation": (
        "endpoint_class",
        "study_context",
        "biological_test_system",
        "result_call",
    ),
    "premutagenic_damage": (
        "endpoint_class",
        "endpoint_subtype",
        "biological_system",
        "result_status",
    ),
    "mutagenicity_mechanism": (
        "assay_method_and_endpoint",
        "biological_system",
        "mechanism_category",
        "result_direction",
    ),
}


def source_routing_rules() -> dict[str, SourceRoutingRules]:
    """Route numeric source pairs without imposing a sign restriction."""
    return {
        source_id: SourceRoutingRules(
            source_id=source_id,
            unit_field="" if source_id == "mutagenicity_outcomes" else "unit_text",
            require_positive_value=False,
        )
        for source_id in SOURCE_IDS
    }


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    """Return reviewed model-visible fields; dose and extra details stay hidden."""
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    fields = ["endpoint_name", "measurement_text"]
    if source_id != "mutagenicity_outcomes":
        fields.append("unit_text")
    fields.append("support_text")
    return (*fields, *_CONTEXT_FIELDS[source_id])


def canonical_endpoint_name(source_id: str, endpoint_name: object) -> str:
    """Resolve only reviewed endpoint identities and refuse exclusions."""
    endpoint = _canonical_endpoint_name(source_id, endpoint_name)
    if not endpoint:
        raise ValueError(
            f"reviewed endpoint exclusion reached extraction: {source_id!r}, "
            f"{endpoint_name!r}"
        )
    return endpoint


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TASK_PROMPT_ROOT)),
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
    """Render the frozen Ames prompt for one source."""
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unknown source_id={source_id!r}")
    return (
        _environment()
        .get_template(TEMPLATE_PATH.name)
        .render(
            source_id=source_id,
            batch_size=batch_size,
            row_fields=list(prompt_row_fields(source_id)),
            has_unit_column=source_id != "mutagenicity_outcomes",
            endpoint_profiles=endpoint_profiles,
        )
    )


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:
    """Pin source payloads and every rendered prompt to the template digest."""
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(TEMPLATE_PATH),
        "template_sha256": file_sha256(TEMPLATE_PATH),
        "batch_size": batch_size,
        "maximum_measurements_per_row": MAX_MEASUREMENTS_PER_ROW,
        "source_row_fields": {
            source_id: list(prompt_row_fields(source_id)) for source_id in SOURCE_IDS
        },
        "rendered_sha256": {
            source_id: hashlib.sha256(
                render_prompt(source_id, batch_size=batch_size).encode("utf-8")
            ).hexdigest()
            for source_id in SOURCE_IDS
        },
    }


def validate_mapping_provenance(mapping_path: str | Path) -> None:
    """Validate the compiled candidate mapping before Stage 2 consumes it."""
    from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.compile_candidate_resolution import (
        validate_compiled_mapping,
    )

    validate_compiled_mapping(Path(mapping_path))


__all__ = [
    "ALLOW_REBATCH_UNATTEMPTED",
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
    "canonical_endpoint_name",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "source_routing_rules",
    "validate_mapping_provenance",
]
