"""Bioavailability measurement-resolution configuration.

The shared resolver operates on all five physical sources.  The initial gold
corpus deliberately covers the four indirect sources (oral exposure, Fa, Fg,
and Fh); direct HF records retain their existing source semantics but use the
same routing contract when a complete mapping is generated later.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from tools.chembl_tool.common.starling.measurement_routing import SourceRoutingRules
from tools.chembl_tool.common.starling.normalization.measurements import (
    canonicalize_endpoint,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_spacing_and_spelling import (
    spacing_and_spelling_endpoint,
)


TASK_ROOT = Path(__file__).resolve().parent

PROMPT_VERSION = "bioavailability_measurement_resolution_prompt.v3"
MAPPING_VERSION = "bioavailability_ma_measurement_resolution.v1"
BATCH_SIZE = 10

DEFAULT_CLEANED_RECORDS = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v7/01_cleaned/records.parquet"
)
DEFAULT_CANONICAL_RECORDS = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
    "starling_normalized_v7/01_cleaned/records.parquet"
)
DEFAULT_MAPPING_PATH = (
    TASK_ROOT / "data_processing/measurement_resolution_v1/measurement_resolution.parquet"
)
DEFAULT_PROFILE_PATH = DEFAULT_CLEANED_RECORDS.parent / "endpoint_unit_profile.json"
TEMPLATE_DIR = TASK_ROOT / "measurement_resolution_templates"
TEMPLATE_NAME = "measurement_resolution_v1.jinja"

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
        )
        for source_id in SOURCE_IDS
    }


def prompt_row_fields(source_id: str) -> tuple[str, ...]:
    fields = ["endpoint_name", "measurement_text"]
    if source_id not in _EMBEDDED_UNIT_SOURCES:
        fields.append("unit_text")
    fields.append("support_text")
    return tuple(fields)


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
        endpoint_profiles=endpoint_profiles,
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
    "PROMPT_VERSION",
    "SOURCE_IDS",
    "canonical_endpoint_name",
    "prompt_manifest",
    "prompt_row_fields",
    "render_prompt",
    "source_routing_rules",
]
