"""Low-reasoning successor for the unresolved Ames measurement tail."""

from __future__ import annotations

import hashlib

from data.processing.evidence_library.versions.v10.tasks.ames import (
    starling_measurement_resolution as legacy,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_measurement_resolution import *  # noqa: F403


PROMPT_VERSION = "ames_measurement_resolution_prompt.v7_defined_units"
MAPPING_VERSION = "ames_measurement_resolution_retry.v2"
TEMPLATE_PATH = legacy.TASK_PROMPT_ROOT / "measurement_resolution_v7.jinja"
REASONING_EFFORT = "low"
MAX_SCHEMA_ATTEMPTS = 2
RETRY_VALIDATION_FEEDBACK = True
AUTO_VALIDATE_GENERATED_MAPPING = False


def render_prompt(
    source_id: str,
    *,
    batch_size: int = BATCH_SIZE,  # noqa: F405
    endpoint_profiles: tuple[str, ...] = (),
) -> str:
    if source_id not in SOURCE_IDS:  # noqa: F405
        raise ValueError(f"unknown source_id={source_id!r}")
    return (
        legacy._environment()
        .get_template(TEMPLATE_PATH.name)
        .render(
            source_id=source_id,
            batch_size=batch_size,
            row_fields=list(prompt_row_fields(source_id)),  # noqa: F405
            has_unit_column=source_id != "mutagenicity_outcomes",
            endpoint_profiles=endpoint_profiles,
        )
    )


def prompt_manifest(*, batch_size: int = BATCH_SIZE) -> dict[str, object]:  # noqa: F405
    return {
        "prompt_version": PROMPT_VERSION,
        "template_path": str(TEMPLATE_PATH),
        "template_sha256": legacy.file_sha256(TEMPLATE_PATH),
        "batch_size": batch_size,
        "maximum_measurements_per_row": MAX_MEASUREMENTS_PER_ROW,  # noqa: F405
        "source_row_fields": {
            source_id: list(prompt_row_fields(source_id))  # noqa: F405
            for source_id in SOURCE_IDS  # noqa: F405
        },
        "rendered_sha256": {
            source_id: hashlib.sha256(
                render_prompt(source_id, batch_size=batch_size).encode("utf-8")
            ).hexdigest()
            for source_id in SOURCE_IDS  # noqa: F405
        },
    }
