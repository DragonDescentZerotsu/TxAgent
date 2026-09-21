"""Carcinogens row-level reference-semantics policy."""

from pathlib import Path

from data.processing.evidence_library.versions.v10.standard_reference_semantics import (
    PROMPT_VERSION,
    standard_reference_semantics_config,
)


MAPPING_VERSION = "carcinogens_reference_semantics.v1"
DEFAULT_MAPPING_PATH = Path(__file__).parent / (
    "data_processing/reference_semantics_v1/reference_semantics.parquet"
)
REFERENCE_SEMANTICS_CONFIG = standard_reference_semantics_config(
    task_id="carcinogens",
    source_ids=(
        "carcinogens_base",
        "carcinogens_v1",
        "carcinogens_v2",
        "carcinogens_v3",
        "carcinogens_v4",
        "carcinogens_v5",
    ),
    mapping_path=DEFAULT_MAPPING_PATH,
)


__all__ = [
    "DEFAULT_MAPPING_PATH",
    "MAPPING_VERSION",
    "PROMPT_VERSION",
    "REFERENCE_SEMANTICS_CONFIG",
]
