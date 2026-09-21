"""DILI row-level reference-semantics policy."""

from pathlib import Path

from data.processing.evidence_library.versions.v10.standard_reference_semantics import (
    PROMPT_VERSION,
    standard_reference_semantics_config,
)


MAPPING_VERSION = "dili_reference_semantics.v1"
DEFAULT_MAPPING_PATH = Path(__file__).parent / (
    "data_processing/reference_semantics_v1/reference_semantics.parquet"
)
REFERENCE_SEMANTICS_CONFIG = standard_reference_semantics_config(
    task_id="dili",
    source_ids=("dili_base", "dili_v1", "dili_v2", "dili_v3", "dili_v4", "dili_v5"),
    mapping_path=DEFAULT_MAPPING_PATH,
)


__all__ = [
    "DEFAULT_MAPPING_PATH",
    "MAPPING_VERSION",
    "PROMPT_VERSION",
    "REFERENCE_SEMANTICS_CONFIG",
]
