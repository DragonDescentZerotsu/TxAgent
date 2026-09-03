"""Compatibility exports for the refactored model and branch payload modules."""

from predict.harnesses.branches.artifacts import read_jsonl_record, write_trace_jsonl
from predict.harnesses.branches.payload import (
    attach_external_condition,
    clean_exact_match,
    clean_shared_assay_context,
    llm_evidence_query_payload,
    llm_query_payload,
)
from predict.llm_engine.pool import load_env_file as _load_env_file
from predict.llm_io.query import (
    EXTERNAL_CONDITION_RENDERER_VERSION,
    NO_REPORTED_EXTERNAL_CONDITION,
    external_condition_sentence,
)


def load_env_file(path) -> None:
    """Preserve the historical rule that an explicit env file overrides values."""
    _load_env_file(path, override=True)
