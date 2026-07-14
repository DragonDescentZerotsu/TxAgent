"""Task-agnostic Starling source adapters."""

from .evidence_library import (
    StarlingSourceProfile,
    build_starling_parquet_evidence_rows,
    write_jsonl,
)

__all__ = [
    "StarlingSourceProfile",
    "build_starling_parquet_evidence_rows",
    "write_jsonl",
]
