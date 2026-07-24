"""Task-agnostic Starling source adapters."""

from .evidence_library import (
    StarlingSourceProfile,
    build_and_write_starling_index,
    build_starling_parquet_evidence_rows,
    starling_molecule_id,
    write_jsonl,
)

__all__ = [
    "StarlingSourceProfile",
    "build_and_write_starling_index",
    "build_starling_parquet_evidence_rows",
    "starling_molecule_id",
    "write_jsonl",
]
