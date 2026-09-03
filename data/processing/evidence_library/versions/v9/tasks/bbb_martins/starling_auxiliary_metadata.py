"""BBB binding for globally reconciled assay context and species metadata."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.auxiliary_metadata import (
    AUXILIARY_ATTACHMENT_VERSION,
    OUTPUT_FIELDS,
)
from data.processing.evidence_library.shared.v2.auxiliary_metadata import (
    AuxiliaryMetadataAttacher as _AuxiliaryMetadataAttacher,
)


MAPPING_VERSION = "bbb_martins_auxiliary.globally_reconciled.v2"
NULL_LIKE = (
    "", "-", "n/a", "na", "nan", "none", "not specified", "not stated",
    "null", "unknown", "unspecified",
)
DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parent
    / "data_processing"
    / "globally_reconciled_auxiliary_value_mapping.json"
)
APPLICABLE_SOURCES = ("direct_bbb", "passive_permeability", "efflux_transport")


class AuxiliaryMetadataAttacher(_AuxiliaryMetadataAttacher):
    def __init__(self, path: str | Path = DEFAULT_MAPPING_PATH):
        super().__init__(
            path,
            mapping_version=MAPPING_VERSION,
            applicable_sources=APPLICABLE_SOURCES,
            null_like=NULL_LIKE,
        )


class PendingAuxiliaryAttacher:
    applicable_sources = APPLICABLE_SOURCES

    def __init__(self, path: str | Path = DEFAULT_MAPPING_PATH):
        self.path = Path(path)

    def attach(self, record: Mapping[str, Any]) -> dict[str, Any]:
        applicable = str(record.get("source_id") or "") in APPLICABLE_SOURCES
        return {
            "global_context": None,
            "global_species_context": None,
            "auxiliary_mapping_status": "not_available" if applicable else "not_applicable",
            "auxiliary_attachment_version": AUXILIARY_ATTACHMENT_VERSION,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "attachment_version": AUXILIARY_ATTACHMENT_VERSION,
            "mapping_version": None,
            "mapping_path": str(self.path),
            "mapping_sha256": None,
            "status": "not_built",
            "output_fields": list(OUTPUT_FIELDS),
            "applicable_sources": list(APPLICABLE_SOURCES),
            "sections": {},
        }

    def coverage_audit(self, records: list[Mapping[str, Any]]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in records:
            status = str(record.get("auxiliary_mapping_status") or "missing")
            counts[status] = counts.get(status, 0) + 1
        return {
            "records": len(records),
            "status_counts": dict(sorted(counts.items())),
            "validations": {
                "all_applicable_records_mapped": False,
                "inapplicable_sources_explicit": True,
            },
        }


__all__ = [
    "APPLICABLE_SOURCES",
    "AUXILIARY_ATTACHMENT_VERSION",
    "AuxiliaryMetadataAttacher",
    "DEFAULT_MAPPING_PATH",
    "MAPPING_VERSION",
    "NULL_LIKE",
    "OUTPUT_FIELDS",
    "PendingAuxiliaryAttacher",
]
