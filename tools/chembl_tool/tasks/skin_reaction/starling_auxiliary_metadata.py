"""Skin_Reaction binding for the shared auxiliary-metadata attacher.

All four sources now receive the comparison metadata needed by their pair
buckets.  Per-source output inventories keep endpoint and severity concepts
explicitly null on sources where they do not apply.

Until the reconciliation pass has been run, ``PendingAuxiliaryAttacher`` lets
stages 01-03 build against an explicit ``not_available`` status rather than a
silently blank context.  A build in that state fails the
``globally_reconciled_auxiliary_coverage`` validation on purpose, so it can
never be mistaken for a complete artifact.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.auxiliary_metadata import (
    AuxiliaryMetadataAttacher as _AuxiliaryMetadataAttacher,
)
from tools.chembl_tool.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
    NULL_LIKE,
)


DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parent
    / "data_processing"
    / "globally_reconciled_auxiliary_value_mapping.json"
)
DEFAULT_PUBLICATION_RECORD = (
    Path(__file__).resolve().parent
    / "data_processing"
    / "species_context_v3"
    / "reconciliation"
    / "PUBLICATION_RECORD.json"
)
APPLICABLE_SOURCES = (
    "direct_skin_reaction",
    "sensitization_aop",
    "phototoxicity_irritation_local_damage",
    "skin_exposure",
)
AUXILIARY_ATTACHMENT_VERSION = "starling_auxiliary_attachment.skin_reaction.v3"
SOURCE_COLUMN_ALIASES = {
    "direct_skin_reaction": {"effect_metric": "measurement_text"},
    "sensitization_aop": {"endpoint_or_target": "endpoint_name"},
}
OUTPUT_FIELDS_BY_SOURCE = {
    "direct_skin_reaction": (
        "global_context",
        "global_species_context",
        "global_severity_grade",
    ),
    "sensitization_aop": (
        "global_context",
        "global_species_context",
        "global_endpoint_context",
    ),
    "phototoxicity_irritation_local_damage": (
        "global_context",
        "global_species_context",
    ),
    "skin_exposure": (
        "global_context",
        "global_species_context",
    ),
}
OUTPUT_FIELDS = tuple(
    dict.fromkeys(
        field
        for fields in OUTPUT_FIELDS_BY_SOURCE.values()
        for field in fields
    )
)


class AuxiliaryMetadataAttacher(_AuxiliaryMetadataAttacher):
    def __init__(self, path: str | Path = DEFAULT_MAPPING_PATH):
        resolved = Path(path)
        if resolved.resolve() == DEFAULT_MAPPING_PATH.resolve():
            _validate_reviewed_publication(resolved)
        super().__init__(
            path,
            mapping_version=MAPPING_VERSION,
            applicable_sources=APPLICABLE_SOURCES,
            null_like=NULL_LIKE,
            output_fields=OUTPUT_FIELDS_BY_SOURCE,
            attachment_version=AUXILIARY_ATTACHMENT_VERSION,
            source_column_aliases=SOURCE_COLUMN_ALIASES,
            non_null_outputs_when_input_present={
                "sensitization_aop": ("global_endpoint_context",),
            },
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_reviewed_publication(mapping_path: Path) -> None:
    """Fail closed while the focused species candidate is under review."""
    if not DEFAULT_PUBLICATION_RECORD.is_file():
        raise ValueError(
            "Skin species context is an unreviewed candidate: publication record "
            f"is absent at {DEFAULT_PUBLICATION_RECORD}"
        )
    record = json.loads(DEFAULT_PUBLICATION_RECORD.read_text(encoding="utf-8"))
    if record.get("publication_status") != "human_approved":
        raise ValueError("Skin species publication record is not human-approved")
    if record.get("runtime_mapping_sha256") != _sha256(mapping_path):
        raise ValueError("Skin species publication record does not match runtime mapping")
    manifest_path = Path(str(record.get("review_manifest_path") or ""))
    if not manifest_path.is_absolute():
        manifest_path = DEFAULT_PUBLICATION_RECORD.parent / manifest_path
    if (
        not manifest_path.is_file()
        or record.get("review_manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError("Skin species publication record does not match review manifest")


class PendingAuxiliaryAttacher:
    """Stand-in used only before the reconciled mapping has been built."""

    applicable_sources = APPLICABLE_SOURCES

    def __init__(self, path: str | Path = DEFAULT_MAPPING_PATH):
        self.path = Path(path)

    def attach(self, record: Mapping[str, Any]) -> dict[str, Any]:
        source_id = str(record.get("source_id") or "")
        applicable = source_id in APPLICABLE_SOURCES
        return {
            **{field: None for field in OUTPUT_FIELDS},
            "auxiliary_mapping_status": (
                "not_available" if applicable else "not_applicable"
            ),
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
            "output_fields_by_source": {
                source: list(fields)
                for source, fields in sorted(OUTPUT_FIELDS_BY_SOURCE.items())
            },
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
    "DEFAULT_PUBLICATION_RECORD",
    "OUTPUT_FIELDS",
    "OUTPUT_FIELDS_BY_SOURCE",
    "SOURCE_COLUMN_ALIASES",
    "PendingAuxiliaryAttacher",
]
