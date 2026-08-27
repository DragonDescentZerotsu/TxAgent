"""Shared schemas for each persisted normalization stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SOURCE_STAGE_VERSION = "starling_source_snapshot.v1"
CLEANING_STAGE_VERSION = "starling_record_cleaning.v9"
NORMALIZATION_STAGE_VERSION = "starling_measurement_normalization.v16"
ORGANIZATION_STAGE_VERSION = "starling_record_organization.v10"

NORMALIZED_RECORD_VERSION = "starling_normalized_record.v6"
NORMALIZED_ARTIFACT_VERSION = "starling_normalized_evidence.v6"
SCALAR_PARSER_VERSION = "lossless_scalar_parser.v6"

STAGE_REQUIRED_COLUMNS = {
    "clean": {
        "cleaned_record_id",
        "source_id",
        "source_row_number",
        "endpoint_name",
        "measurement_text",
        "unit_text",
        "source_smiles",
        "canonical_smiles",
        "structure_status",
    },
    "normalize": {
        "cleaned_record_id",
        "normalized_record_id",
        "spacing_and_spelling_endpoint",
        "spacing_and_spelling_status",
        "canonical_endpoint",
        "canonical_measurement",
        "canonical_unit",
        "finite_scalar_value",
        "is_absolute_and_continuous",
        "normalization_validity_status",
    },
    "organize": {
        "normalized_record_id",
        "duplicate_group_id",
        "duplicate_group_size",
        "retrieval_eligible",
        "organization_status",
    },
}


@dataclass(frozen=True)
class NormalizedSourceProfile:
    """Declarative mapping from one source schema to the common parent schema."""

    source_id: str
    source_name: str
    endpoint_field: str = ""
    endpoint_constant: str = ""
    measurement_field: str = ""
    unit_field: str = ""
    unit_constant: str = ""
    embedded_unit: bool = False
    smiles_field: str = "smiles"
    structure_mode: str = "mapped"
    record_id_field: str = "extraction_id"
    confidence_field: str = "confidence"
    support_text_field: str = "support_text"
    name_fields: tuple[str, ...] = ("molecule_name", "global_identifier")
    context_fields: tuple[str, ...] = ()
    # Some source taxonomies use a generic null token as a real category.
    # Declared fields receive typography/whitespace normalization without the
    # generic textual-null vocabulary.
    literal_text_fields: tuple[str, ...] = ()
    source_path: str = ""
    source_revision: str = ""


@dataclass(frozen=True)
class MeasurementPair:
    """A measurement and unit that must always be transformed together."""

    canonical_measurement: str | None
    canonical_unit: str | None
    status: str
    unit_notation_status: str = "none"
    unit_notation_factor: float | None = None


@dataclass(frozen=True)
class FamilyAssignment:
    group_id: str
    assay_tier: str
    endpoint_group: str
    evidence_role: str
    target_pref_name: str


@dataclass(frozen=True)
class ParsedPoint:
    value: float | None
    variation: float | None
    approximate: bool
    kind: str


@dataclass
class NormalizationResult:
    records: list[dict[str, Any]]
    rejections: list[dict[str, Any]]
    duplicates: list[dict[str, Any]]
    stats: dict[str, Any]
    cleaned_records: list[dict[str, Any]] | None = None
    normalized_records: list[dict[str, Any]] | None = None
