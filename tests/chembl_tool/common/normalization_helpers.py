"""Test helper that runs the public clean, normalize, and organize stages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tools.chembl_tool.common.units import UNIT_NORMALIZER_VERSION

from data.processing.evidence_library.shared.v1.normalization.audit import (
    validate_cleaned_normalized_identity,
    validate_measurement_pairs,
    write_parquet,
)
from data.processing.evidence_library.shared.v1.normalization.cleaning import (
    clean_source_rows,
    clean_text,
    endpoint_inventory_hash,
    file_sha256,
    normalize_endpoint_name,
)
from data.processing.evidence_library.shared.v1.normalization.contracts import (
    NORMALIZED_ARTIFACT_VERSION,
    NORMALIZED_RECORD_VERSION,
    SCALAR_PARSER_VERSION,
    FamilyAssignment,
    MeasurementPair,
    NormalizationResult,
    NormalizedSourceProfile,
    ParsedPoint,
)
from data.processing.evidence_library.shared.v1.normalization.measurements import (
    EndpointNormalizer,
    EndpointOrthography,
    EndpointStandardizer,
    FamilyResolver,
    RecordEnricher,
    canonicalize_endpoint,
    format_number,
    normalize_cleaned_records,
    normalize_measurement_and_unit,
    parse_point_measurement,
    standardize_measurement_pair,
)
from data.processing.evidence_library.shared.v1.normalization.organization import (
    aggregate_molecule_family_records,
    combine_normalization_results,
    deduplicate_within_source,
    examples_text,
    organize_normalized_records,
    representative_examples,
)


def normalize_source_rows(
    rows: Iterable[Mapping[str, Any]],
    profile: NormalizedSourceProfile,
    *,
    smiles_mapping: Mapping[str, str] | None,
    endpoint_normalizer: EndpointNormalizer,
    endpoint_standardizer: EndpointStandardizer | None = None,
    family_resolver: FamilyResolver,
    record_enricher: RecordEnricher | None = None,
    source_sha256: str = "",
    task: str | None = None,
) -> NormalizationResult:
    """Run all record-level stages for one source without semantic row loss."""
    cleaned = clean_source_rows(
        rows,
        profile,
        smiles_mapping=smiles_mapping,
        source_sha256=source_sha256,
    )
    normalized = normalize_cleaned_records(
        cleaned,
        endpoint_normalizer=endpoint_normalizer,
        endpoint_standardizer=endpoint_standardizer,
        family_resolver=family_resolver,
        record_enricher=record_enricher,
        task=task,
    )
    identity_errors = validate_cleaned_normalized_identity(cleaned, normalized)
    if identity_errors:
        raise ValueError(identity_errors[0])
    records, duplicates, exclusions, organization_stats = organize_normalized_records(
        normalized
    )
    stats = {
        "source_id": profile.source_id,
        "n_input_rows": len(cleaned),
        "n_cleaned_records": len(cleaned),
        "n_normalized_records_before_deduplication": len(normalized),
        "n_records": len(records),
        "n_duplicate_records_removed": len(duplicates),
        "n_absolute_and_continuous": sum(
            bool(item.get("is_absolute_and_continuous")) for item in records
        ),
        "n_retrieval_eligible": organization_stats["n_retrieval_eligible"],
        "n_organization_exclusions": len(exclusions),
    }
    return NormalizationResult(
        records,
        exclusions,
        duplicates,
        stats,
        cleaned_records=cleaned,
        normalized_records=normalized,
    )


__all__ = [
    "FamilyAssignment",
    "EndpointOrthography",
    "EndpointStandardizer",
    "MeasurementPair",
    "NORMALIZED_ARTIFACT_VERSION",
    "NORMALIZED_RECORD_VERSION",
    "NormalizationResult",
    "NormalizedSourceProfile",
    "ParsedPoint",
    "SCALAR_PARSER_VERSION",
    "UNIT_NORMALIZER_VERSION",
    "aggregate_molecule_family_records",
    "canonicalize_endpoint",
    "clean_text",
    "combine_normalization_results",
    "deduplicate_within_source",
    "endpoint_inventory_hash",
    "examples_text",
    "file_sha256",
    "format_number",
    "normalize_endpoint_name",
    "normalize_measurement_and_unit",
    "normalize_source_rows",
    "parse_point_measurement",
    "representative_examples",
    "standardize_measurement_pair",
    "validate_cleaned_normalized_identity",
    "validate_measurement_pairs",
    "write_parquet",
]
