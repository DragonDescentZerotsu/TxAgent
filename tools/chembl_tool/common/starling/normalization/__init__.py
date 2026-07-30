"""Layered, one-row-per-source Starling record normalization."""

from .contracts import (
    CLEANING_STAGE_VERSION,
    NORMALIZATION_STAGE_VERSION,
    ORGANIZATION_STAGE_VERSION,
    FamilyAssignment,
    MeasurementPair,
    NormalizationResult,
    NormalizedSourceProfile,
    ParsedPoint,
)
from .measurements import EndpointOrthography, canonicalize_endpoint

__all__ = [
    "CLEANING_STAGE_VERSION",
    "NORMALIZATION_STAGE_VERSION",
    "ORGANIZATION_STAGE_VERSION",
    "FamilyAssignment",
    "EndpointOrthography",
    "MeasurementPair",
    "NormalizationResult",
    "NormalizedSourceProfile",
    "ParsedPoint",
    "canonicalize_endpoint",
]
