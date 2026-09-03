"""Strict source-cleaning and canonical-dimension contracts for Starling v7.

The v7 contract deliberately separates two operations that the v6 vocabulary
called "normalization":

``cleaning``
    Meaning-preserving cleanup of genuine source fields.  Only the four
    universal record roles (endpoint, measurement, unit, and structure) may be
    renamed at this boundary.

``canonicalization``
    A reviewed transformation or extraction used for integration and pair
    bucketing.  A canonical dimension names the concept it represents and
    records the genuine cleaned fields from which it was produced.  This makes
    one-to-many extraction explicit; for example, one ``assay_system`` string
    may produce both ``canonical_assay_context`` and
    ``canonical_species_context`` without inventing a cleaned ``species``
    alias.

This module is intentionally data- and task-agnostic.  Task modules declare
their source schemas and canonical dimensions; builders and tests consume the
same declarations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

from data.processing.evidence_library.shared.v1.normalization.cleaning import clean_scalar, stable_id
from data.processing.evidence_library.shared.v1.categorical_response import (
    ControlledMeasurementSpec,
)


RECORD_CONTRACT_VERSION = "starling_record_contract.v7"
CANONICAL_ARTIFACT_VERSION = "starling_canonical_evidence.v7"
CANONICAL_RECORD_VERSION = "starling_canonical_record.v7"
# The prompt-time wire contract remains v1; v7 changes the field inventory,
# not the evidence-contract envelope consumed by reasoning code.
SOURCE_CONTRACT_VERSION = "source_column_contract.v1"
PAIR_BUCKET_CONTRACT_VERSION = "source_aware_pair_bucket.v7"

ROLE_FIELD_NAMES = {
    "endpoint": "endpoint_name",
    "measurement": "measurement_text",
    "unit": "unit_text",
    "structure": "smiles",
}
ROLE_FIELDS = frozenset(ROLE_FIELD_NAMES.values())

CANONICAL_METHODS = frozenset(
    {
        "deterministic_rule",
        "frozen_mapping",
        "controlled_encoder",
        # A reviewed per-row extraction, frozen offline and joined by record ID.
        # It differs from ``frozen_mapping`` in cardinality: a mapping is keyed
        # by a distinct source *value* and reused across every row sharing it,
        # while an extraction is keyed by one row and never reused.
        "frozen_extraction",
    }
)

_LEGACY_CANONICAL_FIELDS = {
    "canonical_endpoint": "canonical_endpoint_name",
    "canonical_measurement": "canonical_measurement_text",
    "canonical_unit": "canonical_unit_text",
    "global_context": "canonical_assay_context",
    "global_species_context": "canonical_species_context",
    "global_biological_matrix": "canonical_biological_matrix",
    "categorical_encoder_id": "canonical_measurement_scale_id",
    "normalized_record_id": "canonical_record_id",
    "normalization_validity_status": "canonicalization_status",
}

_ROW_ONLY_INTERMEDIATES = frozenset(
    {
        "spacing_and_spelling_endpoint",
        "spacing_and_spelling_status",
        "spacing_and_spelling_reason",
        "spacing_and_spelling_version",
        "normalization_version",
        "measurement_normalization_version",
        "is_absolute_and_continuous",
        "absolute_and_continuous_value",
        "source_payload_json",
        "evidence_context_json",
        "llm_source_contract_json",
        "llm_source_fields_json",
    }
)

_STAGE1_DERIVED_FIELDS = frozenset(
    {
        "canonical_endpoint_name",
        "measurement_resolution_route",
        "measurement_resolution_rule_id",
        "measurement_resolution_exact_measurement",
        "measurement_resolution_exact_unit",
        "measurement_resolution_exact_unit_is_canonical",
        "measurement_routing_version",
    }
)


def _has_text(value: Any) -> bool:
    return value is not None and str(value).strip().casefold() not in {
        "",
        "nan",
        "none",
        "null",
    }


@dataclass(frozen=True)
class CanonicalProducerSpec:
    """One declared way to produce a canonical dimension.

    Most tasks have one producer per dimension and continue to use the legacy
    fields on :class:`CanonicalDimensionSpec`.  A source that conditionally
    uses a scalar parser or a controlled categorical encoder declares both
    producers and persists the selected producer ID on every row.
    """

    producer_id: str
    input_fields: tuple[str, ...]
    method: Literal[
        "deterministic_rule",
        "frozen_mapping",
        "controlled_encoder",
        "frozen_extraction",
    ]
    version: str

    def __post_init__(self) -> None:
        if not self.producer_id:
            raise ValueError("canonical producer_id must be nonempty")
        if not self.input_fields or any(not field for field in self.input_fields):
            raise ValueError(
                f"canonical producer {self.producer_id!r} has no cleaned inputs"
            )
        if len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError(
                f"canonical producer {self.producer_id!r} repeats an input"
            )
        if self.method not in CANONICAL_METHODS:
            raise ValueError(f"unsupported canonicalization method: {self.method!r}")
        if not self.version:
            raise ValueError(
                f"canonical producer {self.producer_id!r} has no version"
            )

    def manifest(self) -> dict[str, Any]:
        return {
            "producer_id": self.producer_id,
            "input_fields": list(self.input_fields),
            "method": self.method,
            "version": self.version,
        }


@dataclass(frozen=True)
class CanonicalDimensionSpec:
    """One final canonical value and the cleaned fields that support it."""

    output_field: str
    semantic_dimension: str
    input_fields: tuple[str, ...]
    method: Literal[
        "deterministic_rule",
        "frozen_mapping",
        "controlled_encoder",
        "frozen_extraction",
    ]
    version: str
    missing_policy: Literal["null", "explicit_unknown"] = "null"
    atomic_group: str | None = None
    depends_on: tuple[str, ...] = ()
    producer_id: str | None = None
    producer_id_field: str | None = None
    producer_variants: tuple[CanonicalProducerSpec, ...] = ()
    # During the v6-to-v7 transition, the reviewed value may already be
    # attached under a historical field name.  This is migration provenance,
    # not a second semantic output.
    legacy_value_field: str | None = None
    # Classification-only dimensions may inspect contextual source fields
    # that remain legitimate residual-heterogeneity dimensions.  The
    # classifier labels the measurement's semantics; it does not consume or
    # normalize those source fields into the canonical value.
    classification_evidence: bool = False

    def __post_init__(self) -> None:
        if not self.output_field.startswith("canonical_"):
            raise ValueError(
                f"canonical output must use canonical_*: {self.output_field!r}"
            )
        if not self.semantic_dimension:
            raise ValueError("canonical semantic_dimension must be nonempty")
        if not self.input_fields or any(not field for field in self.input_fields):
            raise ValueError(
                f"canonical dimension {self.output_field!r} has no cleaned inputs"
            )
        if len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError(
                f"canonical dimension {self.output_field!r} repeats an input"
            )
        if self.method not in CANONICAL_METHODS:
            raise ValueError(f"unsupported canonicalization method: {self.method!r}")
        if not self.version:
            raise ValueError(
                f"canonical dimension {self.output_field!r} has no version"
            )
        if self.missing_policy not in {"null", "explicit_unknown"}:
            raise ValueError(
                f"unsupported missing policy: {self.missing_policy!r}"
            )
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(
                f"canonical dimension {self.output_field!r} repeats a dependency"
            )
        if bool(self.producer_id) != bool(self.producer_id_field):
            raise ValueError(
                f"canonical dimension {self.output_field!r} must declare both "
                "producer_id and producer_id_field, or neither"
            )
        if self.producer_id_field and not self.producer_id_field.startswith("canonical_"):
            raise ValueError(
                f"canonical producer field must use canonical_*: "
                f"{self.producer_id_field!r}"
            )
        producer_ids = [producer.producer_id for producer in self.producers]
        if len(producer_ids) != len(set(producer_ids)):
            raise ValueError(
                f"canonical dimension {self.output_field!r} repeats a producer ID"
            )

    @property
    def producers(self) -> tuple[CanonicalProducerSpec, ...]:
        if self.producer_id is None:
            return ()
        return (
            CanonicalProducerSpec(
                self.producer_id,
                self.input_fields,
                self.method,
                self.version,
            ),
            *self.producer_variants,
        )

    @property
    def all_input_fields(self) -> tuple[str, ...]:
        fields = list(self.input_fields)
        for producer in self.producer_variants:
            for field in producer.input_fields:
                if field not in fields:
                    fields.append(field)
        return tuple(fields)

    def manifest(self) -> dict[str, Any]:
        return {
            "output_field": self.output_field,
            "semantic_dimension": self.semantic_dimension,
            "input_fields": list(self.input_fields),
            "method": self.method,
            "version": self.version,
            "missing_policy": self.missing_policy,
            "atomic_group": self.atomic_group,
            "depends_on": list(self.depends_on),
            "producer_id": self.producer_id,
            "producer_id_field": self.producer_id_field,
            "producers": [producer.manifest() for producer in self.producers],
            "legacy_value_field": self.legacy_value_field,
            "classification_evidence": self.classification_evidence,
        }


@dataclass(frozen=True)
class SourceProfile:
    """One source's raw schema, cleaned role mapping, and final dimensions."""

    source_id: str
    source_columns: tuple[str, ...]
    endpoint_field: str = ""
    endpoint_constant: str = ""
    measurement_field: str = ""
    unit_field: str = ""
    unit_constant: str = ""
    smiles_field: str = "smiles"
    structure_mode: Literal["direct", "mapped"] = "direct"
    structure_identity_field: str = ""
    canonical_dimensions: tuple[CanonicalDimensionSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be nonempty")
        if len(set(self.source_columns)) != len(self.source_columns):
            raise ValueError(f"duplicate raw columns for {self.source_id!r}")
        if bool(self.endpoint_field) == bool(self.endpoint_constant):
            raise ValueError(
                f"{self.source_id!r} must declare exactly one endpoint field or constant"
            )
        if bool(self.unit_field) and bool(self.unit_constant):
            raise ValueError(
                f"{self.source_id!r} cannot declare both unit field and constant"
            )
        raw = set(self.source_columns)
        if self.structure_mode == "mapped":
            if not self.structure_identity_field:
                raise ValueError(
                    f"{self.source_id!r} mapped structure has no identity field"
                )
            if self.structure_identity_field not in raw:
                raise ValueError(
                    f"{self.source_id!r} mapped identity field "
                    f"{self.structure_identity_field!r} is absent from raw schema"
                )
        for role, field in (
            ("endpoint", self.endpoint_field),
            ("measurement", self.measurement_field),
            ("unit", self.unit_field),
            ("structure", self.smiles_field),
        ):
            if field and field not in raw:
                raise ValueError(
                    f"{self.source_id!r} {role} field {field!r} is absent from raw schema"
                )
        cleaned = set(self.cleaned_source_fields)
        outputs: set[str] = set()
        atomic_producer_fields: dict[str, set[str | None]] = {}
        for dimension in self.canonical_dimensions:
            if dimension.output_field in outputs:
                raise ValueError(
                    f"{self.source_id!r} repeats {dimension.output_field!r}"
                )
            outputs.add(dimension.output_field)
            missing = set(dimension.all_input_fields) - cleaned
            if missing:
                raise ValueError(
                    f"{self.source_id!r} canonical dimension "
                    f"{dimension.output_field!r} uses non-cleaned inputs {sorted(missing)}"
                )
            unknown_dependencies = {
                dependency
                for dependency in dimension.depends_on
                if not dependency.startswith("canonical_")
            }
            # An atomic group asserts that its members are produced by one
            # operation.  That is only checkable when every member records the
            # same producer field, so require the declarations to agree here
            # rather than discovering the gap at projection time.
            if dimension.atomic_group:
                atomic_producer_fields.setdefault(
                    dimension.atomic_group, set()
                ).add(dimension.producer_id_field)
            if unknown_dependencies:
                raise ValueError(
                    f"{self.source_id!r} canonical dimension "
                    f"{dimension.output_field!r} has noncanonical dependencies "
                    f"{sorted(unknown_dependencies)}"
                )
        for atomic_group, producer_fields in sorted(atomic_producer_fields.items()):
            if len(producer_fields) != 1:
                raise ValueError(
                    f"{self.source_id!r} atomic group {atomic_group!r} spans "
                    f"different producer fields {sorted(map(str, producer_fields))}; "
                    "members of one atomic group must record the same producer"
                )

    @cached_property
    def raw_role_fields(self) -> frozenset[str]:
        return frozenset(
            field
            for field in (
                self.endpoint_field,
                self.measurement_field,
                self.unit_field,
                self.smiles_field,
            )
            if field
        )

    @cached_property
    def cleaned_source_fields(self) -> tuple[str, ...]:
        """The source-visible Stage-01 schema in deterministic order."""
        output = [
            column for column in self.source_columns if column not in self.raw_role_fields
        ]
        for role in ("endpoint", "measurement", "unit", "structure"):
            cleaned = ROLE_FIELD_NAMES[role]
            raw = getattr(self, f"{role if role != 'structure' else 'smiles'}_field", "")
            constant = getattr(self, f"{role}_constant", "")
            if raw or constant or role in {"measurement", "unit"}:
                if cleaned not in output:
                    output.append(cleaned)
        return tuple(output)

    @cached_property
    def canonical_output_fields(self) -> tuple[str, ...]:
        return tuple(item.output_field for item in self.canonical_dimensions)

    @cached_property
    def source_visible_fields(self) -> tuple[str, ...]:
        """Cleaned fields that are genuine source values, excluding constants."""
        fields = list(self.cleaned_source_fields)
        if self.endpoint_constant:
            fields.remove("endpoint_name")
        if self.unit_constant and "unit_text" in fields:
            fields.remove("unit_text")
        return tuple(fields)

    def manifest(self) -> dict[str, Any]:
        raw_to_cleaned: dict[str, str] = {}
        for role, raw_field in (
            ("endpoint", self.endpoint_field),
            ("measurement", self.measurement_field),
            ("unit", self.unit_field),
            ("structure", self.smiles_field),
        ):
            if raw_field:
                raw_to_cleaned[raw_field] = ROLE_FIELD_NAMES[role]
        return {
            "source_id": self.source_id,
            "raw_source_columns": list(self.source_columns),
            "raw_to_cleaned_role_fields": raw_to_cleaned,
            "cleaned_source_fields": list(self.cleaned_source_fields),
            "source_visible_fields": list(self.source_visible_fields),
            "structure_mode": self.structure_mode,
            "structure_identity_field": self.structure_identity_field or None,
            "canonical_dimensions": [
                dimension.manifest() for dimension in self.canonical_dimensions
            ],
        }


@dataclass(frozen=True)
class PairBucketSpec:
    """Canonical identity fields and cleaned heterogeneity candidates."""

    source_id: str
    canonical_dimensions: tuple[str, ...]
    variance_candidates: tuple[str, ...] = ()
    eligible_reference_scopes: tuple[str, ...] = ()
    reference_basis_required: bool = False
    required_known_dimensions: tuple[str, ...] = ()

    @property
    def additional_dimensions(self) -> tuple[str, ...]:
        """Canonical dimensions appended after endpoint and unit in the key."""
        return tuple(
            field
            for field in self.canonical_dimensions
            if field not in {"canonical_endpoint_name", "canonical_unit_text"}
        )

    @property
    def core_context_dimensions(self) -> tuple[str, ...]:
        """Context dimensions, excluding endpoint/unit and measurement scale."""
        return tuple(
            field
            for field in self.canonical_dimensions
            if field
            not in {
                "canonical_endpoint_name",
                "canonical_unit_text",
                "canonical_measurement_scale_id",
            }
        )

    def validate(self, profile: SourceProfile) -> None:
        if self.source_id != profile.source_id:
            raise ValueError("pair/source profile IDs differ")
        known = set(profile.canonical_output_fields)
        missing = set(self.canonical_dimensions) - known
        if missing:
            raise ValueError(
                f"{self.source_id!r} pair fields lack canonical declarations: "
                f"{sorted(missing)}"
            )
        noncanonical = [
            field
            for field in self.canonical_dimensions
            if not field.startswith("canonical_")
        ]
        if noncanonical:
            raise ValueError(
                f"{self.source_id!r} pair fields are not canonical: {noncanonical}"
            )
        unknown_required = set(self.required_known_dimensions) - set(
            self.canonical_dimensions
        )
        if unknown_required:
            raise ValueError(
                f"{self.source_id!r} required-known fields are absent from pair identity: "
                f"{sorted(unknown_required)}"
            )
        absent_candidates = set(self.variance_candidates) - set(
            profile.cleaned_source_fields
        )
        if absent_candidates:
            raise ValueError(
                f"{self.source_id!r} variance candidates are not cleaned source fields: "
                f"{sorted(absent_candidates)}"
            )
        always_canonical_inputs: set[str] = set()
        conditional_encoder_inputs: set[str] = set()
        for dimension in profile.canonical_dimensions:
            if dimension.classification_evidence:
                continue
            target = (
                conditional_encoder_inputs
                if dimension.method == "controlled_encoder"
                else always_canonical_inputs
            )
            target.update(dimension.input_fields)
            for producer in dimension.producer_variants:
                target = (
                    conditional_encoder_inputs
                    if producer.method == "controlled_encoder"
                    else always_canonical_inputs
                )
                target.update(producer.input_fields)
        # A controlled encoder and a source scalar are mutually exclusive by
        # contract.  Its inputs may therefore remain residual-heterogeneity
        # candidates for the continuous rows, but ordinary canonical inputs
        # may never be reused this way.
        conditional_encoder_inputs.difference_update(always_canonical_inputs)
        reused_candidates = (
            set(self.variance_candidates) & always_canonical_inputs
        )
        if reused_candidates:
            raise ValueError(
                f"{self.source_id!r} variance candidates were already used for "
                f"canonicalization: {sorted(reused_candidates)}"
            )
        if self.eligible_reference_scopes:
            if "canonical_reference_scope" not in self.canonical_dimensions:
                raise ValueError(
                    f"{self.source_id!r} has a reference-scope eligibility policy "
                    "but omits canonical_reference_scope from pair identity"
                )
            if (
                self.reference_basis_required
                and "canonical_reference_basis" not in self.canonical_dimensions
            ):
                raise ValueError(
                    f"{self.source_id!r} requires a reference basis but omits "
                    "canonical_reference_basis from pair identity"
                )


@dataclass(frozen=True)
class StarlingRecordContract:
    """Complete v7 declaration for one task."""

    task_id: str
    sources: Mapping[str, SourceProfile]
    pair_buckets: Mapping[str, PairBucketSpec]
    measurement_scales: Mapping[str, ControlledMeasurementSpec] = field(
        default_factory=dict
    )
    version: str = RECORD_CONTRACT_VERSION

    def __post_init__(self) -> None:
        source_ids = set(self.sources)
        if source_ids != set(self.pair_buckets):
            raise ValueError(
                "record-contract source/pair inventories differ: "
                f"sources={sorted(source_ids)} pairs={sorted(self.pair_buckets)}"
            )
        for source_id, profile in self.sources.items():
            if source_id != profile.source_id:
                raise ValueError(f"source mapping key differs from {profile.source_id!r}")
            self.pair_buckets[source_id].validate(profile)
        for scale_id, scale in self.measurement_scales.items():
            if scale_id != scale.scale_id:
                raise ValueError(
                    f"measurement-scale mapping key differs from {scale.scale_id!r}"
                )
            if scale.source_id not in source_ids:
                raise ValueError(
                    f"measurement scale {scale_id!r} names unknown source "
                    f"{scale.source_id!r}"
                )
        sources_with_controlled_scales = {
            scale.source_id for scale in self.measurement_scales.values()
        }
        for source_id in sources_with_controlled_scales:
            if "canonical_measurement_scale_id" not in self.pair_buckets[
                source_id
            ].canonical_dimensions:
                raise ValueError(
                    f"{source_id!r} has controlled measurement scales but its "
                    "pair identity omits canonical_measurement_scale_id"
                )

    def source(self, source_id: str) -> SourceProfile:
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise ValueError(f"unknown source_id={source_id!r}") from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "record_contract_version": self.version,
            "source_contract_version": SOURCE_CONTRACT_VERSION,
            "task_id": self.task_id,
            "semantics": {
                "cleaning": "meaning-preserving source-visible cleanup",
                "canonicalization": (
                    "reviewed same-field normalization or semantic extraction"
                ),
                "canonical_values_per_dimension": 1,
                "canonical_fallback_in_source_projection": False,
            },
            "sources": {
                source_id: self.sources[source_id].manifest()
                for source_id in sorted(self.sources)
            },
            "pair_buckets": {
                source_id: {
                    "canonical_dimensions": list(spec.canonical_dimensions),
                    "core_context_dimensions": list(
                        spec.core_context_dimensions
                    ),
                    "endpoint_identity_inputs": sorted(
                        {
                            field
                            for dimension in self.sources[
                                source_id
                            ].canonical_dimensions
                            if dimension.output_field
                            == "canonical_endpoint_name"
                            for producer in dimension.producers
                            for field in producer.input_fields
                        }
                    ),
                    "variance_candidates": list(spec.variance_candidates),
                    "eligible_reference_scopes": list(
                        spec.eligible_reference_scopes
                    ),
                    "reference_basis_required": spec.reference_basis_required,
                    "required_known_dimensions": list(
                        spec.required_known_dimensions
                    ),
                }
                for source_id, spec in sorted(self.pair_buckets.items())
            },
            "measurement_scales": {
                scale_id: scale.manifest()
                for scale_id, scale in sorted(self.measurement_scales.items())
            },
        }

    def clean_projection(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Project a v6 in-memory cleaned record into the strict v7 source view."""
        source_id = str(record.get("source_id") or "")
        profile = self.source(source_id)
        # Source attachment makes the complete cleaned schema columnar before
        # projection.  Parse the compatibility payload only on older/resumed
        # rows that actually lack one of those fields.
        payload = (
            record
            if all(field in record for field in profile.source_columns)
            else _source_payload(record)
        )
        foreign_fields = {
            field
            for other_source_id, other_profile in self.sources.items()
            if other_source_id != source_id
            for field in other_profile.source_columns
        } - set(profile.source_columns) - ROLE_FIELDS
        output: dict[str, Any] = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "source_payload_json",
                "evidence_context_json",
                *profile.raw_role_fields,
                *_LEGACY_CANONICAL_FIELDS,
                *_ROW_ONLY_INTERMEDIATES,
                *foreign_fields,
            }
            and (
                not key.startswith("canonical_")
                or key in _STAGE1_DERIVED_FIELDS
                or key == "canonical_smiles"
            )
            and key != "molecule_id"
        }
        # Stage 01 already cleaned these values under the source profile,
        # including its declared literal-text fields.  Re-cleaning during a
        # projection would erase meaningful literals such as "unspecified".
        output["endpoint_name"] = record.get("endpoint_name")
        output["measurement_text"] = record.get("measurement_text")
        output["unit_text"] = record.get("unit_text")
        output["deduplication_context_id"] = stable_id(
            "source_context",
            source_id,
            str(record.get("evidence_context_json") or "{}"),
        )
        # The cleaned source view always preserves the actual source structure
        # field.  A mapped structure belongs only in ``canonical_smiles``.
        output["smiles"] = clean_scalar(payload.get(profile.smiles_field))
        for field in profile.source_columns:
            if field in profile.raw_role_fields:
                continue
            value = record.get(field, payload.get(field))
            output[field] = value
        _validate_source_projection(output, profile)
        return output

    def source_projection(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return only genuine cleaned source fields; never canonical fallback."""
        source_id = str(record.get("source_id") or "")
        profile = self.source(source_id)
        missing = [field for field in profile.source_visible_fields if field not in record]
        if missing:
            raise ValueError(
                f"{source_id!r} record lacks cleaned source fields: {missing}"
            )
        return {
            "contract_version": SOURCE_CONTRACT_VERSION,
            "source_id": source_id,
            "source_or_simply_cleaned": {
                field: True for field in profile.source_visible_fields
            },
            "source_fields": {
                field: _json_value(record.get(field))
                for field in profile.source_visible_fields
            },
        }

    def canonical_projection(
        self,
        record: Mapping[str, Any],
        *,
        cleaned_projection: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Project one internally canonicalized record into the final v7 row."""
        source_id = str(record.get("source_id") or "")
        profile = self.source(source_id)
        if cleaned_projection is None:
            output = self.clean_projection(record)
        else:
            if str(cleaned_projection.get("source_id") or "") != source_id:
                raise ValueError("cleaned/canonical source IDs differ")
            output = dict(cleaned_projection)
            _validate_source_projection(output, profile)
        for key, value in record.items():
            if key in _ROW_ONLY_INTERMEDIATES or key in profile.raw_role_fields:
                continue
            if key in _LEGACY_CANONICAL_FIELDS:
                continue
            output[key] = value
        for legacy, canonical in _LEGACY_CANONICAL_FIELDS.items():
            if legacy in record:
                output[canonical] = record.get(legacy)
        for dimension in profile.canonical_dimensions:
            legacy = dimension.legacy_value_field or dimension.output_field
            output[dimension.output_field] = record.get(legacy)
        output["record_contract_version"] = self.version
        scale_id = str(
            record.get("categorical_encoder_id")
            or record.get("canonical_measurement_scale_id")
            or ""
        )
        scale = self.measurement_scales.get(scale_id) if scale_id else None
        if scale_id and scale is None:
            raise ValueError(
                f"{source_id!r} record uses undeclared measurement scale {scale_id!r}"
            )
        if scale is not None and scale.source_id != source_id:
            raise ValueError(
                f"measurement scale {scale_id!r} belongs to {scale.source_id!r}, "
                f"not {source_id!r}"
            )
        output["measurement_kind"] = (
            scale.kind if scale is not None else _measurement_kind(record)
        )
        if not _has_text(output.get("canonical_unit_text")):
            scope = str(output.get("canonical_reference_scope") or "")
            if scope in {
                "comparator_relative",
                "standardized_control_ratio",
                "endpoint_defined_ratio",
            }:
                output["canonical_unit_text"] = "relative-scalar"
            elif output["measurement_kind"] == "non_scalar":
                output["canonical_unit_text"] = "free-text"
            else:
                output["canonical_unit_text"] = "unresolved-scalar"
        output["canonical_measurement_scale_id"] = scale_id or None
        category = (
            scale.category_for_value(record.get("finite_scalar_value"))
            if scale is not None
            else None
        )
        if scale is not None and scale.kind in {"binary", "ordinal"} and category is None:
            raise ValueError(
                f"{source_id!r} record has an out-of-domain value for {scale_id!r}"
            )
        output["canonical_category_id"] = (
            category.category_id if category is not None else None
        )
        output["canonical_category_rank"] = (
            category.rank if category is not None else None
        )
        output["measurement_parse_kind"] = record.get("measurement_parse_kind")
        _validate_canonical_projection(output, profile)
        return output

    def inflate_cleaned(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Restore the private v6 working aliases used by the existing engine.

        These aliases exist only in memory.  They let the reviewed v6 parsers
        remain unchanged while v7 persists the stricter source-visible schema.
        """
        source_id = str(record.get("source_id") or "")
        profile = self.source(source_id)
        output = dict(record)
        # Parquet resumes materialize the union of every source's columns.
        # Restore the sparse per-source shape used by a fresh Stage-01 build;
        # a populated foreign field is corruption and must not be hidden.
        foreign_fields = {
            field
            for other_source_id, other_profile in self.sources.items()
            if other_source_id != source_id
            for field in other_profile.source_columns
        } - set(profile.source_columns) - ROLE_FIELDS
        populated_foreign = {
            field: output[field]
            for field in foreign_fields
            if output.get(field) is not None
        }
        if populated_foreign:
            raise ValueError(
                f"{source_id!r} resumed row populated foreign source fields "
                f"{sorted(populated_foreign)}"
            )
        for field in foreign_fields:
            output.pop(field, None)
        payload: dict[str, Any] = {}
        role_values = {
            field: value
            for field, value in (
                (profile.endpoint_field, record.get("endpoint_name")),
                (profile.measurement_field, record.get("measurement_text")),
                (profile.unit_field, record.get("unit_text")),
                (profile.smiles_field, record.get("smiles")),
            )
            if field
        }
        for field in profile.source_columns:
            payload[field] = role_values.get(field, record.get(field))
        output["source_payload_json"] = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, default=str
        )
        output.setdefault(
            "source_smiles",
            record.get("canonical_smiles")
            if profile.structure_mode == "mapped"
            else record.get("smiles"),
        )
        output.setdefault("evidence_context_json", "{}")
        return output

    def inflate_canonical(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Restore private legacy aliases for unchanged organization code."""
        output = self.inflate_cleaned(record)
        for legacy, canonical in _LEGACY_CANONICAL_FIELDS.items():
            if canonical in record:
                output[legacy] = record.get(canonical)
        profile = self.source(str(record.get("source_id") or ""))
        for dimension in profile.canonical_dimensions:
            legacy = dimension.legacy_value_field
            if legacy and dimension.output_field in record:
                output[legacy] = record.get(dimension.output_field)
        return output


def _source_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = record.get("source_payload_json")
    if payload is None:
        return record
    try:
        parsed = json.loads(str(payload))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid source_payload_json") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("source_payload_json must contain an object")
    return parsed


def _measurement_kind(record: Mapping[str, Any]) -> str:
    if record.get("finite_scalar_value") is not None:
        return "continuous"
    return "non_scalar"


def _validate_source_projection(
    record: Mapping[str, Any], profile: SourceProfile
) -> None:
    missing = set(profile.cleaned_source_fields) - set(record)
    if missing:
        raise ValueError(
            f"{profile.source_id!r} cleaned projection lacks {sorted(missing)}"
        )
    leaked_roles = profile.raw_role_fields & set(record)
    # A raw field whose spelling already equals the universal role is not a
    # duplicate.  All other raw role aliases must be absent.
    leaked_roles -= ROLE_FIELDS
    if leaked_roles:
        raise ValueError(
            f"{profile.source_id!r} cleaned projection retains raw aliases "
            f"{sorted(leaked_roles)}"
        )
    fake_aliases = {"assay_context", "species_context"} & set(record)
    if fake_aliases and not fake_aliases.issubset(set(profile.source_columns)):
        raise ValueError(
            f"{profile.source_id!r} created semantic cleaned aliases "
            f"{sorted(fake_aliases)}"
        )


def _validate_canonical_projection(
    record: Mapping[str, Any], profile: SourceProfile
) -> None:
    required = {
        "canonical_record_id",
        "canonical_endpoint_name",
        "canonical_measurement_text",
        "canonical_unit_text",
        "canonical_smiles",
        "canonicalization_status",
        "measurement_kind",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(
            f"{profile.source_id!r} canonical projection lacks {sorted(missing)}"
        )
    missing_dimensions = set(profile.canonical_output_fields) - set(record)
    if missing_dimensions:
        raise ValueError(
            f"{profile.source_id!r} canonical projection lacks declared dimensions "
            f"{sorted(missing_dimensions)}"
        )
    selected_producers: dict[str, str] = {}
    atomic_groups: dict[str, set[str]] = {}
    for dimension in profile.canonical_dimensions:
        if not dimension.producer_id_field:
            continue
        selected = str(record.get(dimension.producer_id_field) or "")
        declared = {producer.producer_id: producer for producer in dimension.producers}
        if selected not in declared:
            raise ValueError(
                f"{profile.source_id!r} {dimension.output_field!r} uses undeclared "
                f"producer {selected or '<missing>'!r}"
            )
        missing_inputs = set(declared[selected].input_fields) - set(record)
        if missing_inputs:
            raise ValueError(
                f"{profile.source_id!r} producer {selected!r} lacks cleaned inputs "
                f"{sorted(missing_inputs)}"
            )
        previous = selected_producers.setdefault(dimension.producer_id_field, selected)
        if previous != selected:
            raise ValueError(
                f"{profile.source_id!r} producer field "
                f"{dimension.producer_id_field!r} is not deterministic"
            )
        if dimension.atomic_group:
            atomic_groups.setdefault(dimension.atomic_group, set()).add(selected)
    for atomic_group, producers in atomic_groups.items():
        if len(producers) != 1:
            raise ValueError(
                f"{profile.source_id!r} atomic group {atomic_group!r} spans "
                f"producers {sorted(producers)}"
            )
    legacy = set(_LEGACY_CANONICAL_FIELDS) & set(record)
    if legacy:
        raise ValueError(
            f"{profile.source_id!r} canonical projection retains legacy fields "
            f"{sorted(legacy)}"
        )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    return str(value)


__all__ = [
    "CANONICAL_ARTIFACT_VERSION",
    "CANONICAL_RECORD_VERSION",
    "CanonicalProducerSpec",
    "CanonicalDimensionSpec",
    "PAIR_BUCKET_CONTRACT_VERSION",
    "PairBucketSpec",
    "RECORD_CONTRACT_VERSION",
    "ROLE_FIELDS",
    "SOURCE_CONTRACT_VERSION",
    "SourceProfile",
    "StarlingRecordContract",
]
