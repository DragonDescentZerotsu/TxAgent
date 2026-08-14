"""Controlled source-scoped encoders for non-scalar reported outcomes.

A record reaches a pair bucket only when its ``normalization_validity_status``
is ``valid`` and both ``canonical_endpoint`` and ``canonical_unit`` are
non-empty (``common/starling/pair_buckets.py``).  Categorical outcomes have no
measurement, so they are excluded — for Skin_Reaction that is the single reason
296,567 structurally resolved records never enter a bucket.

The task registry freezes the source, real input fields, parser, measurement
kind, and semantic definition. Binary and ordinal scales additionally freeze
canonical category IDs, ranks, and encoded values. Stage 02 attaches those
identities, Stage 04 owns bucket membership, and v7 Stage 05 calibrates distance
geometry without emitting a transfer label.

Two scales are defined, and they are deliberately different ``canonical_unit``
values so that the bucket key can never pool them:

``logit_response``
    The log-odds of a positive response, ``η = log(p / (1 − p))``.  Use it when
    the record carries a *rate* — either explicit counts or a reviewed ordinal
    grade anchored to a rate.  ``p = sigmoid(η)`` recovers the probability.

``signed_effect_direction``
    A signed hazard direction, negative for a protective effect, zero for no
    effect, positive for a harmful one.  Use it when the source reports a
    direction that is not an intensity — a protective result is the *opposite*
    of a positive one, not a weaker positive, so it cannot share an ordinal
    ladder with it.

An encoder must never invent a measurement.  Records whose outcome is
"inconclusive", "not classified" or otherwise uninformative get **no** value:
mapping them to the midpoint would fabricate evidence of no effect.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


CATEGORICAL_RESPONSE_CONTRACT_VERSION = "starling_categorical_response.v1"

LOGIT_RESPONSE_UNIT = "logit_response"
SIGNED_DIRECTION_UNIT = "signed_effect_direction"
ORDINAL_SEVERITY_UNIT = "ordinal_severity_grade"
BINARY_OUTCOME_UNIT = "binary_outcome_class"
ORDINAL_OUTCOME_UNIT = "ordinal_outcome_class"

# These units name latent scales, not physical dimensions.  A record carrying
# one of them is validated against the encoding contract rather than against a
# physical domain: the unit is deliberately unknown to the unit normalizer, and
# a log-odds or a signed direction is legitimately negative.
ENCODED_UNITS = frozenset(
    {
        BINARY_OUTCOME_UNIT,
        LOGIT_RESPONSE_UNIT,
        ORDINAL_OUTCOME_UNIT,
        SIGNED_DIRECTION_UNIT,
        ORDINAL_SEVERITY_UNIT,
    }
)
# Encoded units whose values may not be negative.
NON_NEGATIVE_ENCODED_UNITS = frozenset({ORDINAL_SEVERITY_UNIT})

# Jeffreys Beta(1/2, 1/2) prior.  Shrinkage is mandatory, not cosmetic: an
# unshrunk rate of 0 or 1 has an infinite logit, and both are common in real
# extraction data (a single subject who did or did not react).
JEFFREYS_PRIOR_ALPHA = 0.5
JEFFREYS_PRIOR_BETA = 0.5


@dataclass(frozen=True)
class CategoricalEncoding:
    """One encoder's verdict for one record."""

    encoder_id: str
    value: float
    unit: str
    measurement_text: str
    inputs: Mapping[str, Any]
    sample_size: int | None = None

    def record_fields(self, *, version: str) -> dict[str, Any]:
        """The fields to merge onto a normalized record."""
        return {
            "canonical_measurement": self.measurement_text,
            "canonical_unit": self.unit,
            "finite_scalar_value": self.value,
            "absolute_and_continuous_value": self.value,
            "is_absolute_and_continuous": True,
            "normalization_validity_status": "valid",
            "categorical_encoder_id": self.encoder_id,
            "categorical_encoding_version": version,
            "categorical_sample_size": self.sample_size,
        }


@dataclass(frozen=True)
class CanonicalCategory:
    """One reviewed member of a binary or ordinal measurement domain."""

    category_id: str
    rank: int
    encoded_value: float


@dataclass(frozen=True)
class ControlledMeasurementSpec:
    """Frozen source/field scope and geometry for one controlled encoder."""

    scale_id: str
    source_id: str
    input_fields: tuple[str, ...]
    encoder: Callable[[Mapping[str, Any]], CategoricalEncoding | None]
    kind: Literal["continuous", "binary", "ordinal"]
    parser_id: str
    definition: str
    categories: tuple[CanonicalCategory, ...] = ()

    def __post_init__(self) -> None:
        if not self.scale_id or not self.source_id or not self.parser_id:
            raise ValueError("controlled measurement IDs must be nonempty")
        if not self.input_fields or len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError(f"invalid controlled inputs for {self.scale_id!r}")
        if self.kind == "continuous":
            if self.categories:
                raise ValueError("continuous controlled measurements have no category domain")
            return
        expected = 2 if self.kind == "binary" else 3
        if len(self.categories) < expected:
            raise ValueError(
                f"{self.kind} scale {self.scale_id!r} requires at least {expected} categories"
            )
        ids = [item.category_id for item in self.categories]
        ranks = [item.rank for item in self.categories]
        values = [float(item.encoded_value) for item in self.categories]
        if len(set(ids)) != len(ids) or len(set(ranks)) != len(ranks):
            raise ValueError(f"duplicate category ID or rank for {self.scale_id!r}")
        if sorted(ranks) != list(range(len(ranks))):
            raise ValueError(f"category ranks must be contiguous for {self.scale_id!r}")
        if any(not math.isfinite(value) for value in values) or len(set(values)) != len(values):
            raise ValueError(f"invalid encoded category values for {self.scale_id!r}")

    def category_for_value(self, value: Any) -> CanonicalCategory | None:
        if self.kind == "continuous":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        for category in self.categories:
            if math.isclose(
                numeric,
                float(category.encoded_value),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                return category
        return None

    def manifest(self) -> dict[str, Any]:
        return {
            "scale_id": self.scale_id,
            "source_id": self.source_id,
            "input_fields": list(self.input_fields),
            "parser_id": self.parser_id,
            "kind": self.kind,
            "definition": self.definition,
            "categories": [
                {
                    "category_id": item.category_id,
                    "rank": item.rank,
                    "encoded_value": item.encoded_value,
                }
                for item in self.categories
            ],
        }


@dataclass(frozen=True)
class CategoricalResponsePolicy:
    """A task's ordered encoders and the version they are frozen under."""

    version: str
    # Ordered by precedence: the first encoder that returns a value wins, so a
    # record backed by real counts is never downgraded to an anchor.
    encoders: Sequence[Callable[[Mapping[str, Any]], CategoricalEncoding | None]] = ()
    controlled_measurements: Sequence[ControlledMeasurementSpec] = ()

    def __post_init__(self) -> None:
        if bool(self.encoders) == bool(self.controlled_measurements):
            raise ValueError(
                "categorical response policy requires exactly one encoder declaration style"
            )
        scale_ids = [item.scale_id for item in self.controlled_measurements]
        if len(scale_ids) != len(set(scale_ids)):
            raise ValueError("controlled measurement scale IDs must be unique")

    @property
    def measurement_scales(self) -> dict[str, ControlledMeasurementSpec]:
        return {item.scale_id: item for item in self.controlled_measurements}

    def encode(self, record: Mapping[str, Any]) -> CategoricalEncoding | None:
        declared = (
            ((None, encoder) for encoder in self.encoders)
            if self.encoders
            else (
                (spec, spec.encoder)
                for spec in self.controlled_measurements
                if str(record.get("source_id") or "") == spec.source_id
            )
        )
        for spec, encoder in declared:
            encoding = encoder(record)
            if encoding is not None:
                if not math.isfinite(encoding.value):
                    raise ValueError(
                        f"encoder {encoding.encoder_id!r} produced a non-finite value"
                    )
                if spec is not None:
                    if encoding.encoder_id != spec.scale_id:
                        raise ValueError(
                            f"encoder returned {encoding.encoder_id!r}; expected {spec.scale_id!r}"
                        )
                    if spec.kind != "continuous" and spec.category_for_value(
                        encoding.value
                    ) is None:
                        raise ValueError(
                            f"encoder {spec.scale_id!r} produced an out-of-domain value"
                        )
                return encoding
        return None

    def manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "contract_version": CATEGORICAL_RESPONSE_CONTRACT_VERSION,
            "controlled_measurements": [
                item.manifest() for item in self.controlled_measurements
            ],
            "first_matching_encoder_wins": True,
            "real_scalar_precedence": True,
        }

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return the fields to merge, or an empty mapping to leave untouched.

        Only records that have no scalar of their own are considered, so a real
        measurement is never overwritten by an encoding.
        """
        if record.get("finite_scalar_value") is not None:
            return {}
        encoding = self.encode(record)
        if encoding is None:
            return {}
        return encoding.record_fields(version=self.version)


def sigmoid(value: float) -> float:
    """Numerically stable logistic, never overflowing on a large magnitude."""
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("sigmoid input must be finite")
    if x >= 0:
        decayed = math.exp(-x)
        return 1.0 / (1.0 + decayed)
    grown = math.exp(x)
    return grown / (1.0 + grown)


def logit(probability: float) -> float:
    """Inverse of :func:`sigmoid`; the caller must have shrunk 0 and 1 away."""
    p = float(probability)
    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError(
            "logit requires a probability strictly inside (0, 1); "
            "shrink the rate before calling"
        )
    return math.log(p / (1.0 - p))


def shrunk_rate(
    positive_count: float,
    total_count: float,
    *,
    alpha: float = JEFFREYS_PRIOR_ALPHA,
    beta: float = JEFFREYS_PRIOR_BETA,
) -> float:
    """Posterior-mean rate under a Beta prior, always strictly inside (0, 1).

    With the Jeffreys default a single positive observation gives 0.75 rather
    than 1.0, while ten out of ten gives 0.955 — so the sample size, not just
    the ratio, moves the encoded value away from the midpoint.
    """
    k = float(positive_count)
    n = float(total_count)
    if not math.isfinite(k) or not math.isfinite(n):
        raise ValueError("counts must be finite")
    if n <= 0:
        raise ValueError("total count must be positive")
    if k < 0 or k > n:
        raise ValueError("positive count must lie within [0, total]")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta prior parameters must be positive")
    return (k + alpha) / (n + alpha + beta)


def count_logit(
    positive_count: float,
    total_count: float,
    *,
    alpha: float = JEFFREYS_PRIOR_ALPHA,
    beta: float = JEFFREYS_PRIOR_BETA,
) -> float:
    """Shrunk log-odds of a rate.  Always finite, by construction."""
    return logit(shrunk_rate(positive_count, total_count, alpha=alpha, beta=beta))


def render_measurement(value: float) -> str:
    """Render an encoded value for the source-facing measurement column."""
    return f"{value:.6g}"


def encoded_unit_validity_status(record: Mapping[str, Any]) -> str | None:
    """Validate a categorically encoded record, or None if it is not one.

    Encoded records bypass the physical-domain checks: their unit is a latent
    scale the unit normalizer does not know, and a log-odds or signed direction
    is legitimately negative.  What is checked instead is that the encoding is
    internally coherent.
    """
    unit = str(record.get("canonical_unit") or "")
    if unit not in ENCODED_UNITS:
        return None
    if not str(record.get("categorical_encoder_id") or ""):
        return "encoded_unit_without_encoder"
    if (
        str(record.get("structure_status") or "") != "resolved"
        or not str(record.get("canonical_smiles") or "")
    ):
        return "unresolved_structure"
    if not str(record.get("canonical_endpoint") or ""):
        return "missing_canonical_endpoint"
    value = record.get("finite_scalar_value")
    if isinstance(value, bool) or value is None:
        return "non_scalar_measurement"
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "non_scalar_measurement"
    if not math.isfinite(scalar):
        return "non_scalar_measurement"
    if unit in NON_NEGATIVE_ENCODED_UNITS and scalar < 0.0:
        return "nonpositive_positive_scalar"
    return "valid"


__all__ = [
    "BINARY_OUTCOME_UNIT",
    "CATEGORICAL_RESPONSE_CONTRACT_VERSION",
    "ENCODED_UNITS",
    "JEFFREYS_PRIOR_ALPHA",
    "JEFFREYS_PRIOR_BETA",
    "LOGIT_RESPONSE_UNIT",
    "NON_NEGATIVE_ENCODED_UNITS",
    "ORDINAL_OUTCOME_UNIT",
    "ORDINAL_SEVERITY_UNIT",
    "SIGNED_DIRECTION_UNIT",
    "encoded_unit_validity_status",
    "CanonicalCategory",
    "CategoricalEncoding",
    "CategoricalResponsePolicy",
    "ControlledMeasurementSpec",
    "count_logit",
    "logit",
    "render_measurement",
    "shrunk_rate",
    "sigmoid",
]
