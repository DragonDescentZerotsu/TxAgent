"""Controlled categorical outcomes for Bioavailability_Ma Starling evidence.

The encoders only fill records that have no usable scalar measurement.  Their
signed anchors are internal distance geometry: they are not benchmark labels,
probabilities, or deterministic oral-bioavailability decisions.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from data.processing.evidence_library.shared.v1.categorical_response import (
    BINARY_OUTCOME_UNIT,
    ORDINAL_OUTCOME_UNIT,
    CanonicalCategory,
    CategoricalEncoding,
    CategoricalResponsePolicy,
    ControlledMeasurementSpec,
    render_measurement,
)
from data.processing.evidence_library.versions.v8.tasks.bioavailability_ma.starling_record_canonicalization import (
    DIRECT_EVIDENCE_SCOPE,
    bioavailability_evidence_scope,
)


CATEGORICAL_RESPONSE_VERSION = "bioavailability_ma_categorical_response.v5"
FG_TARGET_ALIAS_VERSION = "bioavailability_fg_target_aliases.v1"

_DIRECT_AMBIGUOUS_QUALITATIVE_PATTERNS = (
    r"\bvariable\b",
    r"\bunpredictable\b",
    r"\borally bioavailable\b",
    r"\borally available\b",
)
_DIRECT_OUTCOME_NOUN = (
    r"(?:bioavailability|oral bioavailability|systemic bioavailability|"
    r"availability|oral availability|systemic availability)"
)
_DIRECT_POSITIVE_STRENGTH = r"(?:high|very high|good|very good|excellent)"
_DIRECT_NEGATIVE_STRENGTH = (
    r"(?:low|very low|extremely low|poor|very poor|extremely poor|"
    r"negligible|minimal)"
)
_DIRECT_POSITIVE_COMPLETION = (
    r"(?:complete|near complete|near-complete|nearly complete|almost complete|"
    r"virtually complete|essentially complete)"
)
_DIRECT_POSITIVE_FULL = re.compile(
    rf"^(?:{_DIRECT_POSITIVE_STRENGTH}(?: {_DIRECT_OUTCOME_NOUN})?|"
    rf"{_DIRECT_POSITIVE_COMPLETION}(?: {_DIRECT_OUTCOME_NOUN})?|"
    r"(?:completely|nearly completely|almost completely) bioavailable|"
    r"completely available)$"
)
_DIRECT_NEGATIVE_FULL = re.compile(
    rf"^{_DIRECT_NEGATIVE_STRENGTH}(?: {_DIRECT_OUTCOME_NOUN})?$"
)
_DIRECT_MIDDLE_FULL = re.compile(
    r"^(?:moderate|intermediate)(?: "
    r"(?:(?:absolute|oral|systemic|absolute oral|oral absolute|systemic oral) )?"
    r"(?:bioavailability|availability)| ba)?$"
)
_RELATIVE_DIRECT_PATTERN = re.compile(
    r"\b(?:fold|times|relative)\b|"
    r"\b(?:increase|increased|decrease|decreased)\s+by\b|"
    r"\b(?:higher|lower)\s+than\b",
    flags=re.IGNORECASE,
)
_NULL_LIKE = frozenset({"", "nan", "none", "null", "n/a", "na", "unknown", "unspecified"})
_COMPOUND_SEPARATOR = re.compile(r"\s*(?:,|;|\band\b)\s*", flags=re.IGNORECASE)


def _alias_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _source_token(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_")


_TARGET_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "ABCB1": (
        "ABCB1",
        "P-gp",
        "PGP",
        "Pgp",
        "P-glycoprotein",
        "MDR1",
        "P-gp/ABCB1",
        "Pgp/ABCB1",
        "P-glycoprotein/ABCB1",
        "P-glycoprotein (ABCB1)",
        "MDR1/ABCB1",
        "ABCB1/P-gp",
    ),
    "ABCG2": (
        "ABCG2",
        "BCRP",
        "BCRP1",
        "BCRP/ABCG2",
        "ABCG2/BCRP",
        "Bcrp/ABCG2",
        "Bcrp1/ABCG2",
        "BCRP/Bcrp1",
    ),
    "ABCC1": ("ABCC1", "MRP1", "MRP1/ABCC1"),
    "ABCC2": (
        "ABCC2",
        "MRP2",
        "MRP-2",
        "MRP2/ABCC2",
        "MRP-2/ABCC2",
        "MRP2 (ABCC2)",
    ),
    "ABCC3": ("ABCC3", "MRP3", "MRP3/ABCC3"),
    "ABCC4": ("ABCC4", "MRP4", "MRP4/ABCC4"),
    "SLC15A1": (
        "SLC15A1",
        "PEPT1",
        "PepT1",
        "hPEPT1",
        "PEPT1/SLC15A1",
        "PEPT1 (SLC15A1)",
        "hPEPT1 (SLC15A1)",
        "PepT1/PEPT1",
    ),
    "SLC5A1": ("SLC5A1", "SGLT1", "SGLT-1"),
    "SLCO2B1": ("SLCO2B1", "OATP2B1"),
    "SLCO1A2": ("SLCO1A2", "OATP1A2", "OATP-A"),
    "SLC16A1": ("SLC16A1", "MCT1"),
    "SLC22A1": ("SLC22A1", "OCT1"),
    "SLC22A4": ("SLC22A4", "OCTN1"),
    "SLC22A5": ("SLC22A5", "OCTN2"),
    # These exact family/isoform IDs are intentionally distinct.
    "CYP3A": ("CYP3A",),
    "CYP3A4": ("CYP3A4", "CYP 3A4"),
    "CYP3A5": ("CYP3A5",),
    "UGT": ("UGT", "UGTs", "UDP-glucuronosyltransferase (UGT)"),
    "UGT1A1": ("UGT1A1",),
    "NPC1L1": ("NPC1L1",),
}


def _build_target_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for target_id, values in _TARGET_ALIAS_GROUPS.items():
        for value in values:
            key = _alias_key(value)
            previous = aliases.setdefault(key, target_id)
            if previous != target_id:
                raise ValueError(
                    f"Fg target alias {value!r} maps to both {previous!r} and {target_id!r}"
                )
    return aliases


_TARGET_ALIASES = _build_target_aliases()
_EXACT_COMPOUND_TARGETS = {
    _alias_key("ABCB1/ABCG2"): ("ABCB1", "ABCG2"),
}


def canonical_fg_target_id(value: Any) -> str | None:
    """Return a conservative target boundary for one Fg categorical row."""
    raw = str(value or "").strip()
    if raw.casefold() in _NULL_LIKE:
        return None
    key = _alias_key(raw)
    if key in _TARGET_ALIASES:
        return _TARGET_ALIASES[key]
    if key in _EXACT_COMPOUND_TARGETS:
        return "+".join(_EXACT_COMPOUND_TARGETS[key])

    pieces = [piece for piece in _COMPOUND_SEPARATOR.split(raw) if piece.strip()]
    if len(pieces) > 1:
        targets = [canonical_fg_target_id(piece) for piece in pieces]
        if all(targets):
            return "+".join(sorted(set(str(target) for target in targets)))

    token = _source_token(raw)
    return f"source_token:{token}" if token else None


def classify_direct_qualitative_text(value: Any) -> tuple[str | None, str]:
    """Classify only a complete controlled direct-F phrase.

    This evidence-normalization policy intentionally does not interpret
    sentences.  Extra context, negation, comparison, or modality makes the
    source value non-scalar while leaving its original text available as
    evidence.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"\s+", " ", text).strip(" .;:")
    if not text:
        return None, "missing_bioavailability_value"
    if _RELATIVE_DIRECT_PATTERN.search(text):
        return None, "relative_not_absolute_bioavailability"
    if re.search(r"\d", text):
        return None, "numeric_or_compound_not_categorical"
    if any(
        re.search(pattern, text)
        for pattern in _DIRECT_AMBIGUOUS_QUALITATIVE_PATTERNS
    ):
        return None, "qualitative_value_not_threshold_anchored"
    if _DIRECT_MIDDLE_FULL.fullmatch(text):
        return "middle", "explicit_qualitative_middle"
    if _DIRECT_POSITIVE_FULL.fullmatch(text):
        return "high", "explicit_qualitative_high"
    if _DIRECT_NEGATIVE_FULL.fullmatch(text):
        return "low", "explicit_qualitative_low"
    return None, "unmapped_or_ambiguous_qualitative_value"


def _categorical_encoding(
    *,
    encoder_id: str,
    value: float,
    unit: str,
    inputs: Mapping[str, Any],
) -> CategoricalEncoding:
    return CategoricalEncoding(
        encoder_id=encoder_id,
        value=value,
        unit=unit,
        measurement_text=render_measurement(value),
        inputs=inputs,
    )


def encode_direct_oral_bioavailability(
    record: Mapping[str, Any],
) -> CategoricalEncoding | None:
    if (
        str(record.get("source_id") or "") != "hf_bioavailability"
        or bioavailability_evidence_scope(
            record.get("bioavailability_report_type")
        )
        != DIRECT_EVIDENCE_SCOPE
    ):
        return None
    raw = record.get("measurement_text")
    category, _ = classify_direct_qualitative_text(raw)
    if category is None:
        return None
    return _categorical_encoding(
        encoder_id="direct_oral_bioavailability_ordinal.v1",
        value={"low": -1.0, "middle": 0.0, "high": 1.0}[category],
        unit=ORDINAL_OUTCOME_UNIT,
        inputs={"measurement_text": raw},
    )


def encode_fg_substrate_status(
    record: Mapping[str, Any],
) -> CategoricalEncoding | None:
    if str(record.get("source_id") or "") != "fg":
        return None
    raw_status = record.get("substrate_status")
    status = re.sub(
        r"_+",
        "_",
        re.sub(r"[^a-z0-9]+", "_", str(raw_status or "").strip().casefold()),
    ).strip("_")
    if status not in {"substrate", "not_substrate"}:
        return None
    raw_target = record.get("transporter_or_enzyme")
    target_id = canonical_fg_target_id(raw_target)
    if target_id is None:
        return None
    return _categorical_encoding(
        encoder_id="fg_substrate_status_binary.v1",
        value=1.0 if status == "substrate" else -1.0,
        unit=BINARY_OUTCOME_UNIT,
        inputs={
            "substrate_status": raw_status,
            "transporter_or_enzyme": raw_target,
            "canonical_target_id": target_id,
        },
    )


DIRECT_DOMAIN = (
    CanonicalCategory("low", 0, -1.0),
    CanonicalCategory("middle", 1, 0.0),
    CanonicalCategory("high", 2, 1.0),
)
FG_SUBSTRATE_DOMAIN = (
    CanonicalCategory("not_substrate", 0, -1.0),
    CanonicalCategory("substrate", 1, 1.0),
)

CONTROLLED_MEASUREMENTS = (
    ControlledMeasurementSpec(
        scale_id="direct_oral_bioavailability_ordinal.v1",
        source_id="hf_bioavailability",
        input_fields=("measurement_text", "bioavailability_report_type"),
        encoder=encode_direct_oral_bioavailability,
        kind="ordinal",
        parser_id="bioavailability.direct_qualitative_ordinal.v1",
        definition="explicit low, middle, or high direct oral-bioavailability wording",
        categories=DIRECT_DOMAIN,
    ),
    ControlledMeasurementSpec(
        scale_id="fg_substrate_status_binary.v1",
        source_id="fg",
        input_fields=("substrate_status", "transporter_or_enzyme"),
        encoder=encode_fg_substrate_status,
        kind="binary",
        parser_id="bioavailability.fg_substrate_status_binary.v1",
        definition="target-scoped not-substrate versus substrate outcome",
        categories=FG_SUBSTRATE_DOMAIN,
    ),
)
MEASUREMENT_SCALES = {
    item.scale_id: item for item in CONTROLLED_MEASUREMENTS
}
POLICY = CategoricalResponsePolicy(
    version=CATEGORICAL_RESPONSE_VERSION,
    controlled_measurements=CONTROLLED_MEASUREMENTS,
)


def encoding_policy_manifest() -> dict[str, Any]:
    return {
        **POLICY.manifest(),
        "version": CATEGORICAL_RESPONSE_VERSION,
        "units": [BINARY_OUTCOME_UNIT, ORDINAL_OUTCOME_UNIT],
        "direct_claim_scope": "all_explicit_source_claims_with_context_retained",
        "fg_target_alias_policy": {
            "version": FG_TARGET_ALIAS_VERSION,
            "reviewed_alias_groups": {
                key: list(values)
                for key, values in sorted(_TARGET_ALIAS_GROUPS.items())
            },
            "compound_policy": "canonicalize_components_and_sort",
            "unreviewed_nonempty_policy": "preserve_as_normalized_source_token",
            "missing_target_policy": "abstain",
            "family_isoform_collapse": False,
        },
        "abstentions": {
            "direct_ambiguous_relative_numeric_or_conflicting": True,
            "fg_inconclusive_inhibited_missing_or_unknown_status": True,
            "fg_missing_target": True,
        },
        "real_scalar_precedence": True,
        "parseable_unresolved_unit_measurement_precedence": True,
        "encoder_id_is_part_of_the_pair_bucket_key": True,
    }


__all__: Sequence[str] = (
    "CATEGORICAL_RESPONSE_VERSION",
    "CONTROLLED_MEASUREMENTS",
    "FG_TARGET_ALIAS_VERSION",
    "MEASUREMENT_SCALES",
    "POLICY",
    "canonical_fg_target_id",
    "classify_direct_qualitative_text",
    "encode_direct_oral_bioavailability",
    "encode_fg_substrate_status",
    "encoding_policy_manifest",
)
