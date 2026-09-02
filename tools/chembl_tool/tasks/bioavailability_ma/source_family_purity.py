"""Record-level family rules for the current oral-bioavailability retrieval source.

Only source records that emitted a base vote or passed condition review belong
to L1. Parent-level tie/agreement rejection does not erase record-level voting.
Other direct-like oral-F rows remain available as near-direct L2 evidence;
mechanism rows keep their original families.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from tools.chembl_tool.common.source_family_purity import FamilyMove


DIRECT_GROUP = "Observed.direct_oral_bioavailability"
NEAR_DIRECT_GROUP = "Observed.nondirect_oral_bioavailability"
VOTE_PURITY_VERSION = (
    "bioavailability_source_family_purity.legacy_record_supported_v2_vote_pure.v1"
)

_PERCENT = r"[-+]?\d+(?:\.\d+)?\s*%"
_BIOAVAILABILITY_BEFORE_PERCENT = re.compile(
    rf"\b(?:absolute|relative|apparent|oral|mean|median|systemic)?\s*"
    rf"bioavailability(?:\s*\(\s*f\s*\))?[^.;]{{0,100}}{_PERCENT}",
    re.IGNORECASE,
)
_PERCENT_BEFORE_BIOAVAILABILITY = re.compile(
    rf"{_PERCENT}[^.;]{{0,80}}\b(?:absolute|relative|apparent|oral)?\s*bioavailability\b",
    re.IGNORECASE,
)
_F_VALUE = re.compile(
    rf"\bf\s*(?:value)?\s*(?:=|:|of\b|was\b|were\b|is\b)[^.;]{{0,40}}{_PERCENT}",
    re.IGNORECASE,
)
_F_FRACTION_VALUE = re.compile(
    r"(?:\bbioavailability\s*\(\s*f\s*\)|\bf\s*(?:value)?)\s*"
    r"(?:=|:|of\b|was\b|were\b|is\b)\s*(?:about\s*)?"
    r"(?:0(?:\.\d+)?|1(?:\.0+)?)\b",
    re.IGNORECASE,
)
_QUALITATIVE_OVERALL_F = re.compile(
    r"\b(?:very\s+low|low|poor|negligible|minimal|high|good|excellent)\s+"
    r"(?:absolute|oral|systemic)?\s*bioavailability\b|"
    r"\b(?:absolute|oral|systemic)\s+bioavailability\s+(?:was|is|remained)\s+"
    r"(?:very\s+low|low|poor|negligible|minimal|high|good|excellent)\b",
    re.IGNORECASE,
)


def direct_like_bioavailability_reason(record: Mapping[str, Any]) -> str:
    """Return why a nonvoter belongs in near-direct rather than a mechanism family."""

    if _text(record.get("group_id")) == DIRECT_GROUP:
        return ""
    if _text(record.get("canonical_bioavailability_evidence_scope")).lower() == "direct":
        return "canonical_direct_scope_outside_direct_family"

    endpoint = _text(record.get("canonical_endpoint_name"))
    measurement = _text(record.get("canonical_measurement_text"))
    unit = _text(record.get("canonical_unit_text"))
    support = _text(record.get("support_text"))
    combined = " | ".join((endpoint, measurement, support))
    percentages = re.findall(_PERCENT, combined)
    relative_change_only = bool(
        len(percentages) == 1
        and re.search(
            rf"{_PERCENT}\s*(?:higher|lower|greater|less|increase|decrease)",
            combined,
            re.IGNORECASE,
        )
    )

    if (
        "bioavailability" in endpoint.lower()
        and re.search(r"\d", measurement)
        and ("%" in unit or "%" in measurement)
    ):
        return "structured_overall_bioavailability_value"
    if not relative_change_only and _BIOAVAILABILITY_BEFORE_PERCENT.search(combined):
        return "support_reports_overall_bioavailability_percent"
    if not relative_change_only and _PERCENT_BEFORE_BIOAVAILABILITY.search(combined):
        return "support_reports_percent_as_overall_bioavailability"
    if _F_VALUE.search(combined):
        return "support_reports_explicit_f_value"
    if _F_FRACTION_VALUE.search(combined):
        return "support_reports_explicit_fractional_f_value"
    if (
        "bioavailability" in endpoint.lower()
        and re.fullmatch(r"\s*(?:0(?:\.\d+)?|1(?:\.0+)?)\s*", measurement)
    ):
        return "structured_fractional_overall_bioavailability_value"
    if _QUALITATIVE_OVERALL_F.search(combined):
        return "support_reports_qualitative_overall_bioavailability"
    return ""


def upstream_record_key(record: Mapping[str, Any]) -> str:
    """Map a normalized retrieval row to the canonical-source upstream ID."""

    source = _text(record.get("source_id"))
    if source == "hf_bioavailability":
        record_id = _text(record.get("source_record_id"))
        return f"hf:{record_id}" if record_id else ""
    if source == "oral_exposure":
        try:
            source_index = int(record.get("source_row_number")) - 1
        except (TypeError, ValueError):
            return ""
        extraction_id = _text(record.get("extraction_id"))
        return f"local:{source_index}:{extraction_id}" if extraction_id else ""
    return ""


def vote_pure_family_move(
    record: Mapping[str, Any],
    voter_source_record_ids: set[str] | frozenset[str],
) -> FamilyMove:
    """Keep only actual gold-voter source records at L1.

    Every old/direct-like L1 candidate that is not in the exact frozen vote
    ledger moves to near-direct L2.  Mechanism rows that were never candidates
    for L1 remain in their existing families.
    """

    original = _text(record.get("group_id"))
    upstream_id = upstream_record_key(record)
    if upstream_id and upstream_id in voter_source_record_ids:
        if original == DIRECT_GROUP:
            return FamilyMove(new_group="", reason="")
        return FamilyMove(
            new_group=DIRECT_GROUP,
            reason="accepted_gold_vote_promoted_to_l1",
        )
    direct_like_reason = direct_like_bioavailability_reason(record)
    if original == DIRECT_GROUP or direct_like_reason:
        return FamilyMove(
            new_group=NEAR_DIRECT_GROUP,
            reason=(
                "nonvoter_removed_from_l1"
                if original == DIRECT_GROUP
                else f"direct_like_nonvoter_to_l2:{direct_like_reason}"
            ),
        )
    return FamilyMove(new_group="", reason="")


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()
