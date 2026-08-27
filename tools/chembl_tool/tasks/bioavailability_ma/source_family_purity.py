"""Source-family purity rules for oral-bioavailability evidence.

These rules are intentionally about evidence distance, not gold eligibility.
An overall oral-F result is label-proximal even when it was extracted from a
transporter, absorption, or clearance paper.  Population, condition, report
type, and benchmark-vote eligibility remain separate downstream decisions.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


DIRECT_GROUP = "Observed.direct_oral_bioavailability"

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
    """Return why a row must be exposed at the direct-like L1 surface.

    The function does not decide whether a row can vote in the gold benchmark.
    In particular, relative/formulation/conditional F values are still moved to
    L1 while retaining their original scope and condition fields for audit.
    """

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


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()
