"""Conservative, source-only numerical coverage for explicit absolute oral F.

This does not infer F from exposure, assign task labels, or certify attribution.
Unsupported records retain their ordinary features and original prompt text.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Mapping

ABSOLUTE_ORAL_F_VERSION = "absolute_oral_f.v2"
_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
_VALUE = re.compile(
    rf"(?P<op><=|>=|<|>|≤|≥|~|≈)?\s*(?P<first>{_NUMBER})\s*"
    rf"(?P<unit1>%|percent|fraction)?\s*"
    rf"(?:(?P<join>±|\+/-|–|—|-|to)\s*(?P<second>{_NUMBER})\s*"
    rf"(?P<unit2>%|percent|fraction)?)?"
    r"\s*(?:\((?P<statistic>sd|s\.d\.|se|sem|s\.e\.m\.|ci)\))?"
)
_ABSOLUTE = re.compile(r"\babsolute\s+(?:oral\s+)?bioavailability\b")
_RELATIVE = re.compile(
    r"\brelative\s+(?:(?:oral|systemic)\s+)?(?:bioavailability|ba|f)\b|"
    r"\brelative comparison\b|\bbioequivalen\w*\b|\bfold\b|\b(?:increase|decrease|reduction)\s+(?:in|of)\s+bioavailability\b"
)
_MISSING = {"", "none", "null", "nan", "unknown", "unspecified", "not reported"}
_UNIT_SCALE = {"%": 1.0, "percent": 1.0, "percentage": 1.0, "fraction": 100.0}
_VALUE_FIELDS = ("canonical_measurement_text", "measurement_text", "result_value", "quantitative_value")
_UNIT_FIELDS = ("canonical_unit_text", "unit_text", "result_unit", "measurement_unit")


def _text(value: Any) -> str:
    text = " ".join(unicodedata.normalize("NFKC", "" if value is None else str(value)).lower().split())
    return "" if text in _MISSING else text


def _entity_name(description: str, focal: str) -> str | None:
    """Separate a stated name from dose/form wording; unknown aliases stay literal."""
    if focal and not focal.startswith("smiles:") and re.search(r"(?<!\w)" + re.escape(focal) + r"(?!\w)", description):
        return focal
    name = re.sub(r"^(?:the |a |an |oral |prodrug )+", "", description)
    name = re.sub(
        rf"^{_NUMBER}\s*(?:mg|μg|ug|g)(?:\s*(?:/kg|kg[-−]?1))?\s*(?:of\s+)?", "", name
    )
    name = re.split(r"\s+as\s+|\s+administered\s+|\s*\(", name)[0].strip()
    if (not name or name.startswith("smiles:") or re.search(
            r"\b(?:dose|solution|suspension|tablets?|capsules?|bodyweight|body weight|"
            r"drug|prodrug|metabolite|moiety|is|was|were|limited|bioavailability)\b", name)
            or not re.search(r"[a-zα-ω]", name) or "%" in name or len(name.split()) > 3):
        return None
    return name


def _measurement_binding(prepared: Mapping[str, Any], fields: Mapping[str, str], body: str) -> dict:
    """Keep explicit dose/analyte relationships instead of discarding their values.

    Native administered-as fields and explicit source phrases supply candidate
    bindings. Generic wording does not hide an explicit name; disagreements
    stay unresolved. Names/aliases are not resolved to chemical structures.
    """
    payload = prepared.get("input", {})
    focal = _text(payload.get("focal_molecule"))
    support = _text(payload.get("support_text"))
    measured_description = focal
    measured_match = re.search(
        r"\b(?:bioavailability|absolute f)\s+(?:\([^)]*\)\s+)?of\s+(.+?)"
        r"(?=\s+(?:following|after|from|was|were|is|in)\b|[,;.]|$)", support,
    )
    if measured_match and re.search(r"[a-z]", measured_match[1]) and "%" not in measured_match[1]:
        measured_description = measured_match[1].strip()
    measured = _entity_name(measured_description, focal)
    dose_candidates = []
    evidence = []
    if measured_description != focal:
        evidence.append({"field": "support_text", "quote": payload.get("support_text")})
    def candidate(description, field):
        item = {"description": description.strip(), "entity": _entity_name(description.strip(), focal),
                "field": field, "quote": payload.get("support_text") if field == "support_text"
                else payload.get("source_fields", {}).get(field)}
        if item not in dose_candidates:
            dose_candidates.append(item)

    for field in ("qualifying_conditions", "oral_exposure_mode", "molecular_form"):
        match = re.search(r"\b(?:administered|given|dosed) as\s+([^|;.]+)", fields.get(field, ""))
        if match:
            candidate(match[1], field)
    # A generic native "a prodrug" must not hide an explicit name in the source.
    # Retain all explicit candidates so disagreements cannot silently overwrite.
    patterns = (
        r"\boral administration of\s+(.+?)(?=\s+(?:was|were|at|in|to|resulted|showed)\b|[,;.]|$)",
        r"\boral dose of\s+(.+?)(?=\s+(?:is|was|were|at|in|to|resulted|showed|"
        r"demonstrated|yielded|exhibited|following|reaching|co-administered)\b|[,;.]|$)",
        r"\b([a-z][\wβ-]*)\s*,\s*administered as\s+[^.;]+?\bpro[- ]?drug\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, support):
            candidate(match[1], "support_text")
    names = {c["entity"] for c in dose_candidates if c["entity"]}
    dosed = next(iter(names)) if len(names) == 1 else None
    chosen = next((c for c in dose_candidates if c["entity"] == dosed and dosed),
                  dose_candidates[0] if dose_candidates else None)
    dosed_description = chosen["description"] if chosen else ""
    for c in dose_candidates:
        span = {"field": c["field"], "quote": c["quote"]}
        if span not in evidence:
            evidence.append(span)
    scope = "reported_focal_f"
    if len(names) > 1:
        scope = "dosed_entity_conflict"
    elif dosed and measured and dosed != measured:
        scope = "dosed_measured_names_differ"
    elif dosed_description and not dosed:
        scope = "dosed_entity_unresolved"
    elif dosed and not measured:
        scope = "dosed_entity_known_analyte_unspecified"
    elif re.search(r"\bmetabolit\w*\b|\bactive moiety\b", body):
        scope = "metabolite_context"
    elif re.search(r"\bpro[- ]?drug\w*\b", body):
        scope = "prodrug_context"
    return {
        "reported_quantity": "absolute_oral_f",
        "dosed_entity": dosed or None,
        "dosing_description": dosed_description or None,
        "dose_candidates": dose_candidates,
        "measured_entity": measured or None,
        "measurement_entity_description": measured_description or None,
        "reported_record_entity": focal or None,
        "analyte_scope": scope,
        "analyte_fraction": (
            "unbound" if re.search(r"\bunbound\b", body)
            else "free_drug_as_reported" if "free drug concentration" in body
            else "not_specified"
        ),
        "labelled_measurement": bool(re.search(r"\bradioactiv\w*\b|\bradiolabel\w*\b|\b(?:14c|13c\d*)\b", body)),
        "reference": fields.get("comparator", "") or fields.get("comparator_exposure", "") or None,
        "evidence": evidence,
    }


def absolute_oral_f(prepared: Mapping[str, Any]) -> dict:
    """Return source measurements and normalized coverage, or an abstention reason.

    Units must be explicit: a unitless 0.5 is not assumed to mean 50%.
    A stated estimate ± uncertainty covers its central estimate; uncertainty is
    retained separately, not treated as an interval of equally likely values.
    Genuine ranges/bounds share a fixed coverage budget over intersected bins.
    """
    source = prepared.get("input", {}).get("source_fields", {})
    if not source:
        source = {k: v for group in prepared.get("groups", {}).values() for k, v in group.items()}
    fields = {k: _text(v) for k, v in source.items()}
    endpoint = _text(prepared.get("endpoint") or fields.get("canonical_endpoint_name"))
    endpoint = endpoint.replace("_", " ")

    def skip(reason):
        return {"status": "skipped", "reason": reason}

    if "source_requests_context_review" in prepared.get("llm_reasons", []):
        return skip("source_requests_context_review")
    if endpoint not in {"oral bioavailability", "absolute oral bioavailability", "bioavailability"}:
        return skip("not_oral_f_endpoint")
    support = _text(prepared.get("input", {}).get("support_text"))
    body = " | ".join([*fields.values(), support, _text(prepared.get("input", {}).get("focal_molecule"))])
    report_type = fields.get("bioavailability_report_type", "")
    if report_type and report_type not in {"absolute", "extent_f", "systemic_availability"}:
        return skip("nonabsolute_report_type")
    if _RELATIVE.search(body):
        return skip("relative_or_mixed_measurement")
    if re.search(r"\bfraction absorbed\b|\bfraction of (?:the )?dose absorbed\b", body):
        return skip("absorption_vs_availability_unresolved")
    if report_type != "absolute" and not _ABSOLUTE.search(body):
        return skip("absolute_not_explicit")
    if "oral" not in endpoint and not re.search(r"\boral\b|\bp\.?o\.?\b", body):
        return skip("oral_route_not_explicit")
    # Retain named IV references as context; an oral formulation comparator is
    # not an absolute-F reference and cannot silently become one.
    for comparator in (fields.get("comparator", ""), fields.get("comparator_exposure", "")):
        if comparator and not re.match(
            r"(?:(?:vs\.?|versus|compared (?:to|with))\s+)?(?:i\.?v\.?|intravenous)(?=\s|$)",
            comparator,
        ):
            return skip("comparator_needs_review")

    value = next((fields[k] for k in _VALUE_FIELDS if fields.get(k)), "")
    match = _VALUE.fullmatch(value)
    if not match:
        return skip("unsupported_value_expression")
    units = [fields[k] for k in _UNIT_FIELDS if fields.get(k)]
    units.extend(u for u in [match["unit1"], match["unit2"]] if u)
    if not units or any(u not in _UNIT_SCALE for u in units):
        return skip("unit_missing_or_unsupported")
    scales = {_UNIT_SCALE[u] for u in units}
    if len(scales) != 1:
        return skip("conflicting_units")
    scale = scales.pop()
    first = float(match["first"]) * scale
    second = float(match["second"]) * scale if match["second"] else None
    op = match["op"] or "="
    join = match["join"]
    if not math.isfinite(first) or (second is not None and not math.isfinite(second)):
        return skip("nonfinite_value")
    if second is not None and op not in {"=", "~", "≈"}:
        return skip("compound_bound_expression")
    low, high = first, first
    kind, central, uncertainty = "point", first, None
    low_open = high_open = False
    if second is not None:
        if join in {"±", "+/-"}:
            low, high = first - second, first + second
            kind, uncertainty = "estimate_with_uncertainty", second
        else:
            low, high = first, second
            kind, central = "range", None
    elif op in {"<", "<=", "≤"}:
        low, high, high_open = 0.0, first, op == "<"
        kind, central = "upper_bound", None
    elif op in {">", ">=", "≥"}:
        low, high, low_open = first, 100.0, op == ">"
        kind, central = "lower_bound", None
    if (central is not None and not 0 <= central <= 100) or (central is None and not 0 <= low <= high <= 100):
        return skip("outside_absolute_f_range")
    if central is None and low == high and (low_open or high_open):
        return skip("empty_bound")
    # Equal-width, fixed intervals: [0,20), ..., [80,100]. No gold labels used.
    if central is not None or low == high:
        bucket = min(int((central if central is not None else low) // 20), 4)
        shares = {f"{bucket * 20}-{(bucket + 1) * 20}": 1.0}
    else:
        shares = {
            f"{edge}-{edge + 20}": overlap / (high - low)
            for edge in range(0, 100, 20)
            if (overlap := min(high, edge + 20) - max(low, edge)) > 0
        }
    # Do not equate records across reported systems/conditions/comparators.
    context = {
        group: {k: _text(v) for k, v in prepared.get("groups", {}).get(group, {}).items()
                if k not in {"bioavailability_report_type", "extra_details"} and _text(v)}
        for group in ("assay", "system", "exposure", "comparison_role")
    }
    binding = _measurement_binding(prepared, fields, body)
    context["measurement_binding"] = {
        key: binding[key] for key in ("analyte_scope", "analyte_fraction", "labelled_measurement")
    }
    return {
        "status": "eligible", "quantity": "absolute_oral_f", "measurement_binding": binding,
        "lower_percent": low, "upper_percent": high,
        "value_kind": kind, "central_percent": central,
        "uncertainty_percent": uncertainty, "uncertainty_statistic": match["statistic"],
        "lower_open": low_open, "upper_open": high_open,
        "reported_expression": value, "bin": next(iter(shares)) if len(shares) == 1 else None,
        "coverage_shares": shares, "context": context,
    }


def absolute_oral_f_features(prepared: Mapping[str, Any]) -> dict[str, float]:
    """Fixed initial sqrt-gain budget, independent of interval width or precision."""
    result = absolute_oral_f(prepared)
    if result["status"] != "eligible":
        return {}
    context = json.dumps(result["context"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    # The selector uses sqrt coverage. Squared shares make total initial gain
    # sum(sqrt(share**2)) == 1, so a wide interval gets no larger novelty budget.
    return {f"numeric/{ABSOLUTE_ORAL_F_VERSION}/{context}/bin:{name}": share ** 2
            for name, share in result["coverage_shares"].items()}
