"""Assay-transfer-only canonical measurement transformations.

Source-visible measurement fields are retrieval evidence.  The canonical
measurement, unit, and scalar fields are a separate numerical contract for
assay transfer; changing their scale must never change ``retrieval_eligible``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.starling.pair_buckets import raw_pair_bucket_key


POLICY_VERSION = "assay_transfer_canonical_measurement.v1"
CANONICAL_TUPLE_CONTRACT_VERSION = "assay_transfer_canonical_tuple.v1"
LOG10_TRANSFORM = "log10.v1"
RAW_TRANSFORM = "raw.v1"
_PERCENT_UNIT_TARGETS = {
    "%": (0.01, "ratio"),
    "%/d": (0.01, "ratio/d"),
    "%/wk": (0.01, "ratio/wk"),
    "% applied dose": (0.01, "fraction of applied dose"),
    "% control": (0.01, "ratio to control"),
    "% of control": (0.01, "ratio to control"),
    "%·control·of": (0.01, "ratio to control"),
    "%·control·of·untreated": (0.01, "ratio to control"),
    "% recovered material": (0.01, "fraction recovered"),
    "% of applied dose/h": (0.01, "fraction of applied dose/h"),
    "%/cm^2·h": (0.01, "fraction/cm^2·h"),
    "mL/%·m^2·min": (100.0, "mL/fraction·m^2·min"),
}
_FACTOR_RE = re.compile(
    r"(?P<center>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?:\s*[±]\s*(?P<variation>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))?"
    r"\s*\(\s*(?:×|x)\s*10\s*(?:\^\s*)?"
    r"(?P<exponent>[+\-−‐‑]?\s*\d+)\s*[^)]*\)",
    flags=re.IGNORECASE,
)


def load_measurement_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract_version") != POLICY_VERSION:
        raise ValueError(f"unsupported assay-transfer measurement policy: {path}")
    decisions = payload.get("bucket_decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("assay-transfer measurement policy lacks bucket_decisions")
    invalid = {
        key: value
        for key, value in decisions.items()
        if not isinstance(value, Mapping)
        or value.get("transform") not in {"raw", "log10"}
    }
    if invalid:
        raise ValueError(f"invalid assay-transfer bucket decisions: {sorted(invalid)}")
    if not isinstance(payload.get("record_corrections", {}), Mapping):
        raise ValueError("record_corrections must be a mapping")
    ineligibility = payload.get("record_ineligibility", {})
    if not isinstance(ineligibility, Mapping):
        raise ValueError("record_ineligibility must be a mapping")
    invalid_ineligibility = {
        key: value
        for key, value in ineligibility.items()
        if not isinstance(value, (str, Mapping))
        or (isinstance(value, Mapping) and not str(value.get("reason") or ""))
    }
    if invalid_ineligibility:
        raise ValueError(f"invalid record_ineligibility: {sorted(invalid_ineligibility)}")
    return payload


def finalize_assay_transfer_measurement(
    working: Mapping[str, Any],
    projected: Mapping[str, Any],
    *,
    record_contract: Any,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Finalize the reviewed raw/log10 canonical tuple after v7 projection."""
    updated_working = dict(working)
    updated_projected = dict(projected)
    scalar = _finite(projected.get("finite_scalar_value"))
    kind = str(projected.get("measurement_kind") or "")
    raw_key = raw_pair_bucket_key(updated_projected, record_contract=record_contract)
    pretransform_scalar = scalar
    pretransform_measurement = updated_projected.get("canonical_measurement_text")
    pretransform_unit = updated_projected.get("canonical_unit_text")
    pretransform_variation = updated_projected.get("variation_value")
    validity = str(
        working.get("normalization_validity_status")
        or projected.get("canonicalization_status")
        or ""
    )
    decision = (
        policy["bucket_decisions"].get(raw_key, {"transform": "raw"})
        if validity in {"", "valid"}
        else {"transform": "raw"}
    )
    transform = str(decision["transform"])
    if transform == "log10":
        if kind != "continuous" or scalar is None or scalar <= 0:
            raise ValueError(f"log10 policy targets an invalid record in {raw_key}")
        unit = str(updated_projected.get("canonical_unit_text") or "")
        if unit.startswith(("log10(", "-log10(")):
            raise ValueError(f"log10 policy would double-transform {raw_key}")
        scalar = math.log10(scalar)
        _set_log_value(updated_working, updated_projected, scalar, unit)
    elif (
        kind == "continuous"
        and scalar is not None
        and str(working.get("measurement_resolution_status") or "")
        not in {"ok", "relative", "unsure", "unavailable"}
    ):
        text = _measurement_text(scalar, _finite(updated_projected.get("variation_value")))
        updated_working["canonical_measurement"] = text
        updated_projected["canonical_measurement_text"] = text
    provenance = {
        "measurement_kind": kind,
        "assay_transfer_measurement_contract_version": (
            CANONICAL_TUPLE_CONTRACT_VERSION
        ),
        "assay_transfer_transform_id": (
            LOG10_TRANSFORM if transform == "log10" else RAW_TRANSFORM
        ),
        "assay_transfer_policy_key": raw_key,
        "assay_transfer_pretransform_measurement_text": pretransform_measurement,
        "assay_transfer_pretransform_scalar_value": pretransform_scalar,
        "assay_transfer_pretransform_unit_text": pretransform_unit,
        "assay_transfer_pretransform_variation_value": pretransform_variation,
        "assay_transfer_measurement_policy_version": str(policy["policy_version"]),
    }
    updated_working.update(provenance)
    updated_projected.update(provenance)
    return updated_working, updated_projected


def canonicalize_assay_transfer_base(
    record: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Apply base-unit scaling before the task validity/reference recheck."""
    updated = dict(record)
    updated.update(
        {
            "assay_transfer_prebase_measurement_text": record.get(
                "canonical_measurement"
            ),
            "assay_transfer_prebase_unit_text": record.get("canonical_unit"),
            "assay_transfer_prebase_scalar_value": record.get(
                "finite_scalar_value"
            ),
        }
    )
    scalar = _finite(record.get("finite_scalar_value"))
    factor, status, candidates, target_unit = _policy_scale_factor(
        record, scalar, policy
    )
    changed = scalar is not None and factor is not None
    if changed:
        scalar = scalar * factor
        _set_base_scalar(updated, scalar, factor=factor)
        if target_unit:
            updated["canonical_unit"] = target_unit
    updated.update(
        {
            "assay_transfer_scale_factor": factor,
            "assay_transfer_scale_status": status,
            "assay_transfer_support_scale_factors_json": json.dumps(candidates),
            "assay_transfer_measurement_policy_version": str(
                policy["policy_version"]
            ),
        }
    )
    return updated, changed


def validate_final_assay_transfer_measurements(
    records: list[Mapping[str, Any]],
) -> list[str]:
    """Validate the persisted post-scale/post-transform canonical tuple."""
    from tools.chembl_tool.common.starling.normalization.measurements import (
        parse_point_measurement,
    )

    errors: list[str] = []
    for record in records:
        transform = str(record.get("assay_transfer_transform_id") or "")
        if not transform:
            continue
        record_id = str(
            record.get("canonical_record_id")
            or record.get("normalized_record_id")
            or "<missing>"
        )
        scalar = _finite(record.get("finite_scalar_value"))
        if record.get("assay_transfer_measurement_contract_version") != (
            CANONICAL_TUPLE_CONTRACT_VERSION
        ):
            errors.append(f"{record_id}: canonical tuple contract mismatch")
            continue
        measurement_text = (
            record.get("canonical_measurement_text")
            or record.get("canonical_measurement")
        )
        exact = str(record.get("measurement_resolution_status") or "") in {
            "ok",
            "relative",
            "unsure",
            "unavailable",
        }
        parsed_value = _finite(measurement_text) if exact else parse_point_measurement(
            measurement_text
        ).value
        kind = str(record.get("measurement_kind") or "")
        if kind == "continuous" and not _same_optional_number(scalar, parsed_value):
            errors.append(f"{record_id}: canonical text/scalar mismatch")
            continue
        has_absolute_tuple = (
            "is_absolute_and_continuous" in record
            or "absolute_and_continuous_value" in record
        )
        if kind == "continuous" and has_absolute_tuple and (
            not record.get("is_absolute_and_continuous")
            or not _same_optional_number(
                scalar, _finite(record.get("absolute_and_continuous_value"))
            )
        ):
            errors.append(f"{record_id}: absolute continuous scalar mismatch")
            continue
        prevalue = _finite(record.get("assay_transfer_pretransform_scalar_value"))
        preunit = str(record.get("assay_transfer_pretransform_unit_text") or "")
        unit = str(
            record.get("canonical_unit_text") or record.get("canonical_unit") or ""
        )
        if transform == RAW_TRANSFORM:
            if not _same_optional_number(scalar, prevalue) or unit != preunit:
                errors.append(f"{record_id}: raw transform changed value or unit")
        elif transform == LOG10_TRANSFORM:
            if kind != "continuous":
                errors.append(f"{record_id}: log10 transform targets {kind!r}")
                continue
            expected = math.log10(prevalue) if prevalue is not None and prevalue > 0 else None
            if not _same_optional_number(scalar, expected) or unit != f"log10({preunit})":
                errors.append(f"{record_id}: invalid log10 canonical tuple")
        else:
            errors.append(f"{record_id}: unsupported transform {transform!r}")
    return errors


def _same_optional_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _support_scale_factor(
    record: Mapping[str, Any], scalar: float | None
) -> tuple[float | None, str, list[float]]:
    if scalar is None or str(record.get("unit_notation_status") or "") != "none":
        return None, "not_applicable_or_already_applied", []
    measurement = str(record.get("measurement_text") or "")
    if _scientific_factor_present(measurement):
        return None, "factor_already_in_measurement", []
    candidates: list[float] = []
    for match in _FACTOR_RE.finditer(str(record.get("support_text") or "")):
        if not math.isclose(float(match["center"]), scalar, rel_tol=1e-9, abs_tol=1e-12):
            continue
        exponent = int(_normalize_minus(match["exponent"]).replace(" ", ""))
        candidates.append(10.0**exponent)
    unique = sorted(set(candidates))
    if len(unique) == 1 and unique[0] < 1:
        return unique[0], "explicit_support_factor_applied", unique
    if len(unique) == 1:
        return None, "positive_support_factor_requires_review", unique
    if len(unique) > 1:
        return None, "ambiguous_support_factors", unique
    return None, "no_support_factor", []


def _canonical_scale_factor(
    record: Mapping[str, Any], scalar: float | None
) -> tuple[float | None, str, list[float], str | None]:
    canonical_unit = str(
        record.get("canonical_unit_text") or record.get("canonical_unit") or ""
    )
    percent_conversion = _PERCENT_UNIT_TARGETS.get(canonical_unit)
    if scalar is not None and percent_conversion is not None:
        factor, target = percent_conversion
        return factor, "canonical_percent_to_ratio_applied", [factor], target
    factor, status, candidates = _support_scale_factor(record, scalar)
    if factor is not None or status != "no_support_factor" or scalar is None:
        return factor, status, candidates, None
    measurement = str(record.get("measurement_text") or "")
    source_unit = str(record.get("unit_text") or "")
    if canonical_unit == "ratio" and not source_unit and "%" in measurement:
        return 0.01, "explicit_percent_to_ratio_applied", [0.01], "ratio"
    return None, status, candidates, None


def _policy_scale_factor(
    record: Mapping[str, Any], scalar: float | None, policy: Mapping[str, Any]
) -> tuple[float | None, str, list[float], str | None]:
    record_id = str(
        record.get("canonical_record_id") or record.get("normalized_record_id") or ""
    )
    correction = policy.get("record_corrections", {}).get(record_id)
    if correction is None:
        return _canonical_scale_factor(record, scalar)
    if isinstance(correction, list):
        factor, target_unit, status = correction
    else:
        factor = correction["factor"]
        target_unit = correction.get("canonical_unit_text")
        status = correction.get("status")
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"invalid reviewed correction factor for {record_id}")
    return (
        factor,
        str(status or "reviewed_record_factor_applied"),
        [factor],
        str(target_unit or "") or None,
    )


def _set_base_scalar(record: dict[str, Any], scalar: float, *, factor: float) -> None:
    variation = _finite(record.get("variation_value"))
    variation = variation * factor if variation is not None else None
    record.update(
        {
            "canonical_measurement": _measurement_text(scalar, variation),
            "finite_scalar_value": scalar,
            "variation_value": variation,
            "is_absolute_and_continuous": True,
            "absolute_and_continuous_value": scalar,
        }
    )


def _set_log_value(
    working: dict[str, Any], projected: dict[str, Any], scalar: float, unit: str
) -> None:
    text = _format_number(scalar)
    transformed_unit = f"log10({unit})"
    for row in (working, projected):
        row["finite_scalar_value"] = scalar
        row["variation_value"] = None
        row["is_absolute_and_continuous"] = True
        row["absolute_and_continuous_value"] = scalar
    working.update({"canonical_measurement": text, "canonical_unit": transformed_unit})
    projected.update(
        {"canonical_measurement_text": text, "canonical_unit_text": transformed_unit}
    )


def _measurement_text(value: float, variation: float | None) -> str:
    text = _format_number(value)
    return text if variation is None else f"{text} ± {_format_number(variation)}"


def _format_number(value: float) -> str:
    return format(value, ".12g")


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _normalize_minus(value: str) -> str:
    return value.translate(str.maketrans({"−": "-", "‐": "-", "‑": "-"}))


def _scientific_factor_present(value: str) -> bool:
    normalized = _normalize_minus(value)
    return bool(re.search(r"(?:×|x)?\s*10\s*(?:\^\s*)?[+-]?\s*\d+", normalized))


__all__ = [
    "CANONICAL_TUPLE_CONTRACT_VERSION",
    "LOG10_TRANSFORM",
    "POLICY_VERSION",
    "RAW_TRANSFORM",
    "canonicalize_assay_transfer_base",
    "finalize_assay_transfer_measurement",
    "load_measurement_policy",
    "validate_final_assay_transfer_measurements",
]
