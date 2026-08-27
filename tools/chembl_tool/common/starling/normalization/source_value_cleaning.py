"""Audited source-visible measurement/unit cleaning for Starling v7 records.

The immutable source files remain untouched.  This module changes only the
Stage-01 source view, records every changed field in a sidecar, and requires an
exact reviewed repair for changes that cannot be derived from syntax alone.
Support text is immutable after the common ingestion-only Unicode/whitespace
cleanup; it may inform a repair but is never itself a repair target.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .cleaning import clean_measurement_text, clean_text, file_sha256


SOURCE_VALUE_CLEANING_VERSION = "starling_source_value_cleaning.v2"
SUPPORT_TEXT_POLICY_VERSION = "support_text_immutable_after_ingestion.v1"
DECIMAL_COMMA_RULE = "atomic_decimal_comma.v1"
ENCODED_SPACE_RULE = "percent_0020_space.v1"

_ALLOWED_REPAIR_FIELDS = frozenset({"measurement_text", "unit_text"})
_REPAIR_KEYS = frozenset(
    {
        "repair_id",
        "task_id",
        "source_id",
        "source_sha256",
        "source_row_number",
        "source_record_id",
        "before",
        "after",
        "evidence",
    }
)
_DECIMAL_COMMA = re.compile(
    r"(?<![\w,])(?P<integer>[+-]?\d+),(?P<fraction>\d+)(?![\w,])"
)
_POWER_OF_TEN_UNIT = re.compile(
    r"10\s*(?:\^|\*\*)?\s*[+\-−–]?\s*\d+", flags=re.IGNORECASE
)
_SCIENTIFIC_VALUE = re.compile(
    r"(?P<coefficient>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?:[x×*·]\s*)?10\s*(?:\^|\*\*)?\s*"
    r"(?P<exponent>[+\-−–]\s*\d+)",
    flags=re.IGNORECASE,
)
_MIXED_TRAILING_DECIMAL_RANGE = re.compile(
    r"^\s*[+-]?\d+\s*(?:-|‐|–|—|−|to)\s*"
    r"(?P<integer>[1-9]\d*),(?P<fraction>\d{2})\s*%?\s*$",
    flags=re.IGNORECASE,
)
_MEASUREMENT_PREFIX_WORDS = frozenset(
    {
        "about",
        "approx",
        "approximately",
        "ca",
        "ci",
        "confidence",
        "estimated",
        "en",
        "geometric",
        "gmean",
        "interval",
        "mean",
        "median",
        "moyenne",
        "of",
        "range",
        "sd",
        "se",
        "sem",
        "to",
    }
)


@dataclass(frozen=True)
class ReviewedSourceValueRepair:
    repair_id: str
    task_id: str
    source_id: str
    source_sha256: str
    source_row_number: int
    source_record_id: str
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    evidence: Mapping[str, Any]

    @property
    def row_key(self) -> tuple[str, int, str]:
        return self.source_id, self.source_row_number, self.source_record_id


@dataclass(frozen=True)
class SourceValueCleaningResult:
    records: list[dict[str, Any]]
    audit_rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    input_paths: tuple[Path, ...] = ()


def load_reviewed_repairs(
    path: str | Path | None,
    *,
    task_id: str,
) -> list[ReviewedSourceValueRepair]:
    if path is None:
        return []
    target = Path(path)
    repairs: list[ReviewedSourceValueRepair] = []
    for line_number, line in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if set(payload) != _REPAIR_KEYS:
            raise ValueError(
                f"{target}:{line_number} has invalid repair fields: "
                f"{sorted(set(payload) ^ _REPAIR_KEYS)}"
            )
        before = payload["before"]
        after = payload["after"]
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError(f"{target}:{line_number} before/after must be objects")
        if not before or set(before) != set(after):
            raise ValueError(
                f"{target}:{line_number} before/after must name the same fields"
            )
        if not set(before) <= _ALLOWED_REPAIR_FIELDS:
            raise ValueError(f"{target}:{line_number} repairs a forbidden field")
        if all(before[field] == after[field] for field in before):
            raise ValueError(f"{target}:{line_number} repair changes nothing")
        evidence = payload["evidence"]
        if not isinstance(evidence, Mapping) or len(str(evidence.get("note") or "")) < 40:
            raise ValueError(
                f"{target}:{line_number} requires a substantive evidence note"
            )
        repair = ReviewedSourceValueRepair(
            repair_id=str(payload["repair_id"]),
            task_id=str(payload["task_id"]),
            source_id=str(payload["source_id"]),
            source_sha256=str(payload["source_sha256"]),
            source_row_number=int(payload["source_row_number"]),
            source_record_id=str(payload["source_record_id"]),
            before=dict(before),
            after=dict(after),
            evidence=dict(evidence),
        )
        if repair.task_id != task_id:
            raise ValueError(
                f"{target}:{line_number} belongs to task {repair.task_id!r}, "
                f"not {task_id!r}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", repair.source_sha256):
            raise ValueError(f"{target}:{line_number} has an invalid source SHA-256")
        repairs.append(repair)

    repair_ids = [repair.repair_id for repair in repairs]
    row_keys = [repair.row_key for repair in repairs]
    if len(repair_ids) != len(set(repair_ids)):
        raise ValueError(f"{target} contains duplicate repair IDs")
    if len(row_keys) != len(set(row_keys)):
        raise ValueError(f"{target} targets one source row more than once")
    return repairs


def clean_source_values(
    records: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
    reviewed_repairs_path: str | Path | None = None,
    require_all_reviewed_repairs: bool = True,
    require_scientific_scale_review: bool = False,
) -> SourceValueCleaningResult:
    """Return cleaned records and a complete field-level change audit."""
    repairs = load_reviewed_repairs(reviewed_repairs_path, task_id=task_id)
    repairs_by_row = {repair.row_key: repair for repair in repairs}
    applied_repairs: set[str] = set()
    output: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    scale_candidates = 0
    unresolved_scale_candidates: list[str] = []

    for source_record in records:
        # Builder-owned rows are mutable dictionaries.  Reuse them so a full
        # 800k-row task does not temporarily duplicate the entire Stage-01
        # object graph; callers that supply another Mapping still get a dict.
        record = (
            source_record
            if isinstance(source_record, dict)
            else dict(source_record)
        )
        original_support_text = record.get("support_text")
        original_measurement = _optional_text(record.get("measurement_text"))
        measurement, measurement_steps = _clean_measurement(
            original_measurement,
            record.get("unit_text"),
            record.get("support_text"),
        )
        record["measurement_text"] = measurement
        for rule_id, before, after in measurement_steps:
            audit.append(
                _audit_row(record, "measurement_text", before, after, rule_id)
            )
        scale_candidate = _has_scientific_scale_conflict(record)
        scale_candidates += scale_candidate

        row_key = (
            str(record.get("source_id") or ""),
            int(record.get("source_row_number") or 0),
            str(record.get("source_record_id") or ""),
        )
        repair = repairs_by_row.get(row_key)
        if repair is not None:
            actual_sha = str(record.get("source_sha256") or "")
            if actual_sha != repair.source_sha256:
                raise ValueError(
                    f"reviewed repair {repair.repair_id!r} source drift: "
                    f"expected {repair.source_sha256}, found {actual_sha or '<missing>'}"
                )
            for field, expected_before in repair.before.items():
                actual_before = record.get(field)
                if actual_before != expected_before:
                    raise ValueError(
                        f"reviewed repair {repair.repair_id!r} precondition drift for "
                        f"{field}: expected {expected_before!r}, found {actual_before!r}"
                    )
            for field, after in repair.after.items():
                before = record.get(field)
                record[field] = after
                audit.append(
                    _audit_row(
                        record,
                        field,
                        before,
                        after,
                        f"reviewed:{repair.repair_id}",
                        review_status="reviewed",
                        evidence=repair.evidence,
                    )
                )
            applied_repairs.add(repair.repair_id)
        if scale_candidate and _has_scientific_scale_conflict(record):
            unresolved_scale_candidates.append(
                str(record.get("cleaned_record_id") or record.get("source_record_id") or "")
            )
        if record.get("support_text") != original_support_text:
            raise ValueError(
                "source-value cleaning modified immutable support_text for "
                f"{record.get('cleaned_record_id')!r}"
            )
        output.append(record)

    unapplied = sorted({repair.repair_id for repair in repairs} - applied_repairs)
    if require_all_reviewed_repairs and unapplied:
        raise ValueError(
            "reviewed source-value repairs were not applied: " + ", ".join(unapplied)
        )
    if require_scientific_scale_review and unresolved_scale_candidates:
        raise ValueError(
            f"{len(unresolved_scale_candidates)} source value(s) already include the "
            "scientific factor retained in their unit; add reviewed repairs; "
            f"first={unresolved_scale_candidates[0]}"
        )

    changed_records = {str(row["cleaned_record_id"]) for row in audit}
    rule_counts = Counter(str(row["rule_id"]) for row in audit)
    field_counts = Counter(str(row["field"]) for row in audit)
    registry_path = Path(reviewed_repairs_path) if reviewed_repairs_path else None
    manifest = {
        "version": SOURCE_VALUE_CLEANING_VERSION,
        "task_id": task_id,
        "n_input_records": len(records),
        "n_output_records": len(output),
        "n_changed_records": len(changed_records),
        "n_field_changes": len(audit),
        "rule_counts": dict(sorted(rule_counts.items())),
        "field_counts": dict(sorted(field_counts.items())),
        "scientific_scale_review": {
            "n_candidates": scale_candidates,
            "n_resolved_by_reviewed_repair": (
                scale_candidates - len(unresolved_scale_candidates)
            ),
            "n_unresolved": len(unresolved_scale_candidates),
            "all_candidates_resolved": not unresolved_scale_candidates,
        },
        "support_text_policy": {
            "version": SUPPORT_TEXT_POLICY_VERSION,
            "mode": "immutable_after_ingestion",
        },
        "reviewed_repairs": {
            "path": str(registry_path) if registry_path else None,
            "sha256": file_sha256(registry_path) if registry_path else None,
            "n_declared": len(repairs),
            "n_applied": len(applied_repairs),
            "unapplied_repair_ids": unapplied,
            "all_required_repairs_applied": not unapplied,
        },
    }
    return SourceValueCleaningResult(
        records=output,
        audit_rows=audit,
        manifest=manifest,
        input_paths=(registry_path,) if registry_path else (),
    )


def _clean_measurement(
    value: str | None,
    unit_text: Any,
    support_text: Any,
) -> tuple[str | None, list[tuple[str, str, str]]]:
    cleaned, steps = _replace_encoded_spaces(value)
    if cleaned is None:
        return None, steps
    candidate, changed = _decimal_comma_candidate(cleaned)
    if not changed or not _is_atomic_measurement(candidate, unit_text):
        return cleaned, steps
    if _is_uncorroborated_mixed_range(cleaned, support_text):
        return cleaned, steps
    steps.append((DECIMAL_COMMA_RULE, cleaned, candidate))
    return candidate, steps


def _has_scientific_scale_conflict(record: Mapping[str, Any]) -> bool:
    """Flag a nonzero value that support text already expresses with its unit scale."""
    unit = str(record.get("unit_text") or "")
    if _POWER_OF_TEN_UNIT.search(unit) is None:
        return False
    try:
        value = Decimal(str(record.get("measurement_text") or "").strip())
    except (InvalidOperation, ValueError):
        return False
    if not value.is_finite() or value == 0:
        return False
    for match in _SCIENTIFIC_VALUE.finditer(str(record.get("support_text") or "")):
        exponent = match.group("exponent").replace("−", "-").replace("–", "-")
        expressed = Decimal(match.group("coefficient")).scaleb(
            int(exponent.replace(" ", ""))
        )
        if expressed == value:
            return True
    return False


def _is_uncorroborated_mixed_range(value: str, support_text: Any) -> bool:
    """Fail closed for forms such as ``40-70,50``.

    That spelling can mean a decimal endpoint, but it can also be a damaged
    list or table extraction.  A dotted occurrence in the support sentence is
    sufficient independent evidence; otherwise Stage 01 preserves the source
    form for reviewed repair instead of inventing a precise range endpoint.
    """
    match = _MIXED_TRAILING_DECIMAL_RANGE.fullmatch(value)
    if match is None:
        return False
    dotted_endpoint = f"{match.group('integer')}.{match.group('fraction')}"
    support = _optional_text(support_text) or ""
    return re.search(
        rf"(?<![\w.]){re.escape(dotted_endpoint)}(?![\w.])", support
    ) is None


def _replace_encoded_spaces(
    value: str | None,
) -> tuple[str | None, list[tuple[str, str, str]]]:
    if value is None or "%0020" not in value:
        return value, []
    replaced = clean_measurement_text(value.replace("%0020", "% "))
    if replaced == value:
        return value, []
    return replaced, [(ENCODED_SPACE_RULE, value, str(replaced))]


def _decimal_comma_candidate(value: str) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        prefix_words = {
            word.casefold()
            for word in re.findall(r"[A-Za-zµμ]+", value[: match.start()])
        }
        if not prefix_words <= _MEASUREMENT_PREFIX_WORDS:
            return match.group(0)
        integer = match.group("integer")
        fraction = match.group("fraction")
        digits = integer.lstrip("+-")
        # A leading zero cannot be a thousands group.  One- and two-digit
        # suffixes are conventional decimal-comma forms.  A single nonzero
        # three-digit group stays untouched because it may be a real thousands
        # separator (for example 2,180).
        if int(digits) != 0 and len(fraction) > 2:
            return match.group(0)
        changed = True
        return f"{integer}.{fraction}"

    return _DECIMAL_COMMA.sub(replace, value), changed


def _is_atomic_measurement(value: str, unit_text: Any) -> bool:
    # Import lazily to keep the Stage-01 module independent from parser import
    # order while still using the exact grammar that Stage 02 will apply.
    from .measurements import parse_point_measurement, separate_measurement_unit

    display = separate_measurement_unit(value, unit_text)
    parsed = parse_point_measurement(display)
    return parsed.kind in {
        "point",
        "approximate_point",
        "mean_with_variation",
        "mean_with_context",
        "point_with_interval",
        "bound",
        "range",
    }


def _audit_row(
    record: Mapping[str, Any],
    field: str,
    before: Any,
    after: Any,
    rule_id: str,
    *,
    review_status: str = "deterministic",
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = evidence or {}
    return {
        "cleaning_version": SOURCE_VALUE_CLEANING_VERSION,
        "cleaned_record_id": str(record.get("cleaned_record_id") or ""),
        "source_id": str(record.get("source_id") or ""),
        "source_sha256": str(record.get("source_sha256") or ""),
        "source_row_number": int(record.get("source_row_number") or 0),
        "source_record_id": str(record.get("source_record_id") or ""),
        "field": field,
        "before": before,
        "after": after,
        "rule_id": rule_id,
        "review_status": review_status,
        "evidence_id": str(evidence.get("pmid") or evidence.get("source") or ""),
        "rationale": str(evidence.get("note") or ""),
    }


def _optional_text(value: Any) -> str | None:
    return clean_text(value)


__all__ = [
    "DECIMAL_COMMA_RULE",
    "ENCODED_SPACE_RULE",
    "SOURCE_VALUE_CLEANING_VERSION",
    "SUPPORT_TEXT_POLICY_VERSION",
    "SourceValueCleaningResult",
    "clean_source_values",
    "load_reviewed_repairs",
]
