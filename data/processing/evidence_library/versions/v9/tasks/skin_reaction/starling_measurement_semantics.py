"""Reviewed measurement semantics for Skin Reaction numeric sources.

The shared unit parser answers a syntactic question: what dimensions and scale
does a unit string express?  This module answers the task-specific scientific
question: what quantity did a Skin Reaction row measure, and may it participate
in a pair bucket?  Rules are frozen in one JSON registry and are evaluated
after the published auxiliary endpoint mapping is attached.  The published
mapping is input provenance and is never rewritten here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from data.processing.evidence_library.shared.v2.normalization.contracts import MeasurementPair
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    clean_measurement_text,
)
from data.processing.evidence_library.shared.v2.normalization.measurements import (
    parse_point_measurement,
    render_point_measurement,
    standardize_measurement_pair,
)
from tools.chembl_tool.common.units import canonicalize_unit
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.data_processing.auxiliary_mapping_helpers.reconciliation import (
    MAPPING_VERSION,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_auxiliary_metadata import (
    DEFAULT_MAPPING_PATH,
)
from data.processing.evidence_library.versions.v9.tasks.skin_reaction.starling_normalization_sources import (
    EXPECTED_SOURCE_SHA256,
)


MEASUREMENT_SEMANTICS_SCHEMA_VERSION = (
    "skin_reaction_measurement_semantics.schema.v1"
)
MEASUREMENT_SEMANTICS_VERSION = "skin_reaction_measurement_semantics.v1"
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent
    / "data_processing"
    / "canonicalization_v7"
    / "measurement_semantics.json"
)
NUMERIC_SOURCES = frozenset({"sensitization_aop", "skin_exposure"})

_TEXT_FIELDS = (
    "endpoint_name",
    "support_text",
    "extra_details",
    "qualifying_conditions",
    "experimental_conditions",
)
_ALLOWED_STATUSES = frozenset({"approved", "evidence_only"})
_ALLOWED_DOMAINS = frozenset(
    {
        "finite_signed",
        "nonnegative",
        "positive",
        "bounded_0_1",
        "bounded_0_100",
        "nonnegative_unbounded",
    }
)

# Qualifier vocabulary for every unit parsed by this task.
_TASK_VOCAB = "skin_reaction"



def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _casefold(value: Any) -> str:
    return _text(value).casefold()


def _dimension_key(value: Any) -> str:
    result = canonicalize_unit(value, task=_TASK_VOCAB)
    return json.dumps(result.dimension, separators=(",", ":"))


@dataclass(frozen=True)
class MeasurementSemanticRule:
    rule_id: str
    priority: int
    match: Mapping[str, Any]
    action: Mapping[str, Any]
    matched_record_count: int | None
    review_basis: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MeasurementSemanticRule":
        rule_id = _text(payload.get("rule_id"))
        if not rule_id:
            raise ValueError("measurement-semantics rule has no rule_id")
        match = payload.get("match") or payload.get("selector")
        action = payload.get("action") or payload.get("output")
        if not isinstance(match, Mapping) or not isinstance(action, Mapping):
            raise ValueError(f"{rule_id}: match/action must be objects")
        scalar_policy = _text(action.get("scalar_policy"))
        decision = _text(action.get("decision"))
        status = _text(
            action.get("status")
            or (
                "approved"
                if action.get("scalar_eligible") is True
                or decision == "proposed_canonicalization"
                or scalar_policy.startswith("eligible")
                else "evidence_only"
                if action.get("scalar_eligible") is False
                or decision == "evidence_only"
                or scalar_policy.startswith("evidence_only")
                else ""
            )
        )
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"{rule_id}: unsupported status {status!r}")
        normalized_action = dict(action)
        normalized_action["status"] = status
        raw_domain = (
            normalized_action.get("numeric_domain")
            or normalized_action.get("domain")
        )
        domain = _normalize_domain(raw_domain)
        if status == "approved" and domain not in _ALLOWED_DOMAINS:
            raise ValueError(f"{rule_id}: unsupported numeric domain {domain!r}")
        normalized_action["numeric_domain"] = domain or None
        normalized_action.setdefault(
            "canonical_endpoint_name",
            normalized_action.get("refined_endpoint_name"),
        )
        normalized_action.setdefault(
            "semantic_unit_basis",
            normalized_action.get("canonical_basis"),
        )
        unit_policy = _text(normalized_action.get("canonical_unit_policy"))
        unit_by_policy = {
            "percent": "%",
            "fraction": "fraction",
            "fold": "fold",
            "mm": "mm",
            "ppm": "ppm",
            "convert_to_seconds": "s",
            "convert_to_m": "M",
            "convert_to_µg/cm²": "µg/cm^2",
            "convert_to_mol/m²": "mol/m^2",
            "convert_to_µg/cm²/day": "µg/cm^2/d",
            "cells/mm²": "cells/mm^2",
        }
        if not normalized_action.get("canonical_unit_text"):
            # ``casefold()`` maps the micro sign (U+00B5) onto Greek mu (U+03BC), but the
            # keys above use U+00B5 -- so every µ-bearing policy silently missed and the
            # rule derived no target unit. Fold mu back the way ``clean_unit`` does.
            target = unit_by_policy.get(unit_policy.casefold().replace("μ", "µ"))
            if target:
                normalized_action["canonical_unit_text"] = target
        count = payload.get("matched_record_count", payload.get("reviewed_count"))
        if count is not None and (not isinstance(count, int) or count < 0):
            raise ValueError(f"{rule_id}: invalid matched_record_count")
        return cls(
            rule_id=rule_id,
            priority=int(payload.get("priority", 0)),
            match=dict(match),
            action=normalized_action,
            matched_record_count=count,
            review_basis=_text(payload.get("review_basis")),
        )


class SkinMeasurementSemantics:
    """Immutable, fail-closed evaluator for the frozen JSON registry."""

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH):
        self.path = Path(path)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != MEASUREMENT_SEMANTICS_SCHEMA_VERSION:
            raise ValueError("Skin measurement-semantics schema version mismatch")
        if payload.get("policy_version") != MEASUREMENT_SEMANTICS_VERSION:
            raise ValueError("Skin measurement-semantics policy version mismatch")
        mapping = payload.get("frozen_auxiliary_mapping") or {}
        expected_mapping_sha = _text(mapping.get("sha256"))
        from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256

        actual_mapping_sha = file_sha256(DEFAULT_MAPPING_PATH)
        if expected_mapping_sha != actual_mapping_sha:
            raise ValueError(
                "Skin measurement semantics mapping digest mismatch: "
                f"expected {expected_mapping_sha}, found {actual_mapping_sha}"
            )
        actual_mapping_version = json.loads(
            DEFAULT_MAPPING_PATH.read_text(encoding="utf-8")
        ).get("mapping_version")
        if mapping.get("version") != actual_mapping_version:
            raise ValueError(
                "Skin measurement semantics mapping version mismatch: "
                f"expected {mapping.get('version')}, found {actual_mapping_version}"
            )
        source_sha = payload.get("source_sha256") or {}
        for source_id in NUMERIC_SOURCES:
            if source_sha.get(source_id) != EXPECTED_SOURCE_SHA256[source_id]:
                raise ValueError(
                    f"Skin measurement semantics source digest mismatch for {source_id}"
                )
        rules_payload = payload.get("rules")
        if not isinstance(rules_payload, list) or not rules_payload:
            raise ValueError("Skin measurement-semantics registry has no rules")
        self.rules = tuple(
            sorted(
                (MeasurementSemanticRule.from_payload(item) for item in rules_payload),
                key=lambda item: (-item.priority, item.rule_id),
            )
        )
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Skin measurement-semantics rule IDs are not unique")
        self.payload = payload
        self.mapping_sha256 = actual_mapping_sha

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": MEASUREMENT_SEMANTICS_SCHEMA_VERSION,
            "policy_version": MEASUREMENT_SEMANTICS_VERSION,
            "path": str(self.path),
            "frozen_auxiliary_mapping": {
                "version": MAPPING_VERSION,
                "sha256": self.mapping_sha256,
            },
            "source_sha256": dict(sorted(self.payload["source_sha256"].items())),
            "n_rules": len(self.rules),
            "rule_ids": [rule.rule_id for rule in self.rules],
            "fail_closed": True,
            "overlap_policy": "one winner at the highest matching priority",
            "counts_are_human_reviewed": False,
            "review_basis": self.payload.get("review_basis"),
        }

    def _matches(self, rule: MeasurementSemanticRule, record: Mapping[str, Any]) -> bool:
        selector = rule.match
        if not self._matches_clause(selector, record):
            return False
        return True

    def _matches_clause(self, selector: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
        any_of = selector.get("any_of")
        if any_of and not any(self._matches_clause(item, record) for item in any_of):
            return False
        all_of = selector.get("all_of")
        if all_of and not all(self._matches_clause(item, record) for item in all_of):
            return False
        excluded = selector.get("not")
        if isinstance(excluded, Mapping) and self._matches_clause(excluded, record):
            return False
        source_id = _text(record.get("source_id"))
        exact_source = selector.get("source_id") or selector.get("source_id_exact")
        if exact_source and source_id != _text(exact_source):
            return False
        allowed_sources = selector.get("source_ids")
        if allowed_sources and source_id not in {_text(item) for item in allowed_sources}:
            return False
        raw_endpoint = _casefold(record.get("endpoint_name"))
        frozen_endpoint = _casefold(
            record.get("pre_refinement_canonical_endpoint")
            or record.get("global_endpoint_context")
            or record.get("canonical_endpoint")
        )
        raw_unit = _casefold(record.get("unit_text"))
        canonical_unit = _text(record.get("canonical_unit"))
        values = {
            "raw_endpoint": raw_endpoint,
            "frozen_endpoint": frozen_endpoint,
            "raw_unit": raw_unit,
        }
        for key, value_name in (
            ("raw_endpoint_exact", "raw_endpoint"),
            ("endpoint_name_exact", "raw_endpoint"),
            ("frozen_endpoint_exact", "frozen_endpoint"),
            ("raw_unit_exact", "raw_unit"),
            ("unit_text_exact", "raw_unit"),
        ):
            expected = selector.get(key)
            actual = values[value_name]
            expected_value = (
                _normalized_unit(expected)
                if value_name == "raw_unit"
                else _casefold(expected)
            )
            if expected is not None and actual != expected_value:
                return False
        for key, value_name in (
            ("raw_endpoint_in", "raw_endpoint"),
            ("endpoint_exact", "raw_endpoint"),
            ("frozen_endpoint_in", "frozen_endpoint"),
            ("raw_unit_in", "raw_unit"),
            ("unit_text_in", "raw_unit"),
        ):
            expected = selector.get(key)
            accepted = {
                _normalized_unit(item) if value_name == "raw_unit" else _casefold(item)
                for item in (expected or ())
            }
            if expected is not None and values[value_name] not in accepted:
                return False
        for key, value_name in (
            ("raw_endpoint_regex", "raw_endpoint"),
            ("endpoint_name_regex", "raw_endpoint"),
            ("frozen_endpoint_regex", "frozen_endpoint"),
            ("raw_unit_regex", "raw_unit"),
            ("unit_text_regex", "raw_unit"),
        ):
            pattern = selector.get(key)
            if pattern and re.search(str(pattern), values[value_name], re.IGNORECASE) is None:
                return False
        if selector.get("unit_text_is_null") and raw_unit:
            return False
        if selector.get("unit_text_is_nonempty") and not raw_unit:
            return False
        measurement_text = _text(record.get("measurement_text"))
        if selector.get("measurement_text_is_null") and measurement_text:
            return False
        if selector.get("measurement_text_is_nonempty") and not measurement_text:
            return False
        if selector.get("measurement_text_exact") is not None and measurement_text != _text(
            selector.get("measurement_text_exact")
        ):
            return False
        rows = selector.get("source_row_numbers")
        if rows is not None and int(record.get("source_row_number") or -1) not in {
            int(item) for item in rows
        }:
            return False
        pmid = selector.get("pmid_exact")
        if pmid is not None and _text(record.get("pmid")) != _text(pmid):
            return False
        existing_units = selector.get("existing_canonical_unit_exact")
        if existing_units is not None:
            accepted_existing = {
                canonicalize_unit(item, task=_TASK_VOCAB).canonical for item in existing_units
            }
            # Parser-repair selectors intentionally name the old broken
            # canonical result.  Once the raw spelling itself matches, do not
            # require that obsolete result.
            parser_repair = bool(selector.get("raw_unit_regex"))
            if not parser_repair and canonicalize_unit(canonical_unit, task=_TASK_VOCAB).canonical not in accepted_existing:
                return False
        excluded_units = selector.get("raw_unit_not_exact")
        if excluded_units is not None and raw_unit in {
            _normalized_unit(item) for item in excluded_units
        }:
            return False
        dimensions = selector.get("canonical_unit_dimensions")
        if dimensions is not None:
            actual_dimension = _dimension_key(canonical_unit)
            accepted = {
                item if isinstance(item, str) else json.dumps(item, separators=(",", ":"))
                for item in dimensions
            }
            if actual_dimension not in accepted:
                return False
        transforms = selector.get("canonical_unit_transforms")
        if transforms is not None and canonicalize_unit(canonical_unit, task=_TASK_VOCAB).transform not in {
            _text(item) for item in transforms
        }:
            return False
        if selector.get("require_finite_point"):
            value = record.get("finite_scalar_value")
            if value is None or not math.isfinite(float(value)):
                return False
        text_blob = "\n".join(_text(record.get(field)) for field in _TEXT_FIELDS)
        evidence_pattern = selector.get("evidence_text_regex")
        if evidence_pattern and re.search(
            str(evidence_pattern), text_blob, re.IGNORECASE
        ) is None:
            return False
        patterns_any = selector.get("text_regex_any") or selector.get(
            "support_regex_any"
        )
        if patterns_any and not any(
            re.search(str(pattern), text_blob, re.IGNORECASE)
            for pattern in patterns_any
        ):
            return False
        patterns_all = selector.get("text_regex_all")
        if patterns_all and not all(
            re.search(str(pattern), text_blob, re.IGNORECASE)
            for pattern in patterns_all
        ):
            return False
        patterns_none = selector.get("text_regex_none") or selector.get(
            "support_regex_none"
        ) or selector.get("support_text_regex_none")
        if patterns_none and any(
            re.search(str(pattern), text_blob, re.IGNORECASE)
            for pattern in patterns_none
        ):
            return False
        return True

    def select_rule(self, record: Mapping[str, Any]) -> MeasurementSemanticRule | None:
        matches = [rule for rule in self.rules if self._matches(rule, record)]
        if not matches:
            return None
        best_priority = matches[0].priority
        winners = [rule for rule in matches if rule.priority == best_priority]
        if len(winners) != 1:
            raise ValueError(
                "ambiguous Skin measurement-semantics rules at priority "
                f"{best_priority}: {[rule.rule_id for rule in winners]}"
            )
        return winners[0]

    def apply(self, record: Mapping[str, Any]) -> dict[str, Any]:
        source_id = _text(record.get("source_id"))
        if source_id not in NUMERIC_SOURCES:
            return {
                "measurement_semantics_status": "not_applicable",
                "measurement_semantics_rule_id": None,
                "pre_refinement_canonical_endpoint": record.get("canonical_endpoint"),
                "measurement_quantity_kind": None,
                "measurement_semantic_unit_basis": None,
                "measurement_numeric_domain": None,
            }
        frozen_endpoint = (
            record.get("global_endpoint_context")
            if source_id == "sensitization_aop"
            else record.get("canonical_endpoint")
        )
        working = {**record, "pre_refinement_canonical_endpoint": frozen_endpoint}
        rule = self.select_rule(working)
        if rule is None:
            return self._evidence_only(
                working,
                rule_id=None,
                reason="unreviewed_measurement_semantics",
            )
        action = rule.action
        if action["status"] == "evidence_only":
            return self._evidence_only(
                working,
                rule_id=rule.rule_id,
                reason=_text(action.get("exclusion_reason")) or "reviewed_evidence_only",
            )
        return self._approved(working, rule)

    def _evidence_only(
        self,
        record: Mapping[str, Any],
        *,
        rule_id: str | None,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "canonical_endpoint": record.get("pre_refinement_canonical_endpoint"),
            "canonical_measurement": record.get("canonical_measurement"),
            "canonical_unit": None,
            "finite_scalar_value": None,
            "absolute_and_continuous_value": None,
            "is_absolute_and_continuous": False,
            "measurement_unit_status": "semantic_evidence_only",
            "measurement_semantics_status": "evidence_only",
            "measurement_semantics_rule_id": rule_id,
            "measurement_semantics_exclusion_reason": reason,
            "pre_refinement_canonical_endpoint": record.get(
                "pre_refinement_canonical_endpoint"
            ),
            "measurement_quantity_kind": None,
            "measurement_semantic_unit_basis": None,
            "measurement_numeric_domain": None,
        }

    def _approved(
        self, record: Mapping[str, Any], rule: MeasurementSemanticRule
    ) -> dict[str, Any]:
        action = rule.action
        pair = MeasurementPair(
            record.get("canonical_measurement"),
            record.get("canonical_unit"),
            _text(record.get("measurement_unit_status")),
            _text(record.get("unit_notation_status")),
            record.get("unit_notation_factor"),
        )
        target_unit = _text(action.get("canonical_unit_text") or action.get("canonical_unit"))
        measurement_class = _text(
            action.get("measurement_class") or action.get("canonical_unit_kind")
        )
        if measurement_class not in {
            "auc",
            "auc_mass",
            "auc_molar",
            "concentration",
            "molar_concentration",
            "potency",
            "mass_solubility",
            "permeability",
            "clearance",
            "weight_normalized_clearance",
            "intrinsic_clearance",
            "flux",
            "diffusivity",
            "areic_dose",
            "duration",
            "rate_constant",
        }:
            measurement_class = "preserve"
        pair = _apply_exact_transform(pair, record, action)
        if target_unit:
            converted = standardize_measurement_pair(
                pair, targets=(target_unit,), task=_TASK_VOCAB
            )
            if converted.canonical_unit == pair.canonical_unit and target_unit != pair.canonical_unit:
                transform_text = _text(action.get("value_transform"))
                if transform_text == "identity" or "restore" in transform_text:
                    converted = MeasurementPair(
                        pair.canonical_measurement,
                        target_unit,
                        "reviewed_semantic_relabel",
                        pair.unit_notation_status,
                        pair.unit_notation_factor,
                    )
            pair = converted
        elif measurement_class and measurement_class != "preserve":
            pair = standardize_measurement_pair(
                pair, measurement_class=measurement_class, task=_TASK_VOCAB
            )
        if (
            target_unit
            and canonicalize_unit(pair.canonical_unit, task=_TASK_VOCAB).canonical
            != canonicalize_unit(target_unit, task=_TASK_VOCAB).canonical
        ):
            return self._evidence_only(
                record,
                rule_id=rule.rule_id,
                reason="approved_rule_failed_target_unit_conversion",
            )
        parsed = parse_point_measurement(pair.canonical_measurement)
        recognized = canonicalize_unit(pair.canonical_unit, task=_TASK_VOCAB)
        if (
            parsed.value is None
            or not recognized.cleaned
            or recognized.unknown_tokens
        ):
            return self._evidence_only(
                record,
                rule_id=rule.rule_id,
                reason="approved_rule_failed_atomic_conversion",
            )
        domain = _text(action.get("numeric_domain"))
        if not _domain_accepts(parsed.value, domain):
            return {
                **self._evidence_only(
                    record,
                    rule_id=rule.rule_id,
                    reason="outside_reviewed_numeric_domain",
                ),
                "measurement_numeric_domain": domain,
            }
        endpoint = _text(
            action.get("canonical_endpoint_name")
            or action.get("refined_endpoint")
        ) or _text(record.get("pre_refinement_canonical_endpoint"))
        if endpoint.startswith("$"):
            endpoint = _text(record.get("pre_refinement_canonical_endpoint"))
        return {
            "canonical_endpoint": endpoint,
            "canonical_measurement": pair.canonical_measurement,
            "canonical_unit": pair.canonical_unit,
            "finite_scalar_value": parsed.value,
            "variation_value": parsed.variation,
            "absolute_and_continuous_value": parsed.value,
            "is_absolute_and_continuous": True,
            "measurement_unit_status": "reviewed_semantic_pair",
            "measurement_semantics_status": "approved",
            "measurement_semantics_rule_id": rule.rule_id,
            "measurement_semantics_exclusion_reason": None,
            "pre_refinement_canonical_endpoint": record.get(
                "pre_refinement_canonical_endpoint"
            ),
            "measurement_quantity_kind": _text(action.get("quantity_kind")) or None,
            "measurement_semantic_unit_basis": _text(
                action.get("semantic_unit_basis")
            ) or None,
            "measurement_numeric_domain": domain,
        }

    def audit(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        status_counts: Counter[str] = Counter()
        rule_counts: Counter[str] = Counter()
        exclusion_counts: Counter[str] = Counter()
        numeric_records = 0
        for record in records:
            if _text(record.get("source_id")) not in NUMERIC_SOURCES:
                continue
            numeric_records += 1
            result = (
                record
                if _text(record.get("measurement_semantics_status"))
                else self.apply(record)
            )
            status_counts[_text(result.get("measurement_semantics_status"))] += 1
            rule_counts[_text(result.get("measurement_semantics_rule_id")) or "<unmatched>"] += 1
            reason = _text(result.get("measurement_semantics_exclusion_reason"))
            if reason:
                exclusion_counts[reason] += 1
        return {
            "policy_version": MEASUREMENT_SEMANTICS_VERSION,
            "numeric_source_records": numeric_records,
            "status_counts": dict(sorted(status_counts.items())),
            "rule_counts": dict(sorted(rule_counts.items())),
            "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
            "validations": {
                "all_numeric_records_have_explicit_status": sum(status_counts.values())
                == numeric_records,
                "no_ambiguous_top_priority_match": True,
            },
        }


def _domain_accepts(value: float, domain: str) -> bool:
    if not math.isfinite(float(value)):
        return False
    if domain == "finite_signed":
        return True
    if domain in {"nonnegative", "nonnegative_unbounded"}:
        return value >= 0.0
    if domain == "positive":
        return value > 0.0
    if domain == "bounded_0_1":
        return 0.0 <= value <= 1.0
    if domain == "bounded_0_100":
        return 0.0 <= value <= 100.0
    return False


def _normalized_unit(value: Any) -> str:
    return _casefold(clean_measurement_text(value))


def _normalize_domain(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return ""
    if value.get("finite") or value.get("log_finite"):
        return "finite_signed"
    if value.get("nonnegative_percent_or_instrument"):
        return "nonnegative_unbounded"
    minimum = value.get("minimum")
    maximum = value.get("maximum")
    if minimum == 0.0 and maximum == 1.0:
        return "bounded_0_1"
    if minimum == 0.0 and maximum == 100.0:
        return "bounded_0_100"
    if value.get("minimum_exclusive") == 0.0:
        return "positive"
    if minimum == 0.0:
        return "nonnegative"
    if value.get("linear_minimum_exclusive") == 0.0:
        return "finite_signed"
    if value.get("numerator_minimum") == 0:
        return "nonnegative"
    return ""


def _apply_exact_transform(
    pair: MeasurementPair,
    record: Mapping[str, Any],
    action: Mapping[str, Any],
) -> MeasurementPair:
    transform = action.get("transform")
    transforms = action.get("transforms")
    if isinstance(transforms, Mapping):
        raw_unit = _normalized_unit(record.get("unit_text"))
        transform = next(
            (
                value
                for unit, value in transforms.items()
                if _normalized_unit(unit) == raw_unit
            ),
            None,
        )
    if not isinstance(transform, Mapping):
        return pair
    factor = transform.get("multiply_value_by")
    target = _text(transform.get("canonical_unit"))
    parsed = parse_point_measurement(pair.canonical_measurement)
    if factor is None or parsed.value is None or not target:
        return pair
    factor = float(factor)
    return MeasurementPair(
        render_point_measurement(
            pair.canonical_measurement or "",
            parsed.value * factor,
            parsed.variation * factor if parsed.variation is not None else None,
        ),
        target,
        "reviewed_exact_transform",
        pair.unit_notation_status,
        pair.unit_notation_factor,
    )


@lru_cache(maxsize=1)
def default_policy() -> SkinMeasurementSemantics:
    return SkinMeasurementSemantics(DEFAULT_REGISTRY_PATH)


def apply_measurement_semantics(record: Mapping[str, Any]) -> dict[str, Any]:
    return default_policy().apply(record)


def measurement_semantics_manifest() -> dict[str, Any]:
    return default_policy().manifest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen Skin measurement semantics on canonicalized records."
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    frame = pd.read_parquet(args.records)
    audit = default_policy().audit(frame.to_dict(orient="records"))
    rendered = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "MEASUREMENT_SEMANTICS_SCHEMA_VERSION",
    "MEASUREMENT_SEMANTICS_VERSION",
    "SkinMeasurementSemantics",
    "apply_measurement_semantics",
    "default_policy",
    "measurement_semantics_manifest",
]
