"""Build the complete, endpoint-independent DILI V10 exact-unit map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    load_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.dili.starling_measurement_resolution import (
    DEFAULT_CLEANED_RECORDS,
    DEFAULT_MAPPING_PATH,
)
from tools.chembl_tool.common.units import canonicalize_unit

TASK = "dili"
INVENTORY_VERSION = "dili_unit_inventory.v1"
REVIEW_VERSION = "dili_unit_review.v1"
MAPPING_VERSION = "dili_unit_reconciliation.v1"
MAX_PACKET_ITEMS = 64
DEFAULT_OUTPUT = (
    Path(__file__).parent
    / "unit_reconciliation_v1/dili_unit_reconciliation.v1.json"
)

_PRIOR_MAPS = {
    "bbb_v8": Path(__file__).parents[4]
    / "v8/tasks/bbb_martins/data_processing/canonicalization_v8/bbb_unit_reconciliation.v1.json",
    "bbb_v9": Path(__file__).parents[4]
    / "v9/tasks/bbb_martins/data_processing/canonicalization_v8/bbb_unit_reconciliation.v1.json",
    "bbb_v10": Path(__file__).parents[2]
    / "bbb_martins/data_processing/canonicalization_v8/bbb_unit_reconciliation.v1.json",
    "oral_v10": Path(__file__).parents[2]
    / "bioavailability_ma/data_processing/canonicalization_v8/bioavailability_unit_reconciliation.v1.json",
}

_CLEANED_COLUMNS = (
    "cleaned_record_id",
    "source_row_uid",
    "source_id",
    "endpoint_name",
    "measurement_text",
    "unit_text",
    "support_text",
    "measurement_resolution_route",
    "measurement_resolution_exact_measurement",
    "measurement_resolution_exact_unit",
)
_RESOLUTION_COLUMNS = (
    "cleaned_record_id",
    "source_row_uid",
    "source_id",
    "status",
    "measurements_json",
)
_TERMINAL_STATUSES = frozenset({"ok", "relative", "unsure", "unavailable"})


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _json_text(value: Any, *, pretty: bool = False) -> str:
    kwargs = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        kwargs["indent"] = 2
    else:
        kwargs["separators"] = (",", ":")
    return json.dumps(value, **kwargs) + "\n"


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _require_columns(path: Path, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(pq.read_schema(path).names))
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")


def _measurements(value: Any) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(_text(value) or "[]")
    except json.JSONDecodeError as error:
        raise ValueError("invalid measurements_json") from error
    if not isinstance(parsed, list) or any(not isinstance(row, dict) for row in parsed):
        raise ValueError("measurements_json must be a list of objects")
    return parsed


def _load_resolution(path: Path) -> dict[str, dict[str, str]]:
    _require_columns(path, _RESOLUTION_COLUMNS)
    records: dict[str, dict[str, str]] = {}
    for batch in pq.ParquetFile(path).iter_batches(columns=_RESOLUTION_COLUMNS):
        for row in batch.to_pylist():
            record_id = _text(row["cleaned_record_id"])
            status = _text(row["status"])
            entries = _measurements(row["measurements_json"])
            if not record_id or record_id in records or status not in _TERMINAL_STATUSES:
                raise ValueError(f"invalid measurement-resolution identity: {record_id!r}")
            if len(entries) != (1 if status == "ok" else 0):
                raise ValueError(f"status/measurement mismatch: {record_id}")
            measurement = _text(entries[0].get("measurement")) if entries else ""
            unit = _text(entries[0].get("unit")) if entries else ""
            if entries and (not measurement or not unit):
                raise ValueError(f"incomplete resolved pair: {record_id}")
            records[record_id] = {
                "source_row_uid": _text(row["source_row_uid"]),
                "source_id": _text(row["source_id"]),
                "status": status,
                "measurement": measurement,
                "unit": unit,
            }
    return records


def _representative(
    row: Mapping[str, Any], *, origin: str, measurement: str, unit: str
) -> dict[str, str]:
    return {
        "origin": origin,
        "cleaned_record_id": _text(row["cleaned_record_id"]),
        "source_row_uid": _text(row["source_row_uid"]),
        "source_id": _text(row["source_id"]),
        "endpoint_name": _text(row.get("endpoint_name")),
        "source_measurement": _text(row.get("measurement_text"))[:300],
        "source_unit": _text(row.get("unit_text")),
        "resolved_measurement": measurement,
        "resolved_unit": unit,
        "support_excerpt": _text(row.get("support_text"))[:300],
    }


def _add_observation(
    groups: dict[str, dict[str, Any]],
    row: Mapping[str, Any],
    *,
    origin: str,
    measurement: str,
    unit: str,
) -> None:
    if not unit:
        raise ValueError("unit observation is empty")
    group = groups.setdefault(
        unit,
        {
            "origin_counts": Counter(),
            "route_counts": Counter(),
            "representatives": {},
        },
    )
    group["origin_counts"][origin] += 1
    route = _text(row.get("measurement_resolution_route"))
    if origin == "source" and route:
        group["route_counts"][route] += 1
    selector = (
        _text(row["source_id"]),
        _text(row["source_row_uid"]),
        _text(row["cleaned_record_id"]),
    )
    current = group["representatives"].get(origin)
    if current is None or selector < current[0]:
        group["representatives"][origin] = (
            selector,
            _representative(
                row, origin=origin, measurement=measurement, unit=unit
            ),
        )


def _prior_decisions(paths: Mapping[str, Path]) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    by_unit: dict[str, list[dict[str, str]]] = defaultdict(list)
    provenance: dict[str, Any] = {}
    for name, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for entry in payload.get("entries", []):
            if entry.get("canonical_endpoints") != ["*"]:
                continue
            by_unit[_text(entry.get("input_unit"))].append(
                {
                    "source": name,
                    "action": _text(entry.get("action")),
                    "canonical_unit": _text(entry.get("canonical_unit")),
                    "scale": _text(entry.get("scale")),
                }
            )
    return by_unit, provenance


def _parser_evidence(unit: str) -> dict[str, Any]:
    parsed = canonicalize_unit(unit, task=TASK)
    return {
        "cleaned": parsed.cleaned,
        "canonical": parsed.canonical,
        "scale": parsed.scale,
        "dimension": [list(value) for value in parsed.dimension],
        "unknown_tokens": list(parsed.unknown_tokens),
        "transform": parsed.transform,
        "notation_status": parsed.notation_status,
        "notation_factor": parsed.notation_factor,
        "normalizer_version": parsed.normalizer_version,
    }


def build_inventory(
    cleaned_path: Path = DEFAULT_CLEANED_RECORDS,
    resolution_path: Path = DEFAULT_MAPPING_PATH,
    *,
    prior_paths: Mapping[str, Path] = _PRIOR_MAPS,
) -> dict[str, Any]:
    """Return every source, deterministic, and accepted-model unit exactly once."""
    cleaned_path, resolution_path = cleaned_path.resolve(), resolution_path.resolve()
    _require_columns(cleaned_path, _CLEANED_COLUMNS)
    resolution = _load_resolution(resolution_path)
    groups: dict[str, dict[str, Any]] = {}
    joined: set[str] = set()
    seen: set[str] = set()
    for batch in pq.ParquetFile(cleaned_path).iter_batches(columns=_CLEANED_COLUMNS):
        for row in batch.to_pylist():
            record_id = _text(row["cleaned_record_id"])
            if not record_id or record_id in seen:
                raise ValueError(f"duplicate or empty cleaned_record_id: {record_id!r}")
            seen.add(record_id)
            source_unit = _text(row.get("unit_text"))
            if source_unit:
                _add_observation(
                    groups,
                    row,
                    origin="source",
                    measurement=_text(row.get("measurement_text")),
                    unit=source_unit,
                )
            if _text(row.get("measurement_resolution_route")) == "accept":
                exact_unit = _text(row.get("measurement_resolution_exact_unit")) or source_unit
                exact_measurement = _text(row.get("measurement_resolution_exact_measurement"))
                if not exact_unit or not exact_measurement:
                    raise ValueError(f"incomplete deterministic accept: {record_id}")
                _add_observation(
                    groups,
                    row,
                    origin="source_exact",
                    measurement=exact_measurement,
                    unit=exact_unit,
                )
            resolved = resolution.get(record_id)
            if resolved is None:
                continue
            joined.add(record_id)
            if (
                resolved["source_row_uid"] != _text(row["source_row_uid"])
                or resolved["source_id"] != _text(row["source_id"])
                or _text(row.get("measurement_resolution_route")) != "extract"
            ):
                raise ValueError(f"measurement-resolution/source mismatch: {record_id}")
            if resolved["status"] == "ok":
                _add_observation(
                    groups,
                    row,
                    origin="llm",
                    measurement=resolved["measurement"],
                    unit=resolved["unit"],
                )
    if joined != set(resolution):
        missing = sorted(set(resolution) - joined)
        raise ValueError(f"measurement-resolution rows absent from source: {missing[:5]}")
    prior, prior_provenance = _prior_decisions(prior_paths)
    items = []
    item_ids: set[str] = set()
    for unit, group in sorted(groups.items()):
        item_id = "u_" + hashlib.sha256(unit.encode("utf-8")).hexdigest()[:20]
        if item_id in item_ids:
            raise ValueError(f"unit item ID collision: {unit!r}")
        item_ids.add(item_id)
        items.append(
            {
                "item_id": item_id,
                "input_unit": unit,
                "origin_counts": dict(sorted(group["origin_counts"].items())),
                "route_counts": dict(sorted(group["route_counts"].items())),
                "representative_contexts": [
                    value[1]
                    for _, value in sorted(group["representatives"].items())
                ],
                "parser_output": _parser_evidence(unit),
                "prior_reviewed_decisions": sorted(
                    prior.get(unit, []), key=lambda row: row["source"]
                ),
            }
        )
    origin_counts = Counter()
    for item in items:
        origin_counts.update(item["origin_counts"])
    return {
        "version": INVENTORY_VERSION,
        "task": TASK,
        "inputs": {
            "cleaned_records": {
                "path": str(cleaned_path),
                "sha256": file_sha256(cleaned_path),
                "rows": pq.ParquetFile(cleaned_path).metadata.num_rows,
            },
            "measurement_resolution": {
                "path": str(resolution_path),
                "sha256": file_sha256(resolution_path),
                "rows": pq.ParquetFile(resolution_path).metadata.num_rows,
            },
            "prior_unit_maps": prior_provenance,
        },
        "counts": {
            "units": len(items),
            "source_occurrences": origin_counts["source"],
            "source_exact_occurrences": origin_counts["source_exact"],
            "llm_ok_occurrences": origin_counts["llm"],
        },
        "items": items,
    }


def write_inventory(
    output_dir: Path,
    *,
    cleaned_path: Path = DEFAULT_CLEANED_RECORDS,
    resolution_path: Path = DEFAULT_MAPPING_PATH,
    packet_size: int = MAX_PACKET_ITEMS,
) -> dict[str, Any]:
    if not 1 <= packet_size <= MAX_PACKET_ITEMS:
        raise ValueError(f"packet size must be between 1 and {MAX_PACKET_ITEMS}")
    inventory_path = output_dir / "inventory.json"
    manifest_path = output_dir / "inventory.manifest.json"
    if inventory_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to replace unit inventory in {output_dir}")
    inventory = build_inventory(cleaned_path, resolution_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(_json_text(inventory), encoding="utf-8")
    item_count = len(inventory["items"])
    manifest = {
        "version": "dili_unit_review_packets.v1",
        "task": TASK,
        "inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": file_sha256(inventory_path),
        },
        "inputs": inventory["inputs"],
        "item_count": item_count,
        "packet_size": packet_size,
        "packet_count": (item_count + packet_size - 1) // packet_size,
    }
    manifest_path.write_text(_json_text(manifest, pretty=True), encoding="utf-8")
    return manifest


def load_inventory_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != "dili_unit_review_packets.v1" or manifest.get("task") != TASK:
        raise ValueError("unsupported DILI unit inventory manifest")
    reference = manifest.get("inventory") or {}
    inventory_path = Path(_text(reference.get("path")))
    if not inventory_path.is_file() or file_sha256(inventory_path) != reference.get("sha256"):
        raise ValueError("DILI unit inventory hash mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("version") != INVENTORY_VERSION or inventory.get("task") != TASK:
        raise ValueError("unsupported DILI unit inventory")
    items = inventory.get("items")
    if not isinstance(items, list) or len(items) != manifest.get("item_count"):
        raise ValueError("DILI unit inventory count mismatch")
    for reference in inventory["inputs"].values():
        values = reference.values() if "path" not in reference else (reference,)
        for value in values:
            input_path = Path(value["path"])
            if file_sha256(input_path) != value["sha256"]:
                raise ValueError(f"DILI unit input hash mismatch: {input_path}")
    return manifest, inventory


def _load_reviews(path: Path, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {item["item_id"]: item for item in items}
    decisions: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank DILI unit review line: {line_number}")
        row = json.loads(line)
        item_id = _text(row.get("item_id"))
        if item_id not in expected or item_id in decisions:
            raise ValueError(f"invalid or duplicate DILI unit review: {item_id!r}")
        if _text(row.get("input_unit")) != expected[item_id]["input_unit"]:
            raise ValueError(f"DILI unit review key mismatch: {item_id}")
        action = _text(row.get("action"))
        rationale = _text(row.get("rationale"))
        if action not in {"map", "exclude"} or not rationale:
            raise ValueError(f"invalid DILI unit decision: {item_id}")
        canonical = _text(row.get("canonical_unit"))
        if action == "map" and not canonical:
            raise ValueError(f"mapped DILI unit has no target: {item_id}")
        if action == "exclude" and canonical:
            raise ValueError(f"excluded DILI unit has a target: {item_id}")
        decisions[item_id] = {
            "item_id": item_id,
            "input_unit": expected[item_id]["input_unit"],
            "action": action,
            "canonical_unit": canonical,
            "rationale": rationale,
            "review_phase": _text(row.get("review_phase")) or "primary",
        }
    missing = sorted(set(expected) - set(decisions))
    if missing:
        raise ValueError(f"DILI unit review is incomplete: {missing[:5]}")
    return [decisions[item["item_id"]] for item in items]


def _flatten_targets(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_unit = {row["input_unit"]: row for row in decisions}
    output = []
    for row in decisions:
        if row["action"] == "exclude":
            output.append(dict(row))
            continue
        seen = {row["input_unit"]}
        target = row["canonical_unit"]
        while target in by_unit and target != by_unit[target]["canonical_unit"]:
            next_row = by_unit[target]
            if next_row["action"] == "exclude":
                raise ValueError(f"DILI unit target is excluded: {target!r}")
            if target in seen:
                raise ValueError(f"DILI unit alias cycle: {sorted(seen)!r}")
            seen.add(target)
            target = next_row["canonical_unit"]
        resolved = dict(row)
        resolved["canonical_unit"] = target
        if len(seen) > 1:
            resolved["rationale"] += "; flattened reviewed target chain"
        output.append(resolved)
    return output


def build_mapping(
    manifest_path: Path,
    review_path: Path,
    review_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, inventory = load_inventory_manifest(manifest_path)
    receipt = json.loads(review_receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("version") != "dili_unit_review_run.v1"
        or receipt.get("review_sha256") != file_sha256(review_path)
        or receipt.get("inventory_manifest_sha256") != file_sha256(manifest_path)
    ):
        raise ValueError("DILI unit review receipt mismatch")
    decisions = _flatten_targets(_load_reviews(review_path, inventory["items"]))
    item_by_id = {item["item_id"]: item for item in inventory["items"]}
    entries = []
    impact = Counter()
    unit_decisions_by_origin: dict[str, Counter[str]] = defaultdict(Counter)
    occurrence_decisions_by_origin: dict[str, Counter[str]] = defaultdict(Counter)
    excluded_resolved = []
    for row in decisions:
        item = item_by_id[row["item_id"]]
        impact[row["action"]] += sum(item["origin_counts"].values())
        decision = (
            "exclude"
            if row["action"] == "exclude"
            else "alias"
            if row["canonical_unit"] != row["input_unit"]
            else "identity"
        )
        for origin, occurrence_count in item["origin_counts"].items():
            unit_decisions_by_origin[origin][decision] += 1
            occurrence_decisions_by_origin[origin][decision] += occurrence_count
        if row["action"] == "exclude" and (
            item["origin_counts"].get("llm", 0)
            or item["origin_counts"].get("source_exact", 0)
        ):
            excluded_resolved.append(
                {
                    "input_unit": row["input_unit"],
                    "llm_rows": item["origin_counts"].get("llm", 0),
                    "source_exact_rows": item["origin_counts"].get("source_exact", 0),
                    "rationale": row["rationale"],
                }
            )
        entry = {
            "task": TASK,
            "canonical_endpoints": ["*"],
            "input_unit": row["input_unit"],
            "action": row["action"],
            "domain": "any",
            "review_basis": f"deepseek_{row['review_phase']}: {row['rationale']}",
        }
        if row["action"] == "map":
            entry.update(canonical_unit=row["canonical_unit"], scale="1")
        entries.append(entry)
    counts = Counter(row["action"] for row in decisions)
    alias_count = sum(
        row["action"] == "map" and row["canonical_unit"] != row["input_unit"]
        for row in decisions
    )
    payload = {
        "version": "starling_exact_measurement_units.v2",
        "entries": entries,
        "dili_v10_contract": {
            "version": MAPPING_VERSION,
            "scope": "dili_endpoint_independent_all_observed_units",
            "value_transform": "identity",
            "scale_representation": "coefficient_preserving_equivalent_display",
            "endpoint_mapping_applied": False,
            "inputs": inventory["inputs"],
            "inventory_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": file_sha256(manifest_path),
            },
            "review": {
                "path": str(review_path.resolve()),
                "sha256": file_sha256(review_path),
                "receipt_path": str(review_receipt_path.resolve()),
                "receipt_sha256": file_sha256(review_receipt_path),
                "route": receipt["route"],
            },
            "counts": {
                **inventory["counts"],
                "mapped_units": counts["map"],
                "excluded_units": counts["exclude"],
                "alias_units": alias_count,
                "identity_units": counts["map"] - alias_count,
                "excluded_resolved_units": len(excluded_resolved),
                "excluded_llm_ok_rows": sum(row["llm_rows"] for row in excluded_resolved),
                "excluded_source_exact_rows": sum(
                    row["source_exact_rows"] for row in excluded_resolved
                ),
            },
            "validations": {
                "complete_observed_unit_coverage": True,
                "wildcard_endpoint_only": True,
                "identity_value_scale_only": True,
                "alias_targets_flattened": True,
                "frozen_input_hashes_verified": True,
            },
        },
    }
    report = {
        "version": "dili_unit_reconciliation_manifest.v1",
        "task": TASK,
        "mapping_version": MAPPING_VERSION,
        "counts": payload["dili_v10_contract"]["counts"],
        "occurrence_action_counts": dict(sorted(impact.items())),
        "unit_decision_counts_by_origin": {
            origin: dict(sorted(values.items()))
            for origin, values in sorted(unit_decisions_by_origin.items())
        },
        "occurrence_decision_counts_by_origin": {
            origin: dict(sorted(values.items()))
            for origin, values in sorted(occurrence_decisions_by_origin.items())
        },
        "excluded_resolved_units": excluded_resolved,
        "inputs": inventory["inputs"],
        "review": payload["dili_v10_contract"]["review"],
        "validations": payload["dili_v10_contract"]["validations"],
    }
    return payload, report


def write_mapping(
    manifest_path: Path,
    review_path: Path,
    review_receipt_path: Path,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    report_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to replace DILI unit map: {output_path}")
    payload, report = build_mapping(manifest_path, review_path, review_receipt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    report_temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    try:
        temporary.write_text(_json_text(payload, pretty=True), encoding="utf-8")
        load_exact_unit_mapping(temporary)
        report.update(
            mapping={
                "path": str(output_path.resolve()),
                "sha256": file_sha256(temporary),
            }
        )
        report_temporary.write_text(_json_text(report, pretty=True), encoding="utf-8")
        os.replace(temporary, output_path)
        os.replace(report_temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--cleaned-records", type=Path, default=DEFAULT_CLEANED_RECORDS)
    prepare.add_argument("--measurement-resolution", type=Path, default=DEFAULT_MAPPING_PATH)
    prepare.add_argument("--packet-size", type=int, default=MAX_PACKET_ITEMS)
    consolidate = commands.add_parser("consolidate")
    consolidate.add_argument("--inventory-manifest", type=Path, required=True)
    consolidate.add_argument("--reviews", type=Path, required=True)
    consolidate.add_argument("--review-receipt", type=Path, required=True)
    consolidate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = write_inventory(
            args.output_dir,
            cleaned_path=args.cleaned_records,
            resolution_path=args.measurement_resolution,
            packet_size=args.packet_size,
        )
        print(json.dumps({key: result[key] for key in ("item_count", "packet_count")}))
    else:
        result = write_mapping(
            args.inventory_manifest,
            args.reviews,
            args.review_receipt,
            args.output,
        )
        print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "INVENTORY_VERSION",
    "MAPPING_VERSION",
    "MAX_PACKET_ITEMS",
    "TASK",
    "build_inventory",
    "build_mapping",
    "load_inventory_manifest",
    "write_inventory",
    "write_mapping",
]
