"""Build and consolidate complete AMES V10 exact-unit review packets.

The shared unit parser is recorded as review evidence only.  It never chooses a
mapping or a value scale; reviewed mappings default to scale 1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.compile_candidate_resolution import (
    validate_compiled_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.ames.starling_exact_unit_mapping import (
    DECISION_VERSION,
    compile_exact_unit_mapping,
    load_reviewed_unit_decisions,
)
from tools.chembl_tool.common.units import canonicalize_unit

INVENTORY_VERSION = "ames_exact_unit_inventory.v1"
PACKET_VERSION = "ames_exact_unit_review_packet.v1"
PACKET_MANIFEST_VERSION = "ames_exact_unit_review_packets.v1"
COMPLETION_VERSION = "ames_exact_unit_review_completion.v1"
MAX_PACKET_ITEMS = 500
TERMINAL_STATUSES = frozenset({"ok", "relative", "unsure", "unavailable"})
_INPUT_NAMES = frozenset({"cleaned_records", "measurement_resolution"})

_CLEANED_REQUIRED = (
    "cleaned_record_id",
    "source_row_uid",
    "source_id",
    "canonical_endpoint_name",
    "measurement_resolution_route",
    "measurement_resolution_exact_measurement",
    "measurement_resolution_exact_unit",
    "measurement_text",
    "unit_text",
)
_RESOLUTION_REQUIRED = (
    "cleaned_record_id",
    "source_row_uid",
    "source_id",
    "status",
    "measurements_json",
)
_CONTEXT_FIELDS = (
    "endpoint_name",
    "support_text",
    "mutagenicity_result",
    "evidence_basis",
    "test_system",
    "metabolic_activation",
    "endpoint_class",
    "endpoint_subtype",
    "study_context",
    "biological_test_system",
    "biological_system",
    "result_call",
    "result_status",
    "assay_method_and_endpoint",
    "mechanism_category",
    "result_direction",
)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _iter_rows(
    path: Path, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    available = set(pq.read_schema(path).names)
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")
    columns = [*required, *(field for field in optional if field in available)]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        yield from batch.to_pylist()


def _measurement_entries(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        entries = json.loads(_text(row.get("measurements_json")) or "[]")
    except json.JSONDecodeError as error:
        raise ValueError("invalid measurements_json") from error
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) for item in entries
    ):
        raise ValueError("measurements_json must contain a list of objects")
    return entries


def _load_resolution(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for row in _iter_rows(path, _RESOLUTION_REQUIRED):
        record_id = _text(row["cleaned_record_id"])
        status = _text(row["status"])
        entries = _measurement_entries(row)
        expected = 1 if status == "ok" else 0
        if not record_id or record_id in records or status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid measurement-resolution identity: {record_id!r}")
        if len(entries) != expected:
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
        if (
            not records[record_id]["source_row_uid"]
            or not records[record_id]["source_id"]
        ):
            raise ValueError(f"missing measurement-resolution provenance: {record_id}")
    return records


def _source_context(
    row: Mapping[str, Any], origin: str, measurement: str, unit: str
) -> dict[str, str]:
    context = {
        "origin": origin,
        "cleaned_record_id": _text(row["cleaned_record_id"]),
        "source_row_uid": _text(row["source_row_uid"]),
        "source_id": _text(row["source_id"]),
        "source_measurement": _text(row["measurement_text"]),
        "source_unit": _text(row["unit_text"]),
        "resolved_measurement": measurement,
        "resolved_unit": unit,
    }
    context.update(
        (field, _text(row[field])) for field in _CONTEXT_FIELDS if _text(row.get(field))
    )
    return context


def _add_observation(
    groups: dict[tuple[str, str], dict[str, Any]],
    row: Mapping[str, Any],
    origin: str,
    measurement: str,
    unit: str,
) -> None:
    endpoint = _text(row["canonical_endpoint_name"])
    if not endpoint or not unit:
        raise ValueError(
            f"resolved unit lacks an endpoint or unit: {row['cleaned_record_id']}"
        )
    group = groups.setdefault(
        (endpoint, unit),
        {"record_ids": set(), "origin_ids": {}, "representatives": {}},
    )
    record_id = _text(row["cleaned_record_id"])
    group["record_ids"].add(record_id)
    group["origin_ids"].setdefault(origin, set()).add(record_id)
    selector = (_text(row["source_id"]), _text(row["source_row_uid"]), record_id)
    existing = group["representatives"].get(origin)
    if existing is None or selector < existing[0]:
        group["representatives"][origin] = (
            selector,
            _source_context(row, origin, measurement, unit),
        )


def _scan_cleaned(
    path: Path, resolution: Mapping[str, Mapping[str, str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[str] = set()
    joined: set[str] = set()
    expected_extract: set[str] = set()
    for row in _iter_rows(path, _CLEANED_REQUIRED, _CONTEXT_FIELDS):
        record_id = _text(row["cleaned_record_id"])
        if not record_id or record_id in seen:
            raise ValueError(f"duplicate or empty cleaned_record_id: {record_id!r}")
        seen.add(record_id)
        route = _text(row["measurement_resolution_route"])
        if route == "extract":
            expected_extract.add(record_id)
        if route == "accept":
            _add_observation(
                groups,
                row,
                "source_exact",
                _text(row["measurement_resolution_exact_measurement"]),
                _text(row["measurement_resolution_exact_unit"]),
            )
        resolved = resolution.get(record_id)
        if resolved is not None:
            _validate_join(row, resolved)
            joined.add(record_id)
            if route != "extract":
                raise ValueError(f"resolved row is not routed to extract: {record_id}")
            if resolved["status"] == "ok":
                _add_observation(
                    groups, row, "llm", resolved["measurement"], resolved["unit"]
                )
    missing = sorted(set(resolution) - joined)
    if missing:
        raise ValueError(
            f"measurement-resolution rows missing from source: {missing[:5]}"
        )
    unresolved = sorted(expected_extract - set(resolution))
    if unresolved:
        raise ValueError(
            f"extract rows missing measurement resolution: {unresolved[:5]}"
        )
    if not groups:
        raise ValueError("exact-unit inventory is empty")
    return groups


def _validate_join(row: Mapping[str, Any], resolved: Mapping[str, str]) -> None:
    expected = (_text(row["source_row_uid"]), _text(row["source_id"]))
    found = (resolved["source_row_uid"], resolved["source_id"])
    if expected != found:
        raise ValueError(
            f"measurement/source provenance mismatch: {row['cleaned_record_id']}"
        )


def _parser_output(unit: str) -> dict[str, Any]:
    output = asdict(canonicalize_unit(unit, task="ames"))
    output["dimension"] = [list(item) for item in output["dimension"]]
    output["unknown_tokens"] = list(output["unknown_tokens"])
    return output


def _observed_pairs(
    groups: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for (endpoint, unit), group in sorted(groups.items()):
        origin_counts = {
            origin: len(record_ids)
            for origin, record_ids in sorted(group["origin_ids"].items())
        }
        pairs.append(
            {
                "canonical_endpoint": endpoint,
                "input_unit": unit,
                "origins": sorted(origin_counts),
                "rows": len(group["record_ids"]),
                "origin_counts": origin_counts,
                "representative_contexts": [
                    value[1] for _, value in sorted(group["representatives"].items())
                ],
                "parser_output": _parser_output(unit),
            }
        )
    return pairs


def build_unit_inventory(cleaned_path: Path, resolution_path: Path) -> dict[str, Any]:
    """Derive the complete endpoint-unit union from source and LLM resolutions."""
    cleaned_path = cleaned_path.resolve()
    resolution_path = resolution_path.resolve()
    validate_compiled_mapping(resolution_path)
    resolution = _load_resolution(resolution_path)
    pairs = _observed_pairs(_scan_cleaned(cleaned_path, resolution))
    origin_rows = Counter()
    for pair in pairs:
        origin_rows.update(pair["origin_counts"])
    return {
        "version": INVENTORY_VERSION,
        "task": "ames",
        "inputs": {
            "cleaned_records": {
                "path": str(cleaned_path),
                "sha256": file_sha256(cleaned_path),
            },
            "measurement_resolution": {
                "path": str(resolution_path),
                "sha256": file_sha256(resolution_path),
            },
        },
        "parser_role": "advisory_only_no_automatic_mapping",
        "mapping_scale_default": "1",
        "observed_pairs": pairs,
        "counts": {
            "observed_pairs": len(pairs),
            "source_exact_rows": origin_rows["source_exact"],
            "llm_ok_rows": origin_rows["llm"],
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _packet_payload(
    packet_id: str, inventory_sha256: str, items: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "version": PACKET_VERSION,
        "task": "ames",
        "packet_id": packet_id,
        "inventory_sha256": inventory_sha256,
        "item_count": len(items),
        "review_scale_default": "1",
        "items": items,
    }


def _prepare_directory(
    output_dir: Path, inventory: Mapping[str, Any], packet_size: int
) -> dict[str, Any]:
    inventory_path = output_dir / "inventory.json"
    _write_json(inventory_path, inventory)
    inventory_hash = file_sha256(inventory_path)
    rows = inventory["observed_pairs"]
    packets = []
    for start in range(0, len(rows), packet_size):
        items = rows[start : start + packet_size]
        packet_id = f"packet_{len(packets) + 1:04d}"
        relative = Path("packets") / f"{packet_id}.json"
        packet_path = output_dir / relative
        _write_json(packet_path, _packet_payload(packet_id, inventory_hash, items))
        packets.append(
            {
                "packet_id": packet_id,
                "path": str(relative),
                "sha256": file_sha256(packet_path),
                "item_count": len(items),
                "first_key": [items[0]["canonical_endpoint"], items[0]["input_unit"]],
                "last_key": [items[-1]["canonical_endpoint"], items[-1]["input_unit"]],
            }
        )
    manifest = {
        "version": PACKET_MANIFEST_VERSION,
        "task": "ames",
        "inventory": {"path": "inventory.json", "sha256": inventory_hash},
        "inputs": inventory["inputs"],
        "packet_size": packet_size,
        "packet_count": len(packets),
        "item_count": len(rows),
        "unreviewed_item_count": len(rows),
        "packets": packets,
        "validations": {
            "complete_disjoint_inventory": True,
            "maximum_packet_items": MAX_PACKET_ITEMS,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def write_review_packets(
    cleaned_path: Path,
    resolution_path: Path,
    output_dir: Path,
    *,
    packet_size: int = MAX_PACKET_ITEMS,
) -> dict[str, Any]:
    """Freeze a complete deterministic inventory and disjoint review packets."""
    if not 1 <= packet_size <= MAX_PACKET_ITEMS:
        raise ValueError(f"packet size must be in [1, {MAX_PACKET_ITEMS}]")
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to replace frozen review packets: {output_dir}"
        )
    temporary = output_dir.with_name(f"{output_dir.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary review directory already exists: {temporary}")
    inventory = build_unit_inventory(cleaned_path, resolution_path)
    try:
        temporary.mkdir(parents=True)
        manifest = _prepare_directory(temporary, inventory, packet_size)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_review_packet_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the frozen packet manifest and exact inventory coverage."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != PACKET_MANIFEST_VERSION
        or manifest.get("task") != "ames"
    ):
        raise ValueError("unsupported AMES exact-unit packet manifest")
    if not 1 <= int(manifest.get("packet_size") or 0) <= MAX_PACKET_ITEMS:
        raise ValueError("invalid AMES exact-unit packet size")
    root = path.parent.resolve()
    inventory_path = _contained_path(root, manifest["inventory"]["path"], "inventory")
    if file_sha256(inventory_path) != manifest["inventory"]["sha256"]:
        raise ValueError("exact-unit inventory digest mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("version") != INVENTORY_VERSION or inventory.get("task") != "ames":
        raise ValueError("unsupported AMES exact-unit inventory")
    if manifest.get("inputs") != inventory.get("inputs"):
        raise ValueError("exact-unit packet input provenance mismatch")
    input_paths = _validate_inventory_inputs(inventory, inventory_path)
    rebuilt = build_unit_inventory(
        input_paths["cleaned_records"], input_paths["measurement_resolution"]
    )
    if rebuilt != inventory:
        raise ValueError("exact-unit inventory differs from frozen inputs")
    _validate_packet_manifest_claims(manifest, inventory)
    flattened = _load_packet_rows(root, manifest, manifest["inventory"]["sha256"])
    if flattened != inventory.get("observed_pairs"):
        raise ValueError("review packets do not exactly cover the unit inventory")
    return manifest, inventory


def _contained_path(root: Path, value: Any, label: str) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"exact-unit {label} path escapes the packet directory")
    return resolved


def _validate_packet_manifest_claims(
    manifest: Mapping[str, Any], inventory: Mapping[str, Any]
) -> None:
    packets = manifest.get("packets")
    count = len(inventory.get("observed_pairs") or [])
    packet_size = int(manifest["packet_size"])
    expected_packet_count = (count + packet_size - 1) // packet_size
    expected_validations = {
        "complete_disjoint_inventory": True,
        "maximum_packet_items": MAX_PACKET_ITEMS,
    }
    if (
        not isinstance(packets, list)
        or manifest.get("packet_count") != len(packets)
        or len(packets) != expected_packet_count
    ):
        raise ValueError("exact-unit packet count mismatch")
    if (
        manifest.get("item_count") != count
        or manifest.get("unreviewed_item_count") != count
    ):
        raise ValueError("exact-unit packet item count mismatch")
    if manifest.get("validations") != expected_validations:
        raise ValueError("exact-unit packet validation claims mismatch")


def _load_packet_rows(
    root: Path, manifest: Mapping[str, Any], inventory_sha256: str
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    packet_size = int(manifest["packet_size"])
    item_count = int(manifest["item_count"])
    for index, item in enumerate(manifest.get("packets", []), 1):
        packet_path = _contained_path(root, item["path"], "packet")
        if file_sha256(packet_path) != item["sha256"]:
            raise ValueError(f"review packet digest mismatch: {packet_path}")
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        if (
            packet.get("version") != PACKET_VERSION
            or packet.get("task") != "ames"
            or packet.get("packet_id") != f"packet_{index:04d}"
            or item.get("packet_id") != packet.get("packet_id")
            or packet.get("inventory_sha256") != inventory_sha256
            or packet.get("review_scale_default") != "1"
        ):
            raise ValueError(f"unsupported review packet: {packet_path}")
        rows = packet.get("items") or []
        expected_count = min(packet_size, item_count - len(flattened))
        if (
            len(rows) != item["item_count"]
            or packet.get("item_count") != len(rows)
            or len(rows) != expected_count
        ):
            raise ValueError(f"invalid review packet size: {packet_path}")
        first = [rows[0]["canonical_endpoint"], rows[0]["input_unit"]]
        last = [rows[-1]["canonical_endpoint"], rows[-1]["input_unit"]]
        if item.get("first_key") != first or item.get("last_key") != last:
            raise ValueError(f"review packet boundary mismatch: {packet_path}")
        flattened.extend(rows)
    return flattened


def _validate_inventory_inputs(
    inventory: Mapping[str, Any], path: Path
) -> dict[str, Path]:
    inputs = inventory.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != _INPUT_NAMES:
        raise ValueError("exact-unit inventory input provenance is incomplete")
    paths: dict[str, Path] = {}
    for name, item in inputs.items():
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"invalid exact-unit inventory input: {name}")
        input_path = Path(str(item["path"]))
        if not input_path.is_absolute():
            input_path = path.parent / input_path
        actual = file_sha256(input_path)
        if actual != str(item["sha256"]):
            raise ValueError(
                f"exact-unit inventory input digest drift for {name}: "
                f"expected {item['sha256']}, found {actual}"
            )
        paths[name] = input_path.resolve()
    return paths


def _review_key(row: Mapping[str, Any]) -> tuple[str, str]:
    endpoints = row.get("canonical_endpoints")
    endpoint = _text(row.get("canonical_endpoint"))
    if endpoints is not None:
        if endpoint or not isinstance(endpoints, list) or len(endpoints) != 1:
            raise ValueError(f"review must name exactly one endpoint: {row!r}")
        endpoint = _text(endpoints[0])
    unit = _text(row.get("input_unit"))
    if not endpoint or not unit:
        raise ValueError(f"review key is incomplete: {row!r}")
    return endpoint, unit


def _review_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    endpoint, unit = _review_key(row)
    action = _text(row.get("action"))
    basis = _text(row.get("review_basis"))
    if action not in {"map", "exclude"} or not basis:
        raise ValueError(f"review action or basis is invalid: {(endpoint, unit)}")
    entry: dict[str, Any] = {
        "canonical_endpoints": [endpoint],
        "input_unit": unit,
        "action": action,
        "domain": _text(row.get("domain")) or "any",
        "review_basis": basis,
    }
    if action == "exclude":
        if any(field in row for field in ("canonical_unit", "scale")):
            raise ValueError(f"excluded review has mapping fields: {(endpoint, unit)}")
    else:
        canonical = _text(row.get("canonical_unit"))
        if not canonical:
            raise ValueError(f"mapped review has no canonical unit: {(endpoint, unit)}")
        entry.update(canonical_unit=canonical, scale=row.get("scale", "1"))
    for field in (
        "scale_review_basis",
        "domain_review_basis",
        "reviewer",
        "review_round",
        "reviewed_at",
    ):
        if row.get(field) not in (None, ""):
            entry[field] = row[field]
    return entry


def _load_review_entries(
    path: Path, observed: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"review line {line_number} is not an object")
        key = _review_key(row)
        if key in keys:
            raise ValueError(f"duplicate exact-unit review: {key}")
        keys.add(key)
        entries.append(_review_entry(row))
    missing, extra = sorted(observed - keys), sorted(keys - observed)
    if missing or extra:
        raise ValueError(
            f"review coverage mismatch: missing={missing[:5]!r}, extra={extra[:5]!r}"
        )
    entries.sort(key=lambda item: (item["canonical_endpoints"], item["input_unit"]))
    _validate_stable_targets(entries)
    return entries


def _validate_stable_targets(entries: list[dict[str, Any]]) -> None:
    by_key = {
        (entry["canonical_endpoints"][0], entry["input_unit"]): entry
        for entry in entries
    }
    for (endpoint, unit), entry in by_key.items():
        target = _text(entry.get("canonical_unit"))
        if entry["action"] != "map" or target == unit:
            continue
        target_entry = by_key.get((endpoint, target))
        if (
            target_entry is None
            or target_entry["action"] != "map"
            or _text(target_entry.get("canonical_unit")) != target
        ):
            raise ValueError(
                f"canonical unit target must be an observed stable identity: "
                f"{(endpoint, unit)} -> {target!r}"
            )


def _validated_run_receipt(
    packet_manifest_path: Path, review_path: Path
) -> tuple[Path, dict[str, Any]]:
    from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.run_exact_unit_review import (
        review_receipt_path,
        validate_review_run_receipt,
    )

    receipt_path = review_receipt_path(review_path).resolve()
    receipt = validate_review_run_receipt(
        review_path,
        receipt_path=receipt_path,
        packet_manifest_path=packet_manifest_path,
    )
    return receipt_path, receipt


def _completion_manifest(
    packet_manifest_path: Path,
    review_path: Path,
    review_receipt_path: Path,
    output_path: Path,
    output_sha256: str,
    item_count: int,
) -> dict[str, Any]:
    return {
        "version": COMPLETION_VERSION,
        "task": "ames",
        "packet_manifest": {
            "path": str(packet_manifest_path.resolve()),
            "sha256": file_sha256(packet_manifest_path),
        },
        "review_decisions": {
            "path": str(review_path.resolve()),
            "sha256": file_sha256(review_path),
        },
        "review_run_receipt": {
            "path": str(review_receipt_path.resolve()),
            "sha256": file_sha256(review_receipt_path),
        },
        "reviewed_decision_asset": {
            "path": str(output_path.resolve()),
            "sha256": output_sha256,
        },
        "counts": {
            "observed_items": item_count,
            "reviewed_items": item_count,
            "unreviewed_items": 0,
        },
        "validations": {
            "exact_pair_coverage": True,
            "stable_alias_targets": True,
            "unreviewed_scale_policy": "identity_only",
        },
    }


def _validate_completion_reference(
    reference: Any, expected_path: Path, expected_sha256: str, label: str
) -> None:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"invalid exact-unit completion {label} reference")
    path = Path(str(reference["path"]))
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"exact-unit completion {label} path mismatch")
    if reference["sha256"] != expected_sha256 or file_sha256(path) != expected_sha256:
        raise ValueError(f"exact-unit completion {label} hash mismatch")


def _validate_completion_lineage(
    completion: Mapping[str, Any], decision_path: Path, decisions: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    review = decisions.get("review") or {}
    packet_path = Path(str(review.get("packet_manifest_path") or ""))
    review_path = Path(str(review.get("review_path") or ""))
    receipt_path = Path(str(review.get("review_run_receipt_path") or ""))
    _validate_completion_reference(
        completion.get("packet_manifest"),
        packet_path,
        str(review.get("packet_manifest_sha256") or ""),
        "packet manifest",
    )
    _validate_completion_reference(
        completion.get("review_decisions"),
        review_path,
        str(review.get("review_sha256") or ""),
        "review decisions",
    )
    _validate_completion_reference(
        completion.get("review_run_receipt"),
        receipt_path,
        str(review.get("review_run_receipt_sha256") or ""),
        "review run receipt",
    )
    _validate_completion_reference(
        completion.get("reviewed_decision_asset"),
        decision_path,
        file_sha256(decision_path),
        "decision asset",
    )
    return packet_path, review_path, receipt_path


def _validate_completion_inventory(
    packet_path: Path, review_path: Path, decisions: Mapping[str, Any]
) -> int:
    packet, inventory = load_review_packet_manifest(packet_path)
    observed = decisions.get("observed_pairs") or []
    observed_keys = {
        (item["canonical_endpoint"], item["input_unit"]) for item in observed
    }
    if _load_review_entries(review_path, observed_keys) != decisions.get("decisions"):
        raise ValueError("exact-unit decisions differ from the frozen review")
    count = len(observed)
    if (
        packet.get("item_count") != count
        or inventory.get("observed_pairs") != observed
        or inventory.get("inputs") != decisions.get("inputs")
    ):
        raise ValueError("exact-unit completion inventory mismatch")
    return count


def validate_review_completion(
    decision_path: Path, completion_path: Path | None = None
) -> dict[str, Any]:
    """Recursively validate the exact-unit review completion receipt."""
    decision_path = decision_path.resolve()
    completion_path = completion_path or decision_path.with_suffix(".manifest.json")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    decisions = load_reviewed_unit_decisions(decision_path)
    packet_path, review_path, receipt_path = _validate_completion_lineage(
        completion, decision_path, decisions
    )
    validated_receipt_path, _ = _validated_run_receipt(packet_path, review_path)
    if validated_receipt_path != receipt_path.resolve():
        raise ValueError("exact-unit run receipt lineage mismatch")
    count = _validate_completion_inventory(packet_path, review_path, decisions)
    expected_counts = {
        "observed_items": count,
        "reviewed_items": count,
        "unreviewed_items": 0,
    }
    expected_validations = {
        "exact_pair_coverage": True,
        "stable_alias_targets": True,
        "unreviewed_scale_policy": "identity_only",
    }
    if (
        completion.get("version") != COMPLETION_VERSION
        or completion.get("task") != "ames"
    ):
        raise ValueError("unsupported AMES exact-unit completion manifest")
    if completion.get("counts") != expected_counts:
        raise ValueError("exact-unit completion counts mismatch")
    if completion.get("validations") != expected_validations:
        raise ValueError("exact-unit completion validation claims mismatch")
    return completion


def _publish_review_pair(
    decision_temporary: Path,
    decision_path: Path,
    completion_temporary: Path,
    completion_path: Path,
) -> None:
    installed: list[Path] = []
    try:
        os.replace(decision_temporary, decision_path)
        installed.append(decision_path)
        os.replace(completion_temporary, completion_path)
        installed.append(completion_path)
        validate_review_completion(decision_path, completion_path)
    except BaseException:
        for path in reversed(installed):
            path.unlink(missing_ok=True)
        raise


def write_reviewed_decisions(
    packet_manifest_path: Path, review_path: Path, output_path: Path
) -> dict[str, Any]:
    """Consolidate complete reviews into the existing AMES decision schema."""
    completion_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or completion_path.exists():
        raise FileExistsError(f"refusing to replace reviewed unit asset: {output_path}")
    _, inventory = load_review_packet_manifest(packet_manifest_path)
    review_receipt_path, _ = _validated_run_receipt(packet_manifest_path, review_path)
    observed = {
        (row["canonical_endpoint"], row["input_unit"])
        for row in inventory["observed_pairs"]
    }
    entries = _load_review_entries(review_path, observed)
    payload = {
        "version": DECISION_VERSION,
        "task": "ames",
        "inputs": inventory["inputs"],
        "observed_pairs": inventory["observed_pairs"],
        "decisions": entries,
        "review": {
            "packet_manifest_path": str(packet_manifest_path.resolve()),
            "packet_manifest_sha256": file_sha256(packet_manifest_path),
            "review_path": str(review_path.resolve()),
            "review_sha256": file_sha256(review_path),
            "review_run_receipt_path": str(review_receipt_path),
            "review_run_receipt_sha256": file_sha256(review_receipt_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    completion_temporary = completion_path.with_suffix(f"{completion_path.suffix}.tmp")
    if temporary.exists() or completion_temporary.exists():
        raise FileExistsError("temporary reviewed-unit output already exists")
    try:
        _write_json(temporary, payload)
        compile_exact_unit_mapping(temporary)
        completion = _completion_manifest(
            packet_manifest_path,
            review_path,
            review_receipt_path,
            output_path,
            file_sha256(temporary),
            len(observed),
        )
        _write_json(completion_temporary, completion)
        _publish_review_pair(
            temporary, output_path, completion_temporary, completion_path
        )
    finally:
        temporary.unlink(missing_ok=True)
        completion_temporary.unlink(missing_ok=True)
    return completion


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--cleaned-records", type=Path, required=True)
    prepare.add_argument("--measurement-resolution", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--packet-size", type=int, default=MAX_PACKET_ITEMS)
    consolidate = commands.add_parser("consolidate")
    consolidate.add_argument("--packet-manifest", type=Path, required=True)
    consolidate.add_argument("--reviews", type=Path, required=True)
    consolidate.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        result = write_review_packets(
            args.cleaned_records,
            args.measurement_resolution,
            args.output_dir,
            packet_size=args.packet_size,
        )
        print(
            json.dumps(
                {"packets": result["packet_count"], "items": result["item_count"]}
            )
        )
    else:
        result = write_reviewed_decisions(
            args.packet_manifest, args.reviews, args.output
        )
        print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INVENTORY_VERSION",
    "MAX_PACKET_ITEMS",
    "build_unit_inventory",
    "load_review_packet_manifest",
    "validate_review_completion",
    "write_review_packets",
    "write_reviewed_decisions",
]
