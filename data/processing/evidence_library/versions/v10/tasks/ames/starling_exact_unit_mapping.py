"""Compile reviewed Ames source and LLM units into the shared exact map."""

from __future__ import annotations

import argparse
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    EXACT_UNIT_MAPPING_VERSION,
    load_exact_unit_mapping,
)

TASK_ID = "ames"
DECISION_VERSION = "ames_exact_unit_decisions.v1"
TASK_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = TASK_ROOT / "data_processing" / "canonicalization_v10"
REVIEWED_UNIT_DECISIONS_PATH = ASSET_ROOT / "ames_exact_unit_decisions.v1.json"
EXACT_UNIT_MAPPING_PATH = ASSET_ROOT / "ames_exact_unit_mapping.v1.json"
_INPUT_NAMES = frozenset({"cleaned_records", "measurement_resolution"})
_ORIGINS = frozenset({"source_exact", "llm"})
_DOMAINS = frozenset({"any", "nonnegative", "positive"})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"reviewed Ames exact-unit decisions are required but missing: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"exact-unit decision asset must be an object: {path}")
    return payload


def _validate_inputs(payload: dict[str, Any], decision_path: Path) -> None:
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != _INPUT_NAMES:
        raise ValueError(f"decision inputs must be exactly {sorted(_INPUT_NAMES)}")
    for name, item in inputs.items():
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"invalid {name} decision input")
        input_path = Path(str(item["path"]))
        if not input_path.is_absolute():
            input_path = decision_path.parent / input_path
        actual = file_sha256(input_path)
        if actual != str(item["sha256"]):
            raise ValueError(
                f"{name} digest drift: expected {item['sha256']}, found {actual}"
            )


def _observed_pairs(payload: dict[str, Any]) -> set[tuple[str, str]]:
    observed: set[tuple[str, str]] = set()
    for item in payload.get("observed_pairs", []):
        if not isinstance(item, dict):
            raise TypeError(f"invalid observed exact-unit pair: {item!r}")
        key = (
            str(item.get("canonical_endpoint") or ""),
            str(item.get("input_unit") or ""),
        )
        origins = item.get("origins")
        rows = item.get("rows")
        if (
            not all(key)
            or key in observed
            or not isinstance(origins, list)
            or origins != sorted(set(origins))
            or not origins
            or not set(origins) <= _ORIGINS
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
        ):
            raise ValueError(f"invalid observed exact-unit pair: {item!r}")
        observed.add(key)
    if not observed:
        raise ValueError("exact-unit decision asset has no observed pairs")
    return observed


def _decimal_text(value: Any, entry: dict[str, Any]) -> str:
    try:
        scale = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid exact-unit scale: {entry!r}") from error
    if not scale.is_finite() or scale <= 0:
        raise ValueError(f"invalid exact-unit scale: {entry!r}")
    text = format(scale, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _compiled_decision(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise TypeError(f"invalid exact-unit decision: {entry!r}")
    endpoints = entry.get("canonical_endpoints")
    input_unit = str(entry.get("input_unit") or "")
    action = str(entry.get("action") or "")
    domain = str(entry.get("domain") or "any")
    if (
        not isinstance(endpoints, list)
        or endpoints != sorted(set(endpoints))
        or not endpoints
        or not all(isinstance(value, str) and value for value in endpoints)
        or ("*" in endpoints and endpoints != ["*"])
        or not input_unit
        or action not in {"map", "exclude"}
        or domain not in _DOMAINS
        or not str(entry.get("review_basis") or "").strip()
    ):
        raise ValueError(f"invalid exact-unit decision: {entry!r}")
    if action == "exclude":
        if any(name in entry for name in ("canonical_unit", "scale")):
            raise ValueError(f"excluded exact-unit decision has map fields: {entry!r}")
        return {
            "task": TASK_ID,
            "canonical_endpoints": endpoints,
            "input_unit": input_unit,
            "action": action,
            "domain": domain,
            "review_basis": entry["review_basis"],
        }
    return _compiled_map_decision(entry, endpoints, input_unit, domain)


def _compiled_map_decision(
    entry: dict[str, Any], endpoints: list[str], input_unit: str, domain: str
) -> dict[str, Any]:
    canonical_unit = str(entry.get("canonical_unit") or "")
    scale = _decimal_text(entry.get("scale", "1"), entry)
    if not canonical_unit:
        raise ValueError(f"mapped exact-unit decision has no output unit: {entry!r}")
    if scale != "1" and not str(entry.get("scale_review_basis") or "").strip():
        raise ValueError(f"nonidentity exact-unit scale is unreviewed: {entry!r}")
    if domain != "any" and not str(entry.get("domain_review_basis") or "").strip():
        raise ValueError(f"restricted exact-unit domain is unreviewed: {entry!r}")
    output = {
        "task": TASK_ID,
        "canonical_endpoints": endpoints,
        "input_unit": input_unit,
        "action": "map",
        "canonical_unit": canonical_unit,
        "scale": scale,
        "domain": domain,
        "review_basis": entry["review_basis"],
    }
    for field in ("scale_review_basis", "domain_review_basis"):
        if entry.get(field):
            output[field] = entry[field]
    return output


def _decision_keys(entries: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for entry in entries:
        for endpoint in entry["canonical_endpoints"]:
            key = (endpoint, entry["input_unit"])
            if key in keys:
                raise ValueError(f"duplicate exact-unit decision key: {key}")
            keys.add(key)
    return keys


def _validate_coverage(
    observed: set[tuple[str, str]], decision_keys: set[tuple[str, str]]
) -> None:
    used: set[tuple[str, str]] = set()
    missing: list[tuple[str, str]] = []
    for endpoint, unit in sorted(observed):
        selected = (
            (endpoint, unit) if (endpoint, unit) in decision_keys else ("*", unit)
        )
        if selected not in decision_keys:
            missing.append((endpoint, unit))
        else:
            used.add(selected)
    if missing:
        raise ValueError(f"unreviewed observed exact-unit pairs: {missing[:5]!r}")
    unused = sorted(decision_keys - used)
    if unused:
        raise ValueError(
            f"exact-unit decisions outside observed inventory: {unused[:5]!r}"
        )


def load_reviewed_unit_decisions(
    path: str | Path = REVIEWED_UNIT_DECISIONS_PATH,
) -> dict[str, Any]:
    """Load and fully validate the combined source-plus-LLM decision asset."""
    target = Path(path)
    payload = _read_json(target)
    if payload.get("version") != DECISION_VERSION or payload.get("task") != TASK_ID:
        raise ValueError(f"unsupported Ames exact-unit decision asset: {target}")
    _validate_inputs(payload, target)
    observed = _observed_pairs(payload)
    entries = [_compiled_decision(entry) for entry in payload.get("decisions", [])]
    _validate_coverage(observed, _decision_keys(entries))
    return payload


def compile_exact_unit_mapping(
    path: str | Path = REVIEWED_UNIT_DECISIONS_PATH,
) -> dict[str, Any]:
    """Compile reviewed decisions to the shared fail-closed runtime schema."""
    target = Path(path).resolve()
    payload = load_reviewed_unit_decisions(target)
    entries = [_compiled_decision(entry) for entry in payload["decisions"]]
    entries.sort(key=lambda item: (item["input_unit"], item["canonical_endpoints"]))
    return {
        "version": EXACT_UNIT_MAPPING_VERSION,
        "entries": entries,
        "ames_v10_contract": {
            "decision_version": DECISION_VERSION,
            "decision_path": str(target),
            "decision_sha256": file_sha256(target),
            "inputs": payload["inputs"],
            "observed_pair_count": len(payload["observed_pairs"]),
            "unreviewed_unit_policy": "error",
            "unreviewed_scale_policy": "identity_only",
        },
    }


def write_exact_unit_mapping(
    decision_path: str | Path = REVIEWED_UNIT_DECISIONS_PATH,
    output_path: str | Path = EXACT_UNIT_MAPPING_PATH,
) -> Path:
    """Validate and atomically publish a new exact-unit mapping."""
    output = Path(output_path)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    for path in (output, temporary):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite exact-unit output: {path}")
    payload = compile_exact_unit_mapping(decision_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = temporary.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite exact-unit output: {temporary}"
        ) from error
    try:
        with handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        load_exact_unit_mapping(temporary)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite exact-unit output: {output}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    load_exact_unit_mapping(output)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=REVIEWED_UNIT_DECISIONS_PATH)
    parser.add_argument("--output", type=Path, default=EXACT_UNIT_MAPPING_PATH)
    args = parser.parse_args(argv)
    print(write_exact_unit_mapping(args.decisions, args.output))


if __name__ == "__main__":
    main()


__all__ = [
    "DECISION_VERSION",
    "EXACT_UNIT_MAPPING_PATH",
    "REVIEWED_UNIT_DECISIONS_PATH",
    "compile_exact_unit_mapping",
    "load_reviewed_unit_decisions",
    "write_exact_unit_mapping",
]
