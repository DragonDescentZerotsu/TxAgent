"""Reconcile Skin V10 source, exact-route, and extracted units."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
    load_exact_unit_mapping,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction import (
    starling_measurement_resolution,
)


TASK_ROOT = Path(__file__).resolve().parents[1]
BASE_MAPPING = (
    TASK_ROOT / "data_processing/canonicalization_v7/exact_measurement_unit_map.v2.json"
)
DEFAULT_OUTPUT = (
    TASK_ROOT / "data_processing/canonicalization_v7/exact_measurement_unit_map.v3.json"
)
MAPPING_VERSION = "starling_exact_measurement_units.v3"
DEFAULT_CLEANED_RECORDS = starling_measurement_resolution.DEFAULT_CLEANED_RECORDS
DEFAULT_MAPPING_PATH = starling_measurement_resolution.DEFAULT_MAPPING_PATH

_CLEANED_COLUMNS = (
    "cleaned_record_id",
    "canonical_endpoint_name",
    "unit_text",
    "measurement_resolution_route",
    "measurement_resolution_exact_unit",
    "measurement_resolution_exact_unit_is_canonical",
)
_PERCENT_UNITS = {
    "% (incidence)",
    "% incidence",
    "% peptide depletion",
    "% responders",
    "% sensitization",
    "combined cysteine+lysine depletion",
    "positivity",
    "prevalence",
    "sensitization frequency",
    "sensitization rate",
    "SRe",
    "weighted average prevalence",
    "percent positive",
}
_ALIASES = {
    "10^-2 mm ear swelling": "10^-2 mm",
    "10^-3 cm ear thickness increment": "10^-3 cm",
    "10^-3 in": "10^-3 inches",
    "10^-3 inches": "10^-3 inches",
    "10^-4 in.": "10^-4 inches",
    "EC1.5": "%",
    "EC3": "%",
    "EC3 mmol L^-1": "mmol/L",
    "days to reaction": "days",
    "fold SEAP induction": "fold",
    "mg/ml": "mg/mL",
    "nmoles/mg protein per hour": "nmol/mg protein/h",
    "pg/ml": "pg/mL",
    "S.I.": "stimulation index",
    "stimulation indices": "stimulation index",
    "µg/cm^2 EC3": "µg/cm^2",
    "µg/mL minimum induction threshold": "µg/mL",
    "µg/ml": "µg/mL",
    "× 10^-2 mm ear swelling": "10^-2 mm",
    "×10^-4 in.": "10^-4 inches",
    "mm × 10^-2": "10^-2 mm",
    "mm × 10^-2 ear swelling": "10^-2 mm",
}
_EXCLUDED_UNITS = {"patient positive patch test to cobalt chloride"}
_OUT_OF = re.compile(r"\bout of\s+(\d+)\b", re.IGNORECASE)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _decision(unit: str) -> tuple[str, str | None, str | None, str, str]:
    """Return the conservative reviewed decision for a previously unseen unit."""
    if unit in _EXCLUDED_UNITS:
        return "exclude", None, None, "any", "reviewed_non_unit_phrase"
    denominator = _OUT_OF.search(unit)
    if denominator:
        scale = str(Decimal(1) / Decimal(denominator.group(1)))
        return "map", "fraction", scale, "nonnegative", "reviewed_explicit_denominator"
    if unit in _PERCENT_UNITS:
        return "map", "%", "1", "nonnegative", "reviewed_percent_semantics"
    if unit in _ALIASES:
        canonical = _ALIASES[unit]
        domain = "nonnegative" if canonical == "%" else "any"
        return "map", canonical, "1", domain, "reviewed_exact_alias"
    return "map", unit, "1", "any", "reviewed_identity_preservation"


def _resolution_rows(path: Path) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    columns = ("cleaned_record_id", "status", "measurements_json")
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        for row in batch.to_pylist():
            record_id = _text(row["cleaned_record_id"])
            if not record_id or record_id in rows:
                raise ValueError(f"duplicate or empty resolution ID: {record_id!r}")
            rows[record_id] = (_text(row["status"]), _text(row["measurements_json"]))
    return rows


def _required_units(
    cleaned_path: Path,
    resolution_path: Path,
    *,
    allow_out_of_scope_resolution: bool = False,
) -> tuple[dict[tuple[str, str], Counter[str]], dict[str, int]]:
    resolution = _resolution_rows(resolution_path)
    required: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    seen_resolution: set[str] = set()
    counts = Counter()
    for batch in pq.ParquetFile(cleaned_path).iter_batches(columns=_CLEANED_COLUMNS):
        for row in batch.to_pylist():
            record_id = _text(row["cleaned_record_id"])
            endpoint = _text(row["canonical_endpoint_name"])
            source_unit = _text(row["unit_text"])
            if source_unit:
                required[(endpoint, source_unit)]["source"] += 1
                counts["source_occurrences"] += 1
            if _text(row["measurement_resolution_route"]) == "accept":
                unit = _text(row["measurement_resolution_exact_unit"]) or source_unit
                if not unit:
                    raise ValueError(f"deterministic accept has no unit: {record_id}")
                counts["source_exact_occurrences"] += 1
                if not row["measurement_resolution_exact_unit_is_canonical"]:
                    required[(endpoint, unit)]["source_exact"] += 1
            resolved = resolution.get(record_id)
            if (
                _text(row["measurement_resolution_route"]) == "extract"
                and resolved is None
            ):
                raise ValueError(f"extract-routed row has no resolution: {record_id}")
            if resolved is None:
                continue
            seen_resolution.add(record_id)
            if _text(row["measurement_resolution_route"]) != "extract":
                if allow_out_of_scope_resolution:
                    counts["out_of_scope_resolution_rows"] += 1
                    continue
                raise ValueError(f"resolution row is not routed for extraction: {record_id}")
            status, measurements_json = resolved
            if status != "ok":
                continue
            measurements = json.loads(measurements_json)
            if len(measurements) != 1:
                raise ValueError(f"resolved row does not contain one measurement: {record_id}")
            unit = _text(measurements[0].get("unit"))
            if not unit:
                raise ValueError(f"resolved row has no unit: {record_id}")
            required[(endpoint, unit)]["llm"] += 1
            counts["llm_ok_occurrences"] += 1
    extra_resolution = set(resolution) - seen_resolution
    if extra_resolution and not allow_out_of_scope_resolution:
        raise ValueError("measurement-resolution rows are absent from cleaned records")
    counts["out_of_scope_resolution_rows"] += len(extra_resolution)
    counts["endpoint_unit_pairs"] = len(required)
    counts["unique_units"] = len({unit for _, unit in required})
    return required, dict(sorted(counts.items()))


def build_mapping(
    cleaned_path: Path,
    resolution_path: Path,
    *,
    base_mapping_path: Path = BASE_MAPPING,
    allow_out_of_scope_resolution: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = json.loads(base_mapping_path.read_text(encoding="utf-8"))
    if base.get("version") not in {
        "starling_exact_measurement_units.v2",
        MAPPING_VERSION,
    }:
        raise ValueError("unexpected Skin base unit mapping")
    entries = [dict(entry) for entry in base["entries"]]
    required, inventory_counts = _required_units(
        cleaned_path,
        resolution_path,
        allow_out_of_scope_resolution=allow_out_of_scope_resolution,
    )
    covered = {
        (endpoint, _text(entry["input_unit"]))
        for entry in entries
        for endpoint in entry["canonical_endpoints"]
    }
    by_unit: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        by_unit[_text(entry["input_unit"])].append(index)

    additions: dict[
        tuple[str, str, str | None, str | None, str, str], set[str]
    ] = defaultdict(set)
    decision_counts = Counter()
    occurrence_counts = Counter()
    for endpoint, unit in sorted(required):
        if (endpoint, unit) in covered or ("*", unit) in covered:
            continue
        prior = by_unit.get(unit, [])
        prior_decisions = {
            (
                entry["action"],
                entry.get("canonical_unit"),
                entry.get("scale"),
                entry.get("domain", "any"),
            )
            for entry in (entries[index] for index in prior)
        }
        if len(prior_decisions) == 1:
            entries[prior[0]]["canonical_endpoints"] = sorted(
                {*entries[prior[0]]["canonical_endpoints"], endpoint}
            )
            basis = "reused_unambiguous_v2_decision"
            decision_counts[basis] += 1
            occurrence_counts[basis] += sum(required[(endpoint, unit)].values())
            continue
        action, canonical, scale, domain, basis = _decision(unit)
        additions[(unit, action, canonical, scale, domain, basis)].add(endpoint)
        decision_counts[basis] += 1
        occurrence_counts[basis] += sum(required[(endpoint, unit)].values())

    for decision, endpoints in sorted(additions.items()):
        unit, action, canonical, scale, domain, basis = decision
        entry: dict[str, Any] = {
            "task": "skin_reaction",
            "canonical_endpoints": sorted(endpoints),
            "input_unit": unit,
            "action": action,
            "domain": domain,
            "review_basis": basis,
        }
        if action == "map":
            entry.update(canonical_unit=canonical, scale=scale)
        entries.append(entry)
    entries.sort(key=lambda row: (_text(row["input_unit"]), row["canonical_endpoints"]))

    payload = {
        "version": MAPPING_VERSION,
        "entries": entries,
        "skin_v10_reconciliation": {
            "version": "skin_reaction_unit_reconciliation.v1",
            "scope": "all_v10_source_units_noncanonical_exact_units_and_llm_ok_units",
            "inputs": {
                "base_mapping": {
                    "path": str(base_mapping_path),
                    "sha256": file_sha256(base_mapping_path),
                },
                "cleaned_records": {
                    "path": str(cleaned_path),
                    "sha256": file_sha256(cleaned_path),
                },
                "measurement_resolution": {
                    "path": str(resolution_path),
                    "sha256": file_sha256(resolution_path),
                },
            },
            "inventory_counts": inventory_counts,
            "decision_counts": dict(sorted(decision_counts.items())),
            "decision_occurrence_counts": dict(sorted(occurrence_counts.items())),
            "validations": {
                "coefficient_changes_only_for_explicit_denominators": True,
                "complete_required_unit_coverage": True,
                "legacy_decisions_preserved": True,
            },
        },
    }
    return payload, required


def write_mapping(
    cleaned_path: Path,
    resolution_path: Path,
    output: Path,
    *,
    overwrite: bool,
    base_mapping_path: Path = BASE_MAPPING,
    allow_out_of_scope_resolution: bool = False,
) -> None:
    manifest_path = output.with_suffix(".manifest.json")
    if not overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f"refusing to replace Skin unit mapping: {output}")
    payload, required = build_mapping(
        cleaned_path,
        resolution_path,
        base_mapping_path=base_mapping_path,
        allow_out_of_scope_resolution=allow_out_of_scope_resolution,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    manifest_temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        mapping = load_exact_unit_mapping(temporary)
        missing = sorted(
            key
            for key in required
            if ("skin_reaction", key[0], key[1]) not in mapping
            and ("skin_reaction", "*", key[1]) not in mapping
        )
        if missing:
            raise ValueError(f"Skin unit reconciliation remains incomplete: {missing[:5]}")
        manifest = {
            "version": "skin_reaction_unit_reconciliation_manifest.v1",
            "mapping": {"path": str(output), "sha256": file_sha256(temporary)},
            **payload["skin_v10_reconciliation"],
        }
        manifest_temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        os.replace(manifest_temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaned-records", type=Path, default=DEFAULT_CLEANED_RECORDS
    )
    parser.add_argument(
        "--measurement-resolution", type=Path, default=DEFAULT_MAPPING_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-mapping", type=Path, default=BASE_MAPPING)
    parser.add_argument("--allow-out-of-scope-resolution", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    write_mapping(
        args.cleaned_records,
        args.measurement_resolution,
        args.output,
        overwrite=args.overwrite,
        base_mapping_path=args.base_mapping,
        allow_out_of_scope_resolution=args.allow_out_of_scope_resolution,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
