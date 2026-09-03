"""Observed source-unit vocabulary used by strict V8 scalar routing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data.processing.paths import REPO_ROOT, raw_starling_task_root
from data.processing.evidence_library.shared.v2.normalization.cleaning import (
    file_sha256,
)


VOCABULARY_VERSION = "observed_source_units.v1"
DEFAULT_VOCABULARY_PATH = Path(__file__).with_name("unit_vocabulary.v1.json")
TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")


def clean_unit(value: Any) -> str:
    """Apply only the whitespace cleaning allowed before unit reconciliation."""
    return " ".join(str(value or "").split())


def load_unit_vocabulary(
    path: str | Path = DEFAULT_VOCABULARY_PATH,
) -> frozenset[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != VOCABULARY_VERSION:
        raise ValueError(f"unsupported unit vocabulary: {path}")
    units = [str(entry.get("unit") or "") for entry in payload.get("units", [])]
    if not units or any(not unit for unit in units) or units != sorted(set(units)):
        raise ValueError(f"invalid unit vocabulary: {path}")
    return frozenset(units)


def build_unit_vocabulary() -> dict[str, Any]:
    """Collect every nonempty value from every declared source unit column."""
    counts: Counter[str] = Counter()
    sources_by_unit: dict[str, set[str]] = defaultdict(set)
    sources: list[dict[str, Any]] = []
    for task in TASKS:
        module = import_module(
            f"data.processing.evidence_library.versions.v9.tasks.{task}."
            "starling_normalization_sources"
        )
        for profile in module.source_profiles(raw_starling_task_root(task)):
            if not profile.unit_field:
                continue
            path = Path(profile.source_path)
            column = profile.unit_field
            values = pq.read_table(path, columns=[column]).column(column).to_pylist()
            source_counts = Counter(clean_unit(value) for value in values)
            source_counts.pop("", None)
            source_key = f"{task}:{profile.source_id}"
            counts.update(source_counts)
            for unit in source_counts:
                sources_by_unit[unit].add(source_key)
            try:
                stored_path = str(path.resolve().relative_to(REPO_ROOT.resolve()))
            except ValueError:
                stored_path = str(path)
            sources.append(
                {
                    "task": task,
                    "source_id": profile.source_id,
                    "path": stored_path,
                    "unit_column": column,
                    "sha256": file_sha256(path),
                    "rows": len(values),
                    "nonempty_unit_rows": sum(source_counts.values()),
                    "unique_units": len(source_counts),
                }
            )
    return {
        "version": VOCABULARY_VERSION,
        "normalization": "strip_and_collapse_whitespace_case_sensitive",
        "sources": sources,
        "units": [
            {
                "unit": unit,
                "count": counts[unit],
                "sources": sorted(sources_by_unit[unit]),
            }
            for unit in sorted(counts)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_VOCABULARY_PATH)
    args = parser.parse_args(argv)
    payload = build_unit_vocabulary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(payload['units']):,} units to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_VOCABULARY_PATH",
    "VOCABULARY_VERSION",
    "build_unit_vocabulary",
    "clean_unit",
    "load_unit_vocabulary",
]
