"""Audit descriptive Starling evidence metadata coverage."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.evidence_compatibility import (
    COMPATIBILITY_INPUT_VERSION,
    compatibility_inputs_from_row,
)
from tools.chembl_tool.common.json_utils import sha256_file, write_json_atomic


DEFAULT_EVIDENCE_ROOT = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2/evidence"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/paper/final_evidence_surface_record_supported_v2_valid_gpt_oss_120b/metadata_census"
)
SOURCES = {
    "bbb_martins": Path("bbb_starling_full/starling_bbb_evidence.jsonl"),
    "bioavailability_ma": Path(
        "bioavailability_starling_full/starling_factor_evidence.jsonl"
    ),
    "skin_reaction": Path(
        "skin_reaction_starling_full/starling_skin_reaction_evidence.jsonl"
    ),
}
READINESS_FIELDS = (
    "endpoint_match",
    "context_compatibility",
    "measurement_available",
    "measurement_consistency",
    "record_agreement",
    "evidence_scope",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence_root = Path(args.evidence_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(dict.fromkeys(args.tasks))
    summaries = {
        task: audit_evidence_file(evidence_root / SOURCES[task], task=task)
        for task in tasks
    }
    result = {
        "schema_version": "starling.evidence_metadata_census.v1",
        "compatibility_input_version": COMPATIBILITY_INPUT_VERSION,
        "dataset_lineage": "record_supported_v2",
        "evidence_root": str(evidence_root),
        "tasks": summaries,
    }
    write_json_atomic(output_dir / "coverage.json", result)
    _write_field_tsv(output_dir / "field_coverage.tsv", summaries)
    _write_group_tsv(output_dir / "group_coverage.tsv", summaries)
    _write_report(output_dir / "report.md", summaries)
    print(json.dumps({"output_dir": str(output_dir), "tasks": tasks}, indent=2))
    return 0


def audit_evidence_file(path: Path, *, task: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    readiness = Counter()
    roles = Counter()
    scope_keys = Counter()
    endpoint_names = Counter()
    measurement_keys = Counter()
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    n_rows = 0
    n_invalid = 0
    n_multi_record = 0
    n_independent_pmid = 0
    total_source_records = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n_rows += 1
            try:
                row = json.loads(line)
                inputs = compatibility_inputs_from_row(row)
            except (json.JSONDecodeError, TypeError, ValueError):
                n_invalid += 1
                continue
            group_id = str(inputs["endpoint"].get("group_id") or "unknown")
            groups[group_id]["n_rows"] += 1
            for field in READINESS_FIELDS:
                if inputs["readiness"][field]:
                    readiness[field] += 1
                    groups[group_id][field] += 1
            role = str(inputs.get("role") or "unspecified")
            roles[role] += 1
            endpoint_name = str(inputs["endpoint"].get("name") or "")
            if endpoint_name:
                endpoint_names[endpoint_name] += 1
            measurement_key = str(
                inputs["measurement"].get("raw_comparability_key") or ""
            )
            if measurement_key:
                measurement_keys[measurement_key] += 1
            for example in inputs["measurement"].get("example_measurements") or []:
                example_endpoint = str(example.get("endpoint") or endpoint_name)
                example_unit = str(example.get("unit") or "")
                if example_endpoint:
                    measurement_keys[
                        f"{example_endpoint}|{example_unit or '<missing-unit>'}"
                    ] += 1
            for key in inputs["context"].get("scope_keys") or []:
                scope_keys[str(key)] += 1
            support = inputs["support"]
            source_count = int(support.get("source_record_count") or 0)
            total_source_records += source_count
            n_multi_record += int(bool(support.get("has_multi_record_support")))
            n_independent_pmid += int(
                bool(support.get("has_independent_pmid_support"))
            )
    return {
        "task": task,
        "path": str(path),
        "sha256": sha256_file(path),
        "n_rows": n_rows,
        "n_invalid_rows": n_invalid,
        "readiness": {
            field: {
                "n": readiness[field],
                "fraction": _fraction(readiness[field], n_rows),
            }
            for field in READINESS_FIELDS
        },
        "support": {
            "n_multi_record_rows": n_multi_record,
            "multi_record_fraction": _fraction(n_multi_record, n_rows),
            "n_multi_pmid_rows": n_independent_pmid,
            "multi_pmid_fraction": _fraction(n_independent_pmid, n_rows),
            "total_source_record_count": total_source_records,
        },
        "roles": dict(roles.most_common()),
        "scope_keys": dict(scope_keys.most_common()),
        "top_endpoint_names": dict(endpoint_names.most_common(30)),
        "top_measurement_keys": dict(measurement_keys.most_common(30)),
        "groups": {
            group_id: dict(counter)
            for group_id, counter in sorted(groups.items())
        },
    }


def _write_field_tsv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task", "field", "n_available", "n_rows", "fraction"),
            delimiter="\t",
        )
        writer.writeheader()
        for task, summary in summaries.items():
            for field in READINESS_FIELDS:
                row = summary["readiness"][field]
                writer.writerow(
                    {
                        "task": task,
                        "field": field,
                        "n_available": row["n"],
                        "n_rows": summary["n_rows"],
                        "fraction": row["fraction"],
                    }
                )


def _write_group_tsv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    fields = ("task", "group_id", "n_rows", *READINESS_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for task, summary in summaries.items():
            for group_id, counts in summary["groups"].items():
                writer.writerow(
                    {
                        "task": task,
                        "group_id": group_id,
                        **{field: counts.get(field, 0) for field in fields[2:]},
                    }
                )


def _write_report(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Starling evidence metadata coverage 审计",
        "",
        "该报告只检查描述性 metadata 字段是否存在，不计算 label、vote、threshold、transfer score 或 evidence utility。",
        "",
        "| task | evidence molecules | endpoint | context | measurement available | consistency assessable | agreement assessable | scope |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, summary in summaries.items():
        fractions = {
            field: summary["readiness"][field]["fraction"]
            for field in READINESS_FIELDS
        }
        lines.append(
            f"| {task} | {summary['n_rows']:,} | "
            f"{fractions['endpoint_match']:.1%} | "
            f"{fractions['context_compatibility']:.1%} | "
            f"{fractions['measurement_available']:.1%} | "
            f"{fractions['measurement_consistency']:.1%} | "
            f"{fractions['record_agreement']:.1%} | "
            f"{fractions['evidence_scope']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `endpoint` 表示存在 endpoint 描述，不代表它已与 task ontology 匹配。",
            "- `context` 表示存在 scope 或 context text，不代表 species/system 已完成标准化。",
            "- `measurement available` 同时检查聚合 measurement 和 example-level reported values。",
            "- `consistency assessable` 要求至少两个 reported/numeric measurements；跨单位合并仍需 task adapter。",
            "- `agreement assessable` 要求至少两个 source records 或独立 PMID；不表示这些记录已经一致。",
            "- 这些字段只能进入内部 sidecar/audit；不得作为 deterministic label 或隐藏 vote 注入 LLM。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    parser.add_argument("--evidence-root", default=str(DEFAULT_EVIDENCE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
