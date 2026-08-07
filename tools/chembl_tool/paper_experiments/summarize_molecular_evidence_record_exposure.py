"""Summarize LLM-visible record exposure from molecular retrieval artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from tools.chembl_tool.common.evidence_contract import evidence_for_llm


CONTRACT_VERSION = "molecular_evidence_record_exposure.v1"
RUNS_DIRECTORY = "runs_identity_blind_parent_disjoint"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    conditions = [_parse_condition(value) for value in args.condition]
    summaries = [summarize_condition(label, root) for label, root in conditions]
    rows = [row for summary in summaries for row in summary["rows"]]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "counting_unit": {
            "molecule_slot": "one selected molecule in one query-family prompt",
            "visible_record": (
                "one representative example entry in minimal_evidence.v1; repeated "
                "serialization in evidence text is not counted again"
            ),
            "underlying_record": (
                "one retained source record summarized by provenance.source_record_count; "
                "not necessarily shown individually"
            ),
        },
        "conditions": summaries,
    }
    (output_dir / "record_exposure_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_tsv(output_dir / "record_exposure.tsv", rows)
    (output_dir / "report.md").write_text(_report(rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "n_conditions": len(summaries),
                "n_rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


def summarize_condition(label: str, root: Path) -> dict[str, Any]:
    run_root = root if root.name == RUNS_DIRECTORY else root / RUNS_DIRECTORY
    if not run_root.is_dir():
        raise SystemExit(f"Missing run root for condition {label}: {run_root}")

    task_accumulators: dict[str, _Accumulator] = {}
    batch_provenance: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        for batch_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            manifest_path = batch_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = _read_json(manifest_path)
            if not _is_target_batch(manifest):
                continue
            task = task_dir.name
            requested_k = int(manifest.get("top_k_per_group") or 0)
            accumulator = task_accumulators.setdefault(
                task,
                _Accumulator(
                    condition=label,
                    task=task,
                    requested_k=requested_k,
                    min_similarity=float(manifest.get("min_similarity") or 0.0),
                    expected_queries=int(manifest.get("n_items") or 0),
                ),
            )
            if accumulator.requested_k != requested_k:
                raise SystemExit(f"Mixed top-k values under {batch_dir}")
            batch_provenance.append(
                {
                    "task": task,
                    "batch_dir": str(batch_dir),
                    "batch_id": manifest.get("batch_id"),
                    "input_jsonl": manifest.get("input_jsonl"),
                    "n_items": manifest.get("n_items"),
                    "top_k_per_group": requested_k,
                    "min_similarity": manifest.get("min_similarity"),
                    "retrieval_strategy": manifest.get("retrieval_strategy"),
                    "neighbor_identity_policy": manifest.get("neighbor_identity_policy"),
                }
            )
            for retrieval_path in sorted((batch_dir / "runs").glob("*/retrieval.json")):
                accumulator.add_retrieval(_read_json(retrieval_path), retrieval_path)

    if not task_accumulators:
        raise SystemExit(f"No Morgan full-mechanism batches found under {run_root}")
    rows: list[dict[str, Any]] = []
    tasks: dict[str, Any] = {}
    for task, accumulator in sorted(task_accumulators.items()):
        task_rows, task_payload = accumulator.finish()
        rows.extend(task_rows)
        tasks[task] = task_payload
    return {
        "condition": label,
        "root": str(root),
        "run_root": str(run_root),
        "batches": batch_provenance,
        "tasks": tasks,
        "rows": rows,
    }


class _Accumulator:
    def __init__(
        self,
        *,
        condition: str,
        task: str,
        requested_k: int,
        min_similarity: float,
        expected_queries: int,
    ) -> None:
        self.condition = condition
        self.task = task
        self.requested_k = requested_k
        self.min_similarity = min_similarity
        self.expected_queries = expected_queries
        self.query_count = 0
        self.failed_retrievals: list[str] = []
        self.families: dict[str, _FamilyAccumulator] = {}
        self.visible_per_query: list[int] = []
        self.unique_visible_per_query: list[int] = []
        self.unique_molecules_per_query: list[int] = []

    def add_retrieval(self, retrieval: dict[str, Any], path: Path) -> None:
        if retrieval.get("status") != "ok":
            self.failed_retrievals.append(str(path))
            return
        experiment = retrieval.get("experiment") or {}
        feature = experiment.get("retrieval_feature") or {}
        if feature.get("feature") != "morgan_fingerprint":
            raise SystemExit(f"Non-Morgan retrieval in {path}")
        coverage = retrieval.get("coverage") or {}
        if int(coverage.get("top_k_per_group") or 0) != self.requested_k:
            raise SystemExit(f"Retrieval top-k does not match batch manifest: {path}")
        if float(coverage.get("min_similarity") or 0.0) != self.min_similarity:
            raise SystemExit(f"Retrieval similarity floor does not match batch manifest: {path}")

        self.query_count += 1
        query_visible = 0
        query_example_ids: set[str] = set()
        query_molecule_ids: set[str] = set()
        for group in retrieval.get("groups") or []:
            family = str(group.get("group_id") or "unknown")
            family_accumulator = self.families.setdefault(
                family,
                _FamilyAccumulator(self.requested_k),
            )
            result = family_accumulator.add_group(group)
            query_visible += result["visible_examples"]
            query_example_ids.update(result["example_ids"])
            query_molecule_ids.update(result["molecule_ids"])
        self.visible_per_query.append(query_visible)
        self.unique_visible_per_query.append(len(query_example_ids))
        self.unique_molecules_per_query.append(len(query_molecule_ids))

    def finish(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        family_rows = [
            accumulator.row(
                condition=self.condition,
                task=self.task,
                family=family,
                requested_k=self.requested_k,
                min_similarity=self.min_similarity,
                expected_queries=self.expected_queries,
                observed_queries=self.query_count,
            )
            for family, accumulator in sorted(self.families.items())
        ]
        overall = _combine_rows(
            family_rows,
            family_accumulators=list(self.families.values()),
            condition=self.condition,
            task=self.task,
            requested_k=self.requested_k,
            min_similarity=self.min_similarity,
            expected_queries=self.expected_queries,
            observed_queries=self.query_count,
            visible_per_query=self.visible_per_query,
            unique_visible_per_query=self.unique_visible_per_query,
            unique_molecules_per_query=self.unique_molecules_per_query,
        )
        rows = [overall, *family_rows]
        payload = {
            "expected_queries": self.expected_queries,
            "queries_with_retrieval": self.query_count,
            "missing_or_failed_retrievals": self.expected_queries - self.query_count,
            "failed_retrieval_paths": self.failed_retrievals,
            "top_k_per_group": self.requested_k,
            "min_similarity": self.min_similarity,
            "overall": overall,
            "families": {row["family"]: row for row in family_rows},
        }
        return rows, payload


class _FamilyAccumulator:
    def __init__(self, requested_k: int) -> None:
        self.requested_k = requested_k
        self.prompt_count = 0
        self.selected_per_prompt: list[int] = []
        self.visible_per_prompt: list[int] = []
        self.unique_visible_per_prompt: list[int] = []
        self.visible_per_slot: list[int] = []
        self.unique_visible_per_slot: list[int] = []
        self.underlying_per_slot: list[int] = []
        self.evidence_rows_per_slot: list[int] = []
        self.evidence_rows_total = 0
        self.evidence_rows_at_cap = 0
        self.short_prompts = 0
        self.duplicate_molecule_slots = 0

    def add_group(self, group: dict[str, Any]) -> dict[str, Any]:
        self.prompt_count += 1
        neighbors = list(group.get("neighbors") or [])
        self.selected_per_prompt.append(len(neighbors))
        if len(neighbors) < self.requested_k:
            self.short_prompts += 1
        molecule_ids = [str(row.get("molecule_chembl_id") or "") for row in neighbors]
        self.duplicate_molecule_slots += len(molecule_ids) - len(set(molecule_ids))

        prompt_visible = 0
        prompt_example_ids: set[str] = set()
        for neighbor in neighbors:
            rows = list(neighbor.get("evidence_rows") or [])
            visible = 0
            underlying = 0
            slot_example_ids: set[str] = set()
            for row in rows:
                llm_evidence = evidence_for_llm(row)
                examples = list(llm_evidence.get("examples") or [])
                visible += len(examples)
                self.evidence_rows_total += 1
                if len(examples) == 6:
                    self.evidence_rows_at_cap += 1
                provenance = llm_evidence.get("provenance") or {}
                underlying += int(provenance.get("source_record_count") or 0)
                for example in examples:
                    slot_example_ids.add(_example_identity(example))
            self.visible_per_slot.append(visible)
            self.unique_visible_per_slot.append(len(slot_example_ids))
            self.underlying_per_slot.append(underlying)
            self.evidence_rows_per_slot.append(len(rows))
            prompt_visible += visible
            prompt_example_ids.update(slot_example_ids)
        self.visible_per_prompt.append(prompt_visible)
        self.unique_visible_per_prompt.append(len(prompt_example_ids))
        return {
            "visible_examples": prompt_visible,
            "example_ids": prompt_example_ids,
            "molecule_ids": set(molecule_ids),
        }

    def row(
        self,
        *,
        condition: str,
        task: str,
        family: str,
        requested_k: int,
        min_similarity: float,
        expected_queries: int,
        observed_queries: int,
    ) -> dict[str, Any]:
        return {
            "condition": condition,
            "task": task,
            "family": family,
            "top_k_per_group": requested_k,
            "min_similarity": min_similarity,
            "expected_queries": expected_queries,
            "queries_with_retrieval": observed_queries,
            "family_prompts": self.prompt_count,
            "selected_molecule_slots": len(self.visible_per_slot),
            "short_family_prompts": self.short_prompts,
            "duplicate_molecule_slots_within_family": self.duplicate_molecule_slots,
            "selected_molecules_per_prompt": _stats(self.selected_per_prompt),
            "visible_records_per_molecule_slot": _stats(self.visible_per_slot),
            "unique_visible_records_per_molecule_slot": _stats(self.unique_visible_per_slot),
            "visible_records_per_family_prompt": _stats(self.visible_per_prompt),
            "unique_visible_records_per_family_prompt": _stats(self.unique_visible_per_prompt),
            "underlying_records_per_molecule_slot": _stats(self.underlying_per_slot),
            "evidence_rows_per_molecule_slot": _stats(self.evidence_rows_per_slot),
            "evidence_rows": self.evidence_rows_total,
            "fraction_evidence_rows_at_six_record_cap": _fraction(
                self.evidence_rows_at_cap,
                self.evidence_rows_total,
            ),
        }


def _combine_rows(
    family_rows: list[dict[str, Any]],
    *,
    family_accumulators: list[_FamilyAccumulator],
    condition: str,
    task: str,
    requested_k: int,
    min_similarity: float,
    expected_queries: int,
    observed_queries: int,
    visible_per_query: list[int],
    unique_visible_per_query: list[int],
    unique_molecules_per_query: list[int],
) -> dict[str, Any]:
    def pooled(field: str) -> list[int]:
        return [
            value
            for accumulator in family_accumulators
            for value in getattr(accumulator, field)
        ]

    family_prompts = sum(int(row["family_prompts"]) for row in family_rows)
    slots = sum(int(row["selected_molecule_slots"]) for row in family_rows)
    evidence_rows = sum(int(row["evidence_rows"]) for row in family_rows)
    cap_rows = sum(accumulator.evidence_rows_at_cap for accumulator in family_accumulators)
    return {
        "condition": condition,
        "task": task,
        "family": "__all__",
        "top_k_per_group": requested_k,
        "min_similarity": min_similarity,
        "expected_queries": expected_queries,
        "queries_with_retrieval": observed_queries,
        "family_prompts": family_prompts,
        "selected_molecule_slots": slots,
        "short_family_prompts": sum(int(row["short_family_prompts"]) for row in family_rows),
        "duplicate_molecule_slots_within_family": sum(
            int(row["duplicate_molecule_slots_within_family"]) for row in family_rows
        ),
        "selected_molecules_per_prompt": _stats(pooled("selected_per_prompt")),
        "visible_records_per_molecule_slot": _stats(pooled("visible_per_slot")),
        "unique_visible_records_per_molecule_slot": _stats(
            pooled("unique_visible_per_slot")
        ),
        "visible_records_per_family_prompt": _stats(pooled("visible_per_prompt")),
        "unique_visible_records_per_family_prompt": _stats(
            pooled("unique_visible_per_prompt")
        ),
        "visible_records_per_query": _stats(visible_per_query),
        "unique_visible_records_per_query": _stats(unique_visible_per_query),
        "unique_molecules_per_query": _stats(unique_molecules_per_query),
        "underlying_records_per_molecule_slot": _stats(pooled("underlying_per_slot")),
        "evidence_rows_per_molecule_slot": _stats(pooled("evidence_rows_per_slot")),
        "evidence_rows": evidence_rows,
        "fraction_evidence_rows_at_six_record_cap": _fraction(cap_rows, evidence_rows),
    }


def _stats(values: Iterable[int | float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _example_identity(example: dict[str, Any]) -> str:
    fields = (
        example.get("source_id"),
        example.get("source_index"),
        example.get("source_record_id"),
    )
    if any(value not in (None, "") for value in fields):
        return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(example, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_target_batch(manifest: dict[str, Any]) -> bool:
    return (
        manifest.get("experiment_mode") == "full_mechanism"
        and str(manifest.get("retrieval_source") or "").startswith("starling")
        and manifest.get("retrieval_strategy") == "morgan_fingerprint"
        and manifest.get("neighbor_identity_policy") == "parent_disjoint"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = [_flatten_row(row) for row in rows]
    fieldnames = list(flattened[0]) if flattened else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(flattened)


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for statistic, statistic_value in value.items():
                output[f"{key}_{statistic}"] = statistic_value
        else:
            output[key] = value
    return output


def _report(rows: list[dict[str, Any]]) -> str:
    overall = [row for row in rows if row["family"] == "__all__"]
    lines = [
        "# Molecular evidence record exposure",
        "",
        "Visible records are representative `minimal_evidence.v1.examples` entries. "
        "Underlying records are summarized source rows and are not necessarily shown individually.",
        "",
        "| condition | task | k | queries | molecule slots | short prompts | duplicate slots | visible records / molecule | visible records / query | underlying records / molecule |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['condition']} | {row['task']} | {row['top_k_per_group']} | "
            f"{row['queries_with_retrieval']}/{row['expected_queries']} | "
            f"{row['selected_molecule_slots']} | {row['short_family_prompts']} | "
            f"{row['duplicate_molecule_slots_within_family']} | "
            f"{float(row['visible_records_per_molecule_slot']['mean']):.3f} | "
            f"{float(row['visible_records_per_query']['mean']):.3f} | "
            f"{float(row['underlying_records_per_molecule_slot']['mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism-family detail",
            "",
            "| condition | task | family | k | prompts | visible records / molecule | visible records / prompt | max visible / molecule | rows at six-record cap |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["family"] == "__all__":
            continue
        lines.append(
            f"| {row['condition']} | {row['task']} | {row['family']} | "
            f"{row['top_k_per_group']} | {row['family_prompts']} | "
            f"{float(row['visible_records_per_molecule_slot']['mean']):.3f} | "
            f"{float(row['visible_records_per_family_prompt']['mean']):.3f} | "
            f"{row['visible_records_per_molecule_slot']['max']} | "
            f"{float(row['fraction_evidence_rows_at_six_record_cap']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _parse_condition(value: str) -> tuple[str, Path]:
    label, separator, root = value.partition("=")
    if not separator or not label.strip() or not root.strip():
        raise argparse.ArgumentTypeError("--condition must be LABEL=ROOT")
    return label.strip(), Path(root.strip())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="Repeatable LABEL=ROOT condition; ROOT contains runs_identity_blind_parent_disjoint.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
