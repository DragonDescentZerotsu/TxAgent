"""Audit per-query tool-evidence parity across the controlled visibility pair."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.chembl_tool.common.identity_blind import (
    _replace_identity_terms,
    _retrieval_sensitive_terms,
)

from .molecular_evidence_agent import experiments_for_split, paper_root_for_split


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paper_root = Path(args.paper_root) if args.paper_root else paper_root_for_split(args.split)
    experiments = experiments_for_split(args.split)
    output_dir = Path(args.output_dir) if args.output_dir else paper_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        blind_batch = paper_root / "runs" / experiment.task / experiment.name
        visible_batch = (
            paper_root
            / "runs_deployment_visible_prefetched"
            / experiment.task
            / experiment.name
        )
        blind_runs = _prediction_runs(blind_batch)
        visible_runs = _prediction_runs(visible_batch)
        mismatched_indices: list[int] = []
        for query_index in sorted(set(blind_runs) & set(visible_runs)):
            blind_contract = prefetch_contract(blind_runs[query_index])
            visible_contract = prefetch_contract(visible_runs[query_index])
            contract_match = blind_contract == visible_contract
            if not contract_match:
                mismatched_indices.append(query_index)
            rows.append(
                {
                    "experiment": experiment.name,
                    "task": experiment.task,
                    "query_index": query_index,
                    "blind_hash": _hash(blind_contract),
                    "visible_prefetched_hash": _hash(visible_contract),
                    "contract_match": contract_match,
                    "blind_tool_results": _count_results(blind_contract),
                    "visible_prefetched_tool_results": _count_results(visible_contract),
                }
            )
        condition_rows.append(
            _condition_coverage(
                experiment.name,
                experiment.task,
                _jsonl_indices(Path(experiment.input_jsonl)),
                set(blind_runs),
                set(visible_runs),
                set(mismatched_indices),
            )
        )

    _write_tsv(output_dir / "prefetch_contract_audit.tsv", rows)
    _write_tsv(output_dir / "prefetch_contract_conditions.tsv", condition_rows)
    summary = {
        "data_split": args.split,
        "n_conditions_expected": len(experiments),
        "n_conditions_complete": sum(row["complete"] for row in condition_rows),
        "n_expected": sum(row["n_expected"] for row in condition_rows),
        "n_audited": len(rows),
        "n_matched": sum(row["contract_match"] for row in rows),
        "n_mismatched": sum(not row["contract_match"] for row in rows),
        "n_missing_blind": sum(row["n_missing_blind"] for row in condition_rows),
        "n_missing_visible_prefetched": sum(
            row["n_missing_visible_prefetched"] for row in condition_rows
        ),
        "n_extra_blind": sum(row["n_extra_blind"] for row in condition_rows),
        "n_extra_visible_prefetched": sum(
            row["n_extra_visible_prefetched"] for row in condition_rows
        ),
        "complete": bool(condition_rows) and all(row["complete"] for row in condition_rows),
    }
    (output_dir / "prefetch_contract_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return int(not summary["complete"])


def _condition_coverage(
    experiment: str,
    task: str,
    expected: set[int],
    blind: set[int],
    visible_prefetched: set[int],
    mismatched: set[int],
) -> dict[str, Any]:
    missing_blind = expected - blind
    missing_visible = expected - visible_prefetched
    extra_blind = blind - expected
    extra_visible = visible_prefetched - expected
    return {
        "experiment": experiment,
        "task": task,
        "n_expected": len(expected),
        "n_blind": len(blind),
        "n_visible_prefetched": len(visible_prefetched),
        "n_audited": len(blind & visible_prefetched),
        "n_mismatched": len(mismatched),
        "n_missing_blind": len(missing_blind),
        "n_missing_visible_prefetched": len(missing_visible),
        "n_extra_blind": len(extra_blind),
        "n_extra_visible_prefetched": len(extra_visible),
        "missing_blind_indices": _format_indices(missing_blind),
        "missing_visible_prefetched_indices": _format_indices(missing_visible),
        "mismatched_indices": _format_indices(mismatched),
        "complete": not (
            missing_blind
            or missing_visible
            or extra_blind
            or extra_visible
            or mismatched
        ),
    }


def prefetch_contract(run_dir: Path) -> dict[str, Any]:
    single = json.loads((run_dir / "single_molecule_reasoning_output.json").read_text(encoding="utf-8"))
    retrieval_path = run_dir / "retrieval.json"
    contract: dict[str, Any] = {
        "retrieval": json.loads(retrieval_path.read_text(encoding="utf-8")) if retrieval_path.exists() else None,
        "query": ((single.get("llm") or {}).get("tool_results") or []),
        "groups": [],
    }
    raw_groups = run_dir / "group_reasoning_outputs_raw.jsonl"
    group_path = raw_groups if raw_groups.exists() else run_dir / "group_reasoning_outputs.jsonl"
    for branch in _read_jsonl(group_path):
        payload = _initial_user_payload((branch.get("llm") or {}).get("messages") or [])
        group = payload.get("group") or {}
        contract["groups"].append(
            {
                "group_id": group.get("group_id") or branch.get("group_id"),
                "neighbors": [
                    {
                        "rank": neighbor.get("rank"),
                        "prefetched_comparisons": neighbor.get("prefetched_comparisons") or [],
                    }
                    for neighbor in payload.get("neighbors") or []
                ],
            }
        )
    contract["groups"].sort(key=lambda item: str(item["group_id"]))
    identity_terms = _retrieval_sensitive_terms(contract["retrieval"] or {})
    contract["query"] = _replace_identity_terms(contract["query"], identity_terms)
    contract["groups"] = _replace_identity_terms(contract["groups"], identity_terms)
    return contract


def _initial_user_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), str):
            continue
        content = message["content"]
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            object_start = content.find("{")
            if object_start < 0:
                continue
            try:
                payload, _ = json.JSONDecoder().raw_decode(content[object_start:])
            except json.JSONDecodeError:
                continue
        if isinstance(payload, dict) and "neighbors" in payload:
            return payload
    return {}


def _prediction_runs(batch_dir: Path) -> dict[int, Path]:
    predictions = batch_dir / "predictions.jsonl"
    if not predictions.exists():
        return {}
    return {
        int(row["query_index"]): Path(row["run_dir"])
        for row in _read_jsonl(predictions)
    }


def _jsonl_indices(path: Path) -> set[int]:
    return {
        index
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines())
        if line.strip()
    }


def _format_indices(indices: set[int]) -> str:
    return ",".join(str(index) for index in sorted(indices))


def _count_results(contract: dict[str, Any]) -> int:
    return len(contract["query"]) + sum(
        len(neighbor["prefetched_comparisons"])
        for group in contract["groups"]
        for neighbor in group["neighbors"]
    )


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument("--paper-root", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
