"""Artifact-gated progress reporting for long router OOF agent runs."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Iterable

from .contract import TaskSpec, fold_root


def summarize_agent_progress(
    specs: Iterable[TaskSpec],
    *,
    output_root: str | Path,
    folds: dict[str, list[int]],
    recent_window_minutes: float = 10.0,
) -> dict[str, Any]:
    now = time.time()
    recent_cutoff = now - recent_window_minutes * 60
    task_rows = []
    total_expected = 0
    total_none = 0
    total_direct = 0
    total_paired = 0
    recent_final_count = 0
    for spec in specs:
        task_expected = 0
        task_none = 0
        task_direct = 0
        task_paired = 0
        fold_rows = []
        for fold in folds[spec.task]:
            current_root = fold_root(output_root, spec.task, fold)
            expected = _line_count(current_root / "agent_input" / "valid.jsonl")
            none_root = _condition_root(current_root, spec.task, f"{spec.task}__none")
            direct_root = _condition_root(current_root, spec.task, spec.direct_condition)
            none_complete = _complete_indices(none_root, f"{spec.task}__none", expected)
            direct_complete = _complete_indices(direct_root, spec.direct_condition, expected)
            paired = none_complete & direct_complete
            for root, condition in (
                (none_root, f"{spec.task}__none"),
                (direct_root, spec.direct_condition),
            ):
                for index in range(expected):
                    final_path = (
                        root
                        / "runs"
                        / f"{condition}_idx{index:05d}"
                        / "final_reasoning_output.json"
                    )
                    if final_path.exists() and final_path.stat().st_mtime >= recent_cutoff:
                        recent_final_count += 1
            fold_rows.append(
                {
                    "fold": fold,
                    "expected": expected,
                    "none_complete": len(none_complete),
                    "direct_complete": len(direct_complete),
                    "paired_complete": len(paired),
                }
            )
            task_expected += expected
            task_none += len(none_complete)
            task_direct += len(direct_complete)
            task_paired += len(paired)
        task_rows.append(
            {
                "task": spec.task,
                "expected": task_expected,
                "none_complete": task_none,
                "direct_complete": task_direct,
                "paired_complete": task_paired,
                "paired_fraction": task_paired / task_expected if task_expected else 0.0,
                "folds": fold_rows,
            }
        )
        total_expected += task_expected
        total_none += task_none
        total_direct += task_direct
        total_paired += task_paired
    return {
        "schema_version": "router_oof_agent_progress.v1",
        "expected_samples": total_expected,
        "none_complete": total_none,
        "direct_complete": total_direct,
        "paired_complete": total_paired,
        "paired_fraction": total_paired / total_expected if total_expected else 0.0,
        "recent_window_minutes": recent_window_minutes,
        "recent_final_artifacts": recent_final_count,
        "recent_final_artifacts_per_minute": recent_final_count / recent_window_minutes,
        "tasks": task_rows,
    }


def _condition_root(current_root: Path, task: str, condition: str) -> Path:
    return (
        current_root
        / "agent"
        / "runs_identity_blind_parent_disjoint"
        / task
        / condition
    )


def _complete_indices(batch_root: Path, condition: str, expected: int) -> set[int]:
    complete = set()
    for index in range(expected):
        run_dir = batch_root / "runs" / f"{condition}_idx{index:05d}"
        if _run_complete(run_dir):
            complete.add(index)
    return complete


def _run_complete(run_dir: Path) -> bool:
    manifest = _read_json(run_dir / "manifest.json")
    single = _read_json(run_dir / "single_molecule_reasoning_output.json")
    final = _read_json(run_dir / "final_reasoning_output.json")
    if not manifest or not single or not final:
        return False
    if single.get("status") != "ok" or final.get("status") != "ok":
        return False
    expected_groups = list(manifest.get("expected_group_ids") or [])
    group_path = run_dir / "group_reasoning_outputs.jsonl"
    if not group_path.exists():
        return not expected_groups
    groups = [
        json.loads(line)
        for line in group_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    group_ids = [str(row.get("group_id") or "") for row in groups]
    return (
        len(group_ids) == len(expected_groups)
        and sorted(group_ids) == sorted(expected_groups)
        and all(row.get("status") == "ok" for row in groups)
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)
