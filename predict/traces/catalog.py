"""Discover centralized prediction traces for the viewer."""

from __future__ import annotations

import csv
import fcntl
import os
import json
from pathlib import Path
from typing import Any

from predict.traces.io import DEFAULT_TRACE_ROOT, TRACE_SCHEMA_VERSION


def discover_traces(root: str | Path = DEFAULT_TRACE_ROOT) -> list[dict[str, Any]]:
    rows = []
    paths = {
        path for path in Path(root).rglob("*.json")
        if "public" not in path.parts and path.parent.parent.name == "samples"
    }
    for path in sorted(paths):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("schema_version") == TRACE_SCHEMA_VERSION:
            rows.append({**row, "path": str(path)})
    return rows


from predict.utils.json import sha256_file, write_json_atomic

RESULTS_LEDGER_FIELDS = (
    "study", "method", "query_prior", "run_id", "status", "support_mode",
    "prompt_version", "harness_version", "reranking",
    "morgan_primary_parent_width", "l1_molecules", "l1_min_contrast",
    "molecule_description_mode", "indirect_level", "task", "evaluation_subset", "level",
    "n_total", "n_successful", "n_failed_runs", "macro_f1", "accuracy",
    "path", "run_json_sha256", "metrics_path", "metrics_sha256",
    "diagnostics_manifest_sha256", "finished_at",
)


def _refresh_results_catalog(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".results_catalog.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows, ledger = [], []
        for path in sorted(root.glob("*/*/*/*/run.json")):
            document = json.loads(path.read_text())
            if not str(document.get("schema_version", "")).startswith(
                ("progressive_study_run.v", "organized_study_run.v")
            ):
                continue
            run_path = path.parent.relative_to(root).as_posix()
            diagnostics_path = path.parent / "diagnostics_manifest.json"
            rows.append({
                key: document.get(key)
                for key in (
                    "study", "method", "run_id", "batch_id", "status",
                    "prompt_version", "harness_version", "reranking",
                    "query_prior", "l1_molecules", "l1_min_contrast",
                    "morgan_primary_parent_width", "molecule_description_mode",
                    "indirect_level", "support_mode", "metric_status", "tasks",
                    "evaluation_subset",
                )
            } | {"path": run_path})
            base = {
                key: document.get(key)
                for key in RESULTS_LEDGER_FIELDS
                if key not in {
                    "task", "level", "n_total", "n_successful", "n_failed_runs",
                    "macro_f1", "accuracy", "path", "run_json_sha256",
                    "metrics_path", "metrics_sha256",
                    "diagnostics_manifest_sha256",
                }
            } | {
                "path": run_path,
                "run_json_sha256": sha256_file(path),
                "diagnostics_manifest_sha256": (
                    sha256_file(diagnostics_path)
                    if diagnostics_path.is_file()
                    else document.get("diagnostics_manifest_sha256")
                ),
            }
            for task in document.get("tasks") or [None]:
                metric_root = Path(document.get("metric_source_dir") or path.parent)
                metric_paths = sorted(
                    metric_root.glob(f"{task}/levels/level_*/metrics.json")
                ) if task else []
                if task and not metric_paths:
                    metric_paths = sorted(metric_root.glob(f"{task}/*/metrics.json"))
                if not metric_paths:
                    ledger.append(base | {"task": task})
                    continue
                for metrics_path in metric_paths:
                    metrics = json.loads(metrics_path.read_text())
                    ledger.append(base | {
                        "task": task,
                        "level": metrics.get("level"),
                        "n_total": metrics.get("n_total"),
                        "n_successful": metrics.get("n_successful"),
                        "n_failed_runs": metrics.get("n_failed_runs"),
                        "macro_f1": metrics.get("macro_f1"),
                        "accuracy": metrics.get("accuracy"),
                        "metrics_path": (
                            metrics_path.relative_to(path.parent).as_posix()
                            if metric_root == path.parent else str(metrics_path)
                        ),
                        "metrics_sha256": sha256_file(metrics_path),
                    })
        ledger_path = root / "results_ledger.tsv"
        candidate = root / f".results_ledger.tsv.tmp-{os.getpid()}"
        with candidate.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, RESULTS_LEDGER_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(ledger)
        os.replace(candidate, ledger_path)
        write_json_atomic(root / "catalog.json", {
            "schema_version": "progressive_results_catalog.v3",
            "legacy_included": False,
            "results_ledger": {
                "path": ledger_path.name,
                "rows": len(ledger),
                "sha256": sha256_file(ledger_path),
            },
            "runs": rows,
        })
