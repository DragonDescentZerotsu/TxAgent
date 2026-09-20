"""Publish completed top-75 semantic weights without changing historical exports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

import pandas as pd

from semantic_buckets import publish_frozen_weights as frozen
from semantic_buckets import sliding_weight_assignment as weights_io
from semantic_buckets.publication import sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic


ROOT = Path(__file__).resolve().parent
EXPORT_ID = "completed_top75_20260919_v1"
EXPORT_ROOT = ROOT / "provenance/semantic_weight_exports" / EXPORT_ID
V6_ROOT = ROOT / "provenance/semantic_weight_calibration_v6"
RELEASE_ROOT = ROOT / "releases"
TASKS = ("bbb_martins", "bioavailability_ma")
FINAL_RUNS = (
    "v5_completed_levels_top75_final_20260917",
    "v5_bbb_l4_top75_final_20260917",
    "v5_remaining_levels_top75_final_20260917",
)
EXPECTED_FINAL_BUCKETS = 15_032
EXPECTED_FINAL_REQUESTS = 2_145


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_specs() -> list[tuple[str, str, int, Path]]:
    old = V6_ROOT
    return [
        (frozen.V5_RUN, "base", 0, frozen.V5_ROOT / frozen.V5_RUN / "weights.partial.parquet"),
        ("v5_completed_levels_epoch1_local_dgx008_20260917", "v6_top50_epoch1", 1,
         old / "v5_completed_levels_epoch1_local_dgx008_20260917/weights.partial.parquet"),
        ("v5_completed_levels_epoch1_repeat2_20260917", "v6_top50_epoch2", 2,
         old / "v5_completed_levels_epoch1_repeat2_20260917/weights.partial.parquet"),
        ("v5_bbb_l4_epoch2_local_dgx008_20260917", "v6_top50_epoch2", 2,
         old / "v5_bbb_l4_epoch2_local_dgx008_20260917/weights.partial.parquet"),
        *[(run, "v6_top75_complete", 3, old / run / "weights.partial.parquet")
          for run in FINAL_RUNS],
    ]


def _run_receipt(run_id: str) -> dict[str, Any]:
    root = V6_ROOT / run_id
    manifest_path, database = root / "manifest.json", root / "requests.sqlite3"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"complete_unpublished", "complete_candidate"}:
        raise ValueError(f"top-75 run is incomplete: {run_id}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        statuses = dict(connection.execute("SELECT status,count(*) FROM requests GROUP BY status"))
    finally:
        connection.close()
    weights = pd.read_parquet(root / "weights.partial.parquet")
    expected = int(manifest["direct_v6_bucket_count"])
    if quick_check != "ok" or statuses != {"complete": manifest["request_count"]} or (
        len(weights) != expected or weights.semantic_bucket_id.duplicated().any()
    ):
        raise ValueError(f"invalid completed top-75 run: {run_id}")
    names = ("manifest.json", "schedule.parquet", "weights.partial.parquet", "requests.sqlite3")
    return {
        "run_id": run_id, "status": manifest["status"], "request_status_counts": statuses,
        "bucket_count": len(weights), "quick_check": quick_check,
        "files": {name: sha256_file(root / name) for name in names},
        "paths": {name: str(root / name) for name in names},
    }


def _observations() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    receipts = [_run_receipt(run) for run in FINAL_RUNS]
    if sum(row["bucket_count"] for row in receipts) != EXPECTED_FINAL_BUCKETS or sum(
        row["request_status_counts"]["complete"] for row in receipts
    ) != EXPECTED_FINAL_REQUESTS:
        raise ValueError("completed top-75 source totals changed")
    policy = json.loads(frozen.POLICY.read_text(encoding="utf-8"))
    frames = [frozen._observation_frame(*spec) for spec in _source_specs()]
    frames.append(frozen._manual_frame(policy))
    observations = pd.concat(frames, ignore_index=True, sort=False)
    if observations.duplicated(["source_run_id", "semantic_bucket_id"]).any():
        raise ValueError("one source run repeats a semantic bucket")
    return observations.sort_values(["source_priority", "source_run_id"]), receipts


def _changes(task: str, selected: pd.DataFrame) -> pd.DataFrame:
    release = RELEASE_ROOT / task / "v10_main_universe_v1"
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    previous = pd.read_parquet(release / manifest["selected"]["semantic_bucket_weights"])
    keys = ["task", "level", "semantic_bucket_id"]
    old = previous[keys + ["weight", "weight_rank"]].rename(
        columns={"weight": "previous_weight", "weight_rank": "previous_weight_rank"}
    )
    new = selected[selected.task.eq(task)][keys + ["weight", "weight_rank"]].rename(
        columns={"weight": "new_weight", "weight_rank": "new_weight_rank"}
    )
    frame = old.merge(new, on=keys, validate="one_to_one")
    frame["weight_delta"] = frame.new_weight - frame.previous_weight
    frame["weight_rank_delta"] = frame.new_weight_rank - frame.previous_weight_rank
    return frame.sort_values(["level", "new_weight_rank", "semantic_bucket_id"])


def _write_task(task: str, selected: pd.DataFrame, staging: Path) -> dict[str, Any]:
    task_weights = selected[selected.task.eq(task)].copy()
    buckets, records = frozen._integrated_rankings(task, selected)
    changes = _changes(task, selected)
    weights_io._atomic_parquet(task_weights, staging / "semantic_bucket_weights.parquet")
    weights_io._atomic_tsv(task_weights, staging / "semantic_bucket_weights.tsv")
    weights_io._atomic_parquet(buckets, staging / "semantic_bucket_rankings.parquet")
    weights_io._atomic_parquet(records, staging / "record_relevance_rankings.parquet")
    weights_io._atomic_tsv(changes, staging / "weight_changes.tsv")
    names = ("semantic_bucket_weights.parquet", "semantic_bucket_weights.tsv",
             "semantic_bucket_rankings.parquet", "record_relevance_rankings.parquet",
             "weight_changes.tsv")
    bundle = {
        "version": "semantic_weight_completed_top75.v1", "status": "complete_candidate",
        "weight_calibration_status": "completed_top75_with_base_carry_forward",
        "generation": EXPORT_ID, "task": task, "bucket_count": len(task_weights),
        "record_count": len(records),
        "source_counts": task_weights.groupby("source_epoch").size().astype(int).to_dict(),
        "files": {name: sha256_file(staging / name) for name in names},
    }
    write_json_atomic(staging / "manifest.json", bundle)
    return bundle


def build() -> dict[str, Any]:
    if EXPORT_ROOT.exists() or any(
        (RELEASE_ROOT / task / "v10_main_universe_v1/weighting" / EXPORT_ID).exists()
        for task in TASKS
    ):
        raise FileExistsError(EXPORT_ID)
    observations, receipts = _observations()
    policy = json.loads(frozen.POLICY.read_text(encoding="utf-8"))
    selected = frozen._rank_weights(frozen._select_weights(observations, policy))
    stages, bundles = {}, {}
    try:
        for task in TASKS:
            parent = RELEASE_ROOT / task / "v10_main_universe_v1/weighting"
            parent.mkdir(parents=True, exist_ok=True)
            stages[task] = Path(tempfile.mkdtemp(prefix=f".{EXPORT_ID}.", dir=parent))
            bundles[task] = _write_task(task, selected, stages[task])
        export_stage = Path(tempfile.mkdtemp(prefix=f".{EXPORT_ID}.", dir=EXPORT_ROOT.parent))
        _write_export(export_stage, selected, observations, receipts, bundles)
        for task in TASKS:
            stages[task].replace(RELEASE_ROOT / task / "v10_main_universe_v1/weighting" / EXPORT_ID)
        export_stage.replace(EXPORT_ROOT)
        return json.loads((EXPORT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        for path in stages.values():
            shutil.rmtree(path, ignore_errors=True)
        if "export_stage" in locals():
            shutil.rmtree(export_stage, ignore_errors=True)
        raise


def _write_export(root: Path, selected: pd.DataFrame, observations: pd.DataFrame,
                  receipts: list[dict[str, Any]], bundles: dict[str, Any]) -> None:
    selected_ids = set(selected.semantic_bucket_id)
    retired = observations[~observations.semantic_bucket_id.isin(selected_ids)]
    weights_io._atomic_parquet(observations, root / "weight_observations.parquet")
    weights_io._atomic_parquet(selected, root / "semantic_bucket_weights.parquet")
    weights_io._atomic_tsv(selected, root / "semantic_bucket_weights.tsv")
    weights_io._atomic_tsv(retired, root / "retired_bucket_observations.tsv")
    write_json_atomic(root / "source_receipts.json", {"runs": receipts})
    (root / "report.md").write_text(
        "# Completed top-75 semantic weights\n\n"
        f"Selected {len(selected):,} active buckets from completed top-75 calibration, "
        "reviewed incremental assignments, and base-score carry-forward.\n",
        encoding="utf-8",
    )
    names = ("weight_observations.parquet", "semantic_bucket_weights.parquet",
             "semantic_bucket_weights.tsv", "retired_bucket_observations.tsv",
             "source_receipts.json", "report.md")
    manifest = {
        "version": "semantic_weight_completed_export.v1", "status": "complete_candidate",
        "generation": EXPORT_ID, "created_at": _now(), "bucket_count": len(selected),
        "observation_count": len(observations),
        "manual_bucket_count": int(selected.assignment_method.ne("llm_calibration").sum()),
        "tasks": selected.groupby("task").size().astype(int).to_dict(),
        "source_runs": receipts, "task_bundles": bundles,
        "files": {name: sha256_file(root / name) for name in names},
    }
    write_json_atomic(root / "manifest.json", manifest)


def _verified_bundle(task: str) -> tuple[Path, dict[str, Any]]:
    output = RELEASE_ROOT / task / "v10_main_universe_v1/weighting" / EXPORT_ID
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_candidate" or any(
        sha256_file(output / name) != digest for name, digest in manifest["files"].items()
    ):
        raise ValueError(f"candidate bundle changed: {task}")
    return output, manifest


def activate(approved_manifest_sha256: str) -> dict[str, Any]:
    manifest_path = EXPORT_ROOT / "manifest.json"
    export = json.loads(manifest_path.read_text(encoding="utf-8"))
    if export.get("status") != "complete_candidate" or sha256_file(manifest_path) != approved_manifest_sha256:
        raise ValueError("approved completed-export manifest does not match")
    prepared = []
    for task in TASKS:
        output, bundle = _verified_bundle(task)
        release = RELEASE_ROOT / task / "v10_main_universe_v1"
        release_path = release / "manifest.json"
        current = json.loads(release_path.read_text(encoding="utf-8"))
        selected = dict(current["selected"])
        prefix = output.relative_to(release).as_posix()
        selected.update(
            semantic_bucket_weights=f"{prefix}/semantic_bucket_weights.parquet",
            semantic_bucket_rankings=f"{prefix}/semantic_bucket_rankings.parquet",
            record_relevance_rankings=f"{prefix}/record_relevance_rankings.parquet",
        )
        updated = dict(current, status="complete_reviewed", selected=selected,
                       weight_generation=EXPORT_ID,
                       weight_calibration_status="completed_top75_with_base_carry_forward",
                       files={key: {"path": value, "sha256": sha256_file(release / value)}
                              for key, value in selected.items()})
        receipt = {"version": "semantic_weight_selection.v1", "selected_at": _now(),
                   "previous_manifest_sha256": sha256_file(release_path),
                   "previous_selected": current["selected"], "selected": selected}
        prepared.append((task, output, bundle, release_path, updated, receipt))
    for task, output, bundle, release_path, updated, receipt in prepared:
        bundle["status"] = "complete_reviewed"
        write_json_atomic(output / "manifest.json", bundle)
        write_json_atomic(output / "selection_receipt.json", receipt)
        write_json_atomic(release_path, updated)
        export["task_bundles"][task] = {
            **bundle,
            "manifest_sha256": sha256_file(output / "manifest.json"),
        }
    export.update(status="complete_reviewed", activated_at=_now(),
                  release_manifests={task: sha256_file(path) for task, _, _, path, _, _ in prepared})
    write_json_atomic(manifest_path, export)
    return export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "activate"))
    parser.add_argument("--approved-manifest-sha256")
    args = parser.parse_args()
    if args.command == "build":
        result = build()
    else:
        if not args.approved_manifest_sha256:
            parser.error("activate requires --approved-manifest-sha256")
        result = activate(args.approved_manifest_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
