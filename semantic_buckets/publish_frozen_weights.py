"""Freeze resumable V6 ledgers and publish a complete interim weight selection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable

import pandas as pd

from semantic_buckets import calibrate_weights_v6 as calibration
from semantic_buckets import sliding_weight_assignment as weights_io
from semantic_buckets.publication import sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic


ROOT = Path(__file__).resolve().parent
V6_ROOT = ROOT / "provenance/semantic_weight_calibration_v6"
V5_ROOT = ROOT / "provenance/semantic_weight_sliding_v5"
EXPORT_ROOT = ROOT / "provenance/semantic_weight_exports"
RELEASE_ROOT = ROOT / "releases"
POLICY = ROOT / "policies/semantic_weight_incremental_manual_v1.json"
FREEZE_ID = "frozen_top75_20260918_asap_v1"
WEIGHT_GENERATION = FREEZE_ID
TOP75_RUNS = (
    "v5_completed_levels_top75_final_20260917",
    "v5_bbb_l4_top75_final_20260917",
    "v5_remaining_levels_top75_final_20260917",
)
V5_RUN = "openrouter_pro_seed_flash_dgx007_dgx011_rubric_v5_20260917"
TASKS = ("bbb_martins", "bioavailability_ma")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_if_present(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _database_summary(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(path)
    try:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError(f"SQLite quick_check failed: {path}")
        statuses = dict(connection.execute(
            "SELECT status,count(*) FROM requests GROUP BY status"
        ))
    finally:
        connection.close()
    return {"request_status_counts": statuses, "quick_check": "ok"}


def _resume_command(run_id: str, manifest: dict[str, Any]) -> str:
    override = manifest["execution_override"]
    benchmark = (
        V5_ROOT / V5_RUN / "speculative_benchmark_dgx008_first8_v1/manifest.json"
    )
    return (
        "python -m semantic_buckets.calibrate_weights_v6 run "
        f"--run-id {run_id} "
        f"--approved-review-sha256 {manifest['prompt_review_manifest_sha256']} "
        "--local-speculative "
        f"--speculative-benchmark-manifest {benchmark.relative_to(ROOT.parent)} "
        f"--speculative-required-replicas {override['required_replicas']} "
        f"--speculative-fanout {override['fanout']} "
        f"--speculative-base-url {override['endpoint']}"
    )


def _committed_weights(snapshot: Path) -> pd.DataFrame:
    schedule = pd.read_parquet(snapshot / "schedule.parquet")
    connection = calibration.core._request_database(
        snapshot / "requests.sqlite3", journal_mode="DELETE"
    )
    try:
        frame = calibration._score_frame(calibration._completed(connection), schedule)
    finally:
        connection.close()
    if frame.semantic_bucket_id.duplicated().any():
        raise ValueError(f"duplicate committed bucket in {snapshot}")
    return frame


def _freeze_run(run_id: str, output: Path, freeze_id: str) -> dict[str, Any]:
    source = V6_ROOT / run_id
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = output / run_id
    snapshot.mkdir(parents=True)
    names = (
        "manifest.json", "schedule.parquet", "schedule.tsv", "weights.partial.parquet",
        "weights.partial.tsv", "requests.sqlite3", "v5_manifest.snapshot.json",
        "v5_weights.snapshot.parquet", "prompt_review/manifest.json",
    )
    for name in names:
        _copy_if_present(source / name, snapshot / name)
    database = _database_summary(snapshot / "requests.sqlite3")
    committed = _committed_weights(snapshot)
    weights_io._atomic_parquet(committed, snapshot / "weights.committed.parquet")
    weights_io._atomic_tsv(committed, snapshot / "weights.committed.tsv")
    files = {str(path.relative_to(snapshot)): sha256_file(path)
             for path in sorted(snapshot.rglob("*")) if path.is_file()}
    receipt = {
        "run_id": run_id, "source_status": manifest["status"],
        "committed_bucket_count": len(committed), **database,
        "resume_command": _resume_command(run_id, manifest), "files": files,
    }
    if manifest["status"] not in {"complete_unpublished", "complete_candidate"}:
        manifest.update(
            status="paused", paused_at=_now(), freeze_id=freeze_id,
            committed_bucket_count=len(committed), resume_command=receipt["resume_command"],
        )
        write_json_atomic(manifest_path, manifest)
    return receipt


def freeze(run_ids: Iterable[str], freeze_id: str = FREEZE_ID) -> dict[str, Any]:
    root = EXPORT_ROOT / freeze_id / "freeze"
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    receipts = [_freeze_run(run_id, root, freeze_id) for run_id in run_ids]
    manifest = {
        "version": "semantic_weight_freeze.v1", "status": "complete",
        "freeze_id": freeze_id, "created_at": _now(), "runs": receipts,
    }
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _source_specs(freeze_root: Path) -> list[tuple[str, str, int, Path]]:
    old = V6_ROOT
    return [
        (V5_RUN, "base", 0, V5_ROOT / V5_RUN / "weights.partial.parquet"),
        ("v5_completed_levels_epoch1_local_dgx008_20260917", "v6_top50_epoch1", 1,
         old / "v5_completed_levels_epoch1_local_dgx008_20260917/weights.partial.parquet"),
        ("v5_completed_levels_epoch1_repeat2_20260917", "v6_top50_epoch2", 2,
         old / "v5_completed_levels_epoch1_repeat2_20260917/weights.partial.parquet"),
        ("v5_bbb_l4_epoch2_local_dgx008_20260917", "v6_top50_epoch2", 2,
         old / "v5_bbb_l4_epoch2_local_dgx008_20260917/weights.partial.parquet"),
        *[(run_id, "v6_top75_frozen", 3,
           freeze_root / run_id / "weights.committed.parquet") for run_id in TOP75_RUNS],
    ]


def _observation_frame(run_id: str, epoch: str, priority: int, path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = ["semantic_bucket_id", "task", "level", "weight", "rationale"]
    if not set(required) <= set(frame) or frame.semantic_bucket_id.duplicated().any():
        raise ValueError(f"invalid weight source: {path}")
    optional = [
        "request_id", "served_model", "provider_name", "provider_base_url",
        "attempts", "input_tokens", "output_tokens",
    ]
    result = frame[[*required, *[column for column in optional if column in frame]]].copy()
    result["source_run_id"] = run_id
    result["source_epoch"] = epoch
    result["source_priority"] = priority
    result["assignment_method"] = "llm_calibration"
    return result


def _manual_frame(policy: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for task, specification in policy["tasks"].items():
        rows.extend({"task": task, **row} for row in specification["weights"])
    frame = pd.DataFrame(rows)
    frame["source_run_id"] = policy["version"]
    frame["source_epoch"] = "manual_incremental"
    frame["source_priority"] = 4
    frame["assignment_method"] = policy["assignment_method"]
    return frame


def _load_observations(root: Path, policy: dict[str, Any]) -> pd.DataFrame:
    frames = [_observation_frame(*spec) for spec in _source_specs(root / "freeze")]
    frames.append(_manual_frame(policy))
    observations = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["source_run_id", "semantic_bucket_id"]
    if observations.duplicated(keys).any():
        raise ValueError("one source run repeats a semantic bucket")
    return observations.sort_values(["source_priority", "source_run_id"])


def _active_rankings(task: str) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    release = RELEASE_ROOT / task / "v10_main_universe_v1"
    generation = release / "generations/final_active_incremental_v1"
    manifest = json.loads((generation / "generation_manifest.json").read_text())
    rankings = pd.read_parquet(generation / "semantic_bucket_rankings.parquet")
    return release, rankings, manifest


def _select_weights(observations: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    universes = []
    for task in TASKS:
        _, rankings, _ = _active_rankings(task)
        expected = policy["tasks"][task]["semantic_bucket_rankings_sha256"]
        path = RELEASE_ROOT / task / "v10_main_universe_v1/generations/final_active_incremental_v1/semantic_bucket_rankings.parquet"
        if sha256_file(path) != expected:
            raise ValueError(f"manual policy ranking hash changed for {task}")
        universes.append(rankings[[
            "task_id", "level", "semantic_bucket_id", "level_rank"
        ]].rename(
            columns={"task_id": "task"}
        ))
    universe = pd.concat(universes, ignore_index=True)
    selected = observations.sort_values("source_priority").drop_duplicates(
        "semantic_bucket_id", keep="last"
    )
    selected = universe.merge(selected, on=["task", "level", "semantic_bucket_id"],
                              how="left", validate="one_to_one")
    if selected.weight.isna().any():
        missing = selected[selected.weight.isna()].semantic_bucket_id.tolist()
        raise ValueError(f"weight coverage is incomplete: {missing[:10]}")
    return selected.rename(columns={"rationale": "weight_rationale"})


def _rank_weights(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["task", "level", "weight", "level_rank", "semantic_bucket_id"],
        ascending=[True, True, False, True, True], kind="stable",
    ).copy()
    ranked["weight_rank"] = ranked.groupby(["task", "level"]).cumcount() + 1
    sizes = ranked.groupby(["task", "level"]).semantic_bucket_id.transform("size")
    ranked["weight_percentile"] = (sizes - ranked.weight_rank) / (sizes - 1).clip(lower=1) * 100
    return ranked


def _integrated_rankings(task: str, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    release, rankings, _ = _active_rankings(task)
    task_weights = selected[selected.task.eq(task)].drop(columns=["task"])
    keep = [
        "semantic_bucket_id", "weight", "weight_rank", "weight_percentile",
        "weight_rationale",
        "assignment_method", "source_epoch", "source_run_id",
    ]
    buckets = rankings.merge(task_weights[keep], on="semantic_bucket_id", validate="one_to_one")
    source = release / "generations/final_active_incremental_v1/record_relevance_rankings.parquet"
    records = pd.read_parquet(source).merge(
        task_weights[keep], on="semantic_bucket_id", validate="many_to_one"
    )
    return buckets.sort_values(["level", "weight_rank"]), records.sort_values(
        ["level", "weight_rank", "source_row_uid"]
    )


def _publish_task(task: str, selected: pd.DataFrame, root: Path) -> dict[str, Any]:
    release, _, generation_manifest = _active_rankings(task)
    output = release / "weighting" / WEIGHT_GENERATION
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    task_weights = selected[selected.task.eq(task)].copy()
    buckets, records = _integrated_rankings(task, selected)
    weights_io._atomic_parquet(task_weights, output / "semantic_bucket_weights.parquet")
    weights_io._atomic_parquet(buckets, output / "semantic_bucket_rankings.parquet")
    weights_io._atomic_parquet(records, output / "record_relevance_rankings.parquet")
    weights_io._atomic_tsv(task_weights, output / "semantic_bucket_weights.tsv")
    bundle = _weight_bundle_manifest(task, task_weights, records, output, root)
    write_json_atomic(output / "manifest.json", bundle)
    return _activate_release(release, generation_manifest, output, bundle)


def _weight_bundle_manifest(task: str, weights: pd.DataFrame, records: pd.DataFrame,
                            output: Path, root: Path) -> dict[str, Any]:
    names = (
        "semantic_bucket_weights.parquet", "semantic_bucket_weights.tsv",
        "semantic_bucket_rankings.parquet", "record_relevance_rankings.parquet",
    )
    return {
        "version": "semantic_weight_frozen_interim.v1", "status": "complete_reviewed",
        "weight_calibration_status": "frozen_interim_complete_coverage", "task": task,
        "freeze_id": FREEZE_ID, "bucket_count": len(weights), "record_count": len(records),
        "manual_bucket_count": int(weights.assignment_method.ne("llm_calibration").sum()),
        "policy": str(POLICY.relative_to(ROOT.parent)), "policy_sha256": sha256_file(POLICY),
        "freeze_manifest_sha256": sha256_file(root / "freeze/manifest.json"),
        "files": {name: sha256_file(output / name) for name in names},
    }


def _activate_release(release: Path, generation: dict[str, Any], output: Path,
                      bundle: dict[str, Any]) -> dict[str, Any]:
    manifest_path = release / "manifest.json"
    prior_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = dict(manifest["selected"])
    selected = dict(generation["selected"])
    prefix = output.relative_to(release).as_posix()
    selected.update(
        semantic_bucket_weights=f"{prefix}/semantic_bucket_weights.parquet",
        semantic_bucket_rankings=f"{prefix}/semantic_bucket_rankings.parquet",
        record_relevance_rankings=f"{prefix}/record_relevance_rankings.parquet",
    )
    manifest.update(
        status="complete_reviewed", selected=selected,
        generation="final_active_incremental_v1", weight_generation=WEIGHT_GENERATION,
        weight_calibration_status="frozen_interim_complete_coverage",
        files={key: {"path": value, "sha256": sha256_file(release / value)}
               for key, value in selected.items()}, counts=generation["counts"],
    )
    receipt = {
        "version": "semantic_weight_selection.v1", "selected_at": _now(),
        "previous_manifest_sha256": prior_hash, "previous_selected": previous,
        "selected": selected, "weight_bundle": bundle,
    }
    write_json_atomic(output / "selection_receipt.json", receipt)
    write_json_atomic(manifest_path, manifest)
    return {"task": manifest["task"], "release": str(release), "selected": selected}


def publish(freeze_id: str = FREEZE_ID) -> dict[str, Any]:
    root = EXPORT_ROOT / freeze_id
    freeze_manifest = root / "freeze/manifest.json"
    if not freeze_manifest.is_file():
        raise FileNotFoundError(freeze_manifest)
    for task in TASKS:
        output = RELEASE_ROOT / task / "v10_main_universe_v1/weighting" / WEIGHT_GENERATION
        if output.exists():
            raise FileExistsError(output)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    observations = _load_observations(root, policy)
    selected = _rank_weights(_select_weights(observations, policy))
    selected_ids = set(selected.semantic_bucket_id)
    retired = observations[~observations.semantic_bucket_id.isin(selected_ids)].copy()
    weights_io._atomic_parquet(observations, root / "weight_observations.parquet")
    weights_io._atomic_parquet(selected, root / "semantic_bucket_weights.parquet")
    weights_io._atomic_tsv(selected, root / "semantic_bucket_weights.tsv")
    weights_io._atomic_tsv(retired, root / "retired_bucket_observations.tsv")
    (root / "report.md").write_text(
        "# Frozen interim semantic weights\n\n"
        f"Published {len(selected):,} weighted buckets from {len(observations):,} preserved "
        f"observations. {len(retired.semantic_bucket_id.unique()):,} retired bucket IDs remain "
        "audit-only. The paused V6 ledgers can be resumed without repeating completed requests.\n",
        encoding="utf-8",
    )
    publications = [_publish_task(task, selected, root) for task in TASKS]
    summary = _publication_manifest(root, selected, observations, retired, publications)
    write_json_atomic(root / "manifest.json", summary)
    return summary


def _publication_manifest(root: Path, selected: pd.DataFrame, observations: pd.DataFrame,
                          retired: pd.DataFrame, publications: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "semantic_bucket_weights.parquet", "semantic_bucket_weights.tsv",
        "weight_observations.parquet", "retired_bucket_observations.tsv",
    )
    return {
        "version": "semantic_weight_frozen_export.v1", "status": "complete_reviewed",
        "weight_calibration_status": "frozen_interim_complete_coverage",
        "freeze_id": root.name, "published_at": _now(), "bucket_count": len(selected),
        "observation_count": len(observations),
        "manual_bucket_count": int(selected.assignment_method.ne("llm_calibration").sum()),
        "retired_bucket_count": int(retired.semantic_bucket_id.nunique()),
        "tasks": selected.groupby("task").size().astype(int).to_dict(),
        "publications": publications,
        "files": {name: sha256_file(root / name) for name in (*names, "report.md")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "publish"))
    parser.add_argument("--freeze-id", default=FREEZE_ID)
    parser.add_argument("--run-id", action="append", default=[])
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.run_id or TOP75_RUNS, args.freeze_id)
    else:
        result = publish(args.freeze_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
