"""Publish a completed BBB/Oral main-universe semantic refresh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets.artifacts import resolve_semantic_bucket_artifacts
from semantic_buckets.publication import sha256_file
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "main_universe_incremental_semantics.v1"
GENERATION = "main_universe_incremental_v1"


def _readout_map(task: str, workflow: Path, semantic: pd.DataFrame) -> pd.DataFrame:
    active = resolve_semantic_bucket_artifacts(task, "v10")
    current = pd.read_parquet(workflow / "record_relevance_map.parquet")
    current["atom_id"] = [
        core._stable_id("atom", level, source, pair)
        for level, source, pair in current[["level", "source_id", "pair_bucket_key"]]
        .itertuples(index=False, name=None)
    ]
    old = pd.read_parquet(active.record_readout_bucket_map)[
        ["source_row_uid", "level", "readout_bucket_id"]
    ]
    joined = current[["source_row_uid", "level", "atom_id"]].merge(
        old, on=["source_row_uid", "level"], how="left", validate="one_to_one"
    )
    choices = joined.dropna(subset=["readout_bucket_id"]).groupby(
        "atom_id"
    ).readout_bucket_id.agg(lambda values: sorted(set(map(str, values))))
    atom_to_readout = {
        atom: values[0] for atom, values in choices.items() if len(values) == 1
    }
    missing = semantic[~semantic.atom_id.isin(atom_to_readout)]
    for keys, rows in missing.groupby(
        ["semantic_bucket_id", "source_semantic_bucket_id"], sort=True
    ):
        readout = core._stable_id("rb", *map(str, keys), *sorted(rows.atom_id.astype(str)))
        atom_to_readout.update({str(atom): readout for atom in rows.atom_id})
    result = semantic[["level", "source_id", "semantic_bucket_id", "atom_id"]].copy()
    result["readout_bucket_id"] = result.atom_id.map(atom_to_readout)
    result["readout_depth"] = 1
    result["termination_reason"] = result.atom_id.map(
        lambda atom: "frozen_uid_anchor" if atom in choices and len(choices[atom]) == 1
        else "main_universe_incremental_terminal"
    )
    return result


def _record_outputs(workflow: Path, semantic: pd.DataFrame, readout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.read_parquet(workflow / "record_relevance_map.parquet")
    records["atom_id"] = [
        core._stable_id("atom", level, source, pair)
        for level, source, pair in records[
            ["level", "source_id", "pair_bucket_key"]
        ].itertuples(index=False, name=None)
    ]
    assignment = semantic[["atom_id", "semantic_bucket_id"]].merge(
        readout[["atom_id", "readout_bucket_id", "readout_depth"]],
        on="atom_id", validate="one_to_one",
    )
    identity = ["canonical_record_id", "source_row_uid", "level", "source_id", "atom_id"]
    record_readout = records[identity].merge(
        assignment, on="atom_id", validate="many_to_one"
    )
    rankings = records.merge(
        semantic[["atom_id", "semantic_bucket_id"]], on="atom_id", validate="many_to_one"
    ).merge(
        pd.read_parquet(workflow / "final_reviews/semantic_bucket_rankings.parquet"),
        on=["level", "semantic_bucket_id"], validate="many_to_one",
    )
    return record_readout, rankings


def _write(frame: pd.DataFrame, path: Path, sort: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(sort).to_parquet(path, index=False)


def publish(task: str, workflow: Path, output: Path) -> dict:
    status = json.loads((workflow / "manifest.json").read_text())
    if status.get("status") != "complete" or status.get("task_id") != task:
        raise ValueError("semantic refresh is not complete")
    if output.exists():
        raise FileExistsError(output)
    generation = output / "generations" / GENERATION
    eligibility_root = output / "eligibility" / GENERATION
    semantic = pd.read_parquet(workflow / "semantic_bucket_map.parquet")
    readout = _readout_map(task, workflow, semantic)
    record_readout, record_rankings = _record_outputs(workflow, semantic, readout)
    rankings = pd.read_parquet(workflow / "final_reviews/semantic_bucket_rankings.parquet")
    eligibility = pd.read_parquet(workflow / "final_reviews/record_eligibility.parquet")
    _write(semantic, generation / "semantic_bucket_map.parquet", ["level", "semantic_bucket_id", "atom_id"])
    _write(readout, generation / "readout_bucket_map.parquet", ["level", "semantic_bucket_id", "readout_bucket_id"])
    _write(record_readout, generation / "record_readout_bucket_map.parquet", ["level", "source_row_uid"])
    _write(rankings, generation / "semantic_bucket_rankings.parquet", ["level", "level_rank"])
    _write(record_rankings, generation / "record_relevance_rankings.parquet", ["level", "source_row_uid"])
    _write(eligibility, eligibility_root / "record_eligibility.parquet", ["level", "source_row_uid"])
    return _write_manifests(task, workflow, output, generation, eligibility_root, semantic, record_readout)


def _write_manifests(task: str, workflow: Path, output: Path, generation: Path,
                     eligibility: Path, semantic: pd.DataFrame,
                     records: pd.DataFrame) -> dict:
    selected = {
        "semantic_map": f"generations/{GENERATION}/semantic_bucket_map.parquet",
        "semantic_map_manifest": f"generations/{GENERATION}/semantic_bucket_map_manifest.json",
        "readout_bucket_map": f"generations/{GENERATION}/readout_bucket_map.parquet",
        "record_readout_bucket_map": f"generations/{GENERATION}/record_readout_bucket_map.parquet",
        "semantic_bucket_rankings": f"generations/{GENERATION}/semantic_bucket_rankings.parquet",
        "record_relevance_rankings": f"generations/{GENERATION}/record_relevance_rankings.parquet",
        "retrieval_eligibility": f"eligibility/{GENERATION}/record_eligibility.parquet",
    }
    semantic_manifest = {
        "version": VERSION, "task": task, "source_workflow": str(workflow),
        "semantic_atoms": len(semantic), "record_assignments": len(records),
        "source_workflow_manifest_sha256": sha256_file(workflow / "manifest.json"),
    }
    write_json_atomic(generation / "semantic_bucket_map_manifest.json", semantic_manifest)
    files = {key: {"path": value, "sha256": sha256_file(output / value)} for key, value in selected.items()}
    manifest = {
        "schema_version": "semantic_buckets.release.v1", "task": task,
        "status": "complete_reviewed",
        "evidence_library_version": "v10_main_universe_v1", "selected": selected,
        "generation": GENERATION, "files": files,
        "counts": {"semantic_atoms": len(semantic), "record_assignments": len(records)},
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("bbb_martins", "bioavailability_ma"), required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.task, args.workflow, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
