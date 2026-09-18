"""Publish reviewed Gold-v1 semantic deltas without regrouping frozen atoms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import socket
from typing import Any

import pandas as pd


from semantic_buckets import (
    bioavailability_semantic_readout_v1 as core,
)
from semantic_buckets.publication import (
    SCHEMA_VERSION,
    sha256_file,
    tree_inventory,
)
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "gold_v1_protected_incremental_semantics.v1"
GENERATION = "gold_v1_protected_incremental_v1"
LEVELS = {"L2", "L3", "L4"}
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).resolve().parent / "history/pre_gold_v2"
TASKS = {
    "bbb_martins": "bbb_semantic_readout_v10_v3",
    "bioavailability_ma": "bioavailability_semantic_readout_v10_v3_l4_refinement_v1",
}


def _old(task: str) -> tuple[Path, dict[str, Any]]:
    root = ARCHIVE / task / "v10/semantic_buckets"
    return root, json.loads((root / "manifest.json").read_text())


def _selected(root: Path, manifest: dict[str, Any], key: str) -> pd.DataFrame:
    return pd.read_parquet(root / manifest["selected"][key])


def _current(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.read_parquet(
        ROOT / f"data/evidence_libraries/{task}/v10/03_pair_buckets/records.parquet",
        columns=["canonical_record_id", "source_row_uid", "source_id", "pair_bucket_key"],
    )
    levels = pd.read_parquet(
        ROOT / f"data/evidence_libraries/{task}/v10/level_mapping/records.parquet"
    )
    if records.source_row_uid.duplicated().any() or levels.source_row_uid.duplicated().any():
        raise ValueError("current records and levels must be unique by source_row_uid")
    merged = records.merge(levels, on=["source_row_uid", "canonical_record_id"], validate="one_to_one")
    if len(merged) != len(records) or len(levels) != len(records):
        raise ValueError("current Stage-3 records and UID levels do not match exactly")
    merged["level"] = merged.level.map(lambda value: f"L{int(value)}")
    return records, merged


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def publish(task: str, delta_root: Path) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    delta_manifest = json.loads((delta_root / "manifest.json").read_text())
    if delta_manifest.get("status") not in {"delta_complete", "complete"} or delta_manifest.get("task_id") != task:
        raise ValueError("incremental semantic delta is not complete")
    for name in ("alignment", "final_reviews"):
        manifest = json.loads((delta_root / name / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"{name} is not complete")

    old_root, old_manifest = _old(task)
    _, current = _current(task)
    current_uids = set(current.source_row_uid)
    incremental = pd.read_parquet(delta_root / "final_reviews/record_eligibility.parquet")
    semantic_new = pd.read_parquet(delta_root / "semantic_bucket_map.parquet")
    rankings_new = pd.read_parquet(delta_root / "final_reviews/semantic_bucket_rankings.parquet")
    if set(incremental.level) - LEVELS or set(semantic_new.level) - LEVELS:
        raise ValueError("incremental outputs escaped L2-L4")

    old_record_readout = _selected(old_root, old_manifest, "record_readout_bucket_map")
    old_rankings = _selected(old_root, old_manifest, "semantic_bucket_rankings")
    old_records = _selected(old_root, old_manifest, "record_relevance_rankings")

    from semantic_buckets import gold_v2_incremental_semantics as incremental_workflow

    incremental_workflow.configure(task, delta_root / "record_relevance_map.parquet")
    outside = current[~current.level.isin(LEVELS) & current.level.ne("L1")].copy()
    outside = incremental_workflow._with_atom_ids(
        incremental_workflow._semantic_columns(outside)
    )
    semantic_anchors = old_records[
        ~old_records.level.isin(LEVELS) & old_records.source_row_uid.isin(current_uids)
    ][["source_row_uid", "level", "semantic_bucket_id"]]
    outside = outside.merge(
        semantic_anchors, on=["source_row_uid", "level"], how="left", validate="one_to_one"
    )
    semantic_choices = outside.dropna(subset=["semantic_bucket_id"]).groupby(
        "atom_id"
    ).semantic_bucket_id.agg(lambda values: sorted(set(map(str, values))))
    if any(len(values) > 1 for values in semantic_choices):
        raise ValueError("a current nonincremental atom has conflicting frozen semantic parents")
    atom_to_semantic = {atom: values[0] for atom, values in semantic_choices.items()}
    for atom in sorted(set(outside.atom_id) - set(atom_to_semantic)):
        row = outside.loc[outside.atom_id.eq(atom)].iloc[0]
        atom_to_semantic[atom] = core._stable_id("sb", row.level, row.source_id, atom)
    outside["semantic_bucket_id"] = outside.atom_id.map(atom_to_semantic)
    semantic_outside = outside[
        ["level", "source_id", "semantic_bucket_id", "atom_id"]
    ].drop_duplicates("atom_id")
    semantic_outside["source_semantic_bucket_id"] = semantic_outside.semantic_bucket_id
    semantic_outside = semantic_outside[
        ["level", "source_id", "source_semantic_bucket_id", "semantic_bucket_id", "atom_id"]
    ]

    semantic = pd.concat(
        [semantic_new, semantic_outside], ignore_index=True
    ).sort_values(["level", "source_id", "semantic_bucket_id", "atom_id"])
    outside_ids = set(semantic_outside.semantic_bucket_id)
    ranking_outside = old_rankings[
        ~old_rankings.level.isin(LEVELS) & old_rankings.semantic_bucket_id.isin(outside_ids)
    ].copy()
    missing_rank_ids = outside_ids - set(ranking_outside.semantic_bucket_id)
    additions = []
    for level, rows in semantic_outside[
        semantic_outside.semantic_bucket_id.isin(missing_rank_ids)
    ].groupby("level", sort=True):
        start = int(old_rankings.loc[old_rankings.level.eq(level), "level_rank"].max()) + 1
        for offset, bucket in enumerate(sorted(rows.semantic_bucket_id.unique())):
            additions.append({
                "task_id": task, "level": level, "semantic_bucket_id": bucket,
                "bradley_terry_score": 0.0, "level_rank": start + offset,
                "level_percentile": 0.0, "comparison_count": 0,
            })
    if additions:
        ranking_outside = pd.concat([ranking_outside, pd.DataFrame(additions)], ignore_index=True)
    rankings = pd.concat(
        [rankings_new, ranking_outside], ignore_index=True
    ).sort_values(["level", "level_rank", "semantic_bucket_id"])

    old_anchor = old_record_readout[old_record_readout.level.isin(LEVELS)][
        ["source_row_uid", "level", "readout_bucket_id"]
    ]
    anchored = incremental[["source_row_uid", "level", "atom_id"]].merge(
        old_anchor, on=["source_row_uid", "level"], how="left", validate="one_to_one"
    )
    choices = anchored.dropna(subset=["readout_bucket_id"]).groupby("atom_id").readout_bucket_id.agg(
        lambda values: sorted(set(map(str, values)))
    )
    atom_to_readout = {atom: values[0] for atom, values in choices.items() if len(values) == 1}
    unresolved = semantic_new[~semantic_new.atom_id.isin(atom_to_readout)].copy()
    for (_, source_bucket), rows in unresolved.groupby(
        ["semantic_bucket_id", "source_semantic_bucket_id"], sort=True
    ):
        readout_id = core._stable_id("rb", str(source_bucket), *sorted(rows.atom_id.astype(str)))
        atom_to_readout.update({str(atom): readout_id for atom in rows.atom_id})
    readout_new = semantic_new[["level", "source_id", "semantic_bucket_id", "atom_id"]].copy()
    readout_new["readout_bucket_id"] = readout_new.atom_id.map(atom_to_readout)
    readout_new["readout_depth"] = 1
    readout_new["termination_reason"] = readout_new.atom_id.map(
        lambda atom: "frozen_uid_anchor" if atom in choices and len(choices[atom]) == 1
        else "incremental_source_semantic_terminal"
    )
    outside_readout_anchors = outside[["source_row_uid", "level", "atom_id"]].merge(
        old_record_readout[["source_row_uid", "level", "readout_bucket_id"]],
        on=["source_row_uid", "level"], how="left", validate="one_to_one",
    )
    outside_choices = outside_readout_anchors.dropna(subset=["readout_bucket_id"]).groupby(
        "atom_id"
    ).readout_bucket_id.agg(lambda values: sorted(set(map(str, values))))
    outside_atom_to_readout = {
        atom: values[0] for atom, values in outside_choices.items() if len(values) == 1
    }
    for atom in sorted(set(outside.atom_id) - set(outside_atom_to_readout)):
        outside_atom_to_readout[atom] = core._stable_id("rb", atom)
    readout_outside = semantic_outside[
        ["level", "source_id", "semantic_bucket_id", "atom_id"]
    ].copy()
    readout_outside["readout_bucket_id"] = readout_outside.atom_id.map(outside_atom_to_readout)
    readout_outside["readout_depth"] = 1
    readout_outside["termination_reason"] = readout_outside.atom_id.map(
        lambda atom: "frozen_uid_anchor" if atom in outside_choices and len(outside_choices[atom]) == 1
        else "new_non_assay_transfer_atom"
    )
    readout = pd.concat([readout_new, readout_outside], ignore_index=True).sort_values(
        ["level", "semantic_bucket_id", "readout_bucket_id", "atom_id"]
    )

    record_new = incremental[
        ["canonical_record_id", "source_row_uid", "level", "source_id", "atom_id", "semantic_bucket_id"]
    ].merge(
        readout_new[["atom_id", "readout_bucket_id", "readout_depth"]],
        on="atom_id", validate="many_to_one",
    )
    record_outside = outside[
        ["canonical_record_id", "source_row_uid", "level", "source_id", "atom_id", "semantic_bucket_id"]
    ].merge(
        readout_outside[["atom_id", "readout_bucket_id", "readout_depth"]],
        on="atom_id", validate="many_to_one",
    )
    record_readout = pd.concat([record_new, record_outside], ignore_index=True).sort_values(
        ["level", "source_row_uid"]
    )

    group_columns = ["source_row_uid", "source_group_id", "family_key"]
    current_ranked = incremental.drop(columns=["expert_weight", "prior_zero_weight"]).merge(
        current[group_columns], on="source_row_uid", validate="one_to_one"
    )
    outside_ranked = outside.merge(
        ranking_outside, on=["level", "semantic_bucket_id"], validate="many_to_one"
    )
    record_rankings = pd.concat(
        [
            current_ranked.drop(columns=["decision", "reason_code", "retrieval_eligible"]),
            outside_ranked,
        ],
        ignore_index=True, sort=False,
    ).sort_values(["level", "source_row_uid"])
    eligibility = current_ranked.sort_values(["level", "source_row_uid"])

    expected = current[~current.level.eq("L1")]
    if len(record_readout) != len(expected) or set(record_readout.source_row_uid) != set(expected.source_row_uid):
        raise ValueError("record/readout map does not cover every non-L1 record")
    if len(record_rankings) != len(expected) or set(record_rankings.source_row_uid) != set(expected.source_row_uid):
        raise ValueError("record rankings do not cover every non-L1 record")
    if len(eligibility) != len(incremental):
        raise ValueError("eligibility row count changed during publication")

    root = Path(__file__).resolve().parent / "releases" / task / "v10"
    generation = root / "generations" / GENERATION
    eligibility_root = root / "eligibility" / GENERATION
    _write(semantic, generation / "semantic_bucket_map.parquet")
    _write(readout, generation / "readout_buckets_v5_depth1/readout_bucket_map.parquet")
    _write(record_readout, generation / "readout_buckets_v5_depth1/record_readout_bucket_map.parquet")
    _write(rankings, generation / "degree25_rankings/semantic_bucket_rankings.parquet")
    _write(record_rankings, generation / "degree25_rankings/record_relevance_rankings.parquet")
    _write(eligibility, eligibility_root / "record_eligibility.parquet")
    semantic_manifest = {
        "version": VERSION,
        "task": task,
        "levels_rebuilt": sorted(LEVELS),
        "anchored_atoms": int(sum(len(values) == 1 for values in choices)),
        "new_or_conflicting_readout_atoms": len(unresolved),
        "semantic_atoms": len(semantic),
        "record_assignments": len(record_readout),
        "source_delta": str(delta_root),
    }
    write_json_atomic(generation / "semantic_bucket_map_manifest.json", semantic_manifest)

    provenance = generation / "incremental_review_provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for relative in (
        "manifest.json", "prompt_review/manifest.json", "alignment/manifest.json",
        "alignment/decisions.parquet", "alignment/requests.sqlite3",
        "delta_semantic_run/manifest.json", "delta_semantic_run/requests.sqlite3",
        "final_reviews/manifest.json", "final_reviews/bucket_decisions.parquet",
        "final_reviews/requests.sqlite3",
    ):
        source = delta_root / relative
        destination = provenance / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    selected = {
        "semantic_map": str((generation / "semantic_bucket_map.parquet").relative_to(root)),
        "semantic_map_manifest": str((generation / "semantic_bucket_map_manifest.json").relative_to(root)),
        "readout_bucket_map": str((generation / "readout_buckets_v5_depth1/readout_bucket_map.parquet").relative_to(root)),
        "record_readout_bucket_map": str((generation / "readout_buckets_v5_depth1/record_readout_bucket_map.parquet").relative_to(root)),
        "semantic_bucket_rankings": str((generation / "degree25_rankings/semantic_bucket_rankings.parquet").relative_to(root)),
        "record_relevance_rankings": str((generation / "degree25_rankings/record_relevance_rankings.parquet").relative_to(root)),
        "retrieval_eligibility": str((eligibility_root / "record_eligibility.parquet").relative_to(root)),
    }
    from semantic_buckets.publication import _coverage
    release = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_reviewed",
        "task": task,
        "evidence_library_version": "v10",
        "selected": selected,
        "coverage": _coverage(root, selected),
        "upstream": {
            "pair_bucket_records": str((ROOT / f"data/evidence_libraries/{task}/v10/03_pair_buckets/records.parquet").resolve()),
            "pair_bucket_records_sha256": sha256_file(ROOT / f"data/evidence_libraries/{task}/v10/03_pair_buckets/records.parquet"),
        },
        "generations": {GENERATION: tree_inventory(generation)},
        "eligibility_generations": {GENERATION: tree_inventory(eligibility_root)},
        "migration": {"version": VERSION, "hostname": socket.gethostname(), "source_root": str(old_root), "delta_root": str(delta_root)},
    }
    write_json_atomic(root / "manifest.json", release)
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--delta-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.task, args.delta_root), indent=2))


if __name__ == "__main__":
    main()
