"""Reviewed record-level overlays and shared progressive retrieval for new tasks.

Source rows are conserved. Gold membership is read from the actual vote ledger;
TDC labels are never evidence. Review and input hashes bind every build.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.build_runtime import (
    local_input,
    local_workdir,
    publish_file,
    sha256_file,
    worker_pool,
)
from tools.chembl_tool.common.json_utils import read_jsonl, write_json_atomic
from tools.chembl_tool.common.starling.source_gold_review import payload_hash

ROOT = Path("data/starling_data")
BENCHMARK = Path("data/conditioned_benchmark")
DIRECT_SCOPE = "direct_outcome"
VERSION = "reviewed_new_task_progressive_sources.v1"
DIRECTORIES = {"dili": "DILI", "carcinogens": "Carcinogens"}
_state = None


def _hold_matches(row, hold):
    scope = hold.get("scope", "record")
    if scope == "record":
        return row["source_row_uid"] == hold["source_row_uid"]
    if scope == "source_smiles":
        return True
    if scope in (
        "source_smiles_name",
        "source_smiles_and_molecule_name",
        "source_smiles_and_molecule_name_and_text",
    ):
        if row["molecule_name"] != hold["molecule_name"]:
            return False
    if hold.get("required_text_pattern"):
        return bool(
            re.search(hold["required_text_pattern"], row["raw_record_json"], re.I)
        )
    return (
        bool(hold.get("molecule_name"))
        and row["molecule_name"] == hold["molecule_name"]
    )


def _classify_group(job):
    task, path, row_group, destination = job
    policy, voters, reviews, identity_holds = _state
    table = pq.ParquetFile(path).read_row_group(row_group)
    # Python only sees fields used by semantic rules. Arrow retains every source
    # column unchanged, including the complete raw payload.
    counts, moves = Counter(), Counter()
    columns = {
        k: []
        for k in (
            "group_id",
            "retrieval_eligible",
            "is_gold_voter",
            "progressive_level",
            "family_key",
            "level_assignment_reason",
            "heldout_filter_scope",
            "direct_signal",
            "placement_reviewed",
        )
    }
    audit = []
    for row in table.to_pylist():
        uid = row["source_row_uid"]
        voter = uid in voters
        decision = policy.classify(row, is_voter=voter)
        review = reviews.get(uid)
        if review:
            if (
                payload_hash(json.loads(row["raw_record_json"]))
                != review["source_payload_sha256"]
            ):
                raise ValueError("Stale placement review: " + uid)
            decision = {
                **decision,
                "level": review["level"],
                "reason": review["reason"],
            }
        level = int(decision["level"])
        hold = next(
            (
                h
                for h in identity_holds.get(row["source_smiles"], [])
                if _hold_matches(row, h)
            ),
            None,
        )
        if hold:
            level = 0
            decision["reason"] = "reviewed_identity_hold:" + hold["reason"]
            counts["identity_hold_matches"] += 1
        direct = bool(policy.direct_guard(row))
        if not row["retrieval_eligible"]:
            level = 0
            decision["reason"] = (
                row["identity_review_reason"] or "source_identity_ineligible"
            )
        if voter and level != 1:
            raise ValueError(
                "Reviewed/current voter conflict; repair gold first: " + uid
            )
        if level == 1 and not voter:
            raise ValueError("Nonvoter cannot enter L1: " + uid)
        if (
            level > 2
            and direct
            and not (review and review.get("direct_guard_exemption"))
        ):
            level = 2
            decision["reason"] = "broad_direct_containment"
        family = policy.FAMILIES[level] if level else ""
        group = "Group." + family if level else "Excluded." + task
        values = (
            group,
            level > 0,
            voter,
            level,
            family,
            decision["reason"],
            DIRECT_SCOPE if level in (1, 2) else "",
            direct,
            bool(review),
        )
        for key, value in zip(columns, values, strict=True):
            columns[key].append(value)
        counts[f"L{level}"] += 1
        counts["reviewed"] += bool(review)
        counts["voters"] += voter
        counts["source_rows"] += 1
        moves[f"{row['source_id']}->L{level}"] += 1
        audit.append(
            {
                "source_row_uid": uid,
                "source_id": row["source_id"],
                "molecule_identity_key": row["molecule_identity_key"],
                "previous_group_id": row["group_id"],
                "level": level,
                "family_key": family,
                "reason": decision["reason"],
                "direct_signal": direct,
                "reviewed": bool(review),
                "is_gold_voter": voter,
            }
        )
    for key, values in columns.items():
        array = pa.array(values)
        index = table.schema.get_field_index(key)
        table = (
            table.set_column(index, key, array)
            if index >= 0
            else table.append_column(key, array)
        )
    pq.write_table(table, destination, compression="zstd", compression_level=3)
    pq.write_table(
        pa.Table.from_pylist(audit), str(destination) + ".audit", compression="zstd"
    )
    return str(destination), dict(counts), dict(moves)


def build_overlay(task, workers=32):
    current = ROOT / task / "progressive_v1/manifest.json"
    if current.exists() and json.loads(current.read_text()).get("identity_contract"):
        raise ValueError(
            "Reviewed identity-repaired overlay is frozen; use rebuild_current_starling_retrieval restore-records/build for current data"
        )
    global _state
    started = time.monotonic()
    source = ROOT / task / "canonical_v1/records.parquet"
    votes = ROOT / task / "gold_v2/source_votes.jsonl"
    review_path = ROOT / task / "level_review_v1/placement_decisions.jsonl"
    policy = importlib.import_module(f"tools.chembl_tool.tasks.{task}.starling_levels")
    if not review_path.exists():
        raise ValueError("Semantic review is required before final build")
    reviews = read_jsonl(review_path)
    review_map = {r["source_row_uid"]: r for r in reviews}
    if len(review_map) != len(reviews):
        raise ValueError("Duplicate placement reviews")
    voter_ids = {r["source_record_id"] for r in read_jsonl(votes)}
    hold_path = review_path.with_name("identity_holds.jsonl")
    holds = read_jsonl(hold_path) if hold_path.exists() else []
    identity_holds = defaultdict(list)
    for hold in holds:
        identity_holds[hold["source_smiles"]].append(hold)
    paths = [source, votes, review_path, Path(policy.__file__), Path(__file__)]
    if hold_path.exists():
        paths.append(hold_path)
    inputs = {str(p): sha256_file(p) for p in paths}
    output = ROOT / task / "progressive_v1"
    output.mkdir(parents=True, exist_ok=True)
    _state = policy, voter_ids, review_map, identity_holds
    count, transitions = Counter(), Counter()
    with local_workdir() as staging:
        cached = local_input(source)
        n_groups = pq.ParquetFile(cached).num_row_groups
        jobs = [
            (task, cached, i, staging / f"{i:04d}.parquet") for i in range(n_groups)
        ]
        with worker_pool(min(workers, n_groups)) as pool:
            results = list(pool.map(_classify_group, jobs))
        for suffix, filename in [
            ("", "records.parquet"),
            (".audit", "record_audit.parquet"),
        ]:
            writer = None
            try:
                for path, counts, moves in results:
                    if not suffix:
                        count.update(counts)
                        transitions.update(moves)
                    table = pq.read_table(path + suffix)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            staging / filename,
                            table.schema,
                            compression="zstd",
                            compression_level=3,
                        )
                    writer.write_table(table)
            finally:
                if writer:
                    writer.close()
        assert count["source_rows"] == pq.ParquetFile(source).metadata.num_rows
        assert count["voters"] == len(voter_ids) == count["L1"]
        assert count["reviewed"] == len(review_map)
        files = {
            f: publish_file(staging / f, output / f)
            for f in ["records.parquet", "record_audit.parquet"]
        }
    assert all(sha256_file(Path(p)) == h for p, h in inputs.items())
    manifest = {
        "task": task,
        "contract": VERSION,
        "inputs": inputs,
        "files": files,
        "counts": dict(count),
        "source_to_level": dict(sorted(transitions.items())),
        "families": policy.FAMILIES,
        "level_descriptions": policy.LEVEL_DESCRIPTIONS,
        "source_rows_deleted": 0,
        "membership_policy": "exact_actual_source_vote_uid",
        "heldout_filter_scope": DIRECT_SCOPE,
        "record_assignment_unit": "source_record",
        "review_scope": "Codex full record-text reading, not human expert or full-paper review",
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "task": task,
                "counts": dict(count),
                "seconds": manifest["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return manifest


def build_retrieval(task, benchmark_root, workers):
    """Use the canonical shared builder, including the current identity cache."""
    from tools.chembl_tool.paper_experiments import (
        rebuild_current_starling_retrieval as rebuild,
    )

    if Path(benchmark_root).resolve() != Path("data/conditioned_benchmark").resolve():
        raise ValueError(
            "Current retrieval requires data/conditioned_benchmark; historical staging is not an evaluation input"
        )
    rebuild.build_catalogs(rebuild.DEFAULT_ARTIFACT_ROOT, tasks=(task,))
    rebuild.build_indices(rebuild.DEFAULT_ARTIFACT_ROOT, workers=workers, tasks=(task,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=list(DIRECTORIES), default=list(DIRECTORIES)
    )
    parser.add_argument(
        "--phase", choices=["overlay", "retrieval", "all"], default="all"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Total worker budget across task processes",
    )
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    # Separate processes keep each task immutable fork state isolated.
    if len(args.tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(len(args.tasks), args.workers)) as pool:
            futures = [
                pool.submit(
                    subprocess.run,
                    [
                        sys.executable,
                        "-m",
                        "tools.chembl_tool.common.starling.build_progressive_sources",
                        "--tasks",
                        task,
                        "--phase",
                        args.phase,
                        "--workers",
                        str(max(1, args.workers // len(args.tasks))),
                        "--benchmark-root",
                        str(args.benchmark_root),
                    ],
                    check=True,
                )
                for task in args.tasks
            ]
            for future in futures:
                future.result()
    else:
        task = args.tasks[0]
        if args.phase in ["overlay", "all"]:
            build_overlay(task, args.workers)
        if args.phase in ["retrieval", "all"]:
            build_retrieval(task, args.benchmark_root, args.workers)


if __name__ == "__main__":
    main()
