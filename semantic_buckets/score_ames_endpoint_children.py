"""Rescore only the new AMES endpoint-family children with Luna.

Preparation pins the unselected child map, original parent scores, V6 cards,
and exact prompts. Pass 1 independently scores the children; Pass 2 sorts only
those children and runs anchored task-level chains. The existing parent scores
seed each chain but are never candidates or rewritten. The shared official
scorer owns card rendering, request validation, and replica aggregation.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from data.processing import openrouter_provider_pool
from predict.api_client.pool import load_provider_pool_config, preflight_provider_models
from predict.utils.json import sha256_file, write_json_atomic
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import official_two_pass_weights as official
from semantic_buckets import speculative_weight_execution as speculative
from semantic_buckets.artifacts import REPOSITORY_ROOT


CANDIDATE = REPOSITORY_ROOT / (
    "semantic_buckets/provenance/ames_endpoint_family_split_candidate_v1_20260923"
)
PREDECESSOR = REPOSITORY_ROOT / (
    "semantic_buckets/provenance/semantic_weight_two_pass_v4/"
    "ames_morgan100_official_two_pass_v6_cards_task_level_wavefront_"
    "luna_three_of_three_resume_v3_20260922"
)
ROOT = REPOSITORY_ROOT / (
    "semantic_buckets/provenance/semantic_weight_two_pass_v4/"
    "ames_endpoint_family_children_luna_high_v1_20260923"
)
MODEL = "openai/gpt-6-luna"
PROVIDER = "openrouter_gpt6_luna_standard_high"
WIDTHS = {"L2": 1, "L3": 1, "L4": 4}


def _sources() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((CANDIDATE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate_unreviewed_unweighted":
        raise ValueError("AMES endpoint-family candidate is not frozen")
    for name, digest in manifest["files"].items():
        if sha256_file(CANDIDATE / name) != digest:
            raise ValueError(f"Candidate artifact changed: {name}")
    old_manifest = json.loads((PREDECESSOR / "manifest.json").read_text(encoding="utf-8"))
    if (old_manifest.get("status") != "complete_unreviewed_candidate"
            or sha256_file(PREDECESSOR / "weights.parquet")
            != manifest["inputs"][str((PREDECESSOR / "weights.parquet").relative_to(REPOSITORY_ROOT))]):
        raise ValueError("Original AMES parent weights changed")
    lineage = pd.read_parquet(CANDIDATE / "bucket_lineage.parquet")
    children = lineage[lineage.parent_semantic_bucket_id.ne(
        lineage.candidate_semantic_bucket_id)].copy()
    parents = pd.read_parquet(PREDECESSOR / "weights.parquet")
    if len(children) != 384 or children.candidate_semantic_bucket_id.duplicated().any():
        raise ValueError("Unexpected new-child universe")
    return children, parents, manifest


def _profile() -> dict[str, Any]:
    return {
        "version": "openai_provider_pool.v1", "max_failovers": 0,
        "providers": [{
            "name": PROVIDER, "base_url": openrouter_provider_pool.OPENROUTER_URL,
            "model": MODEL, "api_key_env": "OPEN_ROUTER_KEY_TWO",
            "max_inflight": 48, "initial_latency_s": 120, "timeout_s": 3600,
            "request_extra_body": {
                "reasoning_effort_override": "high",
                "allowed_served_models": [MODEL],
                "expected_upstream_provider": "OpenAI",
            },
        }],
    }


def _payloads(children: pd.DataFrame) -> dict[str, Any]:
    original = official.TASK_CONFIGS["ames"]["semantic_root"]
    official.configure_task("ames")
    official.TASK_CONFIGS["ames"]["atom_root"] = original
    official.SEMANTIC_ROOT = CANDIDATE
    official.WORLD_ROOT = ROOT
    official.EXPECTED_BUCKET_COUNT = len(children)
    child_cards = official._payloads()
    with gzip.open(PREDECESSOR / "bucket_prompt_payloads.json.gz", "rt", encoding="utf-8") as handle:
        parent_cards = json.load(handle)
    wanted_parents = set(children.parent_semantic_bucket_id)
    if not wanted_parents <= parent_cards.keys() or set(child_cards) != set(
            children.candidate_semantic_bucket_id):
        raise ValueError("Scoring cards omit a child or parent anchor")
    return {**parent_cards, **child_cards}


def prepare() -> dict[str, Any]:
    if ROOT.exists():
        raise FileExistsError(ROOT)
    children, _, candidate = _sources()
    ROOT.mkdir(parents=True)
    children[["level", "candidate_semantic_bucket_id"]].rename(columns={
        "candidate_semantic_bucket_id": "semantic_bucket_id",
    }).to_parquet(ROOT / "semantic_bucket_world.parquet", index=False)
    payloads = _payloads(children)
    schedule = official._pass1_schedule()
    schedule.to_parquet(ROOT / "pass1_schedule.parquet", index=False)
    with gzip.open(ROOT / "bucket_prompt_payloads.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payloads, handle, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    write_json_atomic(ROOT / "provider_profile.json", _profile())
    prompts = {str(level): official._render(
        json.loads(group.iloc[0].candidate_bucket_ids_json), [], str(level), payloads, {})
        for level, group in schedule.groupby("level", sort=True)}
    write_json_atomic(ROOT / "pass1_prompt_review.json", prompts)
    manifest = {
        "schema_version": "ames_endpoint_children_two_pass.v1",
        "status": "prepared_pass1", "task": "ames", "bucket_count": len(children),
        "pass1_request_count": len(schedule), "model": MODEL, "reasoning_effort": "high",
        "pass1_fanout": 1, "pass2_fanout": 3, "pass2_required": 3,
        "candidate_manifest_sha256": sha256_file(CANDIDATE / "manifest.json"),
        "parent_run_manifest_sha256": sha256_file(PREDECESSOR / "manifest.json"),
        "inputs": candidate["inputs"],
        "files": {name: sha256_file(ROOT / name) for name in (
            "semantic_bucket_world.parquet", "pass1_schedule.parquet",
            "bucket_prompt_payloads.json.gz", "provider_profile.json",
            "pass1_prompt_review.json")},
    }
    write_json_atomic(ROOT / "manifest.json", manifest)
    return manifest


def _load() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    _sources()
    if (manifest["candidate_manifest_sha256"] != sha256_file(CANDIDATE / "manifest.json")
            or manifest["parent_run_manifest_sha256"] != sha256_file(PREDECESSOR / "manifest.json")
            or manifest["reasoning_effort"] != "high"):
        raise ValueError("Prepared scoring inputs changed")
    for name, digest in manifest["files"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Prepared run file changed: {name}")
    official.configure_task("ames")
    with gzip.open(ROOT / "bucket_prompt_payloads.json.gz", "rt", encoding="utf-8") as handle:
        return manifest, json.load(handle)


def _client() -> tuple[Any, dict[str, Any]]:
    config = load_provider_pool_config(ROOT / "provider_profile.json")
    checks = preflight_provider_models(config)
    client = official._pool_adapter(config, "high")
    return client, {"model": MODEL, "reasoning_effort": "high",
                    "max_inflight": 48, "preflight": checks,
                    "profile_sha256": sha256_file(ROOT / "provider_profile.json")}


def _schedule_pass2(pass1: pd.DataFrame, children: pd.DataFrame,
                    old_weights: pd.DataFrame) -> pd.DataFrame:
    parents = children.set_index("candidate_semantic_bucket_id").parent_semantic_bucket_id
    scores = pass1.set_index("semantic_bucket_id").weight
    rows = []
    for level, group in pass1.groupby("level", sort=True):
        buckets = group.sort_values(["weight", "semantic_bucket_id"], ascending=[False, True])[
            "semantic_bucket_id"].tolist()
        width = WIDTHS[str(level)]
        for chain in range(width):
            start, end = len(buckets) * chain // width, len(buckets) * (chain + 1) // width
            segment = buckets[start:end]
            batches = [segment[:12], *[segment[i:i + 7] for i in range(12, len(segment), 7)]]
            for wave, batch in enumerate(batches):
                if wave == 0:
                    parent = str(parents[batch[0]])
                    center = float(scores.loc[batch].median())
                    references = old_weights[old_weights.level.eq(level)].copy()
                    references["distance"] = (references.weight - center).abs()
                    nearby = references[references.semantic_bucket_id.ne(parent)].sort_values(
                        ["distance", "semantic_bucket_id"]).semantic_bucket_id.head(3).tolist()
                    anchors = [parent, *nearby]
                else:
                    anchors = batches[wave - 1][-5 if wave == 1 else -2:]
                rows.append({"stage": "pass2", "level": str(level), "wave": wave,
                             "chain": chain, "batch": chain * 1000 + wave,
                             "candidate_bucket_ids_json": core._canonical_json(batch),
                             "anchor_bucket_ids_json": core._canonical_json(anchors),
                             "execution_backend": "luna_standard_high"})
    return pd.DataFrame(rows).sort_values(["level", "wave", "chain"])


async def _pass1_async(connection: sqlite3.Connection, payloads: dict[str, Any],
                       client: Any, profile_hash: str) -> None:
    semaphore = asyncio.Semaphore(32)
    endpoint = {"name": PROVIDER, "base_url": openrouter_provider_pool.OPENROUTER_URL}
    work = []
    for row in pd.read_parquet(ROOT / "pass1_schedule.parquet").to_dict("records"):
        prompt = official._render(json.loads(row["candidate_bucket_ids_json"]), [],
                                  row["level"], payloads, {})
        request_id = official._queue(connection, row, prompt, endpoint,
                                     model=MODEL, reasoning_effort="high")
        work.append(official._execute_one(connection, request_id, endpoint, client,
                                          profile_hash, semaphore, fanout=1,
                                          required=1, model=MODEL))
    outcomes = await asyncio.gather(*work, return_exceptions=True)
    failures = [item for item in outcomes if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError("Pass 1 has incomplete scoring requests") from failures[0]


def run_pass1() -> dict[str, Any]:
    manifest, payloads = _load()
    if manifest["status"] not in {"prepared_pass1", "incomplete_pass1"}:
        raise ValueError("Pass 1 is not pending")
    client, receipt = _client()
    manifest.update(status="running_pass1", pass1_provider_receipt=receipt)
    write_json_atomic(ROOT / "manifest.json", manifest)
    connection = core._request_database(ROOT / "requests.sqlite3", journal_mode="DELETE")
    speculative.ensure_tables(connection)
    try:
        asyncio.run(_pass1_async(connection, payloads, client, receipt["profile_sha256"]))
        scores = official._scores(connection, "pass1")
        children, parents, _ = _sources()
        if len(scores) != len(children) or scores.semantic_bucket_id.duplicated().any():
            raise ValueError("Pass 1 did not cover every new child")
        scores.to_parquet(ROOT / "pass1_weights.parquet", index=False)
        schedule = _schedule_pass2(scores, children, parents)
        schedule.to_parquet(ROOT / "pass2_schedule.parquet", index=False)
        anchors = parents.set_index("semantic_bucket_id")["weight"].to_dict()
        rationales = parents.set_index("semantic_bucket_id")["rationale"].to_dict()
        previews = {}
        for level, group in schedule[schedule.wave.eq(0)].groupby("level"):
            previews[str(level)] = [official._render(
                json.loads(row.candidate_bucket_ids_json),
                json.loads(row.anchor_bucket_ids_json), str(level), payloads,
                {bucket: {"weight": anchors[bucket], "rationale": rationales[bucket]}
                 for bucket in json.loads(row.anchor_bucket_ids_json)})
                for row in group.itertuples(index=False)]
        write_json_atomic(ROOT / "pass2_prompt_review.json", previews)
        review_sha256 = _write_review_manifest()
        manifest.update(status="awaiting_pass2_prompt_review",
                        pass2_request_count=len(schedule),
                        pass2_schedule_sha256=sha256_file(ROOT / "pass2_schedule.parquet"),
                        pass2_prompt_review_sha256=sha256_file(ROOT / "pass2_prompt_review.json"),
                        pass2_review_manifest_sha256=review_sha256)
    except Exception:
        manifest["status"] = "incomplete_pass1"
        raise
    finally:
        connection.close()
        write_json_atomic(ROOT / "manifest.json", manifest)
    return manifest


def _write_review_manifest() -> str:
    review = {
        "version": "ames_endpoint_children_pass2_prompt_review.v1",
        "status": "ready", "completion_requests_made": 0,
        "pass1_weights_sha256": sha256_file(ROOT / "pass1_weights.parquet"),
        "pass2_schedule_sha256": sha256_file(ROOT / "pass2_schedule.parquet"),
        "rendered_prompts_sha256": sha256_file(ROOT / "pass2_prompt_review.json"),
        "levels": {level: len(prompts) for level, prompts in json.loads(
            (ROOT / "pass2_prompt_review.json").read_text(encoding="utf-8")).items()},
    }
    write_json_atomic(ROOT / "pass2_prompt_review_manifest.json", review)
    return sha256_file(ROOT / "pass2_prompt_review_manifest.json")


async def _chain(rows: pd.DataFrame, connection: sqlite3.Connection,
                 payloads: dict[str, Any], client: Any, profile_hash: str,
                 semaphore: asyncio.Semaphore, parent_scores: dict[str, dict]) -> None:
    endpoint = {"name": PROVIDER, "base_url": openrouter_provider_pool.OPENROUTER_URL}
    for row in rows.sort_values("wave").to_dict("records"):
        existing = official._scores(connection, "pass2")
        scores = {**parent_scores, **(existing.set_index("semantic_bucket_id").to_dict("index")
                                     if len(existing) else {})}
        candidates = json.loads(row["candidate_bucket_ids_json"])
        anchors = json.loads(row["anchor_bucket_ids_json"])
        if not set(anchors) <= scores.keys():
            raise ValueError(f"Pass-2 chain anchor is not complete: {row['level']}/{row['chain']}")
        prompt = official._render(candidates, anchors, row["level"], payloads, scores)
        request_id = official._queue(connection, row, prompt, endpoint,
                                     model=MODEL, reasoning_effort="high")
        await official._execute_one(connection, request_id, endpoint, client,
                                    profile_hash, semaphore, fanout=3,
                                    required=3, model=MODEL)


def run_pass2() -> dict[str, Any]:
    manifest, payloads = _load()
    review = json.loads((ROOT / "pass2_prompt_review_manifest.json").read_text(encoding="utf-8"))
    if (manifest["status"] not in {"awaiting_pass2_prompt_review", "incomplete_pass2"}
            or manifest["pass2_review_manifest_sha256"]
            != sha256_file(ROOT / "pass2_prompt_review_manifest.json")
            or manifest["pass2_prompt_review_sha256"]
            != sha256_file(ROOT / "pass2_prompt_review.json")
            or review["pass1_weights_sha256"] != sha256_file(ROOT / "pass1_weights.parquet")
            or review["pass2_schedule_sha256"] != sha256_file(ROOT / "pass2_schedule.parquet")
            or review["rendered_prompts_sha256"] != manifest["pass2_prompt_review_sha256"]):
        raise ValueError("Frozen Pass-2 prompt review changed")
    if sha256_file(ROOT / "pass2_schedule.parquet") != manifest["pass2_schedule_sha256"]:
        raise ValueError("Frozen Pass-2 schedule changed")
    client, receipt = _client()
    manifest.update(status="running_pass2", pass2_provider_receipt=receipt)
    write_json_atomic(ROOT / "manifest.json", manifest)
    connection = core._request_database(ROOT / "requests.sqlite3", journal_mode="DELETE")
    speculative.ensure_tables(connection)
    parents = pd.read_parquet(PREDECESSOR / "weights.parquet")
    parent_scores = parents.set_index("semantic_bucket_id")[["weight", "rationale"]].to_dict("index")
    schedule = pd.read_parquet(ROOT / "pass2_schedule.parquet")
    async def execute() -> None:
        semaphore = asyncio.Semaphore(12)
        outcomes = await asyncio.gather(*(
            _chain(rows, connection, payloads, client, receipt["profile_sha256"],
                   semaphore, parent_scores)
            for _, rows in schedule.groupby(["level", "chain"])), return_exceptions=True)
        failures = [item for item in outcomes if isinstance(item, BaseException)]
        if failures:
            raise RuntimeError("One or more Pass-2 chains failed") from failures[0]
    try:
        asyncio.run(execute())
        scores = official._scores(connection, "pass2")
        children, _, _ = _sources()
        if len(scores) != len(children) or scores.semantic_bucket_id.duplicated().any():
            raise ValueError("Pass 2 did not cover every new child")
        scores.to_parquet(ROOT / "child_weights.parquet", index=False)
        manifest.update(status="complete_unreviewed_candidate",
                        child_weights_sha256=sha256_file(ROOT / "child_weights.parquet"))
    except Exception:
        manifest["status"] = "incomplete_pass2"
        raise
    finally:
        connection.close()
        write_json_atomic(ROOT / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run-pass1", "run-pass2"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "run-pass1":
        result = run_pass1()
    else:
        result = run_pass2()
    print(json.dumps({"status": result["status"], "run": str(ROOT)}, sort_keys=True))


if __name__ == "__main__":
    main()
