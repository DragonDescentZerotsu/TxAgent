"""Assign official two-pass weights to a frozen semantic-bucket retrieval world."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

from jinja2 import Environment, StrictUndefined
import pandas as pd

from data.processing.llm_api import async_openai_compatible_client
from semantic_buckets import bioavailability_semantic_readout_v1 as core
from semantic_buckets import sliding_weight_assignment as display
from semantic_buckets import speculative_weight_execution as speculative
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "semantic_weight_official_two_pass.v1"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
TARGET = "skin sensitization or allergic contact dermatitis"
MAX_TOKENS = 20_480
FANOUT = 16
REQUIRED = 4
MAX_LOGICAL_INFLIGHT = 128
ENDPOINTS = (
    {"name": "dgx005_50001", "base_url": "http://dgx005:50001/v1"},
    {"name": "dgx011_50001", "base_url": "http://dgx011:50001/v1"},
    {"name": "dgx020_50002", "base_url": "http://dgx020:50002/v1"},
)
SOURCE_LABELS = {
    "direct_skin_reaction": "Direct skin sensitization/contact-allergy evidence",
    "sensitization_aop": "Skin-sensitization AOP and mechanistic evidence",
    "skin_exposure": "Dermal exposure evidence",
}
ROOT = Path(__file__).resolve().parent
SEMANTIC_ROOT = (
    ROOT / "provenance/source_local_semantic_v4/"
    "skin_main_universe_v5_semantic_parent_v1_20260919/semantic_run"
)
WORLD_ROOT = ROOT / "provenance/retrieval_worlds/skin_morgan_top100_valid_test_l2_l3_v1"
PROMPT_ROOT = ROOT / "prompts/semantic_weight_sliding_v2"
OUTPUT_ROOT = ROOT / "provenance/semantic_weight_two_pass_v1"
RUN_ID = "skin_morgan100_official_two_pass_v1_20260919"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def _paths() -> dict[str, Path]:
    return {
        "semantic_manifest": SEMANTIC_ROOT / "semantic_bucket_map_manifest.json",
        "semantic_map": SEMANTIC_ROOT / "semantic_bucket_map.parquet",
        "input_atoms": SEMANTIC_ROOT / "input_atoms.parquet",
        "sample_cards": SEMANTIC_ROOT / "sample_cards.json.gz",
        "world_manifest": WORLD_ROOT / "manifest.json",
        "world": WORLD_ROOT / "semantic_bucket_world.parquet",
        "initial_prompt": PROMPT_ROOT / "initial.jinja",
        "anchored_prompt": PROMPT_ROOT / "anchored.jinja",
    }


def _verify_inputs() -> dict[str, dict[str, str]]:
    paths = _paths()
    semantic = json.loads(paths["semantic_manifest"].read_text(encoding="utf-8"))
    world = json.loads(paths["world_manifest"].read_text(encoding="utf-8"))
    if semantic.get("status") != "complete_reviewed" or semantic.get("task") != "skin_reaction":
        raise ValueError("Skin semantic generation is not reviewed")
    if world.get("status") != "complete" or world.get("counts", {}).get("buckets") != 1244:
        raise ValueError("Skin Morgan-top-100 world is incomplete")
    hashes = {name: {"path": str(path), "sha256": _sha256(path)}
              for name, path in paths.items()}
    if hashes["semantic_map"]["sha256"] != semantic["semantic_bucket_map_sha256"]:
        raise ValueError("reviewed semantic map changed")
    if hashes["input_atoms"]["sha256"] != semantic["input_atoms_sha256"]:
        raise ValueError("reviewed input atoms changed")
    if hashes["world"]["sha256"] != world["files"]["semantic_bucket_world.parquet"]:
        raise ValueError("retrieval world changed")
    return hashes


def _sample_atoms(bucket: str, atoms: pd.DataFrame) -> list[str]:
    ordered = sorted(
        atoms.atom_id.astype(str),
        key=lambda atom: (core._stable_id("weight-sample", bucket, atom), atom),
    )
    sources = dict(zip(atoms.atom_id.astype(str), atoms.source_id.astype(str)))
    selected: list[str] = []
    seen: set[str] = set()
    for new_source_only in (True, False):
        for atom in ordered:
            source = sources[atom]
            if atom in selected or (new_source_only and source in seen):
                continue
            selected.append(atom)
            seen.add(source)
            if len(selected) == 5:
                return selected
    return selected


def _component(source: str, rows: pd.DataFrame) -> dict[str, Any]:
    values = [json.loads(value) for value in rows.values_json]
    fields = sorted({field for item in values for field in item})
    dimensions = {
        field: sorted({str(item.get(field, "__unknown__")) for item in values})
        for field in fields
    }
    return {"source_id": SOURCE_LABELS[source], "canonical_dimensions": dimensions}


def _payloads() -> dict[str, dict[str, Any]]:
    world = pd.read_parquet(_paths()["world"])
    wanted = set(world.semantic_bucket_id.astype(str))
    mapping = pd.read_parquet(_paths()["semantic_map"])
    mapping = mapping[mapping.semantic_bucket_id.astype(str).isin(wanted)]
    atoms = pd.read_parquet(_paths()["input_atoms"])
    atoms = atoms.set_index("atom_id", drop=False)
    with gzip.open(_paths()["sample_cards"], "rt", encoding="utf-8") as handle:
        cards = json.load(handle)
    payloads: dict[str, dict[str, Any]] = {}
    for bucket, members in mapping.groupby("semantic_bucket_id", sort=True):
        rows = atoms.loc[members.atom_id.astype(str)]
        components = [_component(str(source), group) for source, group in rows.groupby("source_id")]
        selected = _sample_atoms(str(bucket), rows)
        samples = [
            {"source_id": SOURCE_LABELS[str(atoms.loc[atom, "source_id"])],
             **{field: value for field, value in cards[atom].items()
                if field != "canonical_record_id"}}
            for atom in selected
        ]
        payloads[str(bucket)] = {
            "identity": {"source_components": components}, "sample_records": samples,
        }
    if set(payloads) != wanted:
        raise ValueError("semantic prompt payload coverage differs from retrieval world")
    return payloads


def _render(candidates: Sequence[str], anchors: Sequence[str], level: str,
            payloads: Mapping[str, Mapping[str, Any]],
            scores: Mapping[str, Mapping[str, Any]]) -> str:
    candidate_cards = "\n\n".join(
        display._card(f"Candidate {index}", payloads[bucket])
        for index, bucket in enumerate(candidates, 1)
    )
    anchor_cards = "\n\n".join(
        display._anchor_card(index, bucket, payloads[bucket], scores[bucket])
        for index, bucket in enumerate(anchors, 1)
    )
    name = "anchored.jinja" if anchors else "initial.jinja"
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        (PROMPT_ROOT / name).read_text(encoding="utf-8")
    )
    return template.render(
        target=TARGET, level=level, candidate_cards=candidate_cards,
        anchor_cards=anchor_cards,
    ).strip()


def _pass1_schedule() -> pd.DataFrame:
    world = pd.read_parquet(_paths()["world"])
    rows = []
    for level, group in world.groupby("level", sort=True):
        buckets = sorted(group.semantic_bucket_id.astype(str))
        for batch, start in enumerate(range(0, len(buckets), 12)):
            rows.append({"stage": "pass1", "level": str(level), "batch": batch,
                         "candidate_bucket_ids_json": core._canonical_json(buckets[start:start + 12]),
                         "anchor_bucket_ids_json": "[]"})
    return pd.DataFrame(rows)


def _pass2_schedule(pass1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, group in pass1.groupby("level", sort=True):
        ordered = group.sort_values(["weight", "semantic_bucket_id"], ascending=[False, True])
        buckets = ordered.semantic_bucket_id.astype(str).tolist()
        rows.append({"stage": "pass2", "level": level, "batch": 0,
                     "candidate_bucket_ids_json": core._canonical_json(buckets[:12]),
                     "anchor_bucket_ids_json": "[]"})
        anchors = buckets[7:12]
        for batch, start in enumerate(range(12, len(buckets), 7), 1):
            candidates = buckets[start:start + 7]
            rows.append({"stage": "pass2", "level": level, "batch": batch,
                         "candidate_bucket_ids_json": core._canonical_json(candidates),
                         "anchor_bucket_ids_json": core._canonical_json(anchors)})
            anchors = candidates[-2:]
    return pd.DataFrame(rows)


def _preview(anchors: Sequence[str]) -> dict[str, dict[str, str]]:
    return {bucket: {"weight": "<locked prior score>",
                     "rationale": "<locked prior rationale>"} for bucket in anchors}


def _write_review(root: Path, schedule: pd.DataFrame,
                  payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    review = root / "prompt_review"
    review.mkdir()
    files = []
    for level, group in schedule.groupby("level", sort=True):
        first = group.iloc[0]
        candidates = json.loads(first.candidate_bucket_ids_json)
        initial = _render(candidates, [], str(level), payloads, {})
        anchors, anchored_candidates = candidates[-5:], candidates[:7]
        anchored = _render(anchored_candidates, anchors, str(level), payloads, _preview(anchors))
        for shape, prompt in (("initial", initial), ("anchored", anchored)):
            path = review / f"{level}_{shape}.txt"
            path.write_text(prompt + "\n", encoding="utf-8")
            files.append({"path": path.name, "sha256": _sha256(path),
                          "characters": len(prompt)})
    manifest = {"version": f"{VERSION}.prompt_review", "status": "ready",
                "completion_requests_made": 0, "files": files}
    write_json_atomic(review / "manifest.json", manifest)
    return manifest


def prepare(run_id: str = RUN_ID) -> dict[str, Any]:
    root = OUTPUT_ROOT / run_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    inputs = _verify_inputs()
    payloads = _payloads()
    schedule = _pass1_schedule()
    _atomic_frame(schedule, root / "pass1_schedule.parquet")
    _atomic_frame(schedule, root / "pass1_schedule.tsv")
    with gzip.open(root / "bucket_prompt_payloads.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payloads, handle, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    review = _write_review(root, schedule, payloads)
    profile = {"model": MODEL, "reasoning_effort": "high", "max_tokens": MAX_TOKENS,
               "fanout": FANOUT, "required_valid_responses": REQUIRED,
               "aggregation": speculative.aggregation_method(REQUIRED),
               "max_logical_inflight_per_endpoint": MAX_LOGICAL_INFLIGHT,
               "endpoints": ENDPOINTS}
    write_json_atomic(root / "execution_profile.json", profile)
    manifest = {"version": VERSION, "status": "awaiting_prompt_review", "run_id": run_id,
                "created_at": _now(), "task": "skin_reaction", "levels": ["L2", "L3"],
                "bucket_count": len(payloads), "pass1_request_count": len(schedule),
                "inputs": inputs, "review": review,
                "files": {name: _sha256(root / name) for name in
                          ("pass1_schedule.parquet", "pass1_schedule.tsv",
                           "bucket_prompt_payloads.json.gz", "execution_profile.json")}}
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _load_run(run_id: str, review_hash: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = OUTPUT_ROOT / run_id
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("run version changed")
    if _sha256(root / "prompt_review/manifest.json") != review_hash:
        raise ValueError("approved prompt-review hash changed")
    _verify_inputs()
    for name, digest in manifest["files"].items():
        if _sha256(root / name) != digest:
            raise ValueError(f"prepared file changed: {name}")
    with gzip.open(root / "bucket_prompt_payloads.json.gz", "rt", encoding="utf-8") as handle:
        payloads = json.load(handle)
    return root, manifest, payloads


def _queue(connection: sqlite3.Connection, row: Mapping[str, Any], prompt: str,
           endpoint: Mapping[str, str]) -> str:
    candidates = json.loads(row["candidate_bucket_ids_json"])
    anchors = json.loads(row["anchor_bucket_ids_json"])
    phase = f"{row['stage']}/{row['level']}/{int(row['batch']):04d}"
    request_id = core._request_id("weight_assignment", phase, prompt)
    validation = {"candidate_aliases": [f"Candidate {i}" for i in range(1, len(candidates) + 1)],
                  "candidate_bucket_ids": candidates, "anchor_bucket_ids": anchors,
                  "requested_model": MODEL, "selected_endpoint": endpoint["name"]}
    return core._queue_request(
        connection, request_id=request_id, kind="weight_assignment", phase=phase,
        prompt=prompt, reasoning_effort="high", max_tokens=MAX_TOKENS,
        validation=validation,
    )


async def _clients() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clients, receipts = {}, []
    for endpoint in ENDPOINTS:
        client, credential = async_openai_compatible_client(
            base_url=endpoint["base_url"], provider="local", env_file=None,
            max_connections=MAX_LOGICAL_INFLIGHT * FANOUT, timeout_s=3600, max_retries=0,
        )
        if credential:
            raise ValueError("local endpoint unexpectedly selected a credential")
        models = sorted(model.id for model in (await client.models.list()).data)
        if models != [MODEL]:
            await client.close()
            raise ValueError(f"endpoint model changed: {endpoint['name']}={models}")
        clients[endpoint["name"]] = client
        receipts.append({**endpoint, "advertised_models": models,
                         "max_inflight": MAX_LOGICAL_INFLIGHT})
    return clients, receipts


async def _execute_one(connection: sqlite3.Connection, request_id: str,
                       endpoint: Mapping[str, str], client: Any,
                       profile_hash: str, semaphore: asyncio.Semaphore) -> None:
    row = speculative._pending_rows(connection, [request_id])
    if not row:
        return
    async with semaphore:
        result = await speculative.execute_request(
            client, row[0], model=MODEL, initial_fanout=FANOUT,
            hedge_seconds=3600, maximum=FANOUT, required=REQUIRED,
        )
    speculative.persist_result(
        connection, result, base_url=endpoint["base_url"], model=MODEL,
        benchmark_sha256=profile_hash,
    )
    if result["status"] != "complete":
        raise RuntimeError(f"request failed: {request_id}: {result['error']}")


def _scores(connection: sqlite3.Connection, stage: str) -> pd.DataFrame:
    aggregates = speculative.aggregate_lookup(connection)
    rows = []
    query = connection.execute(
        "SELECT * FROM requests WHERE status='complete' AND phase LIKE ? ORDER BY phase",
        (f"{stage}/%",),
    )
    for request in query:
        validation, response = json.loads(request["validation_json"]), json.loads(request["response_json"])
        buckets = dict(zip(validation["candidate_aliases"], validation["candidate_bucket_ids"], strict=True))
        aggregate = aggregates[request["request_id"]]
        for score in response["scores"]:
            component = aggregate["components"][score["candidate"]]
            rows.append({"task": "skin_reaction", "level": request["phase"].split("/")[1],
                         "semantic_bucket_id": buckets[score["candidate"]],
                         "weight": float(score["weight"]), "rationale": score["rationale"],
                         "request_id": request["request_id"],
                         "anchor_bucket_ids_json": core._canonical_json(validation["anchor_bucket_ids"]),
                         "endpoint": validation["selected_endpoint"],
                         "weight_stddev": component["sample_stddev"],
                         "weight_min": component["min"], "weight_max": component["max"],
                         "component_weights_json": core._canonical_json(component["component_weights"])})
    return pd.DataFrame(rows)


async def _run_pass1(connection: sqlite3.Connection, root: Path,
                     payloads: Mapping[str, Any], clients: Mapping[str, Any],
                     profile_hash: str) -> pd.DataFrame:
    schedule = pd.read_parquet(root / "pass1_schedule.parquet")
    semaphores = {endpoint["name"]: asyncio.Semaphore(MAX_LOGICAL_INFLIGHT)
                  for endpoint in ENDPOINTS}
    work = []
    for index, row in enumerate(schedule.to_dict("records")):
        endpoint = ENDPOINTS[index % len(ENDPOINTS)]
        candidates = json.loads(row["candidate_bucket_ids_json"])
        prompt = _render(candidates, [], row["level"], payloads, {})
        request_id = _queue(connection, row, prompt, endpoint)
        work.append(_execute_one(connection, request_id, endpoint, clients[endpoint["name"]],
                                 profile_hash, semaphores[endpoint["name"]]))
    await asyncio.gather(*work)
    scores = _scores(connection, "pass1")
    if len(scores) != 1244 or scores.semantic_bucket_id.duplicated().any():
        raise ValueError(f"pass one incomplete: {len(scores)}/1244")
    _atomic_frame(scores, root / "pass1_weights.parquet")
    _atomic_frame(scores, root / "pass1_weights.tsv")
    return scores


async def _run_chain(level_rows: pd.DataFrame, connection: sqlite3.Connection,
                     payloads: Mapping[str, Any], clients: Mapping[str, Any],
                     profile_hash: str, level_offset: int) -> None:
    for position, row in enumerate(level_rows.sort_values("batch").to_dict("records")):
        existing = _scores(connection, "pass2")
        score_map = (existing.set_index("semantic_bucket_id").to_dict("index")
                     if len(existing) else {})
        candidates = json.loads(row["candidate_bucket_ids_json"])
        if set(candidates) <= score_map.keys():
            continue
        anchors = json.loads(row["anchor_bucket_ids_json"])
        if not set(anchors) <= score_map.keys():
            raise ValueError(f"pass-two anchors are incomplete: {row['level']}/{row['batch']}")
        endpoint = ENDPOINTS[(position + level_offset) % len(ENDPOINTS)]
        prompt = _render(candidates, anchors, row["level"], payloads, score_map)
        request_id = _queue(connection, row, prompt, endpoint)
        await _execute_one(connection, request_id, endpoint, clients[endpoint["name"]],
                           profile_hash, asyncio.Semaphore(1))


async def _run_async(connection: sqlite3.Connection, root: Path,
                     payloads: Mapping[str, Any], manifest: dict[str, Any]) -> None:
    clients, preflight = await _clients()
    profile_hash = _sha256(root / "execution_profile.json")
    manifest["endpoint_preflight"] = preflight
    write_json_atomic(root / "manifest.json", manifest)
    try:
        pass1 = await _run_pass1(connection, root, payloads, clients, profile_hash)
        schedule = _pass2_schedule(pass1)
        _atomic_frame(schedule, root / "pass2_schedule.parquet")
        _atomic_frame(schedule, root / "pass2_schedule.tsv")
        manifest["pass2_request_count"] = len(schedule)
        manifest["pass2_schedule_sha256"] = _sha256(root / "pass2_schedule.parquet")
        write_json_atomic(root / "manifest.json", manifest)
        groups = list(schedule.groupby("level", sort=True))
        await asyncio.gather(*(
            _run_chain(rows, connection, payloads, clients, profile_hash, index)
            for index, (_, rows) in enumerate(groups)
        ))
    finally:
        await asyncio.gather(*(client.close() for client in clients.values()))


def _publish(root: Path, connection: sqlite3.Connection,
             manifest: dict[str, Any]) -> dict[str, Any]:
    pass1, final = _scores(connection, "pass1"), _scores(connection, "pass2")
    if len(pass1) != 1244 or len(final) != 1244 or final.semantic_bucket_id.duplicated().any():
        raise ValueError(f"two-pass result incomplete: pass1={len(pass1)}, pass2={len(final)}")
    order = pass1.sort_values(["level", "weight", "semantic_bucket_id"], ascending=[True, False, True])
    order["pass1_order_rank"] = order.groupby("level").cumcount() + 1
    final = final.merge(order[["semantic_bucket_id", "pass1_order_rank"]], on="semantic_bucket_id")
    final = final.sort_values(["level", "weight", "semantic_bucket_id"], ascending=[True, False, True])
    final["final_rank"] = final.groupby("level").cumcount() + 1
    _atomic_frame(final, root / "weights.parquet")
    _atomic_frame(final, root / "weights.tsv")
    ranking = final[["task", "level", "semantic_bucket_id", "final_rank", "weight",
                     "rationale", "pass1_order_rank"]]
    _atomic_frame(ranking, root / "semantic_bucket_rankings.parquet")
    _atomic_frame(ranking, root / "semantic_bucket_rankings.tsv")
    summary = final.groupby("level").weight.agg(["count", "min", "mean", "max"]).reset_index()
    _atomic_frame(summary, root / "summary.tsv")
    manifest.update(status="complete_unselected", completed_at=_now(), completed_bucket_count=len(final),
                    active_policy_changed=False,
                    artifacts={name: _sha256(root / name) for name in
                               ("pass1_weights.parquet", "pass1_weights.tsv", "weights.parquet",
                                "weights.tsv", "semantic_bucket_rankings.parquet",
                                "semantic_bucket_rankings.tsv", "summary.tsv")})
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def run(run_id: str, review_hash: str) -> dict[str, Any]:
    if os.environ.get("DEEPSEEK_API_KEY") != "EMPTY":
        raise ValueError("set DEEPSEEK_API_KEY=EMPTY for local DGX execution")
    root, manifest, payloads = _load_run(run_id, review_hash)
    manifest.update(status="running", started_at=manifest.get("started_at", _now()))
    write_json_atomic(root / "manifest.json", manifest)
    connection = core._request_database(root / "requests.sqlite3", journal_mode="DELETE")
    speculative.ensure_tables(connection)
    try:
        asyncio.run(_run_async(connection, root, payloads, manifest))
        return _publish(root, connection, manifest)
    except Exception:
        manifest["status"] = "incomplete"
        write_json_atomic(root / "manifest.json", manifest)
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--approved-review-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run_id)
    else:
        if not args.approved_review_sha256:
            parser.error("run requires --approved-review-sha256")
        result = run(args.run_id, args.approved_review_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
