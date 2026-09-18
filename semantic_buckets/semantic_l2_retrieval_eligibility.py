"""Classify every V10 BBB/oral L2-L4 semantic bucket for retrieval eligibility."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from jinja2 import Environment, StrictUndefined
import pandas as pd

from semantic_buckets import bioavailability_semantic_degree25 as degree
from semantic_buckets import source_local_semantic_v3 as semantic
from semantic_buckets.artifacts import eligibility_root
from predict.api_client.pool import load_provider_pool_config, select_healthy_providers
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "semantic_indirect_retrieval_eligibility.v2"
TEMPLATE = Path(__file__).with_name("prompts") / "semantic_indirect_retrieval_eligibility_v2.jinja"
WEIGHTS = Path(__file__).resolve().parent / "policies/semantic_weighted_top10_v2.json"
LEVELS = ("L2", "L3", "L4")
BUCKET_COUNTS = {
    "bbb_martins": {"L2": 8_454, "L3": 533, "L4": 4_091},
    "bioavailability_ma": {"L2": 133, "L3": 4_014, "L4": 1_651},
}
RECORD_COUNTS = {
    "bbb_martins": {"L2": 267_544, "L3": 14_693, "L4": 164_050},
    "bioavailability_ma": {"L2": 152_258, "L3": 104_174, "L4": 73_912},
}
TARGETS = {
    "bbb_martins": "whether systemic administration produces meaningful CNS access",
    "bioavailability_ma": "oral bioavailability under the reported condition",
}
REASON_CODES = (
    "potentially_informative",
    "undefined_reference_or_target",
    "malformed_or_uninterpretable_measurement",
    "wrong_property_or_route",
    "non_compound_specific_or_nonexperimental",
    "other_never_transferable",
)


def _sha256(path: Path) -> str:
    return semantic._sha256(path)


def _configure(task: str) -> tuple[dict[str, Any], Any]:
    if task not in TARGETS:
        raise ValueError(f"unsupported task: {task}")
    spec = semantic.configure_rankings(task)
    return spec, degree.semantic


def _prior_zero_buckets(task: str, level: str) -> set[str]:
    document = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    weights = document["tasks"][task]["levels"][level]
    return {
        str(bucket)
        for bucket, weight in zip(weights["bucket_ids"], weights["weights"], strict=True)
        if float(weight) == 0
    }


def _load_inputs(task: str) -> tuple[dict[str, Any], Any, pd.DataFrame, pd.DataFrame]:
    spec, core = _configure(task)
    ranking_root = Path(spec["ranking_root"])
    ranking_manifest = json.loads((ranking_root / "manifest.json").read_text())
    map_manifest = json.loads(degree.FINAL_MAP_MANIFEST.read_text())
    if ranking_manifest.get("status") != "complete" or map_manifest.get("status") != "complete_reviewed":
        raise ValueError("complete reviewed V10 semantic rankings are required")
    mapping = pd.read_parquet(degree.FINAL_MAP)
    mapping = mapping[mapping.level.isin(LEVELS)][["level", "atom_id", "semantic_bucket_id"]]
    rankings = pd.read_parquet(ranking_root / "semantic_bucket_rankings.parquet")
    rankings = rankings[rankings.level.isin(LEVELS)].copy()
    for level in LEVELS:
        mapped = mapping[mapping.level.eq(level)].semantic_bucket_id
        ranked = rankings[rankings.level.eq(level)].semantic_bucket_id
        if mapped.nunique() != BUCKET_COUNTS[task][level] or len(ranked) != BUCKET_COUNTS[task][level]:
            raise ValueError(f"V10 {level} semantic bucket count changed")
        if set(mapped) != set(ranked):
            raise ValueError(f"{level} semantic map and ranking bucket IDs disagree")
    return spec, core, mapping, rankings


def _payloads(core: Any, mapping: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    atoms = pd.read_parquet(degree.INPUT_ROOT / "input_atoms.parquet")
    cards = core._read_gzip_json(degree.INPUT_ROOT / "sample_cards.json.gz")
    lookup = core._atom_lookup(atoms)
    grouped = mapping.groupby(["level", "semantic_bucket_id"], sort=True)["atom_id"].apply(list)
    return {
        (level, bucket): degree._bucket_payload(bucket, members, lookup, cards)
        for (level, bucket), members in grouped.items()
    }


def _prompt(task: str, level: str, payload: dict[str, Any]) -> str:
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        TEMPLATE.read_text(encoding="utf-8")
    )
    return template.render(
        target=TARGETS[task], level=level,
        compact_payload_json=semantic._canonical_json(payload)
    ).strip()


def _input_hashes(spec: dict[str, Any]) -> dict[str, str]:
    paths = [
        degree.FINAL_MAP,
        degree.FINAL_MAP_MANIFEST,
        degree.INPUT_ROOT / "input_atoms.parquet",
        degree.INPUT_ROOT / "sample_cards.json.gz",
        Path(spec["ranking_root"]) / "manifest.json",
        Path(spec["ranking_root"]) / "semantic_bucket_rankings.parquet",
        Path(spec["ranking_root"]) / "record_relevance_rankings.parquet",
        TEMPLATE,
        WEIGHTS,
    ]
    return {str(path.resolve()): _sha256(path) for path in paths}


def _l2_inputs(task: str) -> tuple[pd.DataFrame, dict[str, str]]:
    root = eligibility_root(task, "semantic_l2_retrieval_eligibility_v1")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    decisions_path = root / "bucket_decisions.parquet"
    if (manifest.get("status") != "complete_single_pass"
            or _sha256(decisions_path) != manifest["artifacts"]["bucket_decisions"]["sha256"]):
        raise ValueError(f"complete frozen L2 eligibility artifact required: {task}")
    decisions = pd.read_parquet(decisions_path)
    return decisions, {
        str(manifest_path.resolve()): _sha256(manifest_path),
        str(decisions_path.resolve()): _sha256(decisions_path),
        str((root / "requests.sqlite3").resolve()): _sha256(root / "requests.sqlite3"),
    }


def build(task: str, output: Path | None = None) -> dict[str, Any]:
    spec, core, mapping, rankings = _load_inputs(task)
    l2_decisions, l2_inputs = _l2_inputs(task)
    root = (
        output
        or eligibility_root(task, "semantic_indirect_retrieval_eligibility_v2")
    ).resolve()
    if (root / "manifest.json").is_file():
        return json.loads((root / "manifest.json").read_text())
    root.mkdir(parents=True)
    payloads = _payloads(core, mapping)
    connection = core._request_database(root / "requests.sqlite3")
    schedule = []
    for level, bucket in sorted(payloads):
        if level == "L2":
            continue
        prompt = _prompt(task, level, payloads[(level, bucket)])
        phase = f"{task}|{level}"
        request_id = core._request_id("retrieval_eligibility", phase, prompt)
        core._queue_request(connection, request_id=request_id, kind="retrieval_eligibility",
            phase=phase, prompt=prompt, reasoning_effort="high", max_tokens=core.LOW_MAX_TOKENS,
            validation={"reason_codes": list(REASON_CODES)}, commit=False)
        schedule.append({"task_id": task, "level": level, "semantic_bucket_id": bucket,
                         "request_id": request_id,
                         "prior_zero_weight": bucket in _prior_zero_buckets(task, level)})
    connection.commit()
    connection.close()
    pd.DataFrame(schedule).to_parquet(root / "review_schedule.parquet", index=False)
    manifest = {"version": VERSION, "status": "prepared", "task_id": task,
        "levels": list(LEVELS), "logical_reviews": len(schedule),
        "reused_l2_reviews": len(l2_decisions), "review_passes": 1, "model": core.MODEL,
        "reasoning_effort": "high", "inputs": {**_input_hashes(spec), **l2_inputs},
        "review_schedule_sha256": _sha256(root / "review_schedule.parquet")}
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _verify_inputs(manifest: dict[str, Any]) -> None:
    for path, expected in manifest["inputs"].items():
        if _sha256(Path(path)) != expected:
            raise ValueError(f"eligibility input changed: {path}")


def _completed(connection: sqlite3.Connection, request_ids: list[str]) -> dict[str, sqlite3.Row]:
    wanted = set(request_ids)
    rows = connection.execute("SELECT * FROM requests WHERE status='complete'")
    found = {str(row["request_id"]): row for row in rows if row["request_id"] in wanted}
    if set(found) != wanted:
        raise ValueError(f"eligibility reviews incomplete: {len(wanted) - len(found)} missing")
    return found


def _decision_frame(schedule: pd.DataFrame, requests: dict[str, sqlite3.Row]) -> pd.DataFrame:
    rows = []
    for item in schedule.itertuples(index=False):
        request = requests[item.request_id]
        response = json.loads(request["response_json"])
        eligible = response["decision"] == "include" and not item.prior_zero_weight
        rows.append({**item._asdict(), **response, "retrieval_eligible": eligible,
                     "served_model": request["served_model"], "attempts": request["attempts"],
                     "input_tokens": request["input_tokens"], "output_tokens": request["output_tokens"]})
    return pd.DataFrame(rows).sort_values("semantic_bucket_id")


def _select_provider_endpoints(path: Path, per_endpoint: int, model: str) -> tuple[tuple, dict]:
    config = load_provider_pool_config(path)
    if any(provider.model != model for provider in config.providers):
        raise ValueError(f"provider inventory must serve only {model}")
    if any(provider.api_key_env for provider in config.providers):
        raise ValueError("semantic eligibility supports credential-free local endpoints only")
    capped = replace(
        config,
        providers=tuple(
            replace(provider, max_inflight=per_endpoint)
            for provider in config.providers
        ),
    )
    selection = select_healthy_providers(capped, per_endpoint * len(capped.providers))
    endpoints = tuple({
        "name": provider.name,
        "base_url": provider.base_url,
        "provider": "local",
        "credential_env": "",
        "max_inflight": per_endpoint,
    } for provider in selection.config.providers)
    return endpoints, selection.public_dict()


def _record_frame(task: str, decisions: pd.DataFrame) -> pd.DataFrame:
    rankings = pd.read_parquet(Path(semantic.configure_rankings(task)["ranking_root"])
                               / "record_relevance_rankings.parquet")
    rankings = rankings[rankings.level.isin(LEVELS)].copy()
    columns = ["level", "semantic_bucket_id", "decision", "reason_code", "retrieval_eligible"]
    records = rankings.merge(
        decisions[columns], on=["level", "semantic_bucket_id"], validate="many_to_one"
    )
    counts = records.level.value_counts().to_dict()
    if counts != RECORD_COUNTS[task] or records.retrieval_eligible.isna().any():
        raise ValueError("eligibility decisions do not cover all V10 L2-L4 records")
    return records


def _write_report(root: Path, task: str, decisions: pd.DataFrame, records: pd.DataFrame) -> None:
    summary = decisions.groupby(
        ["level", "decision", "retrieval_eligible"], dropna=False
    ).size().reset_index(name="buckets")
    summary.to_csv(root / "summary.tsv", sep="\t", index=False)
    excluded = decisions[~decisions.retrieval_eligible]
    excluded.to_csv(root / "excluded_buckets.tsv", sep="\t", index=False)
    reasons = excluded.groupby(["level", "reason_code"]).size().sort_values(ascending=False)
    lines = [f"# {task} L2-L4 retrieval eligibility", "",
             f"Classified {len(decisions):,} semantic buckets once; L2 decisions were reused.",
             f"Excluded {len(excluded):,} buckets covering {(~records.retrieval_eligible).sum():,} records.", "",
             "## Exclusion reasons", ""]
    lines += [f"- {level} / {reason}: {count:,}" for (level, reason), count in reasons.items()]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(task: str, output: Path | None = None, parallelism: int = 64,
        base_url: str | None = None,
        provider_pool_config: Path | None = None) -> dict[str, Any]:
    root = (
        output
        or eligibility_root(task, "semantic_indirect_retrieval_eligibility_v2")
    ).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") == "complete_single_pass":
        return manifest
    _, core = _configure(task)
    endpoint_selection = None
    if base_url:
        core.BASE_URL = base_url.rstrip("/")
        core.ENDPOINTS = ({"name": core.BASE_URL, "base_url": core.BASE_URL,
                           "provider": "local", "credential_env": ""},)
    elif provider_pool_config:
        core.ENDPOINTS, endpoint_selection = _select_provider_endpoints(
            provider_pool_config, parallelism, core.MODEL
        )
        core.BASE_URL = core.ENDPOINTS[0]["base_url"]
    _verify_inputs(manifest)
    schedule = pd.read_parquet(root / "review_schedule.parquet")
    connection = core._request_database(root / "requests.sqlite3")
    client, receipts, total = core._build_completion_pool(parallelism)
    manifest.update({
        "status": "running",
        "per_endpoint_parallelism": parallelism,
        "provider_pool_config": (
            {"path": str(provider_pool_config.resolve()),
             "sha256": _sha256(provider_pool_config)}
            if provider_pool_config else None
        ),
        "endpoint_selection": endpoint_selection,
    })
    write_json_atomic(root / "manifest.json", manifest)
    try:
        ids = schedule.request_id.tolist()
        core._run_pending(connection, ids, parallelism=total, client=client,
                          request_slots=threading.BoundedSemaphore(total))
        fresh = _decision_frame(schedule, _completed(connection, ids))
        reused, _ = _l2_inputs(task)
        decisions = pd.concat([reused, fresh], ignore_index=True).sort_values(
            ["level", "semantic_bucket_id"]
        )
        records = _record_frame(task, decisions)
        decisions.to_parquet(root / "bucket_decisions.parquet", index=False)
        records.to_parquet(root / "record_eligibility.parquet", index=False)
        _write_report(root, task, decisions, records)
        manifest.update(_completion(root, decisions, records, receipts, client, total))
    except Exception:
        manifest["status"] = "incomplete"
        raise
    finally:
        connection.close()
        write_json_atomic(root / "manifest.json", manifest)
    return manifest


def _completion(root: Path, decisions: pd.DataFrame, records: pd.DataFrame,
                receipts: list[dict[str, Any]], client: Any, total: int) -> dict[str, Any]:
    conflicts = decisions[decisions.prior_zero_weight & decisions.decision.eq("include")]
    counts = Counter(decisions.reason_code)
    return {"status": "complete_single_pass", "completed_at": time.time(),
        "endpoint_pool_at_start": receipts, "endpoint_pool_final_snapshot": client.snapshot(),
        "total_parallelism": total, "decision_counts": dict(sorted(counts.items())),
        "eligible_buckets": int(decisions.retrieval_eligible.sum()),
        "excluded_buckets": int((~decisions.retrieval_eligible).sum()),
        "bucket_counts_by_level": decisions.level.value_counts().sort_index().to_dict(),
        "record_counts_by_level": records.level.value_counts().sort_index().to_dict(),
        "prior_zero_weight_buckets": int(decisions.prior_zero_weight.sum()),
        "prior_zero_review_conflicts": len(conflicts),
        "artifacts": {name: {"sha256": _sha256(root / path), "rows": rows}
                      for name, path, rows in (
                          ("bucket_decisions", "bucket_decisions.parquet", len(decisions)),
                          ("record_eligibility", "record_eligibility.parquet", len(records)),
                          ("summary", "summary.tsv", None),
                          ("excluded_buckets", "excluded_buckets.tsv", len(decisions[~decisions.retrieval_eligible])),
                          ("report", "REPORT.md", None))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run", "build-and-run"))
    parser.add_argument("--task", choices=tuple(TARGETS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parallelism", type=int, default=64,
                        help="Maximum concurrent requests per selected endpoint.")
    endpoints = parser.add_mutually_exclusive_group()
    endpoints.add_argument("--base-url")
    endpoints.add_argument("--provider-pool-config", type=Path)
    args = parser.parse_args()
    if args.command in {"build", "build-and-run"}:
        result = build(args.task, args.output)
    if args.command in {"run", "build-and-run"}:
        result = run(
            args.task, args.output, args.parallelism, args.base_url,
            args.provider_pool_config,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
