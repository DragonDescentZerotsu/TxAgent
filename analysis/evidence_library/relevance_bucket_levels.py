"""BBB semantic relevance tournaments within imported UID-based levels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import sqlite3
import time
import signal
import threading

import pandas as pd
import pyarrow.parquet as pq

from analysis.evidence_library import relevance_bucket_diagnostic as calls
from analysis.evidence_library import relevance_bucket_luna as graphs
from analysis.evidence_library import relevance_bucket_rounds as ranking
from analysis.evidence_library import relevance_bucket_tournament as bbb
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.evidence_library.level_mappings import evidence_level_mapping_release
from data.processing.llm_api import DEFAULT_ENV_FILE
from tools.chembl_tool.common.json_utils import write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "data/evidence_libraries/bbb_martins/v7/03_pair_buckets/records.parquet"
LEDGER = ROOT / "data/raw/starling/source_row_uid_ledger"
LEVEL_MANIFEST, LEVELS, LEVEL_RECEIPT = evidence_level_mapping_release(
    "bbb_martins", "v10"
)
TEMPLATE = Path(__file__).with_name("prompts") / "relevance_bucket_level_v1.jinja"
COMPACT_TEMPLATE = TEMPLATE.with_name("relevance_bucket_level_v2.jinja")
MODEL = "deepseek/deepseek-v4-flash-0731"
ROUTING = {"provider": {"only": ["baidu/fp8"], "allow_fallbacks": False,
                         "require_parameters": True}}
VERSION = "bbb_relevance_bucket_levels.v1"
RETRIEVAL_ELIGIBILITY_VERSION = "bbb_relevance_retrieval_eligibility.v1"
MINIMUM_RELEVANCE_PERCENTILE = 20.0
DEGREE = 20
BATCH_SIZE = 5


def attach_levels(records: pd.DataFrame, ledger: pd.DataFrame,
                  levels: pd.DataFrame) -> pd.DataFrame:
    """Attach one imported level to each exactly identified acquisition record."""
    if records.canonical_record_id.duplicated().any():
        raise ValueError("Duplicate canonical record IDs")
    if levels.source_row_uid.duplicated().any():
        raise ValueError("Duplicate UID level assignments")
    joined = records.merge(
        ledger[["source_id", "source_record_id", "acquisition_source_row_number", "source_row_uid"]],
        left_on=["source_id", "source_record_id", "source_row_number"],
        right_on=["source_id", "source_record_id", "acquisition_source_row_number"],
        how="left", validate="many_to_one",
    )
    if joined.source_row_uid.isna().any():
        raise ValueError("Canonical records lack an exact acquisition UID")
    if not joined.source_row_uid.str.fullmatch(r"sr_[0-9a-f]{32}").all():
        raise ValueError("Invalid acquisition UID")
    return joined.merge(
        levels[["source_row_uid", "level", "family_key", "source_group_id"]],
        on="source_row_uid", how="left", validate="many_to_one",
    )


def level_edges(keys: list[str], level: str) -> list[tuple[str, str]]:
    keys = sorted(keys)
    if len(keys) <= DEGREE + 1:
        return graphs._complete_edges(keys)
    if len(keys) <= 2 * DEGREE:
        order = sorted(keys, key=lambda k: bbb._hash(VERSION, level, k))
        return [(key, order[(i + offset) % len(order)])
                for offset in range(1, DEGREE // 2 + 1)
                for i, key in enumerate(order)]
    rng = random.Random(f"{VERSION}-{level}")
    used = set()
    edges = []
    for _ in range(DEGREE // 2):
        order = list(keys)
        rng.shuffle(order)
        successors = dict(zip(order, order[1:] + order[:1]))
        counts = Counter(tuple(sorted(e)) for e in successors.items())
        # Swap successors to repair repeated edges, preserving one incoming and
        # outgoing edge per node. The first Hamiltonian cycle ensures connectivity.
        for a in order:
            b = successors[a]
            old = tuple(sorted((a, b)))
            if old not in used and counts[old] == 1 and a != b:
                continue
            for _attempt in range(10_000):
                c = rng.choice(order)
                d = successors[c]
                if len({a, b, c, d}) != 4:
                    continue
                new = [tuple(sorted((a, d))), tuple(sorted((c, b)))]
                if any(e in used or counts[e] for e in new):
                    continue
                counts[old] -= 1
                counts[tuple(sorted((c, d)))] -= 1
                counts.update(new)
                successors[a], successors[c] = d, b
                break
            else:
                raise RuntimeError(f"Unable to repair comparison graph: {level}")
        current = {tuple(sorted(e)) for e in successors.items()}
        if len(current) != len(keys) or current & used:
            raise ValueError(f"Repeated comparison edges: {level}")
        used.update(current)
        edges.extend(successors.items())
    return edges


def semantic_samples(records: list[dict]) -> list[dict]:
    """Choose diverse cards using only outcome-blind semantic attributes."""
    columns = tuple(c for c in bbb.BASE_CARD_COLUMNS if c not in calls.OUTCOME_FIELDS)
    source = records[0]["source_id"]
    columns += tuple(c for c in bbb.SOURCE_CARD_COLUMNS[source]
                     if c not in calls.OUTCOME_FIELDS and c != "needs_more_context")
    candidates = {}
    for record in records:
        card = {c: bbb._clean(record.get(c)) for c in columns}
        card = {c: v for c, v in card.items() if v is not None}
        if card:
            candidates[bbb._canonical_json(card)] = card
    selected = []
    seen = set()
    while candidates and len(selected) < 5:
        key = max(candidates, key=lambda k: (
            len({(c, str(v)) for c, v in candidates[k].items()} - seen), bbb._hash(k)))
        card = candidates.pop(key)
        selected.append(card)
        seen.update((c, str(v)) for c, v in card.items())
    if not selected:
        raise ValueError("Bucket has no semantic sample")
    return selected


def build(output: Path) -> dict:
    """Prepare a reviewable inventory and graph without making API calls."""
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    ledger_manifest = json.loads((LEDGER / "manifest.json").read_text())
    level_manifest = json.loads(LEVEL_MANIFEST.read_text())
    level_path = LEVELS
    if file_sha256(level_path) != LEVEL_RECEIPT["sha256"]:
        raise ValueError("Level mapping hash mismatch")
    for name, expected in ledger_manifest["partition_sha256"].items():
        if file_sha256(LEDGER / name) != expected:
            raise ValueError(f"UID ledger hash mismatch: {name}")
    available = set(pq.read_schema(RECORDS).names)
    columns = sorted(set(bbb._input_columns()) & available | {
        "source_record_id", "source_row_number"})
    records = pd.read_parquet(RECORDS, columns=columns)
    ledger = pq.read_table(LEDGER, filters=[("task_id", "=", "bbb_martins")],
                           ignore_prefixes=[".", "_", "manifest"]).to_pandas()
    levels = pd.read_parquet(level_path)
    joined = attach_levels(records, ledger, levels)
    retained = joined[joined.level.ge(2)].copy()
    retained["level"] = retained.level.map(lambda n: f"L{int(n)}")
    retained["relevance_bucket"] = [bbb.relevance_bucket(r)[0] for r in retained.to_dict("records")]
    retained["node_key"] = [bbb._canonical_json({"level": level, "relevance_bucket": key})
                            for level, key in zip(retained.level, retained.relevance_bucket)]
    output.mkdir(parents=True)
    connection = bbb._connect(output / "requests.sqlite3")
    for node, group in retained.groupby("node_key", sort=True):
        identity = json.loads(group.iloc[0].relevance_bucket)
        identity["level"] = group.iloc[0].level
        samples = semantic_samples(group.to_dict("records"))
        connection.execute("INSERT INTO buckets VALUES (?, ?, ?, ?, ?, 1, NULL)",
                           (node, identity["source_id"], bbb._canonical_json(identity),
                            bbb._canonical_json(samples), len(group)))
    summary = {}
    for level, group in retained.groupby("level", sort=True):
        keys = sorted(group.node_key.unique())
        edges = level_edges(keys, level)
        degree = Counter(k for e in edges for k in e)
        expected = min(DEGREE, len(keys) - 1)
        if any(degree[k] != expected for k in keys):
            raise ValueError(f"Invalid degree in {level}")
        if len(keys) > 1 and not bbb._connected([(a, b, a) for a, b in edges]):
            raise ValueError(f"Disconnected graph in {level}")
        for i, (a, b) in enumerate(edges):
            connection.execute("INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                               (f"{level}-{i:06d}", level, f"{level}-batch-{i // BATCH_SIZE:06d}", a, b))
        summary[level] = {"records": len(group), "acquisition_uids": group.source_row_uid.nunique(),
                          "buckets": len(keys), "comparisons": len(edges),
                          "requests": (len(edges) + BATCH_SIZE - 1) // BATCH_SIZE,
                          "degree": expected,
                          "sources": group.groupby("source_id").node_key.nunique().to_dict()}
    connection.commit()
    record_map = retained[["canonical_record_id", "source_row_uid", "source_id", "level",
                           "source_group_id", "family_key", "relevance_bucket", "node_key"]]
    record_map.to_parquet(output / "record_relevance_map.parquet", index=False)
    # Review examples are disposable; the database stores cards and the frozen schedule.
    for level in summary:
        batch = connection.execute("SELECT batch_id FROM comparisons WHERE phase=? ORDER BY comparison_id LIMIT 1",
                                   (level,)).fetchone()
        if batch:
            (output / f"prompt_{level}.txt").write_text(
                calls.render_request(connection, batch[0], template_path=TEMPLATE))
    connection.close()
    manifest = {
        "version": VERSION, "task": "bbb_martins", "status": "awaiting_prompt_review",
        "ranking_scope": "within_task_and_level_across_sources",
        "universe": "all_v7_canonical_records_with_imported_L2_plus_acquisition_membership",
        "input": str(RECORDS), "input_sha256": file_sha256(RECORDS),
        "level_mapping_sha256": file_sha256(level_path),
        "level_manifest_sha256": file_sha256(LEVEL_MANIFEST),
        "uid_ledger_manifest_sha256": file_sha256(LEDGER / "manifest.json"),
        "record_map_sha256": file_sha256(output / "record_relevance_map.parquet"),
        "upstream_stage3_sha256": level_manifest["inputs"]["stage3_records"][
            "sha256"
        ],
        "level_mapping_evidence_library_version": level_manifest["evidence_library_version"],
        "level_mapping_gold_release": level_manifest["gold_release"],
        "join": "task + source_id + source_record_id + acquisition_source_row_number -> UID -> level",
        "descendant_policy": "canonical measurements retain their acquisition UID's imported level",
        "unmapped_level_records": int(joined.level.isna().sum()),
        "mapped_L1_records": int(joined.level.eq(1).sum()),
        "mapped_L2_plus_records": len(retained),
        "relevance_bucket_columns": bbb.RELEVANCE_BUCKET_COLUMNS,
        "model": MODEL, "provider_routing": ROUTING["provider"],
        "reasoning_effort": "medium", "max_completion_tokens": 8192,
        "batch_size": BATCH_SIZE, "target_degree": DEGREE,
        "prompt": str(TEMPLATE), "prompt_sha256": file_sha256(TEMPLATE),
        "levels": summary,
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def export_rankings(connection, output: Path, manifest: dict) -> None:
    rows = []
    for level in manifest["levels"]:
        outcomes = [tuple(r) for r in connection.execute(
            "SELECT bucket_a, bucket_b, winner FROM comparisons WHERE phase=?", (level,))]
        if any(w is None or w not in (a, b) for a, b, w in outcomes):
            raise ValueError(f"Incomplete or invalid comparisons in {level}")
        keys = [r[0] for r in connection.execute("SELECT relevance_bucket FROM buckets")
                if json.loads(r[0])["level"] == level]
        scores = ranking.fit_bradley_terry(outcomes) if outcomes else {k: 0.0 for k in keys}
        counts = Counter(k for a, b, _ in outcomes for k in (a, b))
        for i, key in enumerate(sorted(keys, key=lambda k: (-scores[k], k))):
            bucket = json.loads(key)["relevance_bucket"]
            rows.append({"task_id": "bbb_martins", "level": level, "node_key": key,
                         "relevance_bucket": bucket, "source_id": json.loads(bucket)["source_id"],
                         "bradley_terry_score": scores[key], "level_rank": i + 1,
                         "level_percentile": 100 * (len(keys) - 1 - i) / max(1, len(keys) - 1),
                         "comparison_count": counts[key]})
    if len(rows) != sum(s["buckets"] for s in manifest["levels"].values()):
        raise ValueError("Ranking coverage mismatch")
    pd.DataFrame(rows).to_parquet(output / "relevance_bucket_rankings.parquet", index=False)


def prepare_degree10(source: Path, output: Path) -> dict:
    """Reuse the reviewed degree-20 inventory, deferring its last five rounds."""
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["target_degree"] != 20 or manifest["version"] != VERSION:
        raise ValueError("Expected the prepared degree-20 inventory")
    if file_sha256(TEMPLATE) != manifest["prompt_sha256"]:
        raise ValueError("Prepared prompt changed")
    if file_sha256(source / "record_relevance_map.parquet") != manifest["record_map_sha256"]:
        raise ValueError("Prepared record map changed")
    with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as original:
        if original.execute("SELECT count(*) FROM requests").fetchone()[0]:
            raise ValueError("Degree reduction requires an unstarted inventory")
        if original.execute("SELECT count(*) FROM comparisons WHERE winner IS NOT NULL").fetchone()[0]:
            raise ValueError("Prepared inventory already contains judgments")
        output.mkdir(parents=True, exist_ok=False)
        connection = bbb._connect(output / "requests.sqlite3")
        original.backup(connection)
    for level, summary in manifest["levels"].items():
        scheduled = connection.execute(
            "SELECT comparison_id,bucket_a,bucket_b FROM comparisons WHERE phase=? ORDER BY comparison_id",
            (level,),
        ).fetchall()
        boundary = summary["buckets"] * 5
        active = scheduled[:boundary]
        edges = [(r[1], r[2]) for r in active]
        degree = Counter(k for edge in edges for k in edge)
        positions = Counter(a for a, _ in edges)
        if (len(degree) != summary["buckets"] or set(degree.values()) != {10}
                or set(positions.values()) != {5}
                or len({tuple(sorted(e)) for e in edges}) != len(edges)
                or not bbb._connected([(a, b, a) for a, b in edges])):
            raise ValueError(f"Invalid degree-10 prefix: {level}")
        connection.executemany("UPDATE comparisons SET phase=? WHERE comparison_id=?",
                               [(f"deferred:{level}", row[0]) for row in scheduled[boundary:]])
        summary.update(degree=10, comparisons=len(active), requests=(len(active) + 4) // 5,
                       deferred_comparisons=len(scheduled) - len(active))
    connection.commit()
    connection.close()
    shutil.copy2(source / "record_relevance_map.parquet", output / "record_relevance_map.parquet")
    manifest.update(target_degree=10, status="prepared", prepared_from=str(source),
                    prepared_manifest_sha256=file_sha256(source / "manifest.json"),
                    schedule_policy="first_five_rounds_of_frozen_degree20_schedule")
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def export_retrieval_eligibility(ranking_dir: Path, output: Path) -> dict:
    """Derive the reviewed percentile-20 record flag from completed rankings."""
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    ranking_manifest_path = ranking_dir / "manifest.json"
    ranking_path = ranking_dir / "relevance_bucket_rankings.parquet"
    record_map_path = ranking_dir / "record_relevance_map.parquet"
    ranking_manifest = json.loads(ranking_manifest_path.read_text())
    if ranking_manifest.get("status") != "complete" or ranking_manifest.get("task") != "bbb_martins":
        raise ValueError("Completed BBB relevance rankings are required")
    for path, expected in (
        (ranking_path, ranking_manifest.get("ranking_sha256")),
        (record_map_path, ranking_manifest.get("record_map_sha256")),
    ):
        if not expected or file_sha256(path) != expected:
            raise ValueError(f"Relevance ranking hash mismatch: {path}")

    rankings = pd.read_parquet(ranking_path, columns=["node_key", "level", "level_percentile"])
    record_map = pd.read_parquet(record_map_path, columns=["canonical_record_id", "node_key", "level"])
    if rankings.duplicated(["node_key", "level"]).any() or record_map.canonical_record_id.duplicated().any():
        raise ValueError("Duplicate relevance ranking or record ID")
    expected_levels = {"L2", "L3", "L4", "L5"}
    if set(rankings.level) != expected_levels or set(record_map.level) != expected_levels:
        raise ValueError("BBB retrieval eligibility requires complete L2-L5 rankings")
    rankings["level_percentile"] = pd.to_numeric(rankings.level_percentile, errors="coerce")
    if rankings.level_percentile.isna().any() or not rankings.level_percentile.between(0, 100).all():
        raise ValueError("Level percentiles must be finite values from 0 through 100")
    flagged = record_map.merge(rankings, on=["node_key", "level"], how="left", validate="many_to_one")
    if flagged.level_percentile.isna().any():
        raise ValueError("Every relevance-mapped record must have a within-level percentile")
    if len(flagged) != ranking_manifest.get("mapped_L2_plus_records"):
        raise ValueError("Record eligibility coverage differs from the completed ranking manifest")
    flagged["relevance_bucket_retrieval_eligible"] = flagged.level_percentile.ge(
        MINIMUM_RELEVANCE_PERCENTILE
    )
    flagged = flagged[["canonical_record_id", "level", "relevance_bucket_retrieval_eligible"]].sort_values(
        "canonical_record_id"
    )

    output.mkdir(parents=True)
    output_path = output / "record_relevance_eligibility.parquet"
    flagged.to_parquet(output_path, index=False)
    counts = {
        level: {
            "eligible": int(group.relevance_bucket_retrieval_eligible.sum()),
            "ineligible": int((~group.relevance_bucket_retrieval_eligible).sum()),
        }
        for level, group in flagged.groupby("level", sort=True)
    }
    manifest = {
        "version": RETRIEVAL_ELIGIBILITY_VERSION,
        "task": "bbb_martins",
        "status": "complete",
        "scope": "ranked imported L2-L5 records; L1 is unaffected",
        "rule": {
            "field": "level_percentile",
            "operator": ">=",
            "value": MINIMUM_RELEVANCE_PERCENTILE,
            "output_field": "relevance_bucket_retrieval_eligible",
        },
        "stage3_input_sha256": ranking_manifest["input_sha256"],
        "ranking_inputs": {
            str(ranking_manifest_path): file_sha256(ranking_manifest_path),
            str(ranking_path): ranking_manifest["ranking_sha256"],
            str(record_map_path): ranking_manifest["record_map_sha256"],
        },
        "records": len(flagged),
        "counts_by_level": counts,
        "output": {"path": str(output_path), "sha256": file_sha256(output_path)},
        "validations": {
            "canonical_record_ids_unique": True,
            "all_records_have_level_percentiles": True,
            "flag_is_binary_and_non_null": True,
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["build", "prepare-degree10", "export-retrieval-eligibility", "run"]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--ranking-dir", type=Path)
    parser.add_argument("--approved-prompt-sha256")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--compact-prompt", action="store_true")
    parser.add_argument("--retry-only", action="store_true", help="Retry cached unfinished requests across selected levels in one pool")
    parser.add_argument("--levels", nargs="+", choices=["L2", "L3", "L4", "L5"])
    parser.add_argument("--endpoint-provider", choices=["baidu/fp8", "relace/fp4", "relace+baidu", "open-inference/fp8", "deepinfra/fp8", "three-provider"])
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.output_dir), indent=2))
        return
    if args.command == "prepare-degree10":
        if args.prepared_dir is None:
            raise ValueError("--prepared-dir is required")
        print(json.dumps(prepare_degree10(args.prepared_dir, args.output_dir), indent=2))
        return
    if args.command == "export-retrieval-eligibility":
        if args.ranking_dir is None:
            raise ValueError("--ranking-dir is required")
        print(json.dumps(export_retrieval_eligibility(args.ranking_dir, args.output_dir), indent=2))
        return
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    selected_levels = [level for level in manifest["levels"] if args.levels is None or level in args.levels]
    compact = args.compact_prompt or manifest.get("compact_prompt", False)
    template = COMPACT_TEMPLATE if compact else TEMPLATE
    if args.approved_prompt_sha256 != file_sha256(template):
        raise ValueError("Reviewed prompt hash required")
    previous_template = COMPACT_TEMPLATE if manifest.get("compact_prompt", False) else TEMPLATE
    if file_sha256(previous_template) != manifest["prompt_sha256"]:
        raise ValueError("Previous prompt contract mismatch")
    if manifest["version"] != VERSION or manifest["model"] != MODEL:
        raise ValueError("Run contract mismatch")
    endpoint = args.endpoint_provider or manifest.get("endpoint_provider") or manifest["provider_routing"]["only"][0]
    if endpoint not in {"baidu/fp8", "relace/fp4", "relace+baidu", "open-inference/fp8", "deepinfra/fp8", "three-provider"}:
        raise ValueError("Unsupported endpoint provider")
    endpoints = ["relace/fp4", "baidu/fp8"] if endpoint == "relace+baidu" else [endpoint]
    if endpoint == "three-provider":
        endpoints = ["open-inference/fp8", "deepinfra/fp8", "relace/fp4"]
    if args.parallelism % len(endpoints):
        raise ValueError("Parallelism must divide evenly across providers")
    routing = {"provider": {**ROUTING["provider"], "only": endpoints}}
    slot_routes = [{"provider": {**ROUTING["provider"], "only": [name]}}
                   for name in endpoints for _ in range(args.parallelism // len(endpoints))]
    if not args.max_cost_usd or args.max_cost_usd <= 0 or args.parallelism < 1:
        raise ValueError("Positive dollar budget and parallelism required")
    for path, digest in [(RECORDS, manifest["input_sha256"]),
                         (args.output_dir / "record_relevance_map.parquet", manifest["record_map_sha256"])]:
        if file_sha256(path) != digest:
            raise ValueError(f"Input hash mismatch: {path}")
    connection = bbb._connect(args.output_dir / "requests.sqlite3")
    connection.execute("CREATE INDEX IF NOT EXISTS comparisons_by_batch ON comparisons(batch_id, comparison_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS comparisons_by_phase ON comparisons(phase)")
    manifest["status"] = "running"
    manifest["parallelism"] = args.parallelism
    manifest["scheduler"] = "rolling_provider_slots.v1"
    manifest["max_cost_usd"] = args.max_cost_usd
    manifest["approved_prompt_sha256"] = args.approved_prompt_sha256
    manifest["max_reason_words"] = None
    manifest["active_levels"] = selected_levels
    manifest["retry_only"] = args.retry_only
    manifest.setdefault("prompt_versions", [manifest["prompt_sha256"]])
    if args.approved_prompt_sha256 not in manifest["prompt_versions"]:
        manifest["prompt_versions"].append(args.approved_prompt_sha256)
    manifest["prompt_sha256"] = args.approved_prompt_sha256
    manifest["compact_prompt"] = compact
    manifest.setdefault("run_history", []).append({
        "started_at": time.time(), "previous_provider_routing": manifest["provider_routing"],
        "levels": selected_levels,
        "prompt_sha256": args.approved_prompt_sha256,
        "provider_routing": routing["provider"],
        "completed_requests_before": connection.execute("SELECT count(*) FROM requests WHERE status='complete'").fetchone()[0],
        "usage_before": list(connection.execute("SELECT coalesce(sum(input_tokens),0), coalesce(sum(output_tokens),0) FROM requests").fetchone()),
        "structured_output": endpoint == "baidu/fp8", "max_reason_words": None,
        "provider_concurrency": dict.fromkeys(endpoints, args.parallelism // len(endpoints)),
    })
    manifest["provider_routing"] = routing["provider"]
    manifest["endpoint_provider"] = endpoint
    manifest["provider_concurrency"] = dict.fromkeys(endpoints, args.parallelism // len(endpoints))
    write_json_atomic(args.output_dir / "manifest.json", manifest)
    stop_event = threading.Event()
    previous_handlers = {sig: signal.signal(sig, lambda *_: stop_event.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        prepared_batch_ids = None
        if args.retry_only:
            prepared_batch_ids = [row[0] for level in selected_levels for row in connection.execute(
                "SELECT batch_id FROM requests WHERE phase=? AND status!='complete' ORDER BY batch_id", (level,))]
        for level in (["retry_all_levels"] if args.retry_only else selected_levels):
            calls.run_condition(
                connection, level, parallelism=args.parallelism, token_budget=10**12,
                env_file=DEFAULT_ENV_FILE, template_path=template, allow_ties=False, compact=compact,
                model=MODEL, reasoning_effort="medium", max_completion_tokens=8192,
                max_tokens_parameter="max_tokens", base_url="https://openrouter.ai/api/v1",
                provider="openrouter", request_extra_body=routing,
                max_cost_usd=args.max_cost_usd, input_cost_per_million=0.14,
                output_cost_per_million=0.28, max_reason_words=None,
                structured_output=endpoint == "baidu/fp8", slot_extra_bodies=slot_routes,
                stop_event=stop_event,
                allow_partial=True,
                prepared_batch_ids=prepared_batch_ids,
            )
        manifest["unjudged_comparisons_by_level"] = {
            level: connection.execute("SELECT count(*) FROM comparisons WHERE phase=? AND winner IS NULL", (level,)).fetchone()[0]
            for level in manifest["levels"]}
        if any(manifest["unjudged_comparisons_by_level"].values()):
            manifest["status"] = "incomplete"
        else:
            export_rankings(connection, args.output_dir, manifest)
            manifest["status"] = "complete"
            manifest["ranking_sha256"] = file_sha256(args.output_dir / "relevance_bucket_rankings.parquet")
    except Exception:
        manifest["status"] = "incomplete"
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        usage = connection.execute("SELECT coalesce(sum(input_tokens),0), coalesce(sum(output_tokens),0) FROM requests").fetchone()
        manifest["usage"] = {"input_tokens": usage[0], "output_tokens": usage[1],
                             "actual_tokens": sum(usage),
                             "undiscounted_cost_estimate_usd": (usage[0] * .14 + usage[1] * .28) / 1e6}
        write_json_atomic(args.output_dir / "manifest.json", manifest)
        connection.close()


if __name__ == "__main__":
    main()
