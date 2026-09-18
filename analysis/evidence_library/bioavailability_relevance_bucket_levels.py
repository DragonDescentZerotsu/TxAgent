"""Build, pilot, and rank V10 oral-bioavailability semantic evidence groups."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import sqlite3
import threading
import time

import pandas as pd
import pyarrow.parquet as pq

from analysis.evidence_library import bioavailability_relevance_bucket_tournament as oral
from analysis.evidence_library import relevance_bucket_diagnostic as calls
from analysis.evidence_library import relevance_bucket_luna as graphs
from analysis.evidence_library import relevance_bucket_rounds as ranking
from analysis.evidence_library import relevance_bucket_tournament as storage
from data.processing.evidence_library.shared.v1.normalization.cleaning import file_sha256
from data.processing.evidence_library.level_mappings import evidence_level_mapping_release
from data.processing.llm_api import openai_compatible_client
from tools.chembl_tool.common.json_utils import write_json_atomic


VERSION = "bioavailability_relevance_bucket_levels.v2"
ENTRYPOINT = Path(__file__)
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
BASE_URL = "http://dgx027:50001/v1"
ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data/evidence_libraries/bioavailability_ma/v10"
RECORDS = RELEASE / "03_pair_buckets/records.parquet"
PAIR_BUCKET_MANIFEST = RELEASE / "03_pair_buckets/manifest.json"
RELEASE_MANIFEST = RELEASE / "manifest.json"
LEVEL_MANIFEST, LEVELS, LEVEL_RECEIPT = evidence_level_mapping_release(
    "bioavailability_ma", "v10"
)
TEMPLATE = (
    ROOT
    / "semantic_buckets/prompts/"
    "bioavailability_relevance_bucket_level_v2.jinja"
)
DEFAULT_OUTPUT = ROOT / "outputs/analysis/evidence_library/bioavailability_relevance_levels_v10_degree10_v2"
DEGREE = 10
SAMPLE_LIMIT = 3
BATCH_SIZE = 1
PILOT_PAIRS_PER_LEVEL = 5
PILOT_MINIMUM_AGREEMENT = 23
PILOT_AGREEMENT_REQUIRED = True
FULL_PARALLELISM = 128
MAX_COMPLETION_TOKENS = 4_096
EXPECTED_INPUT_SHA256 = "593cae2a76990533bab3f535c16e2886bc8683b907af92186d6ec1966e7df74d"
EXPECTED_LEVEL_SHA256 = "ee476fd705856ef0b81c259281d7b093df7fe146b77391f493a10ce6f6b9b72c"
EXPECTED_LEVELS = {
    "L2": {"buckets": 64, "comparisons": 320},
    "L3": {"buckets": 159, "comparisons": 795},
    "L4": {"buckets": 17, "comparisons": 85},
    "L5": {"buckets": 11, "comparisons": 55},
    "L6": {"buckets": 17, "comparisons": 85},
}
SEMANTIC_VERSIONS = {
    "canonical_endpoint_concept": "bioavailability_endpoint_concepts.v1",
    "canonical_bioavailability_report_type": "bioavailability_report_type_normalization.v1",
    "canonical_bioavailability_evidence_scope": "bioavailability_evidence_scope.v1",
    "canonical_biological_matrix": "starling_auxiliary.globally_reconciled.v2",
    "pair_bucket_contract": "bioavailability_ma_pair_buckets.v19",
}


def _level_edges(keys: list[str], level: str) -> list[tuple[str, str]]:
    keys = sorted(keys)
    if len(keys) <= DEGREE + 1:
        return graphs._complete_edges(keys)
    used: set[tuple[str, str]] = set()
    return [
        edge
        for cycle in range(DEGREE // 2)
        for edge in ranking._cycle(keys, f"{VERSION}-{level}-{cycle}", used)
    ]


def _validate_inputs() -> tuple[dict, dict, dict]:
    release = json.loads(RELEASE_MANIFEST.read_text())
    pair_manifest = json.loads(PAIR_BUCKET_MANIFEST.read_text())
    level_manifest = json.loads(LEVEL_MANIFEST.read_text())
    checks = {
        "records": (file_sha256(RECORDS), EXPECTED_INPUT_SHA256),
        "release records": (
            release["canonical_artifact_hashes"]["records.parquet"],
            EXPECTED_INPUT_SHA256,
        ),
        "pair-bucket records": (
            pair_manifest["outputs"]["records.parquet"], EXPECTED_INPUT_SHA256
        ),
        "level mapping": (file_sha256(LEVELS), EXPECTED_LEVEL_SHA256),
        "level manifest": (LEVEL_RECEIPT["sha256"], EXPECTED_LEVEL_SHA256),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{name} hash mismatch: {actual}")
    auxiliary = json.loads(
        (RELEASE / "02_canonicalized/auxiliary_mapping_manifest.json").read_text()
    )
    versions = {
        "canonical_endpoint_concept": release["endpoint_concept_version"],
        "canonical_bioavailability_report_type": release["report_type_normalization_version"],
        "canonical_bioavailability_evidence_scope": release["bioavailability_evidence_scope_version"],
        "canonical_biological_matrix": auxiliary["mapping_version"],
        "pair_bucket_contract": pair_manifest["contract"]["pair_bucket_version"],
    }
    if versions != SEMANTIC_VERSIONS:
        raise ValueError(f"canonical semantic versions changed: {versions}")
    return release, pair_manifest, level_manifest


def _node_source(node: str) -> str:
    return json.loads(json.loads(node)["relevance_bucket"])["source_id"]


def _pilot_rows(connection: sqlite3.Connection, level: str) -> list[sqlite3.Row]:
    rows = list(connection.execute(
        "SELECT comparison_id,bucket_a,bucket_b FROM comparisons WHERE phase=?",
        (level,),
    ))
    rows.sort(key=lambda row: storage._hash("oral-v2-pilot", row["comparison_id"]))
    if level != "L2":
        return rows[:PILOT_PAIRS_PER_LEVEL]
    selected = []
    uncovered = set(oral.RELEVANCE_BUCKET_COLUMNS)
    while rows and len(selected) < PILOT_PAIRS_PER_LEVEL:
        row = max(
            rows,
            key=lambda candidate: (
                len({_node_source(candidate["bucket_a"]), _node_source(candidate["bucket_b"])} & uncovered),
                storage._hash("oral-v2-pilot-greedy", candidate["comparison_id"]),
            ),
        )
        rows.remove(row)
        selected.append(row)
        uncovered -= {_node_source(row["bucket_a"]), _node_source(row["bucket_b"])}
    if uncovered:
        raise ValueError(f"L2 pilot does not cover sources: {sorted(uncovered)}")
    return selected


def _prepare_pilot(output: Path, source: sqlite3.Connection) -> dict:
    pilot_dir = output / "prompt_pilot"
    pilot_dir.mkdir()
    destination = storage._connect(pilot_dir / "requests.sqlite3")
    selected = []
    needed = set()
    for level in EXPECTED_LEVELS:
        for index, row in enumerate(_pilot_rows(source, level)):
            needed.update((row["bucket_a"], row["bucket_b"]))
            selected.append({
                "level": level,
                "index": index,
                "source_comparison_id": row["comparison_id"],
                "bucket_a": row["bucket_a"],
                "bucket_b": row["bucket_b"],
            })
    for key in sorted(needed):
        row = source.execute("SELECT * FROM buckets WHERE relevance_bucket=?", (key,)).fetchone()
        destination.execute("INSERT INTO buckets VALUES (?, ?, ?, ?, ?, ?, ?)", tuple(row))
    for item in selected:
        for phase, left, right in (
            ("pilot_base", item["bucket_a"], item["bucket_b"]),
            ("pilot_reverse", item["bucket_b"], item["bucket_a"]),
        ):
            suffix = f"{item['level']}-{item['index']:02d}"
            destination.execute(
                "INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                (
                    f"{phase}-{suffix}", phase, f"{phase}-{suffix}", left, right,
                    item["source_comparison_id"],
                ),
            )
    for phase in ("pilot_base", "pilot_reverse"):
        calls._prepare_requests(
            destination, phase, template_path=TEMPLATE, model=MODEL,
            compact=True, single_comparison=True,
        )
    request_count = destination.execute("SELECT count(*) FROM requests").fetchone()[0]
    if request_count != 2 * PILOT_PAIRS_PER_LEVEL * len(EXPECTED_LEVELS):
        raise ValueError(f"wrong pilot request count: {request_count}")
    destination.close()
    manifest = {
        "version": f"{VERSION}.prompt_pilot.v1",
        "status": "prepared",
        "pair_count": len(selected),
        "request_count": request_count,
        "pairs_per_level": PILOT_PAIRS_PER_LEVEL,
        "L2_source_coverage": sorted({
            _node_source(key)
            for item in selected if item["level"] == "L2"
            for key in (item["bucket_a"], item["bucket_b"])
        }),
        "selected_pairs": selected,
        "prompt_sha256": file_sha256(TEMPLATE),
        "model": MODEL,
        "base_url": BASE_URL,
        "reasoning_effort": "low",
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "total_token_budget": None,
        "single_comparison_per_request": True,
        "order_reversal": True,
    }
    write_json_atomic(pilot_dir / "manifest.json", manifest)
    return manifest


def _write_prompt_examples(output: Path, connection: sqlite3.Connection) -> str:
    sections = ["# Rendered oral V10 semantic relevance prompt examples\n"]
    for level in EXPECTED_LEVELS:
        batch_id = connection.execute(
            "SELECT batch_id FROM comparisons WHERE phase=? ORDER BY comparison_id LIMIT 1",
            (level,),
        ).fetchone()[0]
        prompt = calls.render_request(
            connection, batch_id, template_path=TEMPLATE,
            compact=True, single_comparison=True,
        )
        sections.extend((f"## {level}\n", "```text\n" + prompt + "\n```\n"))
    path = output / "prompt_examples.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return file_sha256(path)


def _validate_prepared_requests(connection: sqlite3.Connection) -> dict:
    forbidden = {
        "finite_scalar_value", "canonical_category_id", "direct_vote_label",
        "canonical_reference_scope", "measurement_text", "support_text",
    }
    counts = Counter()
    for row in connection.execute(
        "SELECT batch_id,prompt,prompt_sha256,requested_model FROM requests ORDER BY batch_id"
    ):
        comparisons = calls._comparisons(connection, row["batch_id"])
        if len(comparisons) != 1:
            raise ValueError(f"request is not a single comparison: {row['batch_id']}")
        rendered = calls.render_request(
            connection, row["batch_id"], template_path=TEMPLATE,
            compact=True, single_comparison=True,
        )
        if rendered != row["prompt"] or storage._hash(rendered) != row["prompt_sha256"]:
            raise ValueError(f"cached prompt mismatch: {row['batch_id']}")
        if row["requested_model"] != MODEL:
            raise ValueError(f"requested model mismatch: {row['batch_id']}")
        payload = json.JSONDecoder().raw_decode(
            rendered[rendered.index('{"groups":'):]
        )[0]
        if set(payload) != {"groups"} or len(payload["groups"]) != 2:
            raise ValueError(f"prompt payload is not exactly two groups: {row['batch_id']}")
        for group in payload["groups"].values():
            source = group["source"]
            if set(group["identity"]) != {"source_id", *oral.RELEVANCE_BUCKET_COLUMNS[source]}:
                raise ValueError(f"semantic identity field drift: {row['batch_id']}")
            if not 1 <= len(group["sample_records"]) <= SAMPLE_LIMIT:
                raise ValueError(
                    f"sample count outside 1-{SAMPLE_LIMIT}: {row['batch_id']}"
                )
            for sample in group["sample_records"]:
                if not set(sample) <= set(oral.SAMPLE_CARD_COLUMNS[source]):
                    raise ValueError(f"sample field drift: {row['batch_id']}")
                if set(sample) & forbidden:
                    raise ValueError(f"outcome field exposed: {row['batch_id']}")
        schema = calls.response_format(
            comparisons, allow_ties=False, single_comparison=True,
        )["json_schema"]["schema"]
        if schema["required"] != ["winner_bucket_id"] or schema["additionalProperties"]:
            raise ValueError(f"response schema drift: {row['batch_id']}")
        counts["requests"] += 1
        counts["groups_rendered"] += 2
    return {
        "version": f"{VERSION}.prepared_request_validation.v1",
        "status": "passed",
        "requests": counts["requests"],
        "groups_rendered": counts["groups_rendered"],
        "comparisons_per_request": 1,
        "groups_per_request": 2,
        "sample_records_per_group": f"1-{SAMPLE_LIMIT}",
        "exact_winner_only_schema": True,
        "outcome_fields_exposed": [],
    }


def build(output: Path) -> dict:
    """Prepare the immutable full schedule and pilot without model calls."""
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    release, pair_manifest, level_manifest = _validate_inputs()
    schema = set(pq.read_schema(RECORDS).names)
    records = pd.read_parquet(
        RECORDS, columns=[column for column in oral.input_columns() if column in schema]
    )
    levels = pd.read_parquet(LEVELS, columns=["source_row_uid", "level", "family_key"])
    if records.source_row_uid.duplicated().any() or levels.source_row_uid.duplicated().any():
        raise ValueError("records and level mapping require unique source_row_uid values")
    joined = records.merge(levels, on="source_row_uid", how="left", validate="one_to_one")
    if joined.level.isna().any():
        raise ValueError(f"{int(joined.level.isna().sum())} V10 records lack a level")
    mapped_l1 = int(joined.level.eq(1).sum())
    retained = joined[joined.level.ge(2)].copy()
    retained["level"] = retained.level.map(lambda value: f"L{int(value)}")
    retained["relevance_bucket"] = [
        oral.relevance_bucket(row)[0] for row in retained.to_dict("records")
    ]
    retained["node_key"] = [
        storage._canonical_json({"level": level, "relevance_bucket": bucket})
        for level, bucket in zip(retained.level, retained.relevance_bucket)
    ]

    output.mkdir(parents=True)
    connection = storage._connect(output / "requests.sqlite3")
    sample_counts = Counter()
    distinct_pair_bucket_counts = Counter()
    for node, group in retained.groupby("node_key", sort=True):
        identity_json = group.iloc[0].relevance_bucket
        identity = json.loads(identity_json)
        samples = oral.select_diverse_records(
            group.to_dict("records"), limit=SAMPLE_LIMIT
        )
        if not samples:
            raise ValueError(f"semantic group has no sample card: {node}")
        sample_counts[len(samples)] += 1
        distinct_pair_bucket_counts[
            min(SAMPLE_LIMIT, group.pair_bucket_key.nunique())
        ] += 1
        connection.execute(
            "INSERT INTO buckets VALUES (?, ?, ?, ?, ?, 1, NULL)",
            (
                node, identity["source_id"], identity_json,
                storage._canonical_json(samples), len(group),
            ),
        )

    summary = {}
    for level, group in retained.groupby("level", sort=True):
        keys = sorted(group.node_key.unique())
        edges = _level_edges(keys, level)
        degree = Counter(key for edge in edges for key in edge)
        expected_degree = min(DEGREE, len(keys) - 1)
        if set(degree) != set(keys) or any(degree[key] != expected_degree for key in keys):
            raise ValueError(f"invalid degree-{DEGREE} graph: {level}")
        if not storage._connected([(left, right, left) for left, right in edges]):
            raise ValueError(f"disconnected comparison graph: {level}")
        for index, (left, right) in enumerate(edges):
            comparison_id = f"{level}-{index:06d}"
            connection.execute(
                "INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL)",
                (comparison_id, level, comparison_id, left, right),
            )
        summary[level] = {
            "records": len(group),
            "pair_buckets": int(group.pair_bucket_key.nunique()),
            "buckets": len(keys),
            "comparisons": len(edges),
            "requests": len(edges),
            "degree": expected_degree,
            "sources": group.groupby("source_id").node_key.nunique().to_dict(),
        }
        for field, expected in EXPECTED_LEVELS[level].items():
            if summary[level][field] != expected:
                raise ValueError(
                    f"{level} {field} changed: {summary[level][field]} != {expected}"
                )

    for level in EXPECTED_LEVELS:
        calls._prepare_requests(
            connection, level, template_path=TEMPLATE, model=MODEL,
            compact=True, single_comparison=True,
        )
    full_requests = connection.execute("SELECT count(*) FROM requests").fetchone()[0]
    if full_requests != sum(item["comparisons"] for item in EXPECTED_LEVELS.values()):
        raise ValueError(f"wrong full request count: {full_requests}")
    prompt_examples_sha256 = _write_prompt_examples(output, connection)
    validation = _validate_prepared_requests(connection)
    write_json_atomic(output / "build_validation.json", validation)
    pilot = _prepare_pilot(output, connection)
    connection.close()

    record_map = retained[[
        "canonical_record_id", "source_row_uid", "source_id", "level", "family_key",
        "pair_bucket_key", "relevance_bucket", "node_key",
    ]].sort_values("canonical_record_id")
    record_map.to_parquet(output / "record_relevance_map.parquet", index=False)
    manifest = {
        "version": VERSION,
        "task": "bioavailability_ma",
        "status": "prepared_and_awaiting_pilot",
        "supersedes_without_mutation": str(
            ROOT / "outputs/analysis/evidence_library/bioavailability_relevance_levels_baidu_v1"
        ),
        "ranking_scope": "within_task_and_level_across_sources",
        "universe": "all_current_V10_mapped_L2_through_L6_records",
        "input": str(RECORDS),
        "input_sha256": file_sha256(RECORDS),
        "pair_bucket_manifest_sha256": file_sha256(PAIR_BUCKET_MANIFEST),
        "release_manifest_sha256": file_sha256(RELEASE_MANIFEST),
        "level_mapping_sha256": file_sha256(LEVELS),
        "level_manifest_sha256": file_sha256(LEVEL_MANIFEST),
        "record_map_sha256": file_sha256(output / "record_relevance_map.parquet"),
        "mapped_L1_records_excluded": mapped_l1,
        "mapped_L2_through_L6_records": len(retained),
        "relevance_bucket_columns": oral.RELEVANCE_BUCKET_COLUMNS,
        "sample_card_columns": oral.SAMPLE_CARD_COLUMNS,
        "semantic_versions": SEMANTIC_VERSIONS,
        "sample_count_distribution": dict(sorted(sample_counts.items())),
        "distinct_pair_bucket_support_capped_at_sample_limit": dict(
            sorted(distinct_pair_bucket_counts.items())
        ),
        "model": MODEL,
        "base_url": BASE_URL,
        "provider": "local",
        "allow_fallbacks": False,
        "reasoning_effort": "low",
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "total_token_budget": None,
        "batch_size": BATCH_SIZE,
        "single_comparison_per_request": True,
        "target_degree": DEGREE,
        "sample_limit": SAMPLE_LIMIT,
        "full_run_parallelism": FULL_PARALLELISM,
        "prompt": str(TEMPLATE),
        "prompt_sha256": file_sha256(TEMPLATE),
        "prompt_examples_sha256": prompt_examples_sha256,
        "build_validation_sha256": file_sha256(output / "build_validation.json"),
        "implementation_sha256": {
            ENTRYPOINT.name: file_sha256(ENTRYPOINT),
            Path(__file__).name: file_sha256(Path(__file__)),
            Path(oral.__file__).name: file_sha256(Path(oral.__file__)),
            Path(calls.__file__).name: file_sha256(Path(calls.__file__)),
        },
        "levels": summary,
        "pilot": {
            "path": "prompt_pilot",
            "manifest_sha256": file_sha256(output / "prompt_pilot/manifest.json"),
            "pair_count": pilot["pair_count"],
            "request_count": pilot["request_count"],
        },
        "upstream_release_artifact_version": release["artifact_version"],
        "upstream_stage3_version": pair_manifest["version"],
        "upstream_level_stage3_sha256": level_manifest["inputs"]["stage3_records"][
            "sha256"
        ],
        "level_mapping_evidence_library_version": level_manifest["evidence_library_version"],
        "level_mapping_gold_release": level_manifest["gold_release"],
    }
    write_json_atomic(output / "manifest.json", manifest)
    return manifest


def _verify_endpoint_model() -> list[str]:
    client, _ = openai_compatible_client(
        base_url=BASE_URL, provider="local", env_file=None,
        max_connections=1, timeout_s=30,
    )
    model_ids = sorted(model.id for model in client.models.list().data)
    if MODEL not in model_ids:
        raise ValueError(f"endpoint does not serve {MODEL}; available models: {model_ids}")
    return model_ids


def _run_phases(
    connection: sqlite3.Connection,
    phases: list[str],
    *,
    parallelism: int,
    stop: threading.Event,
) -> None:
    for phase in phases:
        calls.run_condition(
            connection,
            phase,
            parallelism=parallelism,
            token_budget=None,
            env_file=None,
            template_path=TEMPLATE,
            allow_ties=False,
            compact=True,
            single_comparison=True,
            model=MODEL,
            reasoning_effort="low",
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            max_tokens_parameter="max_tokens",
            base_url=BASE_URL,
            provider="local",
            max_reason_words=None,
            structured_output=True,
            stop_event=stop,
        )


def run_pilot(output: Path, *, approved_prompt_sha256: str) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    prompt_hash = file_sha256(TEMPLATE)
    if manifest.get("version") != VERSION or approved_prompt_sha256 != prompt_hash:
        raise ValueError("the prepared V2 artifact and reviewed prompt hash are required")
    models = _verify_endpoint_model()
    pilot_dir = output / "prompt_pilot"
    pilot_manifest = json.loads((pilot_dir / "manifest.json").read_text())
    connection = storage._connect(pilot_dir / "requests.sqlite3")
    stop = threading.Event()
    handlers = {
        sig: signal.signal(sig, lambda *_: stop.set())
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        _run_phases(
            connection, ["pilot_base", "pilot_reverse"],
            parallelism=2 * PILOT_PAIRS_PER_LEVEL * len(EXPECTED_LEVELS), stop=stop,
        )
        requests = list(connection.execute(
            "SELECT batch_id,status,served_model,input_tokens,output_tokens FROM requests"
        ))
        complete = sum(row["status"] == "complete" for row in requests)
        exact_model = sum(row["served_model"] == MODEL for row in requests)
        outcomes = {
            (row["phase"], row["audit_of"]): row["winner"]
            for row in connection.execute(
                "SELECT phase,audit_of,winner FROM comparisons ORDER BY comparison_id"
            )
        }
        source_ids = {item["source_comparison_id"] for item in pilot_manifest["selected_pairs"]}
        agreements = sum(
            outcomes[("pilot_base", source_id)] == outcomes[("pilot_reverse", source_id)]
            for source_id in source_ids
        )
        pair_results = []
        for item in pilot_manifest["selected_pairs"]:
            left, right = item["bucket_a"], item["bucket_b"]
            source_id = item["source_comparison_id"]
            base_winner = outcomes[("pilot_base", source_id)]
            reverse_winner = outcomes[("pilot_reverse", source_id)]
            pair_results.append({
                "level": item["level"],
                "source_comparison_id": source_id,
                "candidate_bucket_ids": [calls.bucket_id(left), calls.bucket_id(right)],
                "candidate_identities": [
                    json.loads(json.loads(node)["relevance_bucket"])
                    for node in (left, right)
                ],
                "base_winner_bucket_id": calls.bucket_id(base_winner),
                "reverse_winner_bucket_id": calls.bucket_id(reverse_winner),
                "agrees": base_winner == reverse_winner,
            })
        receipt_rows = list(connection.execute(
            "SELECT receipt_json FROM request_attempt_receipts"
        ))
        reasoning_present = sum(
            bool(json.loads(row[0]).get("reasoning_content")) for row in receipt_rows
        )
        passed = complete == 50 and exact_model == 50 and (
            not PILOT_AGREEMENT_REQUIRED
            or agreements >= PILOT_MINIMUM_AGREEMENT
        )
        report = {
            "version": f"{VERSION}.prompt_pilot_report.v1",
            "status": "passed" if passed else "failed",
            "endpoint_models_at_start": models,
            "requested_model": MODEL,
            "requests_expected": 50,
            "requests_complete": complete,
            "requests_exact_served_model": exact_model,
            "pair_count": len(source_ids),
            "order_reversal_agreements": agreements,
            "minimum_order_reversal_agreements": PILOT_MINIMUM_AGREEMENT,
            "order_reversal_agreement_is_gate": PILOT_AGREEMENT_REQUIRED,
            "pair_results": pair_results,
            "attempt_receipts": len(receipt_rows),
            "attempt_receipts_with_reasoning": reasoning_present,
            "usage": {
                "input_tokens": sum(row["input_tokens"] for row in requests),
                "output_tokens": sum(row["output_tokens"] for row in requests),
            },
            "completed_at": time.time(),
        }
        write_json_atomic(pilot_dir / "pilot_report.json", report)
        report_hash = file_sha256(pilot_dir / "pilot_report.json")
        pilot_manifest.update({
            "status": report["status"], "pilot_report_sha256": report_hash
        })
        write_json_atomic(pilot_dir / "manifest.json", pilot_manifest)
        manifest.update({
            "status": "pilot_passed_awaiting_full_run_approval" if passed else "pilot_failed",
            "pilot_report_sha256": report_hash,
        })
        manifest["pilot"]["manifest_sha256"] = file_sha256(
            pilot_dir / "manifest.json"
        )
        write_json_atomic(output / "manifest.json", manifest)
        return report
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        connection.close()


def _export(connection: sqlite3.Connection, output: Path, manifest: dict) -> None:
    ranking_rows = []
    for level in EXPECTED_LEVELS:
        outcomes = [tuple(row) for row in connection.execute(
            "SELECT bucket_a,bucket_b,winner FROM comparisons WHERE phase=?", (level,)
        )]
        if any(winner not in (left, right) for left, right, winner in outcomes):
            raise ValueError(f"incomplete comparisons: {level}")
        keys = [row[0] for row in connection.execute(
            "SELECT relevance_bucket FROM buckets WHERE relevance_bucket LIKE ?",
            (f'%\"level\":\"{level}\"%',),
        )]
        scores = ranking.fit_bradley_terry(outcomes)
        counts = Counter(key for left, right, _ in outcomes for key in (left, right))
        ordered = sorted(keys, key=lambda key: (-scores[key], key))
        for index, key in enumerate(ordered):
            bucket = json.loads(key)["relevance_bucket"]
            ranking_rows.append({
                "task_id": "bioavailability_ma", "level": level, "node_key": key,
                "relevance_bucket": bucket, "source_id": json.loads(bucket)["source_id"],
                "bradley_terry_score": scores[key], "level_rank": index + 1,
                "level_percentile": 100 * (len(ordered) - 1 - index) / max(1, len(ordered) - 1),
                "comparison_count": counts[key],
            })
    ranking_path = output / "relevance_bucket_rankings.parquet"
    rankings = pd.DataFrame(ranking_rows)
    rankings.to_parquet(ranking_path, index=False)

    comparison_rows = []
    for row in connection.execute(
        "SELECT c.comparison_id,c.phase,c.bucket_a,c.bucket_b,c.winner,"
        "r.requested_model,r.served_model,r.attempts,r.input_tokens,r.output_tokens "
        "FROM comparisons c JOIN requests r USING(batch_id) ORDER BY c.comparison_id"
    ):
        comparison_rows.append({
            "comparison_id": row["comparison_id"],
            "level": row["phase"],
            "bucket_a_id": calls.bucket_id(row["bucket_a"]),
            "bucket_b_id": calls.bucket_id(row["bucket_b"]),
            "winner_bucket_id": calls.bucket_id(row["winner"]),
            "bucket_a_node_key": row["bucket_a"],
            "bucket_b_node_key": row["bucket_b"],
            "winner_node_key": row["winner"],
            "requested_model": row["requested_model"],
            "served_model": row["served_model"],
            "attempts": row["attempts"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
        })
    comparisons_path = output / "comparisons.parquet"
    pd.DataFrame(comparison_rows).to_parquet(comparisons_path, index=False)

    record_map = pd.read_parquet(output / "record_relevance_map.parquet")
    ranked_records = record_map.merge(
        rankings[[
            "level", "node_key", "bradley_terry_score", "level_rank",
            "level_percentile", "comparison_count",
        ]],
        on=["level", "node_key"], how="left", validate="many_to_one",
    )
    if ranked_records.bradley_terry_score.isna().any() or len(ranked_records) != len(record_map):
        raise ValueError("rankings do not cover every L2-L6 record")
    ranked_records_path = output / "record_relevance_rankings.parquet"
    ranked_records.to_parquet(ranked_records_path, index=False)

    manifest["artifacts"] = {
        "comparisons": {
            "path": comparisons_path.name,
            "rows": len(comparison_rows),
            "sha256": file_sha256(comparisons_path),
        },
        "relevance_bucket_rankings": {
            "path": ranking_path.name,
            "rows": len(rankings),
            "sha256": file_sha256(ranking_path),
        },
        "record_relevance_rankings": {
            "path": ranked_records_path.name,
            "rows": len(ranked_records),
            "sha256": file_sha256(ranked_records_path),
        },
    }
    manifest["ranking_sha256"] = manifest["artifacts"]["relevance_bucket_rankings"]["sha256"]


def run_full(
    output: Path,
    *,
    approved_prompt_sha256: str,
    approved_pilot_sha256: str,
    parallelism: int,
) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    report_path = output / "prompt_pilot/pilot_report.json"
    report = json.loads(report_path.read_text())
    if manifest.get("version") != VERSION or approved_prompt_sha256 != file_sha256(TEMPLATE):
        raise ValueError("the prepared V2 artifact and reviewed prompt hash are required")
    if report.get("status") != "passed" or approved_pilot_sha256 != file_sha256(report_path):
        raise ValueError("a separately approved passing pilot report hash is required")
    if parallelism != FULL_PARALLELISM:
        raise ValueError(f"the frozen full-run parallelism is {FULL_PARALLELISM}")
    _verify_endpoint_model()
    connection = storage._connect(output / "requests.sqlite3")
    stop = threading.Event()
    handlers = {
        sig: signal.signal(sig, lambda *_: stop.set())
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    manifest["status"] = "running"
    write_json_atomic(output / "manifest.json", manifest)
    try:
        _run_phases(connection, list(EXPECTED_LEVELS), parallelism=parallelism, stop=stop)
        incomplete = connection.execute(
            "SELECT count(*) FROM comparisons WHERE winner IS NULL"
        ).fetchone()[0]
        if incomplete:
            raise RuntimeError(f"{incomplete} comparisons remain unfinished")
        _export(connection, output, manifest)
        usage = connection.execute(
            "SELECT coalesce(sum(input_tokens),0),coalesce(sum(output_tokens),0) FROM requests"
        ).fetchone()
        manifest.update({
            "status": "complete", "finished_at": time.time(),
            "approved_pilot_sha256": approved_pilot_sha256,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        })
        return manifest
    except Exception:
        manifest["status"] = "incomplete"
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        write_json_atomic(output / "manifest.json", manifest)
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "pilot", "run-full"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--approved-prompt-sha256", default="")
    parser.add_argument("--approved-pilot-sha256", default="")
    parser.add_argument("--parallelism", type=int, default=FULL_PARALLELISM)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.output_dir)
    elif args.command == "pilot":
        result = run_pilot(
            args.output_dir, approved_prompt_sha256=args.approved_prompt_sha256
        )
    else:
        result = run_full(
            args.output_dir,
            approved_prompt_sha256=args.approved_prompt_sha256,
            approved_pilot_sha256=args.approved_pilot_sha256,
            parallelism=args.parallelism,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
