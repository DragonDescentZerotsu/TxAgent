"""Adapt the frozen Skin V9 Gold-v1 direct ranking to ranked_uid_retrieval.v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from predict.retrieval.assay_reranking.build_ranked_retrieval import _clean, _digest
from predict.retrieval.assay_reranking.build_ranked_uid_retrieval import _schema, _write_complete
from predict.retrieval.assay_reranking.ranked_uid_retrieval import CAPACITY, SCHEMA_VERSION
from predict.utils.json import read_jsonl, sha256_file


PROFILE = "ranked_level_retrieval_skin_gold_v1_l1_adapter_v2"
REPO = Path(__file__).resolve().parents[3]
GOLD = REPO / "data/gold_labels/Skin_Reaction/v1/scaffold"
V9 = REPO / (
    "predict/retrieval/cache/assay_reranking/active/"
    "v9_skin_gold_v1_scaffold_morgan100_v1/skin_reaction/scaffold"
)
DEFAULT_OUTPUT = REPO / "data/caches/assay_reranking/active" / PROFILE / "skin_reaction"


def _member_uid(context_id: str, rank: int, record_id: str) -> str:
    digest = hashlib.sha256(f"{context_id}\0{rank}\0{record_id}".encode()).hexdigest()[:24]
    return f"skin_gold_v1_member_{digest}"


def _cards() -> dict[str, dict]:
    rows = read_jsonl(GOLD / "train_molecule_condition_labels.jsonl")
    return {str(row["benchmark_row_id"]): row for row in rows}


def _projection(root: Path, cards: dict[str, dict], context_ids: set[str]) -> Path:
    rows = []
    for context_id in sorted(context_ids):
        card = cards[context_id]
        record_ids = [str(value) for value in card.get("source_record_ids") or []]
        if not record_ids:
            raise ValueError(f"Skin Gold card has no physical source members: {context_id}")
        values = list(card.get("raw_value_examples") or [])
        contexts = list(card.get("context_examples") or card.get("condition_text_examples") or [])
        for rank, record_id in enumerate(record_ids, 1):
            source_row_uid = _member_uid(context_id, rank, record_id)
            fields = {
                "source_record_id": record_id,
                "reported_value": values[min(rank - 1, len(values) - 1)] if values else "",
                "assay_context": contexts[min(rank - 1, len(contexts) - 1)] if contexts else "",
                "gold_vote_label": int(card["Y"]),
            }
            fields = {key: value for key, value in fields.items() if value not in (None, "")}
            payload = {
                "task_id": "skin_reaction", "progressive_level": "L1",
                "family_key": "direct_skin_reaction",
                "source_id": "skin_gold_v1",
                "record_id": record_id,
                "source_row_uid": source_row_uid,
                "measurement_kind": "categorical_label",
                "label_source": "gold_v1", "benchmark_row_id": context_id,
                "canonical_smiles": str(card["drug"]),
                "condition_group": str(card["condition_group"]), "Y": int(card["Y"]),
                "source_record_id": record_id,
                "source_fields": fields,
                "source_contract": {
                    "contract_version": "skin_gold_v1_aggregate_member_adapter.v1",
                    "source_or_simply_cleaned": {key: True for key in fields},
                },
            }
            rows.append({
                "source_row_uid": source_row_uid,
                "external_record_id": record_id,
                "parent_id": str(card["molecule_identity_key"]),
                "parent_smiles": str(card["drug"]), "level": "L1",
                "payload": json.dumps(_clean(payload), sort_keys=True, separators=(",", ":")),
            })
    target = root / "evidence"
    target.mkdir(parents=True)
    records = target / "records.parquet"
    schema = pa.schema([
        ("source_row_uid", pa.string()), ("external_record_id", pa.string()),
        ("parent_id", pa.string()), ("parent_smiles", pa.string()),
        ("level", pa.string()), ("payload", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), records, compression="zstd")
    identity = {
        "schema_version": "ranked_evidence_projection.v1", "status": "complete",
        "task_id": "skin_reaction", "record_count": len(rows),
        "source_identity": "skin_gold_v1_ordered_physical_member_adapter",
        "inputs": {"gold_train_sha256": sha256_file(GOLD / "train_molecule_condition_labels.jsonl")},
    }
    identity["content_id"] = _digest(identity)
    manifest = {**identity, "records": records.name, "records_sha256": sha256_file(records)}
    path = target / "VERSION.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _split(root: Path, subset: str, cards: dict[str, dict], evidence: Path) -> dict:
    source_manifest = V9 / subset / "VERSION.json"
    source = json.loads(source_manifest.read_text())
    rankings_path = source_manifest.with_name(source["rankings"])
    rows = pq.read_table(rankings_path).to_pylist()
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_record_id"]), []).append(row)
    queries = {str(row["benchmark_row_id"]): row for row in read_jsonl(GOLD / f"{subset}.jsonl")}
    target = root / "scaffold" / subset / "L1"
    target.mkdir(parents=True)
    database = target / "rankings.sqlite3"
    connection = sqlite3.connect(database)
    _schema(connection, "L1")
    context_ids: set[str] = set()
    ranking_rows = []
    query_rows = []
    query_counts = {}
    for query_id, query in queries.items():
        candidates = by_query.get(query_id) or []
        if not candidates:
            raise ValueError(f"Skin V9 ranking omits {subset}/{query_id}")
        query_rows.append((
            query_id, str(query["drug"]), str(query["molecule_identity_key"]), str(query["drug"])
        ))
        morgan_order = {
            id(row): rank for rank, row in enumerate(sorted(
                candidates,
                key=lambda row: (
                    int(row["retrieval_parent_rank"]),
                    int(row["retrieval_parent_context_index"]),
                    str(row["retrieval_record_id"]),
                ),
            ), 1)
        }
        assay_order = {
            id(row): rank for rank, row in enumerate(sorted(
                candidates,
                key=lambda row: (-float(row["prob_transfer"]), str(row["retrieval_record_id"])),
            ), 1)
        }
        parents = set()
        member_total = 0
        for row in candidates:
            context_id = str(row["retrieval_record_id"])
            card = cards[context_id]
            count = len(card.get("source_record_ids") or [])
            context_ids.add(context_id)
            parents.add(str(row["retrieval_molecule_identity_key"]))
            member_total += count
            ranking_rows.append((
                query_id, context_id, str(row["retrieval_molecule_identity_key"]),
                str(row["retrieval_smiles"]), float(row["morgan_tanimoto_similarity"]),
                int(row["retrieval_parent_rank"]), int(row["retrieval_parent_context_index"]) + 1,
                morgan_order[id(row)], float(row["prob_transfer"]), assay_order[id(row)],
                context_id, context_id, count, count, str(row["prompt_hash"]),
            ))
        if len(parents) != CAPACITY:
            raise ValueError(f"Skin V9 parent capacity changed: {subset}/{query_id}")
        query_counts[query_id] = {
            "candidate_parents": CAPACITY,
            "morgan_candidate_records": member_total,
            "assay_candidate_records": member_total,
        }
    connection.executemany("INSERT INTO queries VALUES (?,?,?,?)", query_rows)
    connection.executemany("INSERT INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ranking_rows)
    connection.executemany("INSERT INTO contexts VALUES (?,?,?,?)", [
        (context_id, str(cards[context_id]["molecule_identity_key"]),
         str(cards[context_id]["condition_group"]), int(cards[context_id]["Y"]))
        for context_id in sorted(context_ids)
    ])
    connection.executemany("INSERT INTO context_records VALUES (?,?,?)", [
        (context_id, _member_uid(context_id, rank, str(record_id)), rank)
        for context_id in sorted(context_ids)
        for rank, record_id in enumerate(cards[context_id]["source_record_ids"], 1)
    ])
    identity = {
        "task_id": "skin_reaction", "subset": subset, "level": "L1", "pool": "fixed",
        "capacity": CAPACITY, "parent_capacity": CAPACITY, "gold_release": "v1",
        "neighbor_identity_policy": "scaffold_disjoint", "shared_candidate_universe": True,
        "query_count": len(queries), "stored_rows": len(ranking_rows), "query_counts": query_counts,
        "model": source["model"], "assay_transfer_status": "complete",
        "inputs": {
            "v9_manifest_sha256": sha256_file(source_manifest),
            "v9_rankings_sha256": sha256_file(rankings_path),
            "evidence_manifest_sha256": sha256_file(evidence),
        },
    }
    manifest = _write_complete(
        connection, database, task="skin_reaction", subset=subset,
        level="L1", identity=identity, target=target,
    )
    connection.close()
    return manifest


def build(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to replace immutable adapter: {output}")
    cards = _cards()
    context_ids = {
        str(row["retrieval_record_id"])
        for subset in ("valid", "test")
        for row in pq.read_table(V9 / subset / "rankings.parquet", columns=["retrieval_record_id"]).to_pylist()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skin-l1-adapter.", dir=output.parent) as temporary:
        root = Path(temporary)
        evidence = _projection(root, cards, context_ids)
        split_manifests = {subset: _split(root, subset, cards, evidence) for subset in ("valid", "test")}
        evidence_doc = json.loads(evidence.read_text())
        index = {
            "schema_version": "ranked_uid_task_release_index.v1",
            "selection_contract": SCHEMA_VERSION, "profile": PROFILE,
            "task_id": "skin_reaction", "status": "complete", "gold_release": "v1",
            "pool": "all", "parent_capacity": CAPACITY, "levels_independent": True,
            "ranking_modes": ["morgan", "assay-transfer"], "assay_transfer_status": "complete",
            "neighbor_identity_policy_by_level": {"L1": "scaffold_disjoint"},
            "evidence": {
                "manifest": os.path.relpath(evidence, root),
                "manifest_sha256": sha256_file(evidence),
                "content_id": evidence_doc["content_id"], "record_count": evidence_doc["record_count"],
            },
            "splits": {subset: {"levels": {"L1": {
                "manifest": f"scaffold/{subset}/L1/VERSION.json",
                "manifest_sha256": sha256_file(root / f"scaffold/{subset}/L1/VERSION.json"),
                "content_id": split_manifests[subset]["content_id"],
            }}} for subset in ("valid", "test")},
        }
        (root / "RELEASE_INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        os.replace(root, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
