"""Publish Skin's scaffold-disjoint ranking with canonical physical voters."""

from __future__ import annotations

import argparse
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
from predict.retrieval.policies import normalize_molecule_identity
from data.processing.gold_labels.voter_membership import SOURCE_ROW_UID_PATTERN
from data.processing.gold_labels.publish_gold_v1_context_membership import verify_membership
from .runtime import cache_profile_root


PROFILE = "flat_v5/gold_v1/skin_reaction/l1/assay_transfer/v9_0_2/scaffold_disjoint_parent100_v1"
REPO = Path(__file__).resolve().parents[3]
GOLD = REPO / "data/gold_labels/Skin_Reaction/v1/scaffold"
DEFAULT_OUTPUT = REPO / "data/caches/assay_reranking/active" / PROFILE
V9 = DEFAULT_OUTPUT / "scoring"


def _cards(gold: Path = GOLD, library: Path | None = None) -> dict[str, dict]:
    library_root = REPO / "data/evidence_libraries/skin_reaction"
    library = library or library_root / (library_root / "CURRENT").read_text().strip()
    verify_membership(gold, library)
    rows = read_jsonl(gold / "train_molecule_condition_labels.jsonl")
    cards = {str(row["benchmark_row_id"]): row for row in rows}
    membership = pq.read_table(gold / "voter_membership.parquet",
                               filters=[("split", "=", "train")]).to_pylist()
    for card in cards.values():
        card["source_row_uids"] = []
    for row in membership:
        cards[str(row["benchmark_row_id"])]["source_row_uids"].append(str(row["source_row_uid"]))
    for context_id, card in cards.items():
        if len(card["source_row_uids"]) != int(card["source_record_count"]):
            raise ValueError(f"Skin membership differs from frozen Gold vote count: {context_id}")
    return cards


def _projection(root: Path, cards: dict[str, dict], context_ids: set[str], *,
                gold: Path = GOLD, library: Path | None = None) -> Path:
    library_root = REPO / "data/evidence_libraries/skin_reaction"
    library = library or library_root / (library_root / "CURRENT").read_text().strip()
    stage3 = library / "03_pair_buckets/records.parquet"
    source_contract = library / "02_canonicalized/source_contract.json"
    contract = json.loads(source_contract.read_text())
    parents = {}
    for context_id in sorted(context_ids):
        card = cards[context_id]
        for uid in card["source_row_uids"]:
            parent = (str(card["molecule_identity_key"]), str(card["drug"]))
            if uid in parents and parents[uid] != parent:
                raise ValueError(f"Skin physical voter belongs to different Gold parents: {uid}")
            parents[uid] = parent
    uids = set(parents)
    if not uids or any(not SOURCE_ROW_UID_PATTERN.fullmatch(uid) for uid in uids):
        raise ValueError("Skin Gold membership requires canonical physical source_row_uid values")
    physical = pq.read_table(stage3, filters=[("source_row_uid", "in", sorted(uids))]).to_pylist()
    if len(physical) != len(uids) or {str(row["source_row_uid"]) for row in physical} != uids:
        raise ValueError("Skin Gold members are absent or nonunique in the canonical evidence library")
    rows = []
    for raw in sorted(physical, key=lambda row: row["source_row_uid"]):
        parent_id, parent_smiles = parents[str(raw["source_row_uid"])]
        identity = normalize_molecule_identity(str(raw["canonical_smiles"]))
        if identity.status != "ok" or (identity.parent_inchi_key or identity.parent_smiles) != parent_id:
            raise ValueError(f"Skin canonical voter structure differs from frozen Gold parent: {raw['source_row_uid']}")
        source = str(raw["source_id"])
        spec = contract["sources"][source]
        fields = spec.get("source_visible_fields")
        if fields is None:
            fields = [name for name, policy in spec["normalized_artifact_columns"].items()
                      if policy.get("source_or_simply_cleaned") is True]
        source_fields = {name: raw.get(name) for name in fields}
        payload = {
            "task_id": "skin_reaction", "progressive_level": "L1",
            "family_key": "direct_skin_reaction", "source_id": source,
            "source_row_uid": str(raw["source_row_uid"]),
            "record_id": str(raw["source_record_id"]),
            "measurement_kind": str(raw.get("measurement_kind") or ""),
            "canonical_smiles": str(raw["canonical_smiles"]),
            "source_fields": source_fields,
            "source_contract": {
                "contract_version": "skin_gold_canonical_voters.v1",
                "source_or_simply_cleaned": {name: True for name in source_fields},
            },
        }
        rows.append({
            "source_row_uid": str(raw["source_row_uid"]),
            "external_record_id": str(raw["source_record_id"]),
            "parent_id": parent_id, "parent_smiles": parent_smiles, "level": "L1",
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
        "source_identity": "skin_gold_canonical_voters.v1", "evidence_library_release": library.name,
        "inputs": {"records_sha256": sha256_file(stage3),
                   "source_contract_sha256": sha256_file(source_contract),
                   "membership_sha256": sha256_file(gold / "voter_membership.parquet")},
    }
    identity["content_id"] = _digest(identity)
    manifest = {**identity, "records": records.name, "records_sha256": sha256_file(records)}
    path = target / "VERSION.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _split(root: Path, subset: str, cards: dict[str, dict], evidence: Path, *,
           gold: Path = GOLD, ranking_root: Path = V9) -> dict:
    source_manifest = ranking_root / subset / "VERSION.json"
    source = json.loads(source_manifest.read_text())
    if source.get("inputs", {}).get("neighbor_identity_policy") != "scaffold_disjoint":
        raise ValueError("Direct Skin L1 requires scaffold-disjoint source rankings")
    rankings_path = source_manifest.with_name(source["rankings"])
    rows = pq.read_table(rankings_path).to_pylist()
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_record_id"]), []).append(row)
    queries = {str(row["benchmark_row_id"]): row for row in read_jsonl(gold / f"{subset}.jsonl")}
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
            count = len(card["source_row_uids"])
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
        (context_id, uid, rank)
        for context_id in sorted(context_ids)
        for rank, uid in enumerate(cards[context_id]["source_row_uids"], 1)
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


def build(output: Path, *, gold: Path = GOLD, library: Path | None = None,
          ranking_root: Path = V9) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to replace immutable adapter: {output}")
    cards = _cards(gold, library)
    context_ids = {
        str(row["retrieval_record_id"])
        for subset in ("valid", "test")
        for row in pq.read_table(ranking_root / subset / "rankings.parquet", columns=["retrieval_record_id"]).to_pylist()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".skin-l1-adapter.", dir=output.parent) as temporary:
        root = Path(temporary)
        evidence = _projection(root, cards, context_ids, gold=gold, library=library)
        split_manifests = {subset: _split(root, subset, cards, evidence, gold=gold, ranking_root=ranking_root) for subset in ("valid", "test")}
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
    parser.add_argument("--gold-root", type=Path, default=GOLD)
    parser.add_argument("--evidence-library", type=Path)
    parser.add_argument("--ranking-root", type=Path, default=V9)
    args = parser.parse_args()
    build(args.output.resolve(), gold=args.gold_root, library=args.evidence_library, ranking_root=args.ranking_root)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
