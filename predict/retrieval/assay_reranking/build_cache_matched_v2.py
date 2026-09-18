"""Publish one self-contained cache_matched_retrieval.v2 SQLite DB per task/split."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Callable

from rdkit import DataStructs

from data.processing.gold_labels.conditioned_benchmark import split_path
from predict.retrieval.policies import (
    normalize_molecule_identity,
    seeded_rank_tie_key,
    standardize_smiles_and_fp,
)
from predict.utils.json import read_jsonl, sha256_file

from .cache_matched import (
    DEFAULT_GOLD_CONTEXT_MAPPING,
    _load_candidates_v1,
    load_cache_policy,
)
from .cache_matched_v2 import (
    ACTIVE_GOLD_RELEASES,
    DATABASE_NAME,
    FIXED_POOL,
    POOLS,
    SCHEMA_VERSION,
)


from predict.retrieval.assay_reranking.runtime import cache_profile_root

ROOT = cache_profile_root("cache_matched_retrieval_v2")
LEGACY_BUNDLE = Path(__file__).with_name("cache_matched_three_pools_v1.yaml")
L1_MOLECULE_CAPACITY = 10
L1_RECORD_CAPACITY = 10
LATER_RECORD_CAPACITY = 50


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=None)
def _fingerprint(smiles: str):
    _, _, fingerprint = standardize_smiles_and_fp(smiles)
    if fingerprint is None:
        raise ValueError(f"Invalid cached molecule: {smiles}")
    return fingerprint


@lru_cache(maxsize=None)
def _similarity(query_smiles: str, parent_smiles: str) -> float:
    return float(DataStructs.TanimotoSimilarity(
        _fingerprint(query_smiles), _fingerprint(parent_smiles)
    ))


def _source_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    fields = dict(payload.get("source_fields") or {})
    payload.setdefault("family_key", payload.get("level_family"))
    payload.setdefault("source_canonical_smiles", payload.get("canonical_smiles"))
    payload.setdefault("source_contract", {
        "contract_version": "source_column_contract.v2",
        "source_or_simply_cleaned": {name: True for name in fields},
    })
    payload.setdefault("source_projection", "library_source_contract.v2")
    missing = [key for key in ("record_id", "source_id", "progressive_level", "family_key")
               if not payload.get(key)]
    if missing:
        raise ValueError(f"Cached payload lacks semantic identity fields: {missing}")
    return payload


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA user_version=2;
        CREATE TABLE metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE queries(
            query_id INTEGER PRIMARY KEY,
            query_parent_id TEXT NOT NULL,
            query_parent_smiles TEXT NOT NULL
        );
        CREATE TABLE benchmark_queries(
            benchmark_row_id TEXT PRIMARY KEY,
            drug TEXT NOT NULL,
            query_id INTEGER NOT NULL UNIQUE REFERENCES queries(query_id)
        ) WITHOUT ROWID;
        CREATE TABLE records(
            record_key INTEGER PRIMARY KEY,
            external_record_id TEXT NOT NULL UNIQUE,
            level TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE assignments(
            query_id INTEGER NOT NULL REFERENCES queries(query_id),
            pool TEXT NOT NULL,
            level TEXT NOT NULL,
            parent_id TEXT NOT NULL,
            parent_smiles TEXT NOT NULL,
            record_key INTEGER NOT NULL REFERENCES records(record_key),
            within_parent_rank INTEGER NOT NULL CHECK(within_parent_rank > 0),
            morgan_similarity REAL NOT NULL CHECK(morgan_similarity >= 0 AND morgan_similarity <= 1),
            morgan_rank INTEGER NOT NULL CHECK(morgan_rank > 0),
            assay_transfer_score REAL,
            assay_rank INTEGER,
            PRIMARY KEY(query_id,pool,level,record_key),
            CHECK((assay_transfer_score IS NULL) = (assay_rank IS NULL))
        ) WITHOUT ROWID;
        CREATE TABLE selection_counts(
            query_id INTEGER NOT NULL REFERENCES queries(query_id),
            pool TEXT NOT NULL,
            level TEXT NOT NULL,
            candidate_record_count INTEGER NOT NULL,
            candidate_parent_count INTEGER NOT NULL,
            PRIMARY KEY(query_id,pool,level)
        ) WITHOUT ROWID;
        CREATE INDEX assignments_morgan_rank
            ON assignments(query_id,pool,level,morgan_rank);
        CREATE INDEX assignments_assay_rank
            ON assignments(query_id,pool,level,assay_rank);
    """)


class Writer:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.record_keys: dict[str, int] = {}
        self.record_json: dict[str, str] = {}

    def record(self, payload: dict[str, Any]) -> int:
        payload = _source_payload(payload)
        record_id = str(payload["record_id"])
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if record_id in self.record_keys:
            if self.record_json[record_id] != encoded:
                raise ValueError(f"Conflicting cached source payload: {record_id}")
            return self.record_keys[record_id]
        key = len(self.record_keys) + 1
        self.record_keys[record_id] = key
        self.record_json[record_id] = encoded
        self.connection.execute(
            "INSERT INTO records VALUES (?,?,?,?)",
            (key, record_id, str(payload["progressive_level"]), encoded),
        )
        return key

    def assignment(
        self, *, query_id: int, pool: str, level: str, parent_id: str,
        parent_smiles: str, payload: dict[str, Any], within_parent_rank: int,
        similarity: float, morgan_rank: int, assay_score: float | None,
        assay_rank: int | None,
    ) -> None:
        if not math.isfinite(similarity) or not 0 <= similarity <= 1:
            raise ValueError("Invalid Morgan similarity")
        if (assay_score is None) != (assay_rank is None):
            raise ValueError("Assay score and rank must be present together")
        if assay_score is not None and (not math.isfinite(assay_score) or not 0 <= assay_score <= 1):
            raise ValueError("Invalid assay-transfer score")
        self.connection.execute(
            "INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (query_id, pool, level, parent_id, parent_smiles, self.record(payload),
             within_parent_rank, similarity, morgan_rank, assay_score, assay_rank),
        )


def _rank_later_rows(
    rows: list[tuple[str, str, dict[str, Any], float]], *, query_smiles: str,
    task: str, benchmark_row_id: str, level: str,
    parent_smiles_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    parent_forms: dict[str, set[str]] = defaultdict(set)
    for _, parent_id, payload, _ in rows:
        parent_forms[parent_id].update(
            str(value) for value in (
                payload.get("identity_smiles_forms") or [payload["canonical_smiles"]]
            ) if value
        )
    resolved_parent_smiles: dict[str, str] = {}
    similarities: dict[str, float] = {}
    for parent_id, forms in parent_forms.items():
        parent_smiles = str((parent_smiles_by_id or {}).get(parent_id) or "")
        identity = normalize_molecule_identity(parent_smiles or sorted(forms)[0])
        parent_smiles = str(identity.parent_smiles or "")
        identity_key = identity.parent_inchi_key or parent_smiles
        if identity.status != "ok" or not parent_smiles or identity_key != parent_id:
            raise ValueError(f"Cannot resolve cached parent: {parent_id}")
        resolved_parent_smiles[parent_id] = parent_smiles
        similarities[parent_id] = _similarity(query_smiles, parent_smiles)

    staged = []
    for record_id, parent_id, payload, score in rows:
        tie = seeded_rank_tie_key(0, task, benchmark_row_id, level, parent_id, record_id)
        staged.append({
            "record_id": record_id,
            "parent_id": parent_id,
            "parent_smiles": resolved_parent_smiles[parent_id],
            "payload": payload,
            "similarity": similarities[parent_id],
            "score": float(score),
            "tie": tie,
        })
    by_morgan = sorted(staged, key=lambda row: (-row["similarity"], row["tie"]))
    by_assay = sorted(staged, key=lambda row: (-row["score"], row["tie"]))
    morgan_ranks = {row["record_id"]: rank for rank, row in enumerate(by_morgan, 1)}
    assay_ranks = {row["record_id"]: rank for rank, row in enumerate(by_assay, 1)}
    within: dict[str, int] = defaultdict(int)
    for row in by_morgan:
        within[row["parent_id"]] += 1
        row["within_parent_rank"] = within[row["parent_id"]]
        row["morgan_rank"] = morgan_ranks[row["record_id"]]
        row["assay_rank"] = assay_ranks[row["record_id"]]
    return by_morgan


def build(
    *, task: str, subset: str, library: Path, mapper: Path,
    output_root: Path | None = None, legacy_bundle: Path = LEGACY_BUNDLE,
    gold_context_mapping: Path | None = None,
    allow_frozen_l1_vote_scores: bool = False,
    schema_version: str = SCHEMA_VERSION,
    l5_ranker: Callable[[str, str, str], dict[str, Any]] | None = None,
    l5_pool_contract: dict[str, Any] | None = None,
    extra_inputs: dict[str, dict[str, Any]] | None = None,
    canonical_parent_smiles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a cache once from frozen inputs, then publish it atomically."""
    output_root = (output_root or ROOT / task / "scaffold" / subset).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace an existing cache: {output_root}")
    query_path = split_path(task, subset).with_name(
        f"{subset}_molecule_condition_labels.jsonl"
    ).resolve()
    frozen = read_jsonl(query_path)
    queries = {str(row["benchmark_row_id"]): str(row["drug"]) for row in frozen}
    if len(queries) != len(frozen):
        raise ValueError("Duplicate benchmark row IDs")

    legacy_policy = load_cache_policy(legacy_bundle, task, subset, "joint")
    replay_policy = legacy_policy
    if l5_ranker is not None:
        replay_policy = {
            **legacy_policy,
            "stages": {"L1": legacy_policy["stages"]["L1"]},
            "cache_manifests": {"L1": legacy_policy["cache_manifests"]["L1"]},
        }
    l1_molecules, legacy_later, legacy_audit = _load_candidates_v1(
        queries, task=task, subset=subset, library=library, mapper=mapper,
        policy=replay_policy, molecule_limit=10, l1_limit=L1_RECORD_CAPACITY,
        later_limit=LATER_RECORD_CAPACITY, tie_seed=0,
        gold_context_mapping=gold_context_mapping,
        allow_frozen_l1_vote_scores=allow_frozen_l1_vote_scores,
        cache_pool="tool-accepted", _all_l1_candidates=True,
        _accept_pinned_legacy_builder=True,
        _parent_scoped_l5_similarity=l5_ranker is None,
        _filter_l1_record_overlap=l5_ranker is not None,
        _skip_redundant_quick_check=True,
    )
    parent_smiles_by_id = (
        canonical_parent_smiles
        or legacy_audit.pop("_parent_smiles_by_id", {})
    )
    later_manifest_path = Path(legacy_policy["cache_manifests"][
        next(level for level in legacy_policy["cache_manifests"] if level != "L1")
    ]).resolve()
    later_manifest = json.loads(later_manifest_path.read_text())
    later_database = later_manifest_path.with_name("scores.sqlite3")
    if (later_manifest.get("schema_version") != "assay_transfer_three_pools.v1"
            or later_manifest.get("status") != "complete"
            or later_manifest.get("task_id") != task
            or later_manifest.get("subset") != subset):
        raise ValueError("Legacy three-pool cache is incomplete or incompatible")
    with sqlite3.connect(f"file:{later_database}?mode=ro", uri=True) as old:
        old_records = {
            int(record_key): (
                str(record_id), str(parent_id), _source_payload(json.loads(payload))
            )
            for record_key, record_id, parent_id, payload in old.execute(
                "SELECT record_key,external_record_id,parent_id,payload FROM records"
            )
        }
        output_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as tmp:
            tmp_root = Path(tmp)
            database = tmp_root / DATABASE_NAME
            connection = sqlite3.connect(database)
            _schema(connection)
            writer = Writer(connection)
            assignment_counts: dict[str, int] = defaultdict(int)
            try:
                for query_id, (benchmark_row_id, drug) in enumerate(queries.items(), 1):
                    query_identity = normalize_molecule_identity(drug)
                    query_parent_smiles = str(query_identity.parent_smiles or "")
                    if query_identity.status != "ok" or not query_parent_smiles:
                        raise ValueError(f"Invalid frozen query parent: {benchmark_row_id}")
                    query_parent_id = query_identity.parent_inchi_key or query_parent_smiles
                    connection.execute(
                        "INSERT INTO queries VALUES (?,?,?)",
                        (query_id, query_parent_id, query_parent_smiles),
                    )
                    connection.execute(
                        "INSERT INTO benchmark_queries VALUES (?,?,?)",
                        (benchmark_row_id, drug, query_id),
                    )

                    molecules = l1_molecules[benchmark_row_id]
                    if len(molecules) < L1_MOLECULE_CAPACITY:
                        raise ValueError(f"Insufficient L1 cache capacity: {benchmark_row_id}")
                    l1_similarity = {
                        str(molecule["reference_molecule_id"]): _similarity(
                            query_parent_smiles,
                            parent_smiles_by_id.get(
                                str(molecule["reference_molecule_id"]),
                                str(molecule["canonical_smiles"]),
                            ),
                        )
                        for molecule in molecules
                    }
                    original_l1_morgan_rank = {
                        str(molecule["reference_molecule_id"]): int(molecule["morgan_top5_rank"])
                        for molecule in molecules
                    }
                    l1_morgan_rank = {
                        parent_id: rank
                        for rank, parent_id in enumerate(sorted(
                            l1_similarity,
                            key=lambda value: (
                                -l1_similarity[value],
                                original_l1_morgan_rank[value],
                            ),
                        ), 1)
                    }
                    for molecule in molecules:
                        parent_id = str(molecule["reference_molecule_id"])
                        parent_smiles = parent_smiles_by_id.get(
                            parent_id, str(molecule["canonical_smiles"])
                        )
                        morgan_rank = l1_morgan_rank[parent_id]
                        assay_rank = int(molecule["assay_transfer_top5_rank"])
                        similarity = l1_similarity[parent_id]
                        assay_score = float(molecule["transfer_likelihood"])
                        for within_rank, row in enumerate(molecule["l1_records"], 1):
                            writer.assignment(
                                query_id=query_id, pool=FIXED_POOL, level="L1",
                                parent_id=parent_id, parent_smiles=parent_smiles,
                                payload=row["payload"], within_parent_rank=within_rank,
                                similarity=similarity, morgan_rank=morgan_rank,
                                assay_score=assay_score, assay_rank=assay_rank,
                            )
                            assignment_counts["fixed/L1"] += 1
                    connection.execute(
                        "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
                        (query_id, FIXED_POOL, "L1",
                         sum(len(molecule["l1_records"]) for molecule in molecules),
                         len(molecules)),
                    )

                    old_query = old.execute(
                        "SELECT query_id,drug FROM benchmark_queries WHERE benchmark_row_id=?",
                        (benchmark_row_id,),
                    ).fetchone()
                    if old_query is None or old_query[1] != drug:
                        raise ValueError(f"Legacy pool query mismatch: {benchmark_row_id}")
                    for pool in POOLS:
                        for level in sorted(later_manifest["models"], key=lambda value: int(value[1:])):
                            raw_rows = old.execute(
                                """SELECT a.record_key,s.transfer_probability
                                   FROM assignments AS a JOIN scores AS s USING(score_key)
                                   WHERE a.query_id=? AND a.pool=? AND a.level=?""",
                                (old_query[0], pool, level),
                            ).fetchall()
                            rows = [(*old_records[int(record_key)], float(score))
                                    for record_key, score in raw_rows]
                            ranked = _rank_later_rows(
                                rows, query_smiles=query_parent_smiles, task=task,
                                benchmark_row_id=benchmark_row_id, level=level,
                                parent_smiles_by_id=parent_smiles_by_id,
                            )
                            retained = [row for row in ranked
                                        if row["morgan_rank"] <= LATER_RECORD_CAPACITY
                                        or row["assay_rank"] <= LATER_RECORD_CAPACITY]
                            connection.execute(
                                "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
                                (query_id, pool, level, len(rows),
                                 len({parent_id for _, parent_id, _, _ in rows})),
                            )
                            for row in retained:
                                writer.assignment(
                                    query_id=query_id, pool=pool, level=level,
                                    parent_id=row["parent_id"], parent_smiles=row["parent_smiles"],
                                    payload=row["payload"],
                                    within_parent_rank=row["within_parent_rank"],
                                    similarity=row["similarity"], morgan_rank=row["morgan_rank"],
                                    assay_score=row["score"], assay_rank=row["assay_rank"],
                                )
                                assignment_counts[f"{pool}/{level}"] += 1

                    if l5_ranker is None:
                        l5 = legacy_later[benchmark_row_id].get("L5", {}).get("records", [])
                        l5_audit = legacy_audit["query_audits"][benchmark_row_id]["L5"]
                        connection.execute(
                            "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
                            (query_id, FIXED_POOL, "L5", l5_audit["candidate_records"],
                             l5_audit["candidate_molecules"]),
                        )
                        for rank, row in enumerate(l5, 1):
                            payload = _source_payload(row["payload"])
                            parent_id = str(row["reference_molecule_id"])
                            same_parent = sum(
                                1 for earlier in l5[:rank]
                                if str(earlier["reference_molecule_id"]) == parent_id
                            )
                            writer.assignment(
                                query_id=query_id, pool=FIXED_POOL, level="L5",
                                parent_id=parent_id,
                                parent_smiles=str(row["reference_parent_smiles"]),
                                payload=payload, within_parent_rank=same_parent,
                                similarity=float(row["morgan_similarity"]), morgan_rank=rank,
                                assay_score=None, assay_rank=None,
                            )
                            assignment_counts["fixed/L5"] += 1
                    else:
                        for pool in POOLS:
                            ranked = l5_ranker(benchmark_row_id, query_parent_smiles, pool)
                            rows = ranked["records"]
                            if len(rows) < LATER_RECORD_CAPACITY:
                                raise ValueError(
                                    f"Insufficient L5 cache capacity: {benchmark_row_id}/{pool}"
                                )
                            connection.execute(
                                "INSERT INTO selection_counts VALUES (?,?,?,?,?)",
                                (query_id, pool, "L5", ranked["candidate_records"],
                                 ranked["candidate_parents"]),
                            )
                            for row in rows[:LATER_RECORD_CAPACITY]:
                                writer.assignment(
                                    query_id=query_id, pool=pool, level="L5",
                                    parent_id=row["parent_id"],
                                    parent_smiles=row["parent_smiles"],
                                    payload=row["payload"],
                                    within_parent_rank=row["within_parent_rank"],
                                    similarity=row["similarity"],
                                    morgan_rank=row["morgan_rank"],
                                    assay_score=None, assay_rank=None,
                                )
                                assignment_counts[f"{pool}/L5"] += 1
                    connection.commit()

                identity = {
                    "schema_version": schema_version,
                    "task_id": task,
                    "subset": subset,
                    "gold_release": ACTIVE_GOLD_RELEASES[task][1],
                    "query_sha256": sha256_file(query_path),
                    "legacy_bundle_sha256": sha256_file(legacy_bundle),
                    "legacy_later_manifest_sha256": sha256_file(later_manifest_path),
                    "legacy_later_cache_sha256": later_manifest["cache_sha256"],
                    "legacy_selection_inputs_sha256": _digest(legacy_audit["inputs"]),
                    "assignment_counts": dict(sorted(assignment_counts.items())),
                    "record_count": len(writer.record_keys),
                }
                if l5_ranker is not None:
                    identity.update(
                        l5_pool_contract=l5_pool_contract,
                        l5_inputs_sha256=_digest(extra_inputs or {}),
                        l1_record_overlap_exclusions_sha256=_digest(
                            legacy_audit["l1_record_overlap_exclusions"]
                        ),
                    )
                content_id = _digest(identity)
                connection.executemany("INSERT INTO metadata VALUES (?,?)", [
                    ("schema_version", schema_version),
                    ("content_id", content_id),
                    ("task_id", task),
                    ("subset", subset),
                ])
                connection.commit()
                connection.execute("VACUUM")
                if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise ValueError("Published cache failed integrity check")
            finally:
                connection.close()

            manifest = {
                **identity,
                "content_id": content_id,
                "status": "complete",
                "database": DATABASE_NAME,
                "database_sha256": sha256_file(database),
                "tie_seed": 0,
                "pools": list(POOLS),
                "fixed_pool_levels": ["L1"] if l5_ranker is not None else ["L1", "L5"],
                "native_rankings": ["morgan", "assay_transfer"],
                "joint_materialized": False,
                "morgan_fingerprint": {"radius": 2, "bits": 2048, "similarity": "Tanimoto"},
                "morgan_similarity_scope": "query_to_cached_parent_smiles",
                **({"l5_pool_contract": l5_pool_contract} if l5_ranker is not None else {}),
                "capacities": {
                    "l1_molecules": L1_MOLECULE_CAPACITY,
                    "l1_records_per_molecule": L1_RECORD_CAPACITY,
                    "later_records_per_level": LATER_RECORD_CAPACITY,
                },
                "inputs": {
                    "query_ledger": {"path": str(query_path), "sha256": sha256_file(query_path)},
                    "legacy_bundle": {"path": str(Path(legacy_bundle).resolve()), "sha256": sha256_file(legacy_bundle)},
                    "legacy_later_manifest": {"path": str(later_manifest_path), "sha256": sha256_file(later_manifest_path)},
                    "legacy_later_database": {"path": str(later_database), "sha256": later_manifest["cache_sha256"]},
                    **(extra_inputs or {}),
                },
                "legacy_selection_audit": {
                    "selection_policy": legacy_audit["selection_policy"],
                    "inputs": legacy_audit["inputs"],
                    "v9": legacy_audit["v9"],
                    "gold_context_mapping": legacy_audit.get("gold_context_mapping"),
                    **({"l1_record_overlap_exclusions": legacy_audit[
                        "l1_record_overlap_exclusions"
                    ]} if l5_ranker is not None else {}),
                    "neighbor_identity_policy": legacy_audit["neighbor_identity_policy"],
                },
            }
            (tmp_root / "VERSION.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(tmp_root, output_root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("bbb_martins", "bioavailability_ma"))
    parser.add_argument("--subset", required=True, choices=("valid", "test"))
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--mapper", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--legacy-bundle", type=Path, default=LEGACY_BUNDLE)
    parser.add_argument("--gold-context-mapping", type=Path, default=DEFAULT_GOLD_CONTEXT_MAPPING)
    parser.add_argument("--allow-frozen-l1-vote-scores", action="store_true")
    args = parser.parse_args()
    manifest = build(
        task=args.task, subset=args.subset, library=args.library.resolve(),
        mapper=args.mapper.resolve(), output_root=args.output_root,
        legacy_bundle=args.legacy_bundle.resolve(),
        gold_context_mapping=(args.gold_context_mapping.resolve() if args.task == "bbb_martins" else None),
        allow_frozen_l1_vote_scores=args.allow_frozen_l1_vote_scores,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
