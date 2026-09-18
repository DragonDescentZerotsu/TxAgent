"""Publish pooled-L5 cache_matched_retrieval.v3 databases for BBB and oral.

The publisher reuses frozen L1 and scored later-level caches, loads L5 pool
membership from the released calibration, and reads parent fingerprints and
scaffolds from the shared immutable Morgan cache. It writes one self-contained
SQLite database per task and benchmark subset; harness runtime never rebuilds it.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

import pyarrow.parquet as pq
from rdkit import Chem, DataStructs

from data.processing.gold_labels.conditioned_benchmark import split_path
from data.processing.morgan_fingerprint_cache import (
    DEFAULT_ROOT as MORGAN_CACHE_ROOT,
    load_molecule_identity_index,
    load_morgan_fingerprints,
    verify_cache as verify_morgan_cache,
)
from predict.retrieval.policies import (
    normalize_molecule_identity,
    seeded_rank_tie_key,
    standardize_smiles_and_fp,
)
from predict.utils.json import read_jsonl, sha256_file

from . import three_pools, v9
from .build_cache_matched_v2 import (
    DEFAULT_GOLD_CONTEXT_MAPPING,
    LATER_RECORD_CAPACITY,
    LEGACY_BUNDLE,
    _digest,
    build as build_base,
)
from .cache_matched_v2 import DATABASE_NAME, POOLS
from .cache_matched_v3 import SCHEMA_VERSION


from predict.retrieval.assay_reranking.runtime import cache_profile_root

ROOT = cache_profile_root("cache_matched_retrieval_v3")
STARLING_ROOT = Path(__file__).resolve().parents[3].parent / "starling_assay_transfer"
L5_RELEASES = {
    "bbb_martins": STARLING_ROOT / "assay_transfer/record_level/artifacts/bbb_v24_1/hf/L5",
    "bioavailability_ma": STARLING_ROOT / "assay_transfer/record_level/artifacts/oral_v25/hf/L5",
}
L5_POOL_CONTRACT = {
    "level": "L5",
    "ranking": "morgan_record_rank_with_seeded_identity_ties",
    "similarity_scope": "query_parent_to_reference_parent",
    "pools": {
        "tool-accepted": "released_L5_assay_transfer_bucket_eligible",
        "tool-compatible": "finite_scalar_value",
        "all": "all_mapped_L5_records",
    },
    "assay_transfer_score": None,
    "assay_rank": None,
    "tie_seed": 0,
}
REPO_ROOT = Path(__file__).resolve().parents[3]
MORGAN_SOURCE_ARCHIVES = {
    task: REPO_ROOT / f"data/legacy/artifacts/evidence_libraries/v10_before_gold_v2_20260916/{task}/v10/03_pair_buckets/records.parquet"
    for task in ("bbb_martins", "bioavailability_ma")
}
MORGAN_SOURCE_OVERRIDES = {
    str((REPO_ROOT / f"data/evidence_libraries/{task}/v10/03_pair_buckets/records.parquet").resolve()): path
    for task, path in MORGAN_SOURCE_ARCHIVES.items()
}


def _rows_sha256(connection: sqlite3.Connection, sql: str) -> str:
    return _digest(list(connection.execute(sql)))


def build_l1_successor(
    *, task: str, subset: str, base_manifest: Path,
    ranking_manifest: Path, output_root: Path,
) -> dict[str, Any]:
    """Replace only L1 assay ranks in a complete V3 cache."""
    base_manifest = base_manifest.resolve()
    ranking_manifest = ranking_manifest.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace an existing cache: {output_root}")
    base = json.loads(base_manifest.read_text())
    ranking = json.loads(ranking_manifest.read_text())
    base_database = base_manifest.with_name(base.get("database", DATABASE_NAME))
    rankings = ranking_manifest.with_name("rankings.parquet")
    lineage = str(ranking.get("model_lineage") or "v9")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task_id": task,
        "subset": subset,
    }
    if any(base.get(key) != value for key, value in expected.items()):
        raise ValueError("Base cache is not the requested complete V3 artifact")
    if base.get("database_sha256") != sha256_file(base_database):
        raise ValueError("Base cache database hash mismatch")
    if (ranking.get("schema_version") != v9.RANKING_SCHEMA_VERSION
            or ranking.get("status") != "complete"
            or ranking.get("task_id") != task
            or ranking.get("model") != v9.model_profile(task, lineage)
            or ranking.get("prompt_assets") != v9.verify_vendored_assets()
            or ranking.get("reference_provenance", {}) != v9.reference_provenance(task, lineage)
            or ranking.get("rankings_sha256") != sha256_file(rankings)
            or ranking.get("morgan_pool_size") != 100):
        raise ValueError("L1 ranking cache is incomplete or incompatible")

    columns = [
        "query_record_id", "retrieval_record_id", "retrieval_molecule_identity_key",
        "morgan_tanimoto_similarity", "model_rank", "prob_transfer",
    ]
    ranking_rows = pq.read_table(rankings, columns=columns).to_pylist()
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sorted(ranking_rows, key=lambda value: int(value["model_rank"])):
        by_query[str(row["query_record_id"])].setdefault(
            str(row["retrieval_molecule_identity_key"]), row
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as tmp:
        tmp_root = Path(tmp)
        database = tmp_root / DATABASE_NAME
        shutil.copy2(base_database, database)
        with sqlite3.connect(database) as connection:
            later_before = _rows_sha256(
                connection,
                "SELECT * FROM assignments WHERE level!='L1' "
                "ORDER BY query_id,pool,level,record_key",
            )
            registered = {
                str(benchmark_row_id): int(query_id)
                for benchmark_row_id, query_id in connection.execute(
                    "SELECT benchmark_row_id,query_id FROM benchmark_queries"
                )
            }
            if set(registered) != set(by_query):
                raise ValueError("L1 rankings differ from the base query ledger")
            updated = similarity_mismatches = 0
            for benchmark_row_id, query_id in registered.items():
                cached = list(connection.execute(
                    "SELECT parent_id,MIN(morgan_similarity),MAX(morgan_similarity),"
                    "MIN(morgan_rank),MAX(morgan_rank) FROM assignments "
                    "WHERE query_id=? AND pool='fixed' AND level='L1' GROUP BY parent_id",
                    (query_id,),
                ))
                candidates = by_query[benchmark_row_id]
                if len(cached) != 100 or {str(row[0]) for row in cached} != set(candidates):
                    raise ValueError(f"Base cache does not contain the exact L1 parent pool: {benchmark_row_id}")
                ordered = sorted(candidates.items(), key=lambda item: int(item[1]["model_rank"]))
                assay_ranks = {parent_id: rank for rank, (parent_id, _) in enumerate(ordered, 1)}
                for parent_id, low_sim, high_sim, low_rank, high_rank in cached:
                    row = candidates[str(parent_id)]
                    if low_sim != high_sim or low_rank != high_rank:
                        raise ValueError(f"L1 Morgan identity changed: {benchmark_row_id}/{parent_id}")
                    if not math.isclose(
                            float(low_sim), float(row["morgan_tanimoto_similarity"]), abs_tol=1e-10):
                        similarity_mismatches += 1
                    score = float(row["prob_transfer"])
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Invalid L1 transfer probability")
                    cursor = connection.execute(
                        "UPDATE assignments SET assay_transfer_score=?,assay_rank=? "
                        "WHERE query_id=? AND pool='fixed' AND level='L1' AND parent_id=?",
                        (score, assay_ranks[str(parent_id)], query_id, parent_id),
                    )
                    updated += cursor.rowcount
            later_after = _rows_sha256(
                connection,
                "SELECT * FROM assignments WHERE level!='L1' "
                "ORDER BY query_id,pool,level,record_key",
            )
            if later_after != later_before:
                raise ValueError("L2+ assignments changed while publishing the L1 successor")
            identity = {
                "schema_version": SCHEMA_VERSION,
                "task_id": task,
                "subset": subset,
                "base_content_id": base["content_id"],
                "base_database_sha256": base["database_sha256"],
                "l1_ranking_manifest_sha256": sha256_file(ranking_manifest),
                "l1_rankings_sha256": ranking["rankings_sha256"],
                "l1_assignment_count": updated,
                "later_assignments_sha256": later_after,
            }
            content_id = _digest(identity)
            connection.execute(
                "UPDATE metadata SET value=? WHERE key='content_id'", (content_id,)
            )
            connection.commit()
            connection.execute("VACUUM")
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("Published successor cache failed integrity check")

        l1_audit = {
            "model_lineage": lineage,
            "ranking_version": str(ranking_manifest),
            "ranking_version_sha256": sha256_file(ranking_manifest),
            "rankings": str(rankings),
            "rankings_sha256": ranking["rankings_sha256"],
            "candidate_policy": ranking["candidate_policy"],
        }
        manifest = {
            **base,
            **identity,
            "content_id": content_id,
            "database_sha256": sha256_file(database),
            "inputs": {
                **base["inputs"],
                "base_cache_manifest": {
                    "path": str(base_manifest), "sha256": sha256_file(base_manifest),
                },
                "base_cache_database": {
                    "path": str(base_database), "sha256": base["database_sha256"],
                },
                "l1_ranking_manifest": {
                    "path": str(ranking_manifest), "sha256": sha256_file(ranking_manifest),
                },
                "l1_rankings": {
                    "path": str(rankings), "sha256": ranking["rankings_sha256"],
                },
            },
            "legacy_selection_audit": {
                **base["legacy_selection_audit"], "l1_ranking": l1_audit,
            },
            "l1_successor": {
                "contract": "replace_assay_score_and_rank_over_identical_morgan100_parents.v2",
                "later_assignments_sha256": later_after,
                "retained_morgan_fields": "base_parent_scoped_similarity_and_rank",
                "ranking_context_similarity_mismatch_count": similarity_mismatches,
            },
        }
        (tmp_root / "VERSION.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(tmp_root, output_root)
    return manifest


@lru_cache(maxsize=1)
def _morgan_inputs() -> dict[str, dict[str, Any]]:
    audit = verify_morgan_cache(
        MORGAN_CACHE_ROOT, verify_hash=True, source_overrides=MORGAN_SOURCE_OVERRIDES,
    )
    manifest = MORGAN_CACHE_ROOT / "manifest.json"
    database = MORGAN_CACHE_ROOT / "fingerprints.sqlite3"
    inputs = {
        "morgan_manifest": {"path": str(manifest.resolve()), "sha256": sha256_file(manifest)},
        "morgan_database": {"path": str(database.resolve()), "sha256": audit["database_sha256"]},
    }
    inputs.update({f"morgan_source_archive_{task}": {
        "path": str(path.resolve()), "sha256": sha256_file(path),
    } for task, path in MORGAN_SOURCE_ARCHIVES.items()})
    return inputs


@lru_cache(maxsize=2)
def _l5_catalog(task: str) -> dict[str, Any]:
    module = three_pools.MODULES[task]
    release_root = L5_RELEASES[task]
    release = json.loads((release_root / "manifest.json").read_text())
    if release.get("task_id") != task or release.get("level") != "L5":
        raise ValueError(f"Incompatible L5 calibration release: {release_root}")
    calibration_path = release_root / "calibration.json"
    accepted = {
        key for key, value in json.loads(calibration_path.read_text())["accepted_buckets"].items()
        if value.get("level") == "L5"
    }
    mapping_rows = pq.read_table(
        module.LEVEL_MAPPING,
        columns=["source_row_uid", "level", "family_key"],
        filters=[("level", "=", 5)],
    ).to_pylist()
    mapping = {
        str(row["source_row_uid"]): str(row["family_key"])
        for row in mapping_rows
    }
    if not mapping or len(mapping) != len(mapping_rows):
        raise ValueError(f"Invalid L5 UID mapping: {task}")
    identity_index = load_molecule_identity_index(MORGAN_CACHE_ROOT)
    canonical_parent_smiles = {}
    for _, (parent_id, parent_smiles, _) in sorted(identity_index.items()):
        canonical_parent_smiles.setdefault(parent_id, parent_smiles)
    columns = [
        "source_row_uid", "canonical_record_id", "canonical_smiles", "source_id",
        "pair_bucket_key", "finite_scalar_value", "measurement_kind",
    ]
    records: dict[str, dict[str, Any]] = {}
    parent_scaffolds: dict[str, set[str]] = defaultdict(set)
    seen_uids = set()
    canonical_forms = {}
    for batch in pq.ParquetFile(module.STAGE3).iter_batches(
        batch_size=20_000, columns=columns
    ):
        for raw in batch.to_pylist():
            uid = str(raw["source_row_uid"])
            family = mapping.get(uid)
            if family is None:
                continue
            smiles = str(raw["canonical_smiles"] or "")
            if smiles not in canonical_forms:
                molecule = Chem.MolFromSmiles(smiles)
                canonical_forms[smiles] = (
                    Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
                    if molecule is not None else ""
                )
            cached_identity = identity_index.get(canonical_forms[smiles])
            if cached_identity is None:
                raise ValueError(f"Shared Morgan cache lacks L5 identity: {smiles}")
            parent_id, parent_smiles, scaffold = cached_identity
            record_id = str(raw["canonical_record_id"])
            if record_id in records or uid in seen_uids:
                raise ValueError(f"Duplicate mapped L5 record or UID: {record_id}")
            bucket = json.dumps(
                [*json.loads(raw["pair_bucket_key"]), "L5"],
                separators=(",", ":"),
            )
            scalar = raw["finite_scalar_value"]
            records[record_id] = {
                "record_id": record_id,
                "source_row_uid": uid,
                "source_id": str(raw["source_id"]),
                "source_canonical_smiles": smiles,
                "canonical_smiles": parent_smiles,
                "parent_id": parent_id,
                "family_key": family,
                "measurement_kind": raw["measurement_kind"],
                "tool_accepted": bucket in accepted,
                "has_scalar": scalar is not None and math.isfinite(float(scalar)),
            }
            seen_uids.add(uid)
            parent_scaffolds[parent_id].add(scaffold)
    missing_uids = set(mapping) - seen_uids
    if missing_uids:
        raise ValueError(f"Stage 3 lacks {len(missing_uids)} mapped L5 UIDs")

    grouped = {pool: defaultdict(list) for pool in POOLS}
    for row in records.values():
        for pool in three_pools.membership(row):
            grouped[pool][row["parent_id"]].append(row)
    if any(not grouped[pool] for pool in POOLS):
        raise ValueError(f"L5 pool is empty: {task}")

    parent_smiles = {
        parent: canonical_parent_smiles[parent]
        for parent in parent_scaffolds
    }
    for row in records.values():
        row["canonical_smiles"] = parent_smiles[row["parent_id"]]
    fingerprints = load_morgan_fingerprints(parent_smiles.values(), cache_root=MORGAN_CACHE_ROOT)
    snapshots = release["source_snapshot"]
    expected = {
        "records": module.STAGE3,
        "uid_levels": module.LEVEL_MAPPING,
        "uid_manifest": module.LEVEL_MANIFEST,
        "source_contract": module.SOURCE_CONTRACT,
    }
    inputs = {}
    for name, path in expected.items():
        item = snapshots[name]
        if Path(item["path"]).resolve() != Path(path).resolve():
            raise ValueError(f"L5 release points to a different {name}: {item['path']}")
        inputs[f"l5_{name}"] = {"path": str(Path(path).resolve()), "sha256": item["sha256"]}
    inputs.update({
        "l5_calibration": {
            "path": str(calibration_path.resolve()),
            "sha256": sha256_file(calibration_path),
        },
        "l5_release_manifest": {
            "path": str((release_root / "manifest.json").resolve()),
            "sha256": sha256_file(release_root / "manifest.json"),
        },
    })
    inputs.update(_morgan_inputs())
    inputs["v3_builder"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    for name in ("build_cache_matched_v2.py", "cache_matched.py"):
        path = Path(__file__).with_name(name).resolve()
        inputs[f"v3_dependency_{path.stem}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return {
        "grouped": {pool: dict(groups) for pool, groups in grouped.items()},
        "parent_scaffolds": dict(parent_scaffolds),
        "parent_smiles": parent_smiles,
        "canonical_parent_smiles": canonical_parent_smiles,
        "fingerprints": fingerprints,
        "records": records,
        "module": module,
        "inputs": inputs,
    }


def _rank_pool(
    groups: dict[str, list[dict[str, Any]]], similarities: dict[str, float], *,
    task: str, benchmark_row_id: str, limit: int,
) -> list[dict[str, Any]]:
    """Return exact top records without hashing every lower-similarity record."""
    boundary = None
    relevant_parents = []
    count = 0
    for parent in sorted(groups, key=lambda value: (-similarities[value], value)):
        similarity = similarities[parent]
        if boundary is not None and similarity < boundary:
            break
        relevant_parents.append(parent)
        count += len(groups[parent])
        if count >= limit and boundary is None:
            boundary = similarity
    candidates = [
        (parent, row)
        for parent in relevant_parents
        for row in groups[parent]
    ]
    candidates.sort(key=lambda item: (
        -similarities[item[0]],
        seeded_rank_tie_key(
            0, task, benchmark_row_id, "L5", item[0], item[1]["record_id"]
        ),
    ))
    within_parent: dict[str, int] = defaultdict(int)
    ranked = []
    for rank, (parent, payload) in enumerate(candidates[:limit], 1):
        within_parent[parent] += 1
        ranked.append({
            "parent_id": parent,
            "parent_smiles": payload["canonical_smiles"],
            "payload": payload,
            "similarity": similarities[parent],
            "morgan_rank": rank,
            "within_parent_rank": within_parent[parent],
        })
    return ranked


def _load_l5_payloads(
    task: str, record_ids: set[str], catalog: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    module = catalog["module"]
    fields, source_union = module._source_fields()
    core = {
        "source_row_uid", "canonical_record_id", "canonical_smiles", "source_id",
        "measurement_kind", "finite_scalar_value", "canonical_measurement_text",
        "canonical_unit_text", "canonical_measurement_scale_id",
        "canonical_category_id", "categorical_encoder_id",
        "canonical_transporter_identifier",
    }
    schema = set(pq.read_schema(module.STAGE3).names)
    columns = sorted((core | source_union) & schema)
    source_contract = json.loads(module.SOURCE_CONTRACT.read_text())
    if task == "bbb_martins":
        from data.processing.evidence_library.versions.v9.tasks.bbb_martins import (
            semantic_display,
        )
    else:
        semantic_display = None

    payloads = {}
    for batch in pq.ParquetFile(module.STAGE3).iter_batches(
        batch_size=20_000, columns=columns
    ):
        ids = batch.column(batch.schema.get_field_index("canonical_record_id")).to_pylist()
        wanted = [str(record_id) in record_ids for record_id in ids]
        if not any(wanted):
            continue
        for raw in batch.filter(wanted).to_pylist():
            record_id = str(raw["canonical_record_id"])
            info = catalog["records"][record_id]
            if (str(raw["source_row_uid"]) != info["source_row_uid"]
                    or str(raw["source_id"]) != info["source_id"]):
                raise ValueError(f"L5 payload identity changed: {record_id}")
            values = {name: raw.get(name) for name in fields[info["source_id"]]}
            if semantic_display is not None:
                values = semantic_display.semantic_prompt_payload(raw, values)
            payloads[record_id] = {
                "record_id": record_id,
                "source_row_uid": info["source_row_uid"],
                "task_id": task,
                "source_id": info["source_id"],
                "progressive_level": "L5",
                "family_key": info["family_key"],
                "canonical_smiles": info["canonical_smiles"],
                "source_canonical_smiles": info["source_canonical_smiles"],
                "source_fields": values,
                "source_contract": {
                    "contract_version": source_contract["contract_version"],
                    "source_or_simply_cleaned": {name: True for name in values},
                },
                "source_projection": "library_source_contract.v2",
                "measurement_kind": raw["measurement_kind"],
            }
    if set(payloads) != record_ids:
        missing = sorted(record_ids - set(payloads))
        raise ValueError(f"Selected L5 records lack source payloads: {missing[:10]}")
    return payloads


def _l5_ranker(task: str, subset: str):
    catalog = _l5_catalog(task)
    query_path = split_path(task, subset).with_name(
        f"{subset}_molecule_condition_labels.jsonl"
    )
    queries = {str(row["benchmark_row_id"]): row for row in read_jsonl(query_path)}
    query_identities = {
        qid: normalize_molecule_identity(row["drug"])
        for qid, row in queries.items()
    }
    query_smiles = {qid: str(identity.parent_smiles or "")
                    for qid, identity in query_identities.items()}
    if any(identity.status != "ok" or not query_smiles[qid]
           for qid, identity in query_identities.items()):
        raise ValueError(f"Invalid frozen query parent: {task}/{subset}")
    query_fingerprints = {
        qid: standardize_smiles_and_fp(parent_smiles)[2]
        for qid, parent_smiles in query_smiles.items()
    }
    if any(fingerprint is None for fingerprint in query_fingerprints.values()):
        raise ValueError(f"Invalid frozen query fingerprint: {task}/{subset}")
    ranked_by_query_pool = {}
    selected_record_ids = set()
    for benchmark_row_id, query in queries.items():
        identity = query_identities[benchmark_row_id]
        parent_smiles = query_smiles[benchmark_row_id]
        query_parent = str(identity.parent_inchi_key or parent_smiles)
        query_scaffold = str(query.get("bemis_murcko_scaffold") or "")
        all_similarities = {
            parent: float(DataStructs.TanimotoSimilarity(
                query_fingerprints[benchmark_row_id],
                catalog["fingerprints"][reference_smiles],
            ))
            for parent, reference_smiles in catalog["parent_smiles"].items()
        }
        for pool in POOLS:
            groups = {
                parent: rows
                for parent, rows in catalog["grouped"][pool].items()
                if parent != query_parent
                and (not query_scaffold
                     or query_scaffold not in catalog["parent_scaffolds"][parent])
            }
            similarities = {parent: all_similarities[parent] for parent in groups}
            records = _rank_pool(
                groups, similarities, task=task,
                benchmark_row_id=benchmark_row_id, limit=LATER_RECORD_CAPACITY,
            )
            if len(records) < LATER_RECORD_CAPACITY:
                raise ValueError(f"Insufficient L5 pool: {benchmark_row_id}/{pool}")
            selected_record_ids.update(row["payload"]["record_id"] for row in records)
            ranked_by_query_pool[benchmark_row_id, pool] = {
                "records": records,
                "candidate_records": sum(map(len, groups.values())),
                "candidate_parents": len(groups),
            }
    payloads = _load_l5_payloads(task, selected_record_ids, catalog)
    for ranked in ranked_by_query_pool.values():
        for row in ranked["records"]:
            row["payload"] = payloads[row["payload"]["record_id"]]

    def rank(benchmark_row_id: str, parent_smiles: str, pool: str) -> dict[str, Any]:
        if parent_smiles != query_smiles[benchmark_row_id]:
            raise ValueError(f"Frozen query parent changed: {benchmark_row_id}")
        return ranked_by_query_pool[benchmark_row_id, pool]

    return rank, catalog["inputs"], catalog["canonical_parent_smiles"]


def build(
    *, task: str, subset: str, library: Path, mapper: Path,
    output_root: Path | None = None, legacy_bundle: Path = LEGACY_BUNDLE,
    gold_context_mapping: Path | bool | None = None,
    allow_frozen_l1_vote_scores: bool = False,
) -> dict[str, Any]:
    """Publish one immutable v3 task/subset cache."""
    ranker, inputs, parent_smiles = _l5_ranker(task, subset)
    return build_base(
        task=task,
        subset=subset,
        library=library,
        mapper=mapper,
        output_root=(output_root or ROOT / task / "scaffold" / subset),
        legacy_bundle=legacy_bundle,
        gold_context_mapping=gold_context_mapping,
        allow_frozen_l1_vote_scores=allow_frozen_l1_vote_scores,
        schema_version=SCHEMA_VERSION,
        l5_ranker=ranker,
        l5_pool_contract=L5_POOL_CONTRACT,
        extra_inputs=inputs,
        canonical_parent_smiles=parent_smiles,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=tuple(L5_RELEASES))
    parser.add_argument("--subset", required=True, choices=("valid", "test"))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--mapper", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--base-cache", type=Path)
    parser.add_argument("--l1-ranking", type=Path)
    parser.add_argument("--legacy-bundle", type=Path, default=LEGACY_BUNDLE)
    parser.add_argument("--gold-context-mapping", type=Path, default=DEFAULT_GOLD_CONTEXT_MAPPING)
    parser.add_argument("--skip-gold-context-mapping", action="store_true")
    parser.add_argument("--allow-frozen-l1-vote-scores", action="store_true")
    args = parser.parse_args()
    if args.base_cache or args.l1_ranking:
        if not args.base_cache or not args.l1_ranking or not args.output_root:
            parser.error("--base-cache, --l1-ranking, and --output-root are required together")
        result = build_l1_successor(
            task=args.task, subset=args.subset, base_manifest=args.base_cache,
            ranking_manifest=args.l1_ranking, output_root=args.output_root,
        )
    else:
        if not args.library or not args.mapper:
            parser.error("--library and --mapper are required for a full build")
        result = build(
            task=args.task,
            subset=args.subset,
            library=args.library.resolve(),
            mapper=args.mapper.resolve(),
            output_root=args.output_root,
            legacy_bundle=args.legacy_bundle.resolve(),
            gold_context_mapping=(
                (False if args.skip_gold_context_mapping else args.gold_context_mapping.resolve())
                if args.task == "bbb_martins" else None
            ),
            allow_frozen_l1_vote_scores=args.allow_frozen_l1_vote_scores,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
