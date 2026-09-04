"""Build current Stage 3 progressive record-transfer score caches."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from rdkit import DataStructs

from predict.retrieval.assay_reranking.runtime import (
    BACKBONE_DTYPE,
    CACHE_ROOT,
    COMPACT_CACHE_SCHEMA_VERSION,
    LOGIT_EXTRACTION_DTYPE,
    SCORING_CONTRACT_VERSION,
    PromptTask,
    file_sha256,
    load_model,
    model_profile,
    resolve_model_snapshot,
    score_prompt_batch,
)
from predict.retrieval.assay_reranking.v19_1 import (
    ASSET_ROOT,
    PROJECTION_SHA256,
    TRAINING_PROMPT_SHA256,
    V191PromptRenderer,
)
from predict.retrieval.policies import (
    decide_candidate,
    normalize_molecule_identity,
    seeded_rank_tie_key,
    standardize_smiles_and_fp,
)


TASK_CONFIGS = {
    "bbb_martins": {
        "cache_profile": "v19_1_bbb_stage3_progressive_top75_scaffold_disjoint",
        "gold_task": "BBB_Martins",
        "levels": ("L2", "L3", "L4", "L5"),
        "extension_file": "direct_bbb_extension.json",
        "extension_source": "direct_bbb",
        "level_contract": {
            "L1": "current vote-ledger source records; globally excluded here",
            "L2": "current near-direct plus current non-voting former L1",
            "L3": "current passive permeability",
            "L4": "current efflux transport",
            "L5": "current influx transport",
        },
    },
    "bioavailability_ma": {
        "cache_profile": "v19_1_bioavailability_ma_stage3_progressive_top75_scaffold_disjoint",
        "gold_task": "Bioavailability_Ma",
        "levels": ("L2", "L3", "L4", "L5", "L6"),
        "extension_file": "direct_bioavailability_extension.json",
        "extension_source": "hf_bioavailability",
        "level_contract": {
            "L1": "current direct oral-bioavailability voters; globally excluded here",
            "L2": "current nondirect oral bioavailability plus non-voting former L1",
            "L3": "current oral AUC/Cmax exposure",
            "L4": "current intestinal absorption and permeability",
            "L5": "current gut-wall efflux and intestinal metabolism",
            "L6": "current hepatic clearance and metabolic stability",
        },
    },
    "skin_reaction": {
        "cache_profile": "v19_1_skin_reaction_stage3_progressive_top75_scaffold_disjoint",
        "gold_task": "Skin_Reaction",
        "levels": ("L3", "L4"),
        "extension_file": "direct_skin_reaction_extension.json",
        "extension_source": "direct_skin_reaction",
        "level_contract": {
            "L1": "current sensitization/contact-allergy voters; existing V9 cache",
            "L2": "current non-voting skin outcomes; excluded here",
            "L3": "current sensitization AOP evidence",
            "L4": "current phototoxicity, irritation, corrosion, and local skin damage",
        },
    },
}
DEFAULT_CACHE_PROFILE = "v19_1"
V21_CACHE_PROFILE = "v21_bbb_l1_l5"
LUNA_RELEVANCE_CACHE_PROFILE = "v19_1_luna_relevance_top_quartile"
V21_ASSET_ROOT = Path(__file__).with_name("prompts") / "v21_bbb"
V21_CONFIG = {
    "cache_profile": "v21_bbb_stage3_progressive_l1_l5_top75_scaffold_disjoint",
    "gold_task": "BBB_Martins",
    "levels": ("L1", "L2", "L3", "L4", "L5"),
    "candidate_contract": "v21_l1_morgan_top75_plus_frozen_v19_l2_l5_scaffold_disjoint.v1",
    "training_dataset": "jiosephlee/assay-transfer-record-level-v21-bbb-martins-mixed-intern",
    "training_dataset_revision": "bafd9607ef05aefb0dee525b60e3e36ef480e597",
    "l1_training_domain_audit": {
        "total": 8536,
        "v21_record_universe": 3061,
        "outside_v21_record_universe": 5475,
        "v21_split_counts": {"train": 2562, "validation": 2, "test": 497},
    },
    "level_contract": {
        "L1": "current direct BBB vote-ledger source records",
        "L2": "current near-direct plus current non-voting former L1",
        "L3": "current passive permeability",
        "L4": "current efflux transport",
        "L5": "current influx transport",
    },
}
CACHE_PROFILES = (
    DEFAULT_CACHE_PROFILE,
    V21_CACHE_PROFILE,
    LUNA_RELEVANCE_CACHE_PROFILE,
)
POPCOUNT = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


@lru_cache(maxsize=500_000)
def _morgan_fingerprint(smiles: str) -> DataStructs.ExplicitBitVect | None:
    return standardize_smiles_and_fp(smiles)[2]


class ProgressiveV191PromptRenderer(V191PromptRenderer):
    """Use V19.1 with the task's explicit out-of-domain direct-source binding."""

    def __init__(self, task_id: str = "bbb_martins") -> None:
        config = TASK_CONFIGS[task_id]
        super().__init__(task_id)
        extension_path = ASSET_ROOT / str(config["extension_file"])
        extension = json.loads(extension_path.read_text(encoding="utf-8"))
        if extension.get("base_projection_sha256") != PROJECTION_SHA256:
            raise ValueError("Direct-source extension targets another V19.1 projection")
        self.projection["labels"].update(extension["labels"])
        self.projection["tasks"][task_id][str(config["extension_source"])] = extension[
            "binding"
        ]
        merged = json.dumps(
            self.projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        self.projection_hash = hashlib.sha256(merged).hexdigest()


class ProgressiveV21PromptRenderer:
    """Render the source-native V21 BBB prompt with copied query context."""

    def __init__(self, task_id: str = "bbb_martins") -> None:
        if task_id != "bbb_martins":
            raise ValueError("V21 all-record prompting only supports bbb_martins")
        self.task_id = task_id
        self.projection = json.loads(
            (V21_ASSET_ROOT / "projection.json").read_text(encoding="utf-8")
        )
        if self.projection.get("schema_version") != (
            "txagent_assay_transfer_v21_bbb_projection.v1"
        ):
            raise ValueError("Unexpected V21 BBB prompt projection")
        self.template_hash = file_sha256(V21_ASSET_ROOT / "prompt.jinja")
        self.projection_hash = file_sha256(V21_ASSET_ROOT / "projection.json")
        self.environment = Environment(
            loader=FileSystemLoader(str(V21_ASSET_ROOT)),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=False,
            trim_blocks=False,
            lstrip_blocks=False,
            auto_reload=False,
        )

    def prompt_fields(self, source_id: str) -> list[str]:
        try:
            return list(self.projection["source_fields"][source_id])
        except KeyError as exc:
            raise ValueError(f"Unknown V21 BBB source {source_id}") from exc

    def _fields(
        self, record: Mapping[str, Any], names: Sequence[str], *, known: bool
    ) -> list[tuple[str, str]]:
        hidden = set(self.projection["always_hidden"])
        if not known:
            hidden.update(self.projection["query_result_fields"])
        labels = self.projection["labels"]
        fields = []
        for name in names:
            value = record.get(name)
            if name in hidden or value is None or (
                isinstance(value, float) and math.isnan(value)
            ):
                continue
            if isinstance(value, bool):
                text = "yes" if value else "no"
            elif isinstance(value, (list, dict)):
                text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                text = str(value).strip()
            if text:
                fields.append(
                    (str(labels.get(name) or name.replace("_", " ").capitalize()), text)
                )
        return fields

    def render(self, record: Mapping[str, Any], query_smiles: str) -> str:
        if str(record.get("task_id") or "") != self.task_id:
            raise ValueError("V21 prompt record belongs to another task")
        names = self.prompt_fields(str(record.get("source_id") or ""))
        known_smiles = str(
            record.get("source_smiles")
            or record.get("smiles")
            or record.get("canonical_smiles")
            or ""
        )
        return self.environment.get_template("prompt.jinja").render(
            known_smiles=known_smiles,
            query_smiles=query_smiles,
            known_fields=self._fields(record, names, known=True),
            query_fields=self._fields(record, names, known=False),
        ).strip()


def _config(task_id: str, cache_profile: str) -> dict[str, Any]:
    if cache_profile == V21_CACHE_PROFILE:
        if task_id != "bbb_martins":
            raise ValueError("V21 all-record cache only supports bbb_martins")
        return dict(V21_CONFIG)
    if cache_profile == LUNA_RELEVANCE_CACHE_PROFILE:
        if task_id not in {"bbb_martins", "skin_reaction"}:
            raise ValueError("Luna relevance filtering only supports BBB and Skin")
        config = dict(TASK_CONFIGS[task_id])
        config.update(
            {
                "cache_profile": (
                    f"v19_1_{task_id}_stage3_progressive_luna_relevance_"
                    "top_quartile_then_morgan_top75_scaffold_disjoint"
                ),
                "candidate_contract": (
                    "current_stage3_luna_relevance_top_quartile_then_"
                    "morgan_top75_scaffold_disjoint.v1"
                ),
                "record_scope": (
                    "normalized_v7_stage3_non_direct_luna_relevance_"
                    "top_quartile_with_current_vote_pure_level_mapping.v1"
                ),
                "levels": (
                    ("L3", "L4", "L5")
                    if task_id == "bbb_martins"
                    else ("L3", "L4")
                ),
            }
        )
        return config
    if cache_profile != DEFAULT_CACHE_PROFILE:
        raise ValueError(f"unsupported progressive cache profile: {cache_profile}")
    return dict(TASK_CONFIGS[task_id])


def _model_config(task_id: str, cache_profile: str) -> dict[str, str]:
    role = "all_records" if cache_profile == V21_CACHE_PROFILE else "indirect"
    profile = model_profile(task_id, role)
    return {"model": str(profile["model"]), "revision": str(profile["revision"])}


def _paths(
    task_id: str, cache_profile: str = DEFAULT_CACHE_PROFILE
) -> dict[str, Path]:
    config = _config(task_id, cache_profile)
    profile = str(config["cache_profile"])
    output_root = CACHE_ROOT / profile / task_id / "scaffold/valid"
    v19_root = (
        CACHE_ROOT
        / str(TASK_CONFIGS[task_id]["cache_profile"])
        / task_id
        / "scaffold/valid"
    )
    return {
        "records": Path(
            f"data/evidence_libraries/{task_id}/v7/03_pair_buckets/records.parquet"
        ),
        "mapping": Path(
            f"data/artifacts/evidence_library_assets/{task_id}/construction_assets/"
            "progressive_level_mapping_v1/record_levels.parquet"
        ),
        "mapping_manifest": Path(
            f"data/artifacts/evidence_library_assets/{task_id}/construction_assets/"
            "progressive_level_mapping_v1/manifest.json"
        ),
        "v9_source_contract": Path(
            "data/evidence_libraries/bbb_martins/v9/02_canonicalized/source_contract.json"
        ),
        "v19_cache": v19_root / "scores.sqlite3",
        "v19_version": v19_root / "VERSION.json",
        "queries": Path(
            f"data/gold_labels/{config['gold_task']}/v1/scaffold/"
            "valid_molecule_condition_labels.jsonl"
        ),
        "extension": ASSET_ROOT / str(config.get("extension_file") or ""),
        "cache": output_root / "scores.sqlite3",
        "version": output_root / "VERSION.json",
        "journals": output_root / ".scores",
        "relevance_manifest": Path(
            f"data/artifacts/evidence_library_assets/relevance_bucket_luna_v7_progressive_v1/"
            f"{task_id}/manifest.json"
        ),
        "relevance_rankings": Path(
            f"data/artifacts/evidence_library_assets/relevance_bucket_luna_v7_progressive_v1/"
            f"{task_id}/relevance_bucket_rankings.parquet"
        ),
        "relevance_pair_map": Path(
            f"data/artifacts/evidence_library_assets/relevance_bucket_luna_v7_progressive_v1/"
            f"{task_id}/pair_bucket_relevance_map.parquet"
        ),
    }


def load_top_ranked_records(
    task_id: str,
    queries: Mapping[str, str],
    *,
    cache_profile: str = DEFAULT_CACHE_PROFILE,
    levels: Sequence[str] = ("L3", "L4", "L5"),
    limit: int | Mapping[str, int] = 50,
    workers: int = 8,
    ranking: str = "assay_transfer",
    tie_seed: int = 0,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Read the highest-scored Stage 3 records from one finalized cache.

    ``queries`` maps frozen benchmark row IDs to their current split SMILES. The
    cache's pinned query ledger supplies the exact parent-SMILES spelling used as
    the SQLite key. Every cache and source hash is checked before records are read.
    """
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"unsupported progressive cache task: {task_id}")
    config = _config(task_id, cache_profile)
    model = _model_config(task_id, cache_profile)
    if ranking not in {"assay_transfer", "morgan"}:
        raise ValueError(f"unsupported progressive record ranking: {ranking}")
    if workers < 1:
        raise ValueError("workers must be positive")
    if not queries:
        raise ValueError("at least one query is required")
    requested_levels = tuple(dict.fromkeys(str(level) for level in levels))
    if not requested_levels:
        raise ValueError("at least one progressive level is required")
    limits = {
        level: int(limit.get(level, 0) if isinstance(limit, Mapping) else limit)
        for level in requested_levels
    }
    if any(value < 1 for value in limits.values()):
        raise ValueError("every requested level limit must be positive")

    paths = _paths(task_id, cache_profile)
    version = json.loads(paths["version"].read_text(encoding="utf-8"))
    expected = {
        "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
        "profile": config["cache_profile"],
        "task_id": task_id,
        "model": model["model"],
        "model_revision": model["revision"],
        "candidate_contract": config.get(
            "candidate_contract",
            "current_stage3_level_then_morgan_top75_scaffold_disjoint.v1",
        ),
        "record_scope": config.get(
            "record_scope",
            "normalized_v7_stage3_with_current_vote_pure_level_mapping.v1",
        ),
        "neighbor_identity_policy": "scaffold_disjoint",
    }
    for field, value in expected.items():
        if version.get(field) != value:
            raise ValueError(
                f"{task_id} progressive cache has wrong {field}: {version.get(field)!r}"
            )
    cache_status = str(version.get("status") or "")
    if ranking == "assay_transfer" and cache_status != "complete":
        raise ValueError(f"{task_id} assay-transfer cache is not complete")
    if ranking == "morgan" and cache_status not in {"prepared", "complete"}:
        raise ValueError(f"{task_id} Morgan candidate cache is not prepared")
    if cache_status == "complete" and version.get("cache_quick_check") != "ok":
        raise ValueError(f"{task_id} progressive cache failed its quick check")
    if not set(requested_levels) <= set(version.get("levels_in_cache") or []):
        raise ValueError(f"{task_id} progressive cache lacks {requested_levels}")
    cache_hash = file_sha256(paths["cache"])
    if version.get("cache_sha256") and version["cache_sha256"] != cache_hash:
        raise ValueError(f"{task_id} progressive cache hash disagrees with VERSION.json")
    for source, expected_hash in (version.get("inputs") or {}).items():
        source_path = Path(source)
        if not source_path.is_file() or file_sha256(source_path) != expected_hash:
            raise ValueError(f"{task_id} progressive cache input changed: {source_path}")

    frozen_queries = {
        str(row["benchmark_row_id"]): row for row in _read_jsonl(paths["queries"])
    }
    normalized_queries: dict[str, str] = {}
    for query_id, smiles in queries.items():
        frozen = frozen_queries.get(str(query_id))
        if frozen is None:
            raise ValueError(f"query {query_id} is absent from the frozen cache ledger")
        if str(frozen.get("drug") or "") != str(smiles):
            raise ValueError(f"query {query_id} SMILES differs from the frozen cache ledger")
        parent_smiles = str(
            (frozen.get("molecule_identity") or {}).get("parent_smiles") or ""
        )
        if not parent_smiles:
            raise ValueError(f"query {query_id} has no frozen parent SMILES")
        normalized_queries[str(query_id)] = parent_smiles

    assay_transfer_sql = """
        SELECT r.external_record_id, m.molecule_chembl_id,
               s.transfer_probability, r.payload, COUNT(*) OVER ()
        FROM assignments AS a
        JOIN queries AS q USING(query_id)
        JOIN groups_dim AS g USING(group_key)
        JOIN records AS r USING(record_key)
        JOIN molecules AS m USING(molecule_key)
        JOIN scores AS s USING(score_key)
        WHERE q.query_smiles = ? AND g.group_id = ?
        ORDER BY s.transfer_probability DESC, r.external_record_id
        LIMIT ?
    """

    assignment_table = (
        "assignments" if cache_status == "complete" else "prompt_assignments"
    )
    morgan_sql = f"""
        SELECT r.external_record_id, m.molecule_chembl_id, r.payload
        FROM {assignment_table} AS a
        JOIN queries AS q USING(query_id)
        JOIN groups_dim AS g USING(group_key)
        JOIN records AS r USING(record_key)
        JOIN molecules AS m USING(molecule_key)
        WHERE q.query_smiles = ? AND g.group_id = ?
    """

    def load_one(item: tuple[str, str]) -> tuple[str, dict[str, dict[str, Any]]]:
        query_id, parent_smiles = item
        connection = sqlite3.connect(
            f"file:{paths['cache'].resolve()}?mode=ro", uri=True
        )
        try:
            result: dict[str, dict[str, Any]] = {}
            for level in requested_levels:
                level_limit = limits[level]
                if ranking == "assay_transfer":
                    rows = connection.execute(
                        assay_transfer_sql, (parent_smiles, level, level_limit)
                    ).fetchall()
                    available_count = int(rows[0][4]) if rows else 0
                else:
                    query_fp = _morgan_fingerprint(parent_smiles)
                    if query_fp is None:
                        raise ValueError(f"cannot fingerprint query {query_id}")
                    candidates = []
                    for record_id, molecule_id, payload_json in connection.execute(
                        morgan_sql, (parent_smiles, level)
                    ).fetchall():
                        payload = json.loads(payload_json)
                        reference_smiles = str(payload.get("canonical_smiles") or "")
                        reference_fp = _morgan_fingerprint(reference_smiles)
                        if reference_fp is None:
                            raise ValueError(
                                f"cannot fingerprint {task_id} cache record {record_id}"
                            )
                        similarity = float(DataStructs.TanimotoSimilarity(query_fp, reference_fp))
                        candidates.append(
                            (
                                -similarity,
                                seeded_rank_tie_key(
                                    tie_seed,
                                    task_id,
                                    query_id,
                                    level,
                                    molecule_id,
                                    record_id,
                                ),
                                str(record_id),
                                str(molecule_id),
                                similarity,
                                payload,
                            )
                        )
                    candidates.sort(key=lambda row: row[:3])
                    available_count = len(candidates)
                    rows = candidates[:level_limit]
                if len(rows) != level_limit:
                    raise ValueError(
                        f"{task_id} query {query_id} has {len(rows)} {level} records; "
                        f"exactly {level_limit} are required"
                    )
                records = []
                for row in rows:
                    if ranking == "assay_transfer":
                        record_id, molecule_id, score_value, payload_json, _ = row
                        payload = json.loads(payload_json)
                        score_field = {"transfer_likelihood": float(score_value)}
                    else:
                        _, _, record_id, molecule_id, score_value, payload = row
                        score_field = {"morgan_similarity": float(score_value)}
                    if (
                        str(payload.get("record_id") or "") != str(record_id)
                        or str(payload.get("progressive_level") or "") != level
                    ):
                        raise ValueError(
                            f"{task_id} cache payload disagrees with {record_id} at {level}"
                        )
                    records.append(
                        {
                            "record_id": str(record_id),
                            "reference_molecule_id": str(molecule_id),
                            **score_field,
                            "payload": payload,
                        }
                    )
                result[level] = {
                    "available_record_count": available_count,
                    "records": records,
                }
            return query_id, result
        finally:
            connection.close()

    unique_queries = dict.fromkeys(normalized_queries.values())
    work_items = (
        list(normalized_queries.items())
        if ranking == "morgan"
        else [(smiles, smiles) for smiles in unique_queries]
    )
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(work_items))
    ) as pool:
        for key, result in pool.map(load_one, work_items):
            loaded[key] = result
    selected = loaded if ranking == "morgan" else {
        query_id: loaded[parent_smiles]
        for query_id, parent_smiles in normalized_queries.items()
    }
    return selected, {
        "version": str(paths["version"]),
        "version_sha256": file_sha256(paths["version"]),
        "cache": str(paths["cache"]),
        "cache_sha256": cache_hash,
        "cache_status": cache_status,
        "profile": version["profile"],
        "model": version["model"],
        "model_revision": version["model_revision"],
        "cache_profile": cache_profile,
        "levels": list(requested_levels),
        "records_per_level": (
            next(iter(limits.values()))
            if len(set(limits.values())) == 1
            else limits
        ),
        "n_query_rows": len(queries),
        "n_unique_query_parents": len(unique_queries),
        "candidate_contract": version["candidate_contract"],
        "record_scope": version["record_scope"],
        "neighbor_identity_policy": version["neighbor_identity_policy"],
        "ranking": ranking,
        "ranking_tie_seed": tie_seed if ranking == "morgan" else None,
        "assay_transfer_scores_used": ranking == "assay_transfer",
        "morgan_fingerprint": (
            {"radius": 2, "bits": 2048, "similarity": "Tanimoto"}
            if ranking == "morgan"
            else None
        ),
        "inputs": dict(version.get("inputs") or {}),
    }


def load_v21_molecule_card_records(
    queries: Mapping[str, str],
    *,
    molecule_limit: int = 10,
    l1_record_limit: int = 4,
    l2_record_limit: int = 2,
    workers: int = 8,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Group V21-ranked L1/L2 records into query-specific molecule cards."""
    if min(molecule_limit, l1_record_limit, l2_record_limit, workers) < 1:
        raise ValueError("V21 molecule-card limits and workers must be positive")
    _, cache_audit = load_top_ranked_records(
        "bbb_martins",
        queries,
        cache_profile=V21_CACHE_PROFILE,
        levels=("L1", "L2"),
        limit=1,
        workers=workers,
        ranking="assay_transfer",
    )
    paths = _paths("bbb_martins", V21_CACHE_PROFILE)
    frozen_queries = {
        str(row["benchmark_row_id"]): row for row in _read_jsonl(paths["queries"])
    }
    parent_smiles = {
        query_id: str(
            (frozen_queries[query_id].get("molecule_identity") or {}).get(
                "parent_smiles"
            )
            or ""
        )
        for query_id in queries
    }
    sql = """
        SELECT g.group_id, m.molecule_chembl_id, r.external_record_id,
               s.transfer_probability, r.payload
        FROM assignments AS a
        JOIN queries AS q USING(query_id)
        JOIN groups_dim AS g USING(group_key)
        JOIN records AS r USING(record_key)
        JOIN molecules AS m USING(molecule_key)
        JOIN scores AS s USING(score_key)
        WHERE q.query_smiles = ? AND g.group_id IN ('L1', 'L2')
        ORDER BY g.group_id, s.transfer_probability DESC, r.external_record_id
    """

    def load_one(item: tuple[str, str]) -> tuple[str, list[dict[str, Any]]]:
        query_id, query_parent = item
        connection = sqlite3.connect(
            f"file:{paths['cache'].resolve()}?mode=ro", uri=True
        )
        try:
            by_level: dict[str, dict[str, list[dict[str, Any]]]] = {
                "L1": defaultdict(list),
                "L2": defaultdict(list),
            }
            for level, molecule_id, record_id, score, payload_json in connection.execute(
                sql, (query_parent,)
            ):
                payload = json.loads(payload_json)
                if (
                    str(payload.get("record_id") or "") != str(record_id)
                    or str(payload.get("progressive_level") or "") != str(level)
                ):
                    raise ValueError(
                        f"V21 cache payload disagrees with {record_id} at {level}"
                    )
                by_level[str(level)][str(molecule_id)].append(
                    {
                        "record_id": str(record_id),
                        "transfer_likelihood": float(score),
                        "payload": payload,
                    }
                )
            ranked_molecules = sorted(
                by_level["L1"],
                key=lambda molecule_id: (
                    -by_level["L1"][molecule_id][0]["transfer_likelihood"],
                    molecule_id,
                ),
            )[:molecule_limit]
            if len(ranked_molecules) != molecule_limit:
                raise ValueError(
                    f"BBB query {query_id} has {len(ranked_molecules)} V21 L1 molecules; "
                    f"exactly {molecule_limit} are required"
                )
            cards = []
            for selection_rank, molecule_id in enumerate(ranked_molecules):
                l1 = by_level["L1"][molecule_id]
                l2 = by_level["L2"].get(molecule_id, [])
                cards.append(
                    {
                        "reference_molecule_id": molecule_id,
                        "canonical_smiles": str(l1[0]["payload"]["canonical_smiles"]),
                        "transfer_likelihood": float(
                            l1[0]["transfer_likelihood"]
                        ),
                        "selection_rank": selection_rank,
                        "available_l1": len(l1),
                        "available_l2": len(l2),
                        "l1_records": l1[:l1_record_limit],
                        "l2_records": l2[:l2_record_limit],
                    }
                )
            return query_id, cards
        finally:
            connection.close()

    selected: dict[str, list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(parent_smiles))
    ) as pool:
        for query_id, cards in pool.map(load_one, parent_smiles.items()):
            selected[query_id] = cards
    wanted_record_ids = {
        str(record["record_id"])
        for cards in selected.values()
        for card in cards
        for level in ("l1_records", "l2_records")
        for record in card[level]
    }
    stage3_candidates = [
        Path(path)
        for path in cache_audit["inputs"]
        if Path(path).name == "records.parquet"
        and Path(path).parent.name == "03_pair_buckets"
    ]
    if len(stage3_candidates) != 1:
        raise ValueError("V21 cache provenance lacks one Stage 3 records artifact")
    stage3 = pq.ParquetFile(stage3_candidates[0])
    canonical_columns = [
        name
        for name in stage3.schema_arrow.names
        if name == "canonical_record_id" or name.startswith("canonical_")
    ]
    canonical_by_id: dict[str, dict[str, Any]] = {}
    for batch in stage3.iter_batches(batch_size=16_384, columns=canonical_columns):
        for row in batch.to_pylist():
            record_id = str(row["canonical_record_id"])
            if record_id in wanted_record_ids:
                canonical_by_id[record_id] = row
    if set(canonical_by_id) != wanted_record_ids:
        missing = next(iter(wanted_record_ids - set(canonical_by_id)), None)
        raise ValueError(f"V21 selected record lacks Stage 3 canonical fields: {missing}")
    for cards in selected.values():
        for card in cards:
            for level in ("l1_records", "l2_records"):
                for record in card[level]:
                    record["payload"].update(canonical_by_id[record["record_id"]])
    l1_counts = [len(card["l1_records"]) for cards in selected.values() for card in cards]
    l2_counts = [len(card["l2_records"]) for cards in selected.values() for card in cards]
    return selected, {
        **cache_audit,
        "selection_unit": "molecule_then_record",
        "molecule_ranking": "maximum_v21_l1_record_transfer_likelihood",
        "within_molecule_record_ranking": "v21_transfer_likelihood_descending",
        "molecules_per_query": molecule_limit,
        "l1_record_limit_per_molecule": l1_record_limit,
        "l2_record_limit_per_molecule": l2_record_limit,
        "selected_l1_records": sum(l1_counts),
        "selected_l2_records": sum(l2_counts),
        "molecules_without_l2_records": sum(count == 0 for count in l2_counts),
        "canonical_stage3_hydrated_records": len(canonical_by_id),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _level_by_record(
    mapping_path: Path, levels_in_cache: Sequence[str]
) -> tuple[dict[str, str], Counter]:
    mapping = pq.read_table(
        mapping_path,
        columns=[
            "canonical_record_id",
            "progressive_level",
            "source_id",
            "finite_scalar_value_present",
            "score_cache_candidate_eligible",
            "scoring_domain_flags",
        ],
    ).to_pylist()
    levels: dict[str, str] = {}
    counts: Counter = Counter()
    for row in mapping:
        level = str(row.get("progressive_level") or "")
        if level not in {"L1", *levels_in_cache}:
            continue
        record_id = str(row["canonical_record_id"])
        if record_id in levels:
            raise ValueError(f"duplicate mapping record: {record_id}")
        levels[record_id] = level
        counts[
            (level, str(row["source_id"]), bool(row["finite_scalar_value_present"]))
        ] += 1
    return levels, counts


def _tanimoto_vector(query_fp: Any, packed: np.ndarray, bit_counts: np.ndarray) -> np.ndarray:
    query = np.frombuffer(DataStructs.BitVectToBinaryText(query_fp), dtype=np.uint8)
    intersection = POPCOUNT[np.bitwise_and(packed, query)].sum(axis=1, dtype=np.uint16)
    union = bit_counts.astype(np.int32) + int(POPCOUNT[query].sum()) - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(packed), dtype=np.float64),
        where=union != 0,
    )


def _open_build_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE cache_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE queries(query_id INTEGER PRIMARY KEY, query_smiles TEXT UNIQUE NOT NULL);
        CREATE TABLE groups_dim(group_key INTEGER PRIMARY KEY, group_id TEXT UNIQUE NOT NULL);
        CREATE TABLE molecules(
            molecule_key INTEGER PRIMARY KEY, molecule_chembl_id TEXT UNIQUE NOT NULL
        );
        CREATE TABLE records(
            record_key INTEGER PRIMARY KEY,
            external_record_id TEXT UNIQUE NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE refs(
            record_id TEXT NOT NULL,
            query_smiles TEXT NOT NULL,
            group_id TEXT NOT NULL,
            molecule_id TEXT NOT NULL,
            PRIMARY KEY(record_id, query_smiles, group_id, molecule_id)
        ) WITHOUT ROWID;
        CREATE TABLE prompt_tasks(
            prompt_key INTEGER PRIMARY KEY,
            cache_key TEXT UNIQUE NOT NULL,
            prompt TEXT NOT NULL
        );
        CREATE TABLE prompt_assignments(
            query_id INTEGER NOT NULL,
            group_key INTEGER NOT NULL,
            molecule_key INTEGER NOT NULL,
            record_key INTEGER NOT NULL,
            prompt_key INTEGER NOT NULL,
            PRIMARY KEY(query_id, group_key, molecule_key, record_key)
        ) WITHOUT ROWID;
        """
    )
    return connection


def _dimension_key(
    connection: sqlite3.Connection,
    table: str,
    key_column: str,
    value_column: str,
    value: str,
) -> int:
    connection.execute(
        f"INSERT OR IGNORE INTO {table}({value_column}) VALUES (?)", (value,)
    )
    return int(
        connection.execute(
            f"SELECT {key_column} FROM {table} WHERE {value_column}=?", (value,)
        ).fetchone()[0]
    )


def _prepare_candidates(
    connection: sqlite3.Connection,
    paths: Mapping[str, Path],
    records_by_level: Mapping[str, Mapping[int, list[str]]],
    molecules: Sequence[Mapping[str, Any]],
    packed: np.ndarray,
    pool_size: int,
    levels_in_cache: Sequence[str],
    require_full_pool: bool = True,
) -> tuple[int, Counter]:
    molecule_ids = [str(row["molecule_id"]) for row in molecules]
    bit_counts = POPCOUNT[packed].sum(axis=1, dtype=np.uint16)
    level_sets = {level: set(rows) for level, rows in records_by_level.items()}
    queries = _read_jsonl(paths["queries"])
    counts: Counter = Counter()
    for query_index, query in enumerate(queries):
        stored_identity = query.get("molecule_identity") or {}
        query_smiles = str(stored_identity.get("parent_smiles") or "")
        if not query_smiles:
            raise ValueError(f"Query {query_index} lacks its frozen parent SMILES")
        current_identity = normalize_molecule_identity(str(query["drug"]))
        frozen_parent = str(stored_identity.get("parent_inchi_key") or query_smiles)
        current_parent = (
            current_identity.parent_inchi_key or current_identity.parent_smiles
        )
        if current_parent != frozen_parent:
            raise ValueError(f"Query {query_index} parent identity drifted from its gold row")
        _, _, query_fp = standardize_smiles_and_fp(query_smiles)
        if query_fp is None:
            raise ValueError(f"Invalid query SMILES at row {query_index}")
        query_identity = normalize_molecule_identity(query_smiles)
        similarities = _tanimoto_vector(query_fp, packed, bit_counts)
        ranked = sorted(
            range(len(molecules)),
            key=lambda index: (-float(similarities[index]), molecule_ids[index]),
        )
        selected = {level: 0 for level in levels_in_cache}
        for molecule_index in ranked:
            relevant = [
                level
                for level in levels_in_cache
                if selected[level] < pool_size and molecule_index in level_sets[level]
            ]
            if not relevant:
                if all(selected[level] == pool_size for level in levels_in_cache):
                    break
                continue
            if decide_candidate(
                query_identity, molecules[molecule_index], "scaffold_disjoint"
            ).excluded:
                for level in relevant:
                    counts[(level, "identity_excluded")] += 1
                continue
            for level in relevant:
                record_ids = records_by_level[level][molecule_index]
                connection.executemany(
                    "INSERT OR IGNORE INTO refs VALUES (?, ?, ?, ?)",
                    (
                        (record_id, query_smiles, level, molecule_ids[molecule_index])
                        for record_id in record_ids
                    ),
                )
                selected[level] += 1
                counts[(level, "candidate_molecules")] += 1
                counts[(level, "record_assignments")] += len(record_ids)
        if require_full_pool and any(
            selected[level] != pool_size for level in levels_in_cache
        ):
            raise ValueError(f"Query {query_index} lacks {pool_size} candidates: {selected}")
        if not require_full_pool and any(
            selected[level] < 1 for level in levels_in_cache
        ):
            raise ValueError(f"Query {query_index} has an empty candidate level: {selected}")
        if (query_index + 1) % 25 == 0:
            connection.commit()
            print(f"prepared candidates for {query_index + 1}/{len(queries)} queries", flush=True)
    connection.commit()
    return len(queries), counts


def _record_payload(
    renderer: Any,
    row: Mapping[str, Any],
    level: str,
    parent_smiles: str,
    task_id: str,
) -> dict[str, Any]:
    source_id = str(row.get("source_id") or "")
    fields = renderer.prompt_fields(source_id)
    source_fields = {field: row.get(field) for field in fields}
    return {
        "record_id": str(row["canonical_record_id"]),
        "task_id": task_id,
        "source_id": source_id,
        "progressive_level": level,
        "canonical_smiles": parent_smiles,
        "source_canonical_smiles": str(row.get("canonical_smiles") or ""),
        "measurement_kind": str(
            row.get("measurement_kind")
            or ("numeric" if row.get("finite_scalar_value") is not None else "nonnumeric")
        ),
        "training_measurement_kind_supported": row.get("finite_scalar_value") is not None,
        "source_contract": {
            "source_id": source_id,
            "record_contract_version": "starling_record_contract.v7",
        },
        "source_fields": source_fields,
        **source_fields,
    }


def _relevance_value(value: Any) -> str:
    if value is None or (
        isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
    ):
        return "__unknown__"
    text = str(value).strip()
    return text or "__unknown__"


def _relevance_identity(
    row: Mapping[str, Any], columns: Sequence[str]
) -> tuple[str, ...]:
    return tuple(_relevance_value(row.get(column)) for column in columns)


def _select_luna_relevance_records(
    stage3: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    relevance_manifest: Mapping[str, Any],
    levels_in_cache: Sequence[str],
) -> tuple[set[str], dict[str, Any]]:
    """Keep every record in the top-ranked quarter of each level/source pool."""
    if relevance_manifest.get("status") != "complete":
        raise ValueError("Luna relevance ranking is not complete")
    columns_by_source = {
        str(source): tuple(str(column) for column in columns)
        for source, columns in relevance_manifest["relevance_bucket_columns"].items()
    }
    scores: dict[tuple[str, tuple[str, ...]], tuple[float, str]] = {}
    for row in ranking_rows:
        encoded = str(row["relevance_bucket"])
        identity = json.loads(encoded)
        source = str(row["source_id"])
        if str(identity.get("source_id") or "") != source:
            raise ValueError("Luna ranking source disagrees with its bucket identity")
        key = (
            source,
            tuple(
                _relevance_value(identity.get(column))
                for column in columns_by_source[source]
            ),
        )
        if key in scores:
            raise ValueError(f"duplicate Luna relevance ranking: {encoded}")
        scores[key] = (float(row["bradley_terry_score"]), encoded)

    stage3_by_record = {
        str(row["canonical_record_id"]): row for row in stage3
    }
    if len(stage3_by_record) != len(stage3):
        raise ValueError("Stage 3 contains duplicate canonical record IDs")
    v7_buckets: set[tuple[str, tuple[str, ...]]] = set()
    bucket_by_record: dict[str, tuple[str, tuple[str, ...]]] = {}
    for record_id, row in stage3_by_record.items():
        source = str(row.get("source_id") or "")
        if source not in columns_by_source or row.get("assay_transfer_eligible") is not True:
            continue
        key = (source, _relevance_identity(row, columns_by_source[source]))
        v7_buckets.add(key)
        bucket_by_record[record_id] = key
    ranked_buckets = set(scores)
    if v7_buckets != ranked_buckets:
        missing = next(iter(v7_buckets - ranked_buckets), None)
        extra = next(iter(ranked_buckets - v7_buckets), None)
        raise ValueError(
            "Stage 3 and Luna ranking relevance identities differ; "
            f"first_unranked_stage3={missing}, first_ranking_only={extra}"
        )

    records_by_pool: dict[tuple[str, str], list[str]] = defaultdict(list)
    buckets_by_pool: dict[
        tuple[str, str], set[tuple[str, tuple[str, ...]]]
    ] = defaultdict(set)
    allowed_levels = set(levels_in_cache)
    for row in mapping:
        record_id = str(row["canonical_record_id"])
        level = str(row.get("progressive_level") or "")
        if (
            level not in allowed_levels
            or not bool(row.get("score_cache_candidate_eligible"))
            or record_id not in bucket_by_record
        ):
            continue
        key = bucket_by_record[record_id]
        pool = (level, key[0])
        records_by_pool[pool].append(record_id)
        buckets_by_pool[pool].add(key)

    selected: set[str] = set()
    pools: dict[str, Any] = {}
    for (level, source), buckets in sorted(buckets_by_pool.items()):
        ordered = sorted(buckets, key=lambda key: (-scores[key][0], scores[key][1]))
        keep_count = math.ceil(len(ordered) * 0.25)
        kept = set(ordered[:keep_count])
        kept_records = {
            record_id
            for record_id in records_by_pool[(level, source)]
            if bucket_by_record[record_id] in kept
        }
        selected.update(kept_records)
        pools[f"{level}:{source}"] = {
            "input_relevance_buckets": len(buckets),
            "selected_relevance_buckets": keep_count,
            "input_records": len(records_by_pool[(level, source)]),
            "selected_records": len(kept_records),
        }
    if not selected:
        raise ValueError("Luna relevance filtering selected no Stage 3 records")
    return selected, {
        "selection_fraction": 0.25,
        "rounding": "ceil_per_progressive_level_and_source",
        "ranking_scope": "source_local_bradley_terry",
        "selection_unit": "relevance_bucket",
        "record_policy": "retain_all_records_in_selected_relevance_buckets",
        "stage3_ranking_relevance_identity_sets_equal": True,
        "ranked_relevance_buckets": len(ranked_buckets),
        "selected_records": len(selected),
        "pools": pools,
    }


def _hydrate_and_render(
    connection: sqlite3.Connection,
    paths: Mapping[str, Path],
    renderer: Any,
    level_by_record: Mapping[str, str],
    record_parent_smiles: Mapping[str, str],
    task_id: str,
    cache_profile: str,
) -> tuple[int, int, int]:
    wanted = {
        str(row[0]) for row in connection.execute("SELECT DISTINCT record_id FROM refs")
    }
    found = 0
    parquet = pq.ParquetFile(paths["records"])
    for batch in parquet.iter_batches(batch_size=16_384):
        rows = []
        for raw in batch.to_pylist():
            record_id = str(raw["canonical_record_id"])
            if record_id not in wanted:
                continue
            payload = _record_payload(
                renderer,
                raw,
                level_by_record[record_id],
                record_parent_smiles[record_id],
                task_id,
            )
            rows.append(
                (record_id, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
            )
        if rows:
            connection.executemany(
                "INSERT OR REPLACE INTO records(external_record_id, payload) VALUES (?, ?)",
                rows,
            )
            connection.commit()
            found += len(rows)
    if found != len(wanted):
        raise ValueError(f"Record hydration mismatch: expected={len(wanted)}, found={found}")

    profile = _model_config(task_id, cache_profile)
    assignments = duplicates = 0
    query = """
        SELECT refs.query_smiles, refs.group_id, refs.molecule_id,
               records.record_key, records.payload
        FROM refs JOIN records ON refs.record_id = records.external_record_id
        ORDER BY refs.query_smiles, refs.group_id, refs.molecule_id, refs.record_id
    """
    for query_smiles, group_id, molecule_id, record_key, payload_json in connection.execute(query):
        record = json.loads(payload_json)
        prompt = renderer.render(record, str(query_smiles))
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        identity = {
            "prompt_hash": prompt_hash,
            "model": profile["model"],
            "model_revision": profile["revision"],
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "template_hash": renderer.template_hash,
            "projection_hash": renderer.projection_hash,
        }
        cache_key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        query_id = _dimension_key(
            connection, "queries", "query_id", "query_smiles", str(query_smiles)
        )
        group_key = _dimension_key(
            connection, "groups_dim", "group_key", "group_id", str(group_id)
        )
        molecule_key = _dimension_key(
            connection, "molecules", "molecule_key", "molecule_chembl_id", str(molecule_id)
        )
        inserted = connection.execute(
            "INSERT OR IGNORE INTO prompt_tasks(cache_key, prompt) VALUES (?, ?)",
            (cache_key, prompt),
        ).rowcount
        duplicates += int(not inserted)
        prompt_key = int(
            connection.execute(
                "SELECT prompt_key FROM prompt_tasks WHERE cache_key=?", (cache_key,)
            ).fetchone()[0]
        )
        connection.execute(
            "INSERT OR IGNORE INTO prompt_assignments VALUES (?, ?, ?, ?, ?)",
            (query_id, group_key, molecule_key, int(record_key), prompt_key),
        )
        assignments += 1
        if assignments % 100_000 == 0:
            connection.commit()
            print(f"rendered {assignments} score assignments", flush=True)
    connection.commit()
    prompts = int(connection.execute("SELECT COUNT(*) FROM prompt_tasks").fetchone()[0])
    return found, assignments, duplicates


def prepare(
    task_id: str = "bbb_martins",
    pool_size: int = 75,
    cache_profile: str = DEFAULT_CACHE_PROFILE,
) -> dict[str, Any]:
    config = _config(task_id, cache_profile)
    model = _model_config(task_id, cache_profile)
    levels_in_cache = tuple(config["levels"])
    paths = _paths(task_id, cache_profile)
    required = [
        paths[key]
        for key in (
            "records", "mapping", "mapping_manifest", "queries",
        )
    ]
    if cache_profile == V21_CACHE_PROFILE:
        required += [
            V21_ASSET_ROOT / "prompt.jinja",
            V21_ASSET_ROOT / "projection.json",
            paths["v9_source_contract"],
            paths["v19_cache"],
            paths["v19_version"],
        ]
    else:
        required += [
            ASSET_ROOT / "prompt.jinja",
            ASSET_ROOT / "prompt_projection.json",
            paths["extension"],
        ]
    if cache_profile == LUNA_RELEVANCE_CACHE_PROFILE:
        required += [
            paths["relevance_manifest"],
            paths["relevance_rankings"],
            paths["relevance_pair_map"],
        ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing progressive cache input(s): " + ", ".join(missing))
    output_root = paths["cache"].parent
    output_root.mkdir(parents=True, exist_ok=True)
    if paths["cache"].exists() or paths["version"].exists():
        raise FileExistsError(f"Progressive cache already exists: {output_root}")

    renderer = (
        ProgressiveV21PromptRenderer(task_id)
        if cache_profile == V21_CACHE_PROFILE
        else ProgressiveV191PromptRenderer(task_id)
    )
    level_by_record, level_counts = _level_by_record(
        paths["mapping"], levels_in_cache
    )
    mapping = pq.read_table(
        paths["mapping"],
        columns=[
            "canonical_record_id",
            "progressive_level",
            "source_id",
            "score_cache_candidate_eligible",
        ],
    ).to_pylist()
    eligible = {
        str(row["canonical_record_id"])
        for row in mapping
        if bool(row["score_cache_candidate_eligible"])
        or (
            cache_profile == V21_CACHE_PROFILE
            and row.get("progressive_level") == "L1"
            and row.get("source_id") == "direct_bbb"
        )
    }
    relevance_manifest: dict[str, Any] | None = None
    relevance_selection: dict[str, Any] | None = None
    stage3_columns = [
        "canonical_record_id",
        "canonical_smiles",
        "source_id",
        "assay_transfer_eligible",
    ]
    if cache_profile == LUNA_RELEVANCE_CACHE_PROFILE:
        relevance_manifest = json.loads(
            paths["relevance_manifest"].read_text(encoding="utf-8")
        )
        stage3_columns.extend(
            column
            for columns in relevance_manifest["relevance_bucket_columns"].values()
            for column in columns
            if column not in stage3_columns
        )
    stage3 = pq.read_table(
        paths["records"], columns=stage3_columns
    ).to_pylist()
    if cache_profile == LUNA_RELEVANCE_CACHE_PROFILE:
        ranking_rows = pq.read_table(paths["relevance_rankings"]).to_pylist()
        eligible, relevance_selection = _select_luna_relevance_records(
            stage3,
            mapping,
            ranking_rows,
            relevance_manifest,
            levels_in_cache,
        )
    molecule_index: dict[str, int] = {}
    molecules: list[dict[str, Any]] = []
    fingerprints: list[np.ndarray] = []
    record_parent_smiles: dict[str, str] = {}
    records_by_level: dict[str, dict[int, list[str]]] = {
        level: defaultdict(list) for level in levels_in_cache
    }
    for row in stage3:
        record_id = str(row["canonical_record_id"])
        if record_id not in eligible:
            continue
        level = level_by_record[record_id]
        identity = normalize_molecule_identity(str(row.get("canonical_smiles") or ""))
        if identity.status != "ok" or not identity.parent_smiles:
            raise ValueError(f"cannot normalize Stage 3 parent for record {record_id}")
        molecule_id = identity.parent_inchi_key or identity.parent_smiles
        if molecule_id not in molecule_index:
            index = len(molecules)
            _, _, fingerprint = standardize_smiles_and_fp(identity.parent_smiles)
            if fingerprint is None:
                raise ValueError(f"cannot fingerprint Stage 3 parent for record {record_id}")
            molecule_index[molecule_id] = index
            molecules.append(
                {
                    "molecule_id": molecule_id,
                    "canonical_smiles": identity.parent_smiles,
                    "molecule_identity": identity.to_dict(),
                }
            )
            fingerprints.append(
                np.frombuffer(
                    DataStructs.BitVectToBinaryText(fingerprint), dtype=np.uint8
                )
            )
        index = molecule_index[molecule_id]
        records_by_level[level][index].append(record_id)
        record_parent_smiles[record_id] = identity.parent_smiles
    if set(record_parent_smiles) != eligible:
        missing = next(iter(eligible - set(record_parent_smiles)), None)
        raise ValueError(f"Stage 3 mapping coverage mismatch; first missing={missing}")
    packed = np.stack(fingerprints)
    connection = _open_build_db(paths["cache"])
    try:
        selected_levels = (
            ("L1",) if cache_profile == V21_CACHE_PROFILE else levels_in_cache
        )
        n_queries, candidate_counts = _prepare_candidates(
            connection,
            paths,
            records_by_level,
            molecules,
            packed,
            pool_size,
            selected_levels,
            require_full_pool=cache_profile != LUNA_RELEVANCE_CACHE_PROFILE,
        )
        if cache_profile == V21_CACHE_PROFILE:
            v19_version = json.loads(paths["v19_version"].read_text(encoding="utf-8"))
            if (
                v19_version.get("status") != "complete"
                or v19_version.get("cache_sha256") != file_sha256(paths["v19_cache"])
            ):
                raise ValueError("Frozen V19.1 BBB cache failed provenance validation")
            connection.execute("ATTACH DATABASE ? AS v19", (str(paths["v19_cache"]),))
            connection.execute(
                """
                INSERT INTO refs(record_id, query_smiles, group_id, molecule_id)
                SELECT r.external_record_id, q.query_smiles, g.group_id,
                       m.molecule_chembl_id
                FROM v19.assignments a
                JOIN v19.queries q USING(query_id)
                JOIN v19.groups_dim g USING(group_key)
                JOIN v19.molecules m USING(molecule_key)
                JOIN v19.records r USING(record_key)
                WHERE g.group_id IN ('L2', 'L3', 'L4', 'L5')
                """
            )
            connection.commit()
            connection.execute("DETACH DATABASE v19")
            for level in ("L2", "L3", "L4", "L5"):
                assignments = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM refs WHERE group_id=?", (level,)
                    ).fetchone()[0]
                )
                candidate_counts[(level, "candidate_molecules")] = n_queries * pool_size
                candidate_counts[(level, "record_assignments")] = assignments
                candidate_counts[(level, "identity_excluded")] = int(
                    v19_version["level_candidate_counts"][level]["identity_excluded"]
                )
                coverage = connection.execute(
                    """
                    SELECT MIN(n), MAX(n) FROM (
                        SELECT query_smiles, COUNT(DISTINCT molecule_id) AS n
                        FROM refs WHERE group_id=? GROUP BY query_smiles
                    )
                    """,
                    (level,),
                ).fetchone()
                if tuple(coverage) != (pool_size, pool_size):
                    raise ValueError(f"Frozen V19.1 {level} pool coverage changed: {coverage}")
        n_records, n_assignments, n_duplicates = _hydrate_and_render(
            connection,
            paths,
            renderer,
            level_by_record,
            record_parent_smiles,
            task_id,
            cache_profile,
        )
        n_prompts = int(
            connection.execute("SELECT COUNT(*) FROM prompt_tasks").fetchone()[0]
        )
        n_unique_query_smiles = int(
            connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        )
        candidate_molecule_coverage = {}
        for level in levels_in_cache:
            minimum, maximum, query_count = connection.execute(
                """
                SELECT MIN(n), MAX(n), COUNT(*) FROM (
                    SELECT query_smiles, COUNT(DISTINCT molecule_id) AS n
                    FROM refs WHERE group_id=? GROUP BY query_smiles
                )
                """,
                (level,),
            ).fetchone()
            if int(query_count) != n_unique_query_smiles:
                raise ValueError(f"{level} candidate coverage omits query rows")
            candidate_molecule_coverage[level] = {
                "minimum": int(minimum),
                "maximum": int(maximum),
            }
        version = {
            "schema_version": COMPACT_CACHE_SCHEMA_VERSION,
            "status": "prepared",
            "profile": config["cache_profile"],
            "task_id": task_id,
            "model": model["model"],
            "model_revision": model["revision"],
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "backbone_dtype": BACKBONE_DTYPE,
            "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "template_hash": renderer.template_hash,
            "training_artifact_template_hash": (
                renderer.projection["training_prompt_sha256"]
                if cache_profile == V21_CACHE_PROFILE
                else TRAINING_PROMPT_SHA256
            ),
            "training_template_difference": (
                "one terminal newline; rendered prompts are identical after strip"
            ),
            "training_artifact_projection_hash": (
                renderer.projection["training_projection_sha256"]
                if cache_profile == V21_CACHE_PROFILE
                else PROJECTION_SHA256
            ),
            "projection_hash": renderer.projection_hash,
            "template_profile": (
                "v21_bbb_source_native_context_copy"
                if cache_profile == V21_CACHE_PROFILE
                else "v19_1_retrieval_context_copy"
            ),
            "training_dataset": config.get("training_dataset"),
            "training_dataset_revision": config.get("training_dataset_revision"),
            "l1_training_domain_audit": config.get("l1_training_domain_audit"),
            "candidate_contract": config.get(
                "candidate_contract",
                "current_stage3_level_then_morgan_top75_scaffold_disjoint.v1",
            ),
            "record_scope": config.get(
                "record_scope",
                "normalized_v7_stage3_with_current_vote_pure_level_mapping.v1",
            ),
            "revised_level_contract": config["level_contract"],
            "levels_in_cache": list(levels_in_cache),
            "l1_cache": (
                None
                if cache_profile == V21_CACHE_PROFILE
                else str(
                    CACHE_ROOT
                    / f"v9_direct_gold_morgan100/{task_id}/scaffold/valid/rankings.parquet"
                )
            ),
            "l1_morgan_width": pool_size,
            "assay_transfer_initial_morgan_filter": pool_size,
            "candidate_molecule_pool_policy": (
                f"up to {pool_size} per query and level; at least one required"
                if cache_profile == LUNA_RELEVANCE_CACHE_PROFILE
                else f"exactly {pool_size} per query and level"
            ),
            "min_similarity": 0.0,
            "neighbor_identity_policy": "scaffold_disjoint",
            "global_source_exclusion_policy": (
                "none; L1 direct voters are explicitly included"
                if cache_profile == V21_CACHE_PROFILE
                else "direct sources and bottom 75% relevance buckets per level/source"
                if cache_profile == LUNA_RELEVANCE_CACHE_PROFILE
                else "L1 records only"
            ),
            "relevance_selection": relevance_selection,
            "unresolved_endpoint_policy": "exclude unresolved-endpoint records",
            "unresolved_parent_identity_policy": (
                "exclude records that cannot be Morgan-ranked or scaffold-filtered"
            ),
            "other_out_of_domain_policy": "retain and flag in the level mapping",
            "fresh_stage3_parent_molecules": len(molecules),
            "n_queries": n_queries,
            "n_unique_query_smiles": n_unique_query_smiles,
            "n_catalog_records": n_records,
            "n_score_assignments": n_assignments,
            "n_prompt_scores": n_prompts,
            "n_duplicate_prompt_tasks_collapsed": n_duplicates,
            "level_record_counts": {
                level: {
                    "total": sum(
                        count for (current, _, _), count in level_counts.items() if current == level
                    ),
                    "numeric": sum(
                        count
                        for (current, _, numeric), count in level_counts.items()
                        if current == level and numeric
                    ),
                }
                for level in ("L1", *levels_in_cache)
            },
            "level_candidate_counts": {
                level: {
                    name: int(candidate_counts[(level, name)])
                    for name in (
                        "candidate_molecules",
                        "record_assignments",
                        "identity_excluded",
                    )
                }
                for level in levels_in_cache
            },
            "candidate_molecule_coverage_by_level": candidate_molecule_coverage,
            "inputs": {str(path): file_sha256(path) for path in required},
            "cache": str(paths["cache"]),
        }
        metadata_keys = (
            "schema_version", "status", "task_id", "profile", "model", "model_revision",
            "scoring_contract_version", "backbone_dtype", "logit_extraction_dtype",
            "template_hash", "projection_hash", "candidate_contract", "record_scope",
        )
        connection.executemany(
            "INSERT INTO cache_metadata(key, value) VALUES (?, ?)",
            ((key, json.dumps(version[key], sort_keys=True)) for key in metadata_keys),
        )
        connection.execute("DROP TABLE refs")
        connection.commit()
    finally:
        connection.close()
    _write_json(paths["version"], version)
    return version


def _shard_rows(
    connection: sqlite3.Connection, shard_index: int, num_shards: int
) -> list[tuple[int, str, str]]:
    return connection.execute(
        """
        SELECT prompt_key, cache_key, prompt FROM prompt_tasks
        WHERE (prompt_key - 1) % ? = ? ORDER BY length(prompt), prompt_key
        """,
        (num_shards, shard_index),
    ).fetchall()


def score(
    *,
    task_id: str = "bbb_martins",
    cache_profile: str = DEFAULT_CACHE_PROFILE,
    shard_index: int,
    num_shards: int,
    device: int,
    batch_size: int = 128,
) -> dict[str, Any]:
    paths = _paths(task_id, cache_profile)
    version = json.loads(paths["version"].read_text())
    if version.get("status") != "prepared":
        raise ValueError("Progressive cache is not prepared")
    connection = sqlite3.connect(f"file:{paths['cache'].resolve()}?mode=ro", uri=True)
    rows = _shard_rows(connection, shard_index, num_shards)
    connection.close()
    paths["journals"].mkdir(exist_ok=True)
    journal = paths["journals"] / f"scores-{shard_index:02d}-of-{num_shards:02d}.jsonl"
    completed = _read_jsonl(journal) if journal.exists() else []
    expected_keys = {row[1] for row in rows}
    completed_keys = [row["cache_key"] for row in completed]
    if len(completed_keys) != len(set(completed_keys)) or not set(completed_keys) <= expected_keys:
        raise ValueError(f"Score journal contains duplicate or foreign keys: {journal}")
    completed_key_set = set(completed_keys)
    rows = [row for row in rows if row[1] not in completed_key_set]
    total_rows = len(completed) + len(rows)
    profile = _model_config(task_id, cache_profile)
    if (version.get("model"), version.get("model_revision")) != (
        profile["model"],
        profile["revision"],
    ):
        raise ValueError("Progressive cache model disagrees with its selected profile")
    snapshot = resolve_model_snapshot(
        str(profile["model"]), str(profile["revision"]), local_files_only=True
    )
    model, tokenizer = load_model(snapshot, device=device)
    with journal.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(rows), batch_size):
            batch_rows = rows[offset : offset + batch_size]
            tasks = [
                PromptTask(
                    cache_key=cache_key,
                    prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                    prompt=prompt,
                    task_id=task_id,
                    query_smiles="",
                    group_id="",
                    molecule_id="",
                    record_id="",
                    model=str(profile["model"]),
                    model_revision=str(profile["revision"]),
                    scoring_contract_version=SCORING_CONTRACT_VERSION,
                    template_hash=str(version["template_hash"]),
                    projection_hash=str(version["projection_hash"]),
                )
                for _, cache_key, prompt in batch_rows
            ]
            for result in score_prompt_batch(model, tokenizer, tasks, device=device):
                handle.write(
                    json.dumps(
                        {
                            "cache_key": result.cache_key,
                            "transfer_probability": result.transfer_probability,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
            print(
                f"shard {shard_index + 1}/{num_shards}: "
                f"{len(completed) + offset + len(tasks)}/{total_rows}",
                flush=True,
            )
    return {"status": "complete", "journal": str(journal), "n_scores": total_rows}


def finalize(
    num_shards: int,
    task_id: str = "bbb_martins",
    cache_profile: str = DEFAULT_CACHE_PROFILE,
) -> dict[str, Any]:
    paths = _paths(task_id, cache_profile)
    version = json.loads(paths["version"].read_text())
    connection = sqlite3.connect(paths["cache"])
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE prompt_scores(prompt_key INTEGER PRIMARY KEY, transfer_probability REAL NOT NULL)"
    )
    try:
        for shard_index in range(num_shards):
            expected = _shard_rows(connection, shard_index, num_shards)
            journal = paths["journals"] / f"scores-{shard_index:02d}-of-{num_shards:02d}.jsonl"
            observed = _read_jsonl(journal) if journal.exists() else []
            observed_by_key = {
                row["cache_key"]: float(row["transfer_probability"]) for row in observed
            }
            if len(observed_by_key) != len(observed) or set(observed_by_key) != {
                row[1] for row in expected
            }:
                raise ValueError(f"Incomplete or mismatched score journal: {journal}")
            connection.executemany(
                "INSERT INTO prompt_scores VALUES (?, ?)",
                (
                    (prompt_key, observed_by_key[cache_key])
                    for prompt_key, cache_key, _ in expected
                ),
            )
            connection.commit()
        expected_count = int(
            connection.execute("SELECT COUNT(*) FROM prompt_tasks").fetchone()[0]
        )
        observed_count = int(
            connection.execute("SELECT COUNT(*) FROM prompt_scores").fetchone()[0]
        )
        if expected_count != observed_count:
            raise ValueError(
                f"Prompt score coverage mismatch: expected={expected_count}, found={observed_count}"
            )
        connection.executescript(
            """
            CREATE TABLE scores(
                score_key INTEGER PRIMARY KEY, transfer_probability REAL NOT NULL
            );
            INSERT INTO scores SELECT prompt_key, transfer_probability FROM prompt_scores;
            CREATE TABLE assignments(
                query_id INTEGER NOT NULL,
                group_key INTEGER NOT NULL,
                molecule_key INTEGER NOT NULL,
                record_key INTEGER NOT NULL,
                score_key INTEGER NOT NULL,
                PRIMARY KEY(query_id, group_key, molecule_key, record_key)
            ) WITHOUT ROWID;
            INSERT INTO assignments
                SELECT query_id, group_key, molecule_key, record_key, prompt_key
                FROM prompt_assignments;
            DROP TABLE prompt_assignments;
            DROP TABLE prompt_scores;
            DROP TABLE prompt_tasks;
            """
        )
        connection.execute(
            "UPDATE cache_metadata SET value=? WHERE key='status'", (json.dumps("complete"),)
        )
        connection.commit()
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    finally:
        connection.close()
    for journal in paths["journals"].glob("*.jsonl"):
        journal.unlink()
    paths["journals"].rmdir()
    connection = sqlite3.connect(paths["cache"])
    connection.execute("VACUUM")
    connection.close()
    version.update(
        status="complete",
        n_cached_scores=observed_count,
        cache_quick_check=quick_check,
        cache_sha256=file_sha256(paths["cache"]),
    )
    _write_json(paths["version"], version)
    return version


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="bbb_martins")
    prepare_parser.add_argument(
        "--cache-profile", choices=CACHE_PROFILES, default=DEFAULT_CACHE_PROFILE
    )
    prepare_parser.add_argument("--pool-size", type=int, default=75)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="bbb_martins")
    score_parser.add_argument(
        "--cache-profile", choices=CACHE_PROFILES, default=DEFAULT_CACHE_PROFILE
    )
    score_parser.add_argument("--shard-index", type=int, required=True)
    score_parser.add_argument("--num-shards", type=int, required=True)
    score_parser.add_argument("--device", type=int, required=True)
    score_parser.add_argument("--batch-size", type=int, default=128)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--task", choices=sorted(TASK_CONFIGS), default="bbb_martins")
    finalize_parser.add_argument(
        "--cache-profile", choices=CACHE_PROFILES, default=DEFAULT_CACHE_PROFILE
    )
    finalize_parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            task_id=args.task,
            pool_size=args.pool_size,
            cache_profile=args.cache_profile,
        )
    elif args.command == "score":
        result = score(
            task_id=args.task,
            cache_profile=args.cache_profile,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        result = finalize(
            args.num_shards,
            task_id=args.task,
            cache_profile=args.cache_profile,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
