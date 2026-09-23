"""Pinned models, prompt scoring, and read-only assay-transfer caches."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BACKBONE_DTYPE = "bfloat16"
LOGIT_EXTRACTION_DTYPE = "float32"
SCORING_CONTRACT_VERSION = (
    "assay_transfer_chat_first_divergent_token_bf16_backbone_fp32_head.v2"
)
COMPACT_CACHE_SCHEMA_VERSION = "txagent_assay_transfer_compact_cache.v1"
REQUIRED_TRANSFORMERS_VERSION = "4.57.6"
CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache" / "assay_reranking"
ACTIVE_CACHE_ROOT = CACHE_ROOT / "active"
ARCHIVE_CACHE_ROOT = CACHE_ROOT / "archive"
DATA_ACTIVE_CACHE_ROOT = Path(__file__).resolve().parents[3] / "data" / "caches" / "assay_reranking" / "active"
CANONICAL_CACHE_ROOT = DATA_ACTIVE_CACHE_ROOT / "flat_v5"
ACTIVE_CACHE_PROFILES = frozenset(
    {
        "cache_matched_retrieval_v3",
        "cache_matched_retrieval_v3_l1_v10_3_v1",
        "cache_matched_retrieval_v3_gold_v2_v1",
        "cache_matched_retrieval_v3_l1_v10_4_v1",
        "l1_context_morgan_semantic_l2_v1",
        "l1_context_morgan25_v10_4_gold_v2_v1",
        "l1_context_morgan50_v10_4_gold_v2_v1",
        "l1_context_morgan100_v10_4_gold_v2_v1",
        "l1_context_semantic_l2_v1",
        "l1_context_semantic_weighted_l2_v1",
        "recent_models_three_pools_morgan100_v2_gold_v1",
        "v24_1_bbb_uid_levels_morgan75",
        "v25_oral_uid_levels_morgan75",
        "v25_oral_uid_levels_morgan75_l2",
        "v25_oral_uid_levels_morgan75_l3",
        "flat_v5/gold_v1/bbb_martins/l2plus/assay_transfer/v24_1/morgan75",
        "flat_v5/gold_v1/bioavailability_ma/l2plus/assay_transfer/v25/morgan75",
        "flat_v5/gold_v1/bioavailability_ma/l2plus/assay_transfer/"
        "v25/morgan75_l2_builder",
        "flat_v5/gold_v1/bioavailability_ma/l2plus/assay_transfer/"
        "v25/morgan75_l3_builder",
        "flat_v5/gold_v1/skin_reaction/l2plus/assay_transfer/v27/training_candidate_copy_shared_parent100_v2",
        "flat_v5/tdc_v1/skin_reaction/l2plus/assay_transfer/v27/training_candidate_copy_shared_parent100_v2",
        "flat_v5/gold_v1/ames/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent100_v2",
        "flat_v5/gold_v1/dili/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent100_v2",
        "flat_v5/gold_v1/carcinogens/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent100_v2",
        "flat_v5/gold_v1/carcinogens/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent50_v1",
        "flat_v5/gold_v1/carcinogens/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent40_v1",
        "flat_v5/gold_v1/carcinogens/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent40_partial_snapshot_v1",
        "flat_v5/tdc_v1/ames/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent100_v2",
        "flat_v5/tdc_v1/ames/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent50_v1",
        "flat_v5/tdc_v1/ames/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent40_v1",
        "flat_v5/tdc_v1/ames/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent40_partial_snapshot_v1",
        "flat_v5/tdc_v1/dili/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent100_v2",
        "flat_v5/tdc_v1/carcinogens/l2plus/assay_transfer/v27/general_candidate_copy_shared_parent100_v2",
        "flat_v5/tdc_v1/ames/l1/assay_transfer/v10_3/tdc_pinned_v1",
        "flat_v5/tdc_v1/dili/l1/assay_transfer/v10_3/tdc_pinned_v1",
        "flat_v5/tdc_v1/carcinogens/l1/assay_transfer/v10_3/tdc_pinned_v1",
        "v10_3_direct_gold_morgan100_v1",
        "v10_3_best_scaffold_morgan100_v1",
        "v10_3_best_parent_morgan100_v1",
        "ranked_level_retrieval_v2",
        "ranked_level_retrieval_v3",
        "ranked_level_retrieval_v4",
        "ranked_level_retrieval_tdc_v1",
        "ranked_level_retrieval_tdc_v1_assay_v10_3_best_v1",
        "ranked_level_retrieval_tdc_v1_assay_skin_v9_v1",
        "ranked_level_retrieval_tdc_v1_l1_assay_task_best_v1",
        "ranked_level_retrieval_tdc_v1_indirect_v1",
        "ranked_level_retrieval_tdc_v1_indirect_v2",
        "ranked_level_retrieval_tdc_v1_gold_v1_mixed_l1_assay_v10_3_best_v1",
        "ranked_level_retrieval_tdc_v1_gold_v1_mixed_l1_assay_v10_3_best_v2",
        "ranked_level_retrieval_tdc_v1_gold_v1_mixed_l1_assay_v10_3_best_v3",
        "ranked_level_retrieval_gold_v1_addon_v1",
        "ranked_level_retrieval_gold_v1_addon_v2",
        "ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1",
        "ranked_level_retrieval_skin_v27_gold_v1",
        "ranked_level_retrieval_skin_v27_tdc_v1",
        "ranked_level_retrieval_skin_gold_v1_l1_adapter_v2",
        "tdc_mixed_l1_v1",
        "v10_4_direct_gold_morgan100_v1",
        "v9_skin_gold_v1_morgan100_v1",
        "v9_skin_gold_v1_scaffold_morgan100_v1",
    }
)


def cache_profile_root(profile: str) -> Path:
    """Return the explicit active or archived root for one cache profile."""
    if str(profile).startswith("flat_v5/"):
        parts = tuple(str(profile).split("/"))
        if len(parts) != 7 or any(not part or part in {".", ".."} for part in parts):
            raise ValueError(f"Invalid canonical flat-v5 cache profile: {profile!r}")
        return CANONICAL_CACHE_ROOT.joinpath(*parts[1:])
    from predict.retrieval.assay_reranking.artifact_bundle import (
        canonical_cache_alias,
        canonical_cache_path,
    )

    canonical = canonical_cache_alias(str(profile))
    if canonical is not None:
        candidate = canonical_cache_path(canonical)
        if candidate.exists():
            return candidate
    if profile in {
        "ranked_level_retrieval_v3",
        "ranked_level_retrieval_v4",
        "ranked_level_retrieval_tdc_v1",
        "ranked_level_retrieval_tdc_v1_assay_v10_3_best_v1",
        "ranked_level_retrieval_tdc_v1_assay_skin_v9_v1",
        "ranked_level_retrieval_tdc_v1_l1_assay_task_best_v1",
        "ranked_level_retrieval_tdc_v1_indirect_v1",
        "ranked_level_retrieval_tdc_v1_indirect_v2",
        "ranked_level_retrieval_tdc_v1_gold_v1_mixed_l1_assay_v10_3_best_v1",
        "ranked_level_retrieval_tdc_v1_gold_v1_mixed_l1_assay_v10_3_best_v2",
        "ranked_level_retrieval_tdc_v1_gold_v1_mixed_l1_assay_v10_3_best_v3",
        "ranked_level_retrieval_gold_v1_addon_v1",
        "ranked_level_retrieval_gold_v1_addon_v2",
        "ranked_level_retrieval_gold_v1_addon_l1_assay_safety_best_v1",
        "ranked_level_retrieval_skin_v27_gold_v1",
        "ranked_level_retrieval_skin_v27_tdc_v1",
        "ranked_level_retrieval_skin_gold_v1_l1_adapter_v2",
        "tdc_mixed_l1_v1",
    }:
        return DATA_ACTIVE_CACHE_ROOT / profile
    parent = ACTIVE_CACHE_ROOT if profile in ACTIVE_CACHE_PROFILES else ARCHIVE_CACHE_ROOT
    return parent / profile


def canonical_cache_profile(profile: str, *, benchmark: str = "gold_v1") -> str | None:
    """Map a historical profile name to its semantic flat-v5 successor."""
    from predict.retrieval.assay_reranking.artifact_bundle import canonical_cache_alias

    return canonical_cache_alias(profile, benchmark=benchmark)
MODEL_ROLES = (
    "direct", "direct_v10_3", "direct_v10_3_0_2", "direct_v10_4",
    "direct_task_best", "direct_tdc_best",
    "indirect", "all_records",
)
MODEL_PROFILES = {
    "bbb_martins": {
        "direct_task_best": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-tdc-bbb-martins-best",
            "revision": "42cb4afa224c102898d75fe4ad48e166254952b5",
            "prompt_profile": "tdc_binary_same_different_parent_smiles.v1",
            "checkpoint_step": 130,
        },
        "direct": {
            "model": "jiosephlee/assay-transfer-tool-soft-v9.0.2-bbb-martins-vote-mean",
            "revision": "06b9900222e887597ca06f0015a09fa87b8eb509",
            "prompt_profile": "v9",
        },
        "direct_v10_3": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-bbb-martins-best",
            "revision": "7678f7a1c43932f7612cfd49a8f4872d6e2f4cab",
            "prompt_profile": "v10_3",
            "checkpoint_step": 120,
        },
        "direct_v10_4": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-4-bbb-martins-best",
            "revision": "4c064f571a53234c5eb1c9d4c0e068bdd10860c1",
            "prompt_profile": "v10_4",
            "checkpoint_step": 70,
        },
        "indirect": {
            "model": "jiosephlee/intern-s1-mini-assay-transfer-v19-1-bbb-martins-numeric-best",
            "revision": "b93ebfb909de5689fe3b50978d65172a6096b974",
            "prompt_profile": "v19_1",
        },
        "all_records": {
            "model": "jiosephlee/intern-s1-mini-assay-transfer-v21-bbb-martins-mixed-best",
            "revision": "2521166a95cd36bc78fb9b90669cc097ead758ca",
            "prompt_profile": "v21_bbb",
        },
    },
    "bioavailability_ma": {
        "direct_task_best": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-tdc-v2-hp-bioavailability-ma-best",
            "revision": "a1cb6960e9ea4b18c69a87cf10bb7ef695192208",
            "prompt_profile": "tdc_binary_same_different_parent_smiles.v1",
            "checkpoint_step": 80,
        },
        "direct": {
            "model": "jiosephlee/assay-transfer-tool-soft-v9-bioavailability-ma-mixed-continuous",
            "revision": "6f3aefabc9a07b357066aaf7ca0f69ab63785240",
            "prompt_profile": "v9",
        },
        "direct_v10_3": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-bioavailability-ma-best",
            "revision": "e8837889f707d97f8aeb7e54e74fcf2c7c2968de",
            "prompt_profile": "v10_3",
            "checkpoint_step": 140,
        },
        "direct_v10_3_0_2": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-0-2-bioavailability-ma-BEST",
            "revision": "29cc74f02df160b1f153edb700da66d1c30ef4ed",
            "prompt_profile": "v10_3_0_2",
            "checkpoint_step": 160,
        },
        "direct_v10_4": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-4-bioavailability-ma-best",
            "revision": "e3a5b31ce8c47b4aa950c974cf9c3eb80c67fcae",
            "prompt_profile": "v10_4",
            "checkpoint_step": 200,
        },
        "indirect": {
            "model": "jiosephlee/intern-s1-mini-assay-transfer-v19-1-bioavailability-ma-numeric-best",
            "revision": "612cd794583e2129a664defaf9229b26d94b9a69",
            "prompt_profile": "v19_1",
        },
    },
    "skin_reaction": {
        "direct_task_best": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v9-0-2-tdc-v2-skin-reaction-best",
            "revision": "870f401c451b4324a93af5049bec02c529119c20",
            "prompt_profile": "tdc_binary_same_different_parent_smiles.v1",
            "checkpoint_step": 80,
        },
        "direct": {
            "model": "jiosephlee/assay-transfer-tool-soft-v9.0.2-skin-reaction-mixed-continuous",
            "revision": "e4e894af28151760d2041275c5ecf136971c3925",
            "prompt_profile": "v9",
        },
        "indirect": {
            "model": "jiosephlee/intern-s1-mini-assay-transfer-v19-1-skin-reaction-numeric-best",
            "revision": "f29d100ca2112490d22913736d4efa5bc5308cb6",
            "prompt_profile": "v19_1",
        },
    },
    "ames": {
        "direct_tdc_best": {
            "model": "jiosephlee/intern-s1-mini-ames-v10-3-tdc-mixed-canonical-best",
            "revision": "392dd01912090040cfc0427f9f680d16d47eb27b",
            "prompt_profile": "tdc_binary_same_different_parent_smiles.v1",
        },
        "direct_task_best": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-ames-best",
            "revision": "d27f5f44328a43ccfea650841dba9fcf4729ec69",
            "prompt_profile": "gold_v1_context.v10.3",
            "checkpoint_step": 50,
        },
    },
    "dili": {
        "direct_tdc_best": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-tdc-pinned-dili-xnkoqrux",
            "revision": "bb5248609a729ac7dc6d6c9a3cccba1566a3d19e",
            "prompt_profile": "tdc_binary_same_different_parent_smiles.v1",
        },
        "direct_task_best": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-1-dili-best",
            "revision": "b594bd7be81d140a9ebdf137aea5b979359e0b7f",
            "prompt_profile": "gold_v1_context.v10.3",
            "checkpoint_step": 140,
        },
    },
    "carcinogens": {
        "direct_tdc_best": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-tdc-pinned-carcinogens-2xcwcpul",
            "revision": "8be3376cd1d84e46c18b61617f63ad597aeaff52",
            "prompt_profile": "tdc_binary_same_different_parent_smiles.v1",
        },
        "direct_task_best": {
            "model": "jiosephlee/intern-s1-mini-context-conditioned-molecule-transfer-v10-3-carcinogens-step160",
            "revision": "204d66ca9f7692ae76938ef6c5ebaa84737fd5f2",
            "prompt_profile": "gold_v1_context.v10.3",
            "checkpoint_step": 160,
        },
    },
}


class AssayTransferCacheMiss(RuntimeError):
    """A frozen inference run requested a score absent from its cache."""


@dataclass(frozen=True)
class PromptTask:
    cache_key: str
    prompt_hash: str
    prompt: str
    task_id: str
    query_smiles: str
    group_id: str
    molecule_id: str
    record_id: str
    model: str
    model_revision: str
    scoring_contract_version: str
    template_hash: str
    projection_hash: str


@dataclass(frozen=True)
class PromptScore:
    cache_key: str
    logp_transfer: float
    logp_not_transfer: float
    transfer_probability: float


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_immutable_revision(revision: str) -> str:
    value = str(revision or "").strip().lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Model revision must be an immutable 40-character SHA: {revision!r}")
    return value


def model_profile(task_id: str, role: str) -> dict[str, Any]:
    """Return one runnable pinned model, failing closed when none exists."""
    if role not in MODEL_ROLES:
        raise ValueError(f"Unknown assay-reranking model role: {role}")
    if task_id not in MODEL_PROFILES:
        raise ValueError(f"Unknown assay-reranking task: {task_id}")
    if role not in MODEL_PROFILES[task_id]:
        raise ValueError(f"No {role} reranker for {task_id}")
    profile = dict(MODEL_PROFILES[task_id][role])
    profile["revision"] = require_immutable_revision(str(profile["revision"]))
    return profile


def build_prompt_task(
    renderer: Any,
    record: Mapping[str, Any],
    *,
    query_smiles: str,
    group_id: str,
    molecule_id: str,
    model: str,
    model_revision: str,
) -> PromptTask:
    prompt = renderer.render(record, query_smiles)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    revision = require_immutable_revision(model_revision)
    identity = {
        "prompt_hash": prompt_hash,
        "model": model,
        "model_revision": revision,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "template_hash": renderer.template_hash,
        "projection_hash": renderer.projection_hash,
    }
    cache_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PromptTask(
        cache_key=cache_key,
        prompt_hash=prompt_hash,
        prompt=prompt,
        task_id=renderer.task_id,
        query_smiles=query_smiles,
        group_id=group_id,
        molecule_id=molecule_id,
        record_id=str(record["record_id"]),
        model=model,
        model_revision=revision,
        scoring_contract_version=SCORING_CONTRACT_VERSION,
        template_hash=renderer.template_hash,
        projection_hash=renderer.projection_hash,
    )


def prompt_task_to_dict(task: PromptTask) -> dict[str, Any]:
    return asdict(task)


def prompt_task_from_dict(payload: Mapping[str, Any]) -> PromptTask:
    return PromptTask(**dict(payload))


class CompactScoreCache:
    """Read finalized scores keyed by query, group, molecule, and record."""

    def __init__(self, path: str | Path, *, task_id: str):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.connection = sqlite3.connect(
            f"file:{self.path.resolve()}?mode=ro", uri=True, timeout=60.0
        )
        self.connection.row_factory = sqlite3.Row
        self.metadata = {
            str(row["key"]): json.loads(str(row["value"]))
            for row in self.connection.execute("SELECT key, value FROM cache_metadata")
        }
        if self.metadata.get("schema_version") != COMPACT_CACHE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported assay-reranking cache: {self.path}")
        if self.metadata.get("task_id") != task_id:
            raise ValueError("Assay-reranking cache task mismatch")
        if self.metadata.get("status") != "complete":
            raise ValueError("Assay-reranking cache is not finalized")

    def close(self) -> None:
        self.connection.close()

    def records_for_candidates(
        self, query_smiles: str, group_id: str, molecule_ids: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        ids = list(dict.fromkeys(str(value) for value in molecule_ids))
        output = {value: [] for value in ids}
        for offset in range(0, len(ids), 400):
            self._append_records(output, query_smiles, group_id, ids[offset : offset + 400])
        return output

    def _append_records(
        self,
        output: dict[str, list[dict[str, Any]]],
        query_smiles: str,
        group_id: str,
        ids: list[str],
    ) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        sql = f"""
            SELECT m.molecule_chembl_id, r.external_record_id, r.payload,
                   s.transfer_probability
            FROM assignments a
            JOIN queries q USING(query_id)
            JOIN groups_dim g USING(group_key)
            JOIN molecules m USING(molecule_key)
            JOIN records r USING(record_key)
            JOIN scores s USING(score_key)
            WHERE q.query_smiles = ? AND g.group_id = ?
              AND m.molecule_chembl_id IN ({placeholders})
            ORDER BY m.molecule_chembl_id, r.external_record_id
        """
        for row in self.connection.execute(sql, [query_smiles, group_id, *ids]):
            output[str(row["molecule_chembl_id"])].append({
                "record_id": str(row["external_record_id"]),
                "transfer_probability": float(row["transfer_probability"]),
                "payload": json.loads(str(row["payload"])),
            })


class CachedAssayReranker:
    """Attach scores from one immutable compact cache to retrieved candidates."""

    name = "assay_transfer"

    def __init__(
        self,
        *,
        task_id: str,
        cache_path: str | Path,
        model: str,
        model_revision: str,
        renderer: Any,
        profile_name: str,
        template_profile: str,
        query_context_policy: str,
        allow_missing: bool = False,
        **_: Any,
    ):
        self.task_id = task_id
        self.model = model
        self.model_revision = require_immutable_revision(model_revision)
        self.renderer = renderer
        self.profile_name = profile_name
        self.template_profile = template_profile
        self.query_context_policy = query_context_policy
        self.allow_missing = allow_missing
        self.compact_cache = CompactScoreCache(cache_path, task_id=task_id)
        self.cache = self.compact_cache
        self._validate_provenance()

    def _validate_provenance(self) -> None:
        expected = {
            "model": self.model,
            "model_revision": self.model_revision,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "backbone_dtype": BACKBONE_DTYPE,
            "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "template_hash": self.renderer.template_hash,
            "projection_hash": self.renderer.projection_hash,
        }
        mismatches = {
            key: {"expected": value, "observed": self.cache.metadata.get(key)}
            for key, value in expected.items()
            if self.cache.metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Assay-reranking cache provenance mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "profile": self.profile_name,
            "task_id": self.task_id,
            "model": self.model,
            "model_revision": self.model_revision,
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "backbone_dtype": BACKBONE_DTYPE,
            "logit_extraction_dtype": LOGIT_EXTRACTION_DTYPE,
            "template_profile": self.template_profile,
            "template_hash": self.renderer.template_hash,
            "projection_hash": self.renderer.projection_hash,
            "query_context_policy": self.query_context_policy,
            "candidate_contract": self.cache.metadata.get("candidate_contract"),
        }

    def rerank_records(
        self, *, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_molecule = {str(row["molecule_chembl_id"]): row for row in candidates}
        found = self.cache.records_for_candidates(query_smiles, group_id, by_molecule)
        output, missing = [], []
        for molecule_id, candidate in by_molecule.items():
            rows = found.get(molecule_id) or []
            if not rows:
                missing.append(molecule_id)
            for scored in rows:
                output.append(_scored_candidate(candidate, scored))
        if missing and not self.allow_missing:
            raise AssayTransferCacheMiss(
                f"Cache is missing {len(missing)} of {len(by_molecule)} candidates for {group_id}"
            )
        output.sort(key=lambda row: (
            -float(row["transfer_selection_score"]),
            -float(row.get("similarity") or 0.0),
            str(row.get("molecule_chembl_id") or ""),
            str(row.get("transfer_winning_record_id") or ""),
        ))
        for rank, row in enumerate(output, start=1):
            row["transfer_selection_rank"] = rank
        return output

    def rerank(
        self, *, query_smiles: str, group_id: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return self.rerank_records(
            query_smiles=query_smiles, group_id=group_id, candidates=candidates
        )


def _scored_candidate(
    candidate: Mapping[str, Any], scored: Mapping[str, Any]
) -> dict[str, Any]:
    record = dict(scored["payload"])
    record_id = str(scored["record_id"])
    for evidence in candidate.get("evidence_rows") or []:
        ids = list(evidence.get("_representative_record_ids") or [])
        if record_id not in ids:
            continue
        examples = list((evidence.get("minimal_evidence") or {}).get("examples") or [])
        position = ids.index(record_id)
        if position < len(examples) and examples[position].get("resolved_measurement_display"):
            record["resolved_measurement_display"] = examples[position][
                "resolved_measurement_display"
            ]
        break
    return {
        **candidate,
        "transfer_selection_score": scored["transfer_probability"],
        "transfer_winning_record_id": record_id,
        "transfer_winning_record": record,
        "transfer_scored_record_count": 1,
    }


def probability_from_logits(logit_a: float, logit_b: float) -> float:
    maximum = max(logit_a, logit_b)
    a = math.exp(logit_a - maximum)
    b = math.exp(logit_b - maximum)
    return a / (a + b)


def resolve_model_snapshot(
    model: str, revision: str, *, local_files_only: bool = False
) -> str:
    """Download or resolve exactly one immutable Hugging Face revision."""
    from huggingface_hub import HfApi, snapshot_download

    immutable = require_immutable_revision(revision)
    if not local_files_only:
        observed = require_immutable_revision(
            HfApi().model_info(model, revision=immutable).sha
        )
        if observed != immutable:
            raise ValueError(
                f"Configured model revision changed: expected={immutable}, observed={observed}"
            )
    return snapshot_download(
        repo_id=model,
        revision=immutable,
        local_files_only=local_files_only,
    )


def load_model(snapshot: str, *, device: int = 0) -> tuple[Any, Any]:
    """Load one InternS1 checkpoint replica and its pinned tokenizer."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Assay reranking requires transformers {REQUIRED_TRANSFORMERS_VERSION}; "
            f"observed {transformers.__version__}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        local_files_only=True,
    ).to(f"cuda:{device}")
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def score_prompt_batch(
    model: Any, tokenizer: Any, tasks: Sequence[PromptTask], *, device: int = 0,
    tokenized: tuple[list[list[int]], list[int], list[int]] | None = None,
) -> list[PromptScore]:
    """Score the first token where the `(A)` and `(B)` answers diverge."""
    import torch

    prefixes, a_tokens, b_tokens = tokenized or _answer_prefixes(tokenizer, tasks)
    encoded = tokenizer.pad({"input_ids": prefixes}, padding=True, return_tensors="pt")
    encoded = {key: value.to(f"cuda:{device}") for key, value in encoded.items()}
    positions = torch.arange(
        encoded["attention_mask"].shape[1], device=encoded["input_ids"].device
    )
    last = (encoded["attention_mask"] * positions).max(dim=1).values
    with torch.inference_mode():
        hidden = model.get_decoder()(**encoded, use_cache=False).last_hidden_state
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    last_hidden = hidden[rows, last].float()
    output_head = model.get_output_embeddings().weight
    output = []
    for index, task in enumerate(tasks):
        logit_a = float(
            (last_hidden[index] * output_head[a_tokens[index]].float()).sum().item()
        )
        logit_b = float(
            (last_hidden[index] * output_head[b_tokens[index]].float()).sum().item()
        )
        output.append(
            PromptScore(
                cache_key=task.cache_key,
                logp_transfer=logit_a,
                logp_not_transfer=logit_b,
                transfer_probability=probability_from_logits(logit_a, logit_b),
            )
        )
    return output


def _answer_prefixes(
    tokenizer: Any, tasks: Sequence[PromptTask]
) -> tuple[list[list[int]], list[int], list[int]]:
    prefixes, a_tokens, b_tokens = [], [], []
    for task in tasks:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": task.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        a_ids = tokenizer(prompt + "(A)", add_special_tokens=False).input_ids
        b_ids = tokenizer(prompt + "(B)", add_special_tokens=False).input_ids
        divergent = next(
            (i for i, pair in enumerate(zip(a_ids, b_ids)) if pair[0] != pair[1]),
            None,
        )
        if divergent is None:
            raise ValueError("A/B answer tokenizations do not diverge")
        prefixes.append(a_ids[:divergent])
        a_tokens.append(int(a_ids[divergent]))
        b_tokens.append(int(b_ids[divergent]))
    return prefixes, a_tokens, b_tokens
