"""Render frozen molecular evidence and run the FINAL flat harness. Historical prompt renderers support offline replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from data.processing.paths import REPO_ROOT
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from predict.prompts import ROOT as PROMPT_ROOT, bundle_directory, flat_task_profiles, template_environment
import yaml

from optimization.selection.pool import load_coverage
from predict.harnesses.branches.assay_transfer import (
    ASSAY_TRANSFER_DIVERSITY_NONE,
    ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
    ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
)
from predict.harnesses.branches.retrieval import (
    ASSAY_TRANSFER_TOOL_STRATEGY,
    BranchRetrievalConfig,
    RetrievalReranker,
    flatten_retrieval_groups,
    retrieve_mechanism_evidence,
    retrieval_coverage,
)
from predict.retrieval.policies import NeighborIdentityPolicy, SIMILARITY_SELECTOR

from predict.retrieval.policies import normalize_molecule_identity, selector_metadata
from predict.tasks.bbb_martins import EVIDENCE_POLICY as BBB_EVIDENCE_POLICY, prediction_only_uids
from predict.utils.evidence import minimal_evidence_from_row
from predict.utils.response import validate_query_prior_output
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic
from predict.harnesses.branches.manifests import (run_directory, read_record, write_batch, scientific_settings,
    require_query_ids, require_artifact_versions, canonical_location, PREDICTION_ROOT,
    prediction_run_root, prediction_batch_directory, query_prior_batch)
from data.processing.gold_labels.conditioned_benchmark import split_path, tdc_split_path, tdc_config, TDC_MANIFEST
from predict.retrieval.assay_reranking.artifact_bundle import (
    ArtifactBundleError,
    bundle_manifest_path,
    bundle_receipt_path,
    require_receipt,
    cache_task,
    cache_level_limit,
    resolve_cache_set,
)


DEFAULT_PROVIDER_CONFIG = (
    Path(__file__).resolve().parents[2] / "api_client/providers"
    / "deepseek_v4_flash_0731_openrouter_auto_128_high.json"
)

TIANANG_PROMPT_VERSION = "tianang_flat_v1"
JOSEPH_V1_PROMPT_VERSION = "joseph_flat_v1"
JOSEPH_PROMPT_VERSION = "joseph_flat_v2"
CONTEXT_V4_PROMPT_VERSION = "joseph_flat_context_v4_v1"
CONTEXT_V5_PROMPT_VERSION = "full_flat_context_v5"
CONTEXT_V6_PROMPT_VERSION = "full_flat_context_v6_oral_high_low"
CONTEXT_V7_PROMPT_VERSION = "full_flat_context_v7_oral_final_only"
CONTEXT_V8_PROMPT_VERSION = "full_flat_context_v8_context_scoped_transfer_v1"
CONTEXT_V5_SIX_TASKS_PROMPT_VERSION = "full_flat_context_v5_six_tasks_v1"
CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION = "full_flat_context_v5_six_tasks_upstream_v1"
CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION = "full_flat_context_v5_six_tasks_upstream_v2"
CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION = "full_flat_context_v5_six_tasks_upstream_v3"
TDC_SKIN_LLNA_PROMPT_VERSION = "full_flat_context_v5_six_tasks_tdc_skin_llna_v1"
TDC_SKIN_LLNA_SCORES_PROMPT_VERSION = "full_flat_context_v5_six_tasks_tdc_skin_llna_scores_v2"
TDC_V2_MIXED_PROMPT_VERSION = "full_flat_context_v5_six_tasks_tdc_v2_mixed_v1"
DILI_TDC_V2_MIXED_V3_PROMPT_VERSION = "full_flat_context_v5_six_tasks_dili_tdc_v2_mixed_v3"
DILI_TDC_V2_POLARIZED_PROMPT_VERSION = "full_flat_context_v5_six_tasks_dili_tdc_v2_polarized_v1"
CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION = "full_flat_context_v5_six_tasks_upstream_scores_v2"
FINAL_FULL_FLAT_PROMPT_VERSION = "final_full_flat"
FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION = "final_full_flat_tdc_carc_direct_priority_v1"
FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION = "final_full_flat_tdc_carc_strong_direct_v1"
FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION = "final_full_flat_tdc_carc_direct_positive_gate_v1"
FINAL_FULL_FLAT_PROMPT_VERSIONS = {
    FINAL_FULL_FLAT_PROMPT_VERSION, FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION, FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
}
CONTEXT_CLAIMS_PROMPT_VERSIONS = {
    CONTEXT_V5_PROMPT_VERSION,
    CONTEXT_V6_PROMPT_VERSION,
    CONTEXT_V8_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
    TDC_SKIN_LLNA_PROMPT_VERSION,
    TDC_SKIN_LLNA_SCORES_PROMPT_VERSION,
    TDC_V2_MIXED_PROMPT_VERSION,
    DILI_TDC_V2_MIXED_V3_PROMPT_VERSION,
    DILI_TDC_V2_POLARIZED_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION,
    FINAL_FULL_FLAT_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION, FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
}
CONTEXT_NUMBERED_PROMPT_VERSIONS = {
    *CONTEXT_CLAIMS_PROMPT_VERSIONS,
    CONTEXT_V7_PROMPT_VERSION,
}
JOSEPH_PROMPT_VERSIONS = (
    JOSEPH_V1_PROMPT_VERSION,
    JOSEPH_PROMPT_VERSION,
    CONTEXT_V4_PROMPT_VERSION,
    CONTEXT_V5_PROMPT_VERSION,
    CONTEXT_V6_PROMPT_VERSION,
    CONTEXT_V7_PROMPT_VERSION,
    CONTEXT_V8_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
    TDC_SKIN_LLNA_PROMPT_VERSION,
    TDC_SKIN_LLNA_SCORES_PROMPT_VERSION,
    TDC_V2_MIXED_PROMPT_VERSION,
    DILI_TDC_V2_MIXED_V3_PROMPT_VERSION,
    DILI_TDC_V2_POLARIZED_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION,
    FINAL_FULL_FLAT_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION, FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
)
# Backward-compatible default for direct prompt-library callers. The public
# The unified branches CLI selects Joseph flat behavior below.
PROMPT_VERSION = TIANANG_PROMPT_VERSION
PUBLIC_HARNESS_VERSION = "final-full-flat"
CONTEXT_V4_HARNESS_VERSION = "joseph-flat-context-v4-v1"
CONTEXT_V5_HARNESS_VERSION = "full-flat-context-v5"
CONTEXT_V6_HARNESS_VERSION = "full-flat-context-v6-oral-high-low"
CONTEXT_V7_HARNESS_VERSION = "full-flat-context-v7-oral-final-only"
CONTEXT_V8_HARNESS_VERSION = "full-flat-context-v8-context-scoped-transfer-v1"
CONTEXT_V5_SIX_TASKS_HARNESS_VERSION = "full-flat-context-v5-six-tasks-v1"
CONTEXT_V5_SIX_TASKS_UPSTREAM_HARNESS_VERSION = "full-flat-context-v5-six-tasks-upstream-v1"
CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_HARNESS_VERSION = "full-flat-context-v5-six-tasks-upstream-v2"
CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_HARNESS_VERSION = "full-flat-context-v5-six-tasks-upstream-v3"
TDC_SKIN_LLNA_HARNESS_VERSION = "full-flat-context-v5-six-tasks-tdc-skin-llna-v1"
TDC_SKIN_LLNA_SCORES_HARNESS_VERSION = "full-flat-context-v5-six-tasks-tdc-skin-llna-scores-v2"
TDC_V2_MIXED_HARNESS_VERSION = "full-flat-context-v5-six-tasks-tdc-v2-mixed-v1"
DILI_TDC_V2_MIXED_V3_HARNESS_VERSION = "full-flat-context-v5-six-tasks-dili-tdc-v2-mixed-v3"
DILI_TDC_V2_POLARIZED_HARNESS_VERSION = "full-flat-context-v5-six-tasks-dili-tdc-v2-polarized-v1"
CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_HARNESS_VERSION = "full-flat-context-v5-six-tasks-upstream-scores-v2"
FINAL_FULL_FLAT_HARNESS_VERSION = "final-full-flat"
FINAL_FULL_FLAT_TDC_CARC_HARNESS_VERSION = "final-full-flat-tdc-carc-direct-priority-v1"
FINAL_FULL_FLAT_TDC_CARC_STRONG_HARNESS_VERSION = "final-full-flat-tdc-carc-strong-direct-v1"
FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_HARNESS_VERSION = "final-full-flat-tdc-carc-direct-positive-gate-v1"
JOSEPH_V1_HARNESS_VERSION = "joseph-flat-v1"
LEGACY_HARNESS_VERSION = "tianang-flat-v1"


def require_current_cache_bundle(path: Path) -> None:
    """Keep numeric recipes and historical formats out of active prediction."""
    from predict.retrieval.assay_reranking.artifact_bundle import CACHE_SET_SCHEMA
    if yaml.safe_load(path.read_text()).get("schema_version") != CACHE_SET_SCHEMA:
        raise ValueError("Use a named retrieval cache set with --retrieval-cache; numbered configurations require historical replay")


JOSEPH_HARNESS_PROMPTS = {
    JOSEPH_V1_HARNESS_VERSION: JOSEPH_V1_PROMPT_VERSION,
    "joseph-flat-v2": JOSEPH_PROMPT_VERSION,
    CONTEXT_V4_HARNESS_VERSION: CONTEXT_V4_PROMPT_VERSION,
    CONTEXT_V5_HARNESS_VERSION: CONTEXT_V5_PROMPT_VERSION,
    CONTEXT_V6_HARNESS_VERSION: CONTEXT_V6_PROMPT_VERSION,
    CONTEXT_V7_HARNESS_VERSION: CONTEXT_V7_PROMPT_VERSION,
    CONTEXT_V8_HARNESS_VERSION: CONTEXT_V8_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_HARNESS_VERSION: CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_HARNESS_VERSION: CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_HARNESS_VERSION: CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_HARNESS_VERSION: CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
    TDC_SKIN_LLNA_HARNESS_VERSION: TDC_SKIN_LLNA_PROMPT_VERSION,
    TDC_SKIN_LLNA_SCORES_HARNESS_VERSION: TDC_SKIN_LLNA_SCORES_PROMPT_VERSION,
    TDC_V2_MIXED_HARNESS_VERSION: TDC_V2_MIXED_PROMPT_VERSION,
    DILI_TDC_V2_MIXED_V3_HARNESS_VERSION: DILI_TDC_V2_MIXED_V3_PROMPT_VERSION,
    DILI_TDC_V2_POLARIZED_HARNESS_VERSION: DILI_TDC_V2_POLARIZED_PROMPT_VERSION,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_HARNESS_VERSION: CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION,
    FINAL_FULL_FLAT_HARNESS_VERSION: FINAL_FULL_FLAT_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_HARNESS_VERSION: FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_STRONG_HARNESS_VERSION: FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION,
    FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_HARNESS_VERSION: FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
}
JOSEPH_PROMPT_HARNESSES = {
    prompt: harness for harness, prompt in JOSEPH_HARNESS_PROMPTS.items()
}
ACTIVE_HARNESS_PROMPTS = {
    harness: prompt for harness, prompt in JOSEPH_HARNESS_PROMPTS.items()
    if prompt in {
        FINAL_FULL_FLAT_PROMPT_VERSION,
        FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION,
        FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
        TDC_V2_MIXED_PROMPT_VERSION,
        DILI_TDC_V2_MIXED_V3_PROMPT_VERSION,
        DILI_TDC_V2_POLARIZED_PROMPT_VERSION,
    }
}
PROMPT_DIR = PROMPT_ROOT
MORGAN_VARIANT = "morgan"
ASSAY_TRANSFER_VARIANT = "assay-transfer"
JOINT_VARIANT = "joint"
VARIANTS = (MORGAN_VARIANT, ASSAY_TRANSFER_VARIANT, JOINT_VARIANT)
CONTEXT_V4_VARIANTS = (
    MORGAN_VARIANT,
    "morgan-contrastive",
    ASSAY_TRANSFER_VARIANT,
    "assay-transfer-contrastive",  # Frozen prompt metadata; selection is retired.
    "assay-transfer-within-morgan",
)
CONTEXT_V4_LAYOUTS = ("global", "level-grouped")
CONTEXT_V4_SELECTION_CONTRACT = "joseph_flat_context_retrieval.v1"
CONTEXT_V5_SELECTION_CONTRACT = "joseph_flat_context_retrieval.v2"
PRESELECTED_UID_SCHEMA = "flat_preselected_uids.v1"
PRESELECTED_MIXED_SCHEMA = "gold_mixed_selection.v1"
TDC_V2_MIXED_SCHEMA = "tdc_mixed_selection"
DILI_TDC_V2_MIXED_SCHEMA = "tdc_dili_geomean_mixed_selection"
TDC_V2_BBB_FILTERED_SCHEMA = "tdc_bbb_filtered_selection"
PRESELECTED_DIRECT_SCHEMA = "gold_direct_context_selection.v1"
PRESELECTED_DIRECT_SCHEMAS = {
    PRESELECTED_DIRECT_SCHEMA,
    "direct_context_selection.v2",
}
CONTEXT_PROMPT_VERSIONS = {
    CONTEXT_V4_PROMPT_VERSION,
    *CONTEXT_NUMBERED_PROMPT_VERSIONS,
}
DEFAULT_QUERY_PRIOR_ROOT = Path(
    "outputs/paper/legacy/"
    "starling_conditioned_gold_l1_deepseek_v4_flash_nvfp4_query_prior/"
    "runs_deployment_visible_parent_disjoint"
)
QUERY_PRIOR_CURRENT = Path(
    "outputs/paper/assay_transfer_harness/joseph/query_priors/CURRENT.json"
)


def default_query_prior_root(prompt_version: str, benchmark: str, subset: str) -> Path:
    """Choose the immutable upstream Gold prior batch for its matching prompt and split."""
    if benchmark == "gold" and prompt_version in {
        CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
        TDC_V2_MIXED_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION,
        FINAL_FULL_FLAT_PROMPT_VERSION,
        FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION,
        FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION, FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
    }:
        current = json.loads(QUERY_PRIOR_CURRENT.read_text(encoding="utf-8"))
        if current.get("schema_version") != "query_prior_current.v1":
            raise ValueError(f"Invalid query-prior CURRENT: {QUERY_PRIOR_CURRENT}")
        entry = current["gold_v1"][subset]
        root = Path(entry["batch_root"])
        execution = canonical_location(root / "execution.json")
        if sha256_file(execution) != entry["execution_sha256"]:
            raise ValueError(f"Query-prior CURRENT hash mismatch: {execution}")
        details = json.loads(execution.read_text(encoding="utf-8"))
        if (details.get("status"), details.get("subset"), details.get("prompt_version")) != (
            "complete", subset, CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION
        ):
            raise ValueError(f"Query-prior CURRENT contract mismatch: {execution}")
        return root
    return DEFAULT_QUERY_PRIOR_ROOT


MOLECULE_DESCRIPTION_ARTIFACTS = {
    "v1": (
        Path("canonical_smiles_quotient.parquet"),
        "abd6f6d31ee74d854fd42330e816516c39a5ca63c67df56965cc8ec44468f183",
    ),
    "v2": (
        Path("canonical_smiles_quotient_v2.parquet"),
        "3e1ca4ece0f137b492c4d6c60713cf9771fbab7871a11b4c6c126df674b0abb7",
    ),
}
MOLECULE_DESCRIPTION_COLUMNS = {
    "raw": "description_raw",
    "motif": "description_motif",
    "coarse": "description_coarse",
}
PROMPT_VARIANTS = {
    TIANANG_PROMPT_VERSION: (MORGAN_VARIANT, ASSAY_TRANSFER_VARIANT),
    JOSEPH_V1_PROMPT_VERSION: VARIANTS,
    JOSEPH_PROMPT_VERSION: VARIANTS,
    CONTEXT_V4_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V5_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V6_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V7_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V8_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V5_SIX_TASKS_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    TDC_SKIN_LLNA_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    TDC_SKIN_LLNA_SCORES_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    TDC_V2_MIXED_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    DILI_TDC_V2_MIXED_V3_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    DILI_TDC_V2_POLARIZED_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    FINAL_FULL_FLAT_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
    FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION: CONTEXT_V4_VARIANTS,
}
TASKS = {
    "bbb_martins": 5,
    "bioavailability_ma": 6,
    "skin_reaction": 3,
    "ames": 5,
    "dili": 2,
    "carcinogens": 2,
}


RECORD_POOLS = {
    "assay-transfer-trained": "tool-accepted",
    "all_transfer_eligible": "tool-compatible",
    "all": "all",
}
SQLITE_SELECTION_CONTRACTS = {
    "cache_matched_retrieval.v2": "cache_matched_v2.py",
    "cache_matched_retrieval.v3": "cache_matched_v3.py",
    "ranked_evidence_retrieval.v1": "ranked_retrieval.py",
    "ranked_level_retrieval.v2": "ranked_level_retrieval.py",
    "ranked_uid_retrieval.v1": "ranked_uid_retrieval.py",
}
EVIDENCE_PROJECTION = "source_contract_complete_semantics.v1"
EXTRA_DETAILS_POLICY = "raw_nonempty_visible"
MOLECULE_NAME_FIELDS = {
    "compound_aliases",
    "compound_name",
    "compound_names",
    "drug_name",
    "molecule_aliases",
    "molecule_name",
    "molecule_names",
    "preferred_name",
    "source_molecule_name",
    "source_molecule_names",
}
NONSEMANTIC_SOURCE_FIELDS = {
    "canonical_record_id",
    "canonical_smiles",
    "confidence",
    "doi",
    "extraction_id",
    "global_identifier",
    "needs_more_context",
    "paragraph_idx",
    "pmid",
    "record_id",
    "smiles",
    "source_doi",
    "source_id",
    "source_index",
    "source_name",
    "source_record_id",
    "source_row_number",
    "source_row_uid",
    "source_smiles",
    "source_url",
}
CORE_FIELD_SOURCES = {
    "assay_context": (
        "assay_context",
        "assay_description",
        "assay_model",
        "assay_system",
    ),
    "endpoint": (
        "endpoint_name",
        "standard_type",
        "measurement_type",
        "endpoint",
        "endpoint_type",
        "parameter_name",
        "metric_type",
        "bioavailability_report_type",
    ),
    "reported_value": (
        "measurement_text",
        "standard_value",
        "reported_value",
        "parameter_value",
        "quant_value",
        "value",
    ),
    "reported_unit": (
        "unit_text",
        "standard_units",
        "reported_units",
        "parameter_units",
        "quant_units",
        "unit",
    ),
    "species": ("species", "organism", "species_or_population"),
    "qualifying_conditions": (
        "qualifying_conditions",
        "experimental_conditions",
    ),
    "support_text": ("support_text", "description"),
}
FLAT_V2_CARD_FIELDS = (
    "card_id",
    "assay_context",
    "endpoint",
    "reported_value",
    "reported_unit",
    "species",
    "qualifying_conditions",
    "experimental_details",
    "extra_details",
    "support_text",
    "transfer_likelihood",
)


def _source_semantic_prompt(version: str) -> bool:
    return version in {JOSEPH_PROMPT_VERSION, *CONTEXT_PROMPT_VERSIONS}
FLAT_V2_MOLECULE_SCORE_FIELDS = (
    "morgan_similarity",
    "transfer_likelihood",
    "morgan_top5_rank",
    "assay_transfer_top5_rank",
)


def prompt_directory(version: str) -> Path:
    """Resolve one explicit flat prompt bundle without fallback."""
    if Path(version).name != version or version not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown flat prompt version: {version!r}")
    directory = bundle_directory(version)
    if not directory.is_dir():
        raise ValueError(f"Flat prompt bundle is missing: {directory}")
    return directory


@lru_cache(maxsize=None)
def prompt_assets(version: str = PROMPT_VERSION) -> dict[str, Any]:
    """Load and validate one version-owned flat group prompt contract."""
    directory = prompt_directory(version)
    tasks = flat_task_profiles(version)
    modes = {} if version in CONTEXT_NUMBERED_PROMPT_VERSIONS else {
        mode: yaml.safe_load(
            (directory / "modes" / mode / "user.yaml").read_text(encoding="utf-8")
        )
        for mode in PROMPT_VARIANTS[version]
    }
    provenance = json.loads(
        (directory / "provenance.json").read_text(encoding="utf-8")
    )
    levels = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "levels").glob("L*.yaml"))
    } if (directory / "levels").is_dir() else {}
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"Flat prompt tasks are invalid: {directory / 'tasks.yaml'}")
    if any(
        not isinstance(mode, dict) or not mode.get("guidance")
        for mode in modes.values()
    ):
        raise ValueError(f"Flat prompt modes are invalid: {directory / 'modes'}")
    if provenance.get("version") != version or set(tasks) != set(
        provenance.get("supports_tasks") or []
    ):
        raise ValueError(f"Flat prompt provenance does not match {directory}")
    for task_id, contract in tasks.items():
        source = (provenance.get("tasks") or {}).get(task_id) or {}
        if source.get("prompt_profile") != contract.get("prompt_profile"):
            raise ValueError(f"Flat prompt profile provenance mismatch: {task_id}")
    return {
        "tasks": tasks,
        "modes": modes,
        "provenance": provenance,
        "levels": levels,
    }


def prompt_asset_manifest(version: str = PROMPT_VERSION) -> dict[str, Any]:
    """Fingerprint the complete prompt bundle and its assembly code."""
    prompt_assets(version)
    directory = prompt_directory(version)
    hashes = {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.suffix in {".jinja", ".json", ".yaml"}
    }
    if not (directory / "tasks.yaml").is_file():
        hashes.update({f"tasks/{task}/profiles.yaml": _sha256(PROMPT_ROOT / "tasks" / task / "profiles.yaml") for task in TASKS})
    assembly = {"flat.py": _sha256(Path(__file__)), "prompts/__init__.py": _sha256(PROMPT_ROOT / "__init__.py")}
    digest = hashlib.sha256(
        json.dumps(
            {"files_sha256": hashes, "assembly_files_sha256": assembly},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "version": version,
        "directory": str(directory),
        "files_sha256": hashes,
        "assembly_files_sha256": assembly,
        "sha256": digest,
    }


def flat_prompt_variant(
    retrieval_strategy: str,
    *,
    prompt_version: str = PROMPT_VERSION,
    flat_reranking: str = "",
) -> str:
    """Resolve the prompt mode from the legacy or cache-matched CLI."""
    if prompt_version in JOSEPH_PROMPT_VERSIONS:
        if flat_reranking not in PROMPT_VARIANTS[prompt_version]:
            raise ValueError(
                f"{prompt_version} does not support reranking {flat_reranking!r}"
            )
        return flat_reranking
    if retrieval_strategy == "morgan_fingerprint":
        return MORGAN_VARIANT
    if retrieval_strategy == ASSAY_TRANSFER_TOOL_STRATEGY:
        return ASSAY_TRANSFER_VARIANT
    raise ValueError(
        "tianang_flat_v1 requires morgan_fingerprint or assay_transfer_tool retrieval"
    )


def apply_flat_group_prompt(
    payload: Mapping[str, Any],
    *,
    task_id: str,
    task_prompt_profile: str,
    include_query_tool_guidance: bool,
    prompt_version: str = PROMPT_VERSION,
    retrieval_strategy: str,
    flat_reranking: str = "",
) -> dict[str, Any]:
    """Apply a frozen task contract and retrieval surface to a payload copy."""
    contract = _task_contract(task_id, task_prompt_profile, prompt_version)
    output = deepcopy(dict(payload))
    group = output.get("group") or {}
    if group.get("group_id") != "Flat.all_evidence":
        raise ValueError("tianang_flat_v1 requires the Flat.all_evidence group")

    variant = flat_prompt_variant(
        retrieval_strategy,
        prompt_version=prompt_version,
        flat_reranking=flat_reranking,
    )
    output["task"] = str(contract["task"])
    output["required_json_schema"] = deepcopy(contract["required_json_schema"])
    instruction_key = (
        "with_comparisons" if include_query_tool_guidance else "without_comparisons"
    )
    instructions: list[str] = []
    for entry in contract["instructions"]:
        if isinstance(entry, str):
            instructions.append(entry)
        elif isinstance(entry, dict) and entry.get(instruction_key):
            selected = entry[instruction_key]
            instructions.extend(
                [str(line) for line in selected]
                if isinstance(selected, list)
                else [str(selected)]
            )
    if prompt_version == JOSEPH_PROMPT_VERSION:
        if output.get("prompt_transport"):
            raise ValueError(
                "joseph_flat_v2 refuses prompt transport truncation; "
                "all selected semantic fields must remain visible"
            )
        output = {
            "task": str(contract["task"]),
            "query": _query_without_tools(output.get("query") or {}),
            "molecules": _flat_v2_prompt_molecules(output.get("neighbors") or []),
            "instructions": instructions,
            "required_json_schema": deepcopy(contract["required_json_schema"]),
        }
    elif prompt_version == TIANANG_PROMPT_VERSION and variant == MORGAN_VARIANT:
        _remove_keys(output, _is_assay_transfer_key)
    elif prompt_version == TIANANG_PROMPT_VERSION:
        if isinstance(group, dict):
            group.pop("assay_transfer_score_policy", None)
        for neighbor in output.get("neighbors") or []:
            if isinstance(neighbor, dict):
                neighbor.pop("similarity", None)
                neighbor.pop("similarity_bucket", None)
                neighbor.pop("similarity_metric", None)
    elif prompt_version == JOSEPH_V1_PROMPT_VERSION:
        # Cache-matched selection records the method and its applicable scores
        # per physical evidence row. An outer molecule score would be ambiguous
        # after several levels are merged into one molecule card.
        for neighbor in output.get("neighbors") or []:
            if isinstance(neighbor, dict):
                neighbor.pop("similarity", None)
                neighbor.pop("similarity_bucket", None)
                neighbor.pop("similarity_metric", None)
    if prompt_version not in CONTEXT_NUMBERED_PROMPT_VERSIONS:
        instructions.append(str(prompt_assets(prompt_version)["modes"][variant]["guidance"]))
    output["instructions"] = instructions
    return output


def render_flat_group_messages(
    payload: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
    task_id: str,
    task_prompt_profile: str,
    group_tools_enabled: bool,
    prompt_version: str = PROMPT_VERSION,
    retrieval_strategy: str,
    flat_reranking: str = "",
) -> list[dict[str, str]]:
    """Render the version-owned flat group system and JSON user messages."""
    if prompt_version == JOSEPH_PROMPT_VERSION and group_tools_enabled:
        raise ValueError("joseph_flat_v2 does not expose group comparison tools")
    include_comparisons = bool(group.get("tools_prefetched")) or group_tools_enabled
    contract = _task_contract(task_id, task_prompt_profile, prompt_version)
    system = _environment().get_template(
        f"{prompt_version}/system.jinja"
    ).render(
        system_role=str(contract["system_role"]).strip(),
        tools_prefetched=bool(group.get("tools_prefetched")),
        group_tools_enabled=group_tools_enabled,
        identity_blind=bool(group.get("identity_blind")),
    )
    user = apply_flat_group_prompt(
        payload,
        task_id=task_id,
        task_prompt_profile=task_prompt_profile,
        include_query_tool_guidance=include_comparisons,
        prompt_version=prompt_version,
        retrieval_strategy=retrieval_strategy,
        flat_reranking=flat_reranking,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def flat_group_validation(
    task_id: str,
    *,
    task_prompt_profile: str,
    prompt_version: str = PROMPT_VERSION,
    group: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the frozen response fields required by one task bundle."""
    contract = _task_contract(task_id, task_prompt_profile, prompt_version)
    validation: dict[str, Any] = {
        "required_fields": tuple(contract["required_fields"])
    }
    if prompt_version == JOSEPH_PROMPT_VERSION and group is not None:
        card_ids = {
            str((row.get("prompt_evidence") or {}).get("card_id") or "")
            for neighbor in group.get("neighbors") or []
            for row in neighbor.get("evidence_rows") or []
        }
        card_ids.discard("")
        validation.update(
            forbidden_field_names=("key_evidence", "tool_summary"),
            content_validator=lambda content: _claim_errors(content, card_ids),
        )
    return validation


def flat_prompt_provenance(
    task_id: str,
    *,
    retrieval_strategy: str,
    task_prompt_profile: str,
    prompt_version: str = PROMPT_VERSION,
    flat_reranking: str = "",
) -> dict[str, Any]:
    """Describe and fingerprint one frozen task and retrieval specialization."""
    variant = flat_prompt_variant(
        retrieval_strategy,
        prompt_version=prompt_version,
        flat_reranking=flat_reranking,
    )
    _task_contract(task_id, task_prompt_profile, prompt_version)
    assets = prompt_assets(prompt_version)
    manifest = prompt_asset_manifest(prompt_version)
    upstream = assets["provenance"]
    contract = {
        "prompt_version": prompt_version,
        "variant": variant,
        "task_id": task_id,
        "task_prompt_profile": task_prompt_profile,
        "upstream_commit": upstream.get("upstream_commit", ""),
        "upstream_sources": upstream["tasks"][task_id].get("upstream_sources", []),
        "asset_manifest_sha256": manifest["sha256"],
        "asset_files_sha256": manifest["files_sha256"],
        "assembly_files_sha256": manifest["assembly_files_sha256"],
        "base_prompt_contract": "task_standard_group",
        "flat_group_id": "Flat.all_evidence",
    }
    if prompt_version not in CONTEXT_NUMBERED_PROMPT_VERSIONS:
        contract["retrieval_guidance"] = assets["modes"][variant]["guidance"]
    if _source_semantic_prompt(prompt_version):
        contract.update(
            evidence_projection=EVIDENCE_PROJECTION,
            extra_details_policy=EXTRA_DETAILS_POLICY,
            molecule_name_visible=False,
            group_tools_enabled=False,
        )
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return contract


def _task_contract(
    task_id: str,
    task_prompt_profile: str,
    prompt_version: str,
) -> Mapping[str, Any]:
    tasks = prompt_assets(prompt_version)["tasks"]
    try:
        contract = tasks[task_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported {prompt_version} task: {task_id}") from exc
    expected_profile = str(contract.get("prompt_profile") or "")
    if task_prompt_profile != expected_profile:
        raise ValueError(
            f"{prompt_version} requires task prompt profile {expected_profile!r} "
            f"for {task_id}, got {task_prompt_profile!r}"
        )
    return contract


@lru_cache(maxsize=1)
def _environment() -> Environment:
    return template_environment()


def _compact_prompt(text: str) -> str:
    text = re.sub(r"(?m)(^- .*)\n[ \t]*\n(?=- )", r"\1\n", text)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text).strip()


def _is_assay_transfer_key(key: str) -> bool:
    normalized = key.lower()
    return "assay_transfer" in normalized or "assay-transfer" in normalized


def _remove_keys(value: Any, predicate: Callable[[str], bool]) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if predicate(str(key)):
                value.pop(key)
            else:
                _remove_keys(value[key], predicate)
    elif isinstance(value, list):
        for item in value:
            _remove_keys(item, predicate)


def _query_without_tools(query: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(query))
    output.pop("prefetched_molecule_properties", None)
    output.pop("tools_prefetched", None)
    return output


def _flat_v2_prompt_molecules(
    neighbors: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    molecules: list[dict[str, Any]] = []
    seen_cards: set[str] = set()
    for neighbor in neighbors:
        scores: dict[str, Any] = {}
        cards: list[dict[str, Any]] = []
        for raw_card in neighbor.get("evidence_rows") or []:
            if not isinstance(raw_card, Mapping):
                raise ValueError("joseph_flat_v2 evidence card is not an object")
            source = deepcopy(dict(raw_card))
            for name, value in dict(source.pop("_molecule_scores", {})).items():
                _set_consistent(scores, name, value)
            card = {
                name: source[name]
                for name in FLAT_V2_CARD_FIELDS
                if name in source and _is_nonempty(source[name])
            }
            card_id = str(card.get("card_id") or "")
            if not card_id or card_id in seen_cards:
                raise ValueError(
                    f"joseph_flat_v2 card ID is missing or duplicated: {card_id!r}"
                )
            seen_cards.add(card_id)
            cards.append(card)
        if not cards:
            continue
        molecule = {
            "canonical_smiles": str(neighbor.get("canonical_smiles") or ""),
            **{
                name: scores[name]
                for name in FLAT_V2_MOLECULE_SCORE_FIELDS
                if name in scores
            },
            "evidence_cards": cards,
        }
        molecules.append(molecule)
    return molecules


def build_flat_context_request(
    retrieval: Mapping[str, Any],
    *,
    task_id: str,
    task_prompt_profile: str,
    layout: str,
    reranking: str,
    query_prior: Mapping[str, Any] | None,
    prompt_version: str = CONTEXT_V4_PROMPT_VERSION,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Render one all-level prediction request and its exact visible-ID index."""
    if layout not in CONTEXT_V4_LAYOUTS:
        raise ValueError(f"Unknown flat context layout: {layout!r}")
    if reranking not in CONTEXT_V4_VARIANTS:
        raise ValueError(f"Unsupported flat context reranking: {reranking!r}")
    groups = retrieval.get("groups") or []
    if len(groups) != 1 or groups[0].get("group_id") != "Flat.all_evidence":
        raise ValueError("Flat context prediction requires one Flat.all_evidence group")
    if prompt_version not in CONTEXT_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported flat context prompt: {prompt_version!r}")
    contract = _task_contract(task_id, task_prompt_profile, prompt_version)
    assets = prompt_assets(prompt_version)
    if prompt_version in FINAL_FULL_FLAT_PROMPT_VERSIONS:
        benchmark = retrieval.get("benchmark")
        if benchmark not in {"gold_v1", "tdc"}:
            raise ValueError("final_full_flat requires an explicit Gold-v1 or TDC benchmark")
        if benchmark == "tdc" and (contract.get("tdc_task_instructions") or contract.get("tdc_v2_task_instructions")):
            contract = {**contract, "task_instructions": (contract.get("tdc_task_instructions") or contract["tdc_v2_task_instructions"])}
    if (prompt_version == CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION
            and retrieval.get("tdc_score_visibility") in {"high", "middle", "low"}
            and task_id == "bbb_martins"):
        contract = {**contract, "task_instructions":
                    (contract.get("tdc_task_instructions") or contract["tdc_v2_task_instructions"])}
    sections, aliases, references = _flat_context_sections(
        groups[0], layout=layout, prompt_version=prompt_version
    )
    if task_id == "skin_reaction" and prompt_version == CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION:
        # Preserve the Skin request surface used by the original V3 evaluation.
        for section in sections:
            for molecule in section["molecules"]:
                molecule.pop("morgan_similarity", None)
    for section in sections:
        level = str(section.get("level") or "")
        level_contract = ((assets.get("levels") or {}).get(level) or {}).get(task_id, {})
        section["description"] = str(
            level_contract.get("tdc_description", level_contract.get("tdc_v2_description", level_contract.get("description", "")))
            if prompt_version in FINAL_FULL_FLAT_PROMPT_VERSIONS
            and retrieval.get("benchmark") == "tdc"
            else level_contract.get("description", "")
        )
    query = deepcopy(dict(retrieval.get("query") or {}))
    query.pop("fingerprint", None)
    payload = {
        "task_id": task_id,
        "query": query,
        "query_prior": deepcopy(dict(query_prior or {})),
        "sections": sections,
        "required_json_schema": _flat_context_schema(contract),
    }
    system_arguments = {
        **contract,
        "mode": (
            "morgan" if reranking.startswith("morgan")
            or (retrieval.get("score_visibility") or retrieval.get("tdc_score_visibility")) in {"low", "middle"}
            else reranking
        ),
        "query_prior_visible": bool(query_prior),
    }
    if prompt_version not in CONTEXT_NUMBERED_PROMPT_VERSIONS:
        system_arguments["ranking_guidance"] = assets["modes"][reranking]["guidance"]
    system = _environment().get_template(
        f"{prompt_version}/system.jinja"
    ).render(**system_arguments)
    user = _environment().get_template(
        f"{prompt_version}/user.jinja"
    ).render(**payload)
    if prompt_version in {
        CONTEXT_V5_SIX_TASKS_UPSTREAM_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
        TDC_SKIN_LLNA_PROMPT_VERSION,
        TDC_SKIN_LLNA_SCORES_PROMPT_VERSION,
        TDC_V2_MIXED_PROMPT_VERSION,
        DILI_TDC_V2_MIXED_V3_PROMPT_VERSION,
        DILI_TDC_V2_POLARIZED_PROMPT_VERSION,
        CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION,
        FINAL_FULL_FLAT_PROMPT_VERSION,
        FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION,
        FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION, FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
    }:
        system, user = _compact_prompt(system), _compact_prompt(user)
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ], {
        "card_alias_map": aliases,
        "reasoning_reference_contract": deepcopy(
            assets["provenance"]["reasoning_reference_contracts"][layout]
        ),
        "reasoning_reference_index": references,
    }


def flat_context_validation(
    task_id: str, *, task_prompt_profile: str,
    prompt_version: str = CONTEXT_V4_PROMPT_VERSION,
    reference_index: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate the shared two-field flat prediction."""
    contract = _task_contract(
        task_id, task_prompt_profile, prompt_version
    )
    prediction_field = str(contract["prediction_field"])
    expected_fields = set(contract["required_fields"])

    visible = {
        str(row["visible_id"]): str(row["unit_kind"])
        for row in (reference_index or [])
    }

    def content_errors(content: Mapping[str, Any]) -> list[str]:
        if set(content) != expected_fields:
            return ["invalid_output_fields"]
        if prompt_version not in CONTEXT_CLAIMS_PROMPT_VERSIONS:
            return []
        errors: list[str] = []
        claims = content.get("claims")
        if not isinstance(claims, list) or not claims:
            return ["claims_must_be_nonempty_array"]
        for index, claim in enumerate(claims):
            if not isinstance(claim, Mapping) or set(claim) != {
                "claim", "molecule_ids", "record_ids", "evidence_role"
            }:
                errors.append(f"invalid_claim_fields:{index}")
                continue
            if not isinstance(claim["claim"], str) or not claim["claim"].strip():
                errors.append(f"invalid_claim_text:{index}")
            role = claim["evidence_role"]
            if role not in {"supportive", "contradictory"}:
                errors.append(f"invalid_evidence_role:{index}")
            cited: list[str] = []
            for name, kind in (("molecule_ids", "molecule"), ("record_ids", "record")):
                values = claim[name]
                if not isinstance(values, list) or any(
                    not isinstance(value, str) for value in values
                ):
                    errors.append(f"invalid_identifier_array:{index}:{name}")
                    continue
                if len(values) != len(set(values)):
                    errors.append(f"duplicate_identifiers:{index}:{name}")
                invalid = sorted(value for value in values if visible.get(value) != kind)
                if invalid:
                    errors.append(
                        f"unknown_identifiers:{index}:{name}:{','.join(invalid)}"
                    )
                cited.extend(values)
            if not cited:
                errors.append(f"claim_without_evidence:{index}")
        return errors

    return {
        "required_fields": tuple(contract["required_fields"]),
        "allowed_values": {
            prediction_field: {
                str(contract["positive_prediction"]),
                str(contract["negative_prediction"]),
            },
        },
        "content_validator": content_errors,
    }


def derive_flat_claim_evidence(content: Mapping[str, Any]) -> dict[str, list[str]]:
    """Derive diagnostic role inventories from a validated V5 response."""
    output = {
        "supportive_molecule_ids": [],
        "supportive_record_ids": [],
        "contradictory_molecule_ids": [],
        "contradictory_record_ids": [],
    }
    for claim in content.get("claims") or []:
        role = str(claim["evidence_role"])
        for source, suffix in (("molecule_ids", "molecule_ids"), ("record_ids", "record_ids")):
            target = f"{role}_{suffix}"
            output[target].extend(
                value for value in claim[source] if value not in output[target]
            )
    return output


def native_flat_prediction(
    task_id: str, prediction: str, *, prompt_version: str
) -> tuple[str, str]:
    """Map a portable prompt label to the task's benchmark label."""
    mapping = (prompt_assets(prompt_version)["provenance"].get("label_mapping") or {}).get(
        task_id
    )
    if not isinstance(mapping, dict) or prediction not in mapping:
        raise ValueError(
            f"No native prediction mapping for {prompt_version}/{task_id}/{prediction}"
        )
    digest = hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return str(mapping[prediction]), digest


def _flat_context_schema(contract: Mapping[str, Any]) -> dict[str, Any]:
    schema = deepcopy(dict(contract.get("required_json_schema") or {}))
    if "summary" in contract.get("required_fields", ()):
        schema["summary"] = "concise overall conclusion"
    schema[str(contract["prediction_field"])] = (
        f"{contract['positive_prediction']} | {contract['negative_prediction']}"
    )
    return schema


def _flat_context_sections(
    group: Mapping[str, Any], *, layout: str,
    prompt_version: str = CONTEXT_V4_PROMPT_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    neighbors = list(group.get("neighbors") or [])
    levels = sorted({
        str((row.get("selection_provenance") or {}).get("level") or "")
        for neighbor in neighbors for row in neighbor.get("evidence_rows") or []
    }, key=lambda value: int(value[1:]) if value.startswith("L") else 99)
    section_levels: list[str | None] = [None] if layout == "global" else levels
    sections: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    references: list[dict[str, Any]] = []
    molecule_number = 0
    for section_level in section_levels:
        molecules = []
        for neighbor in neighbors:
            rows = [
                row for row in neighbor.get("evidence_rows") or []
                if section_level is None
                or str((row.get("selection_provenance") or {}).get("level")) == section_level
            ]
            if not rows:
                continue
            molecule_number += 1
            parent_id = str(neighbor.get("standard_inchi_key") or neighbor.get("molecule_chembl_id") or "")
            stable_molecule_id = (
                parent_id if section_level is None else f"{parent_id}@{section_level}"
            )
            cards = []
            molecule_scores: dict[str, Any] = {}
            for record_number, row in enumerate(rows, 1):
                provenance = dict(row.get("selection_provenance") or {})
                card = deepcopy(dict(row.get("prompt_evidence") or {}))
                alias = (
                    f"Record {molecule_number}-{record_number}"
                    if prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS
                    else str(card.get("card_id") or "")
                )
                if prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS:
                    card["card_id"] = f"{molecule_number}-{record_number}"
                stable_card_id = str(row.get("evidence_id") or "")
                if not alias or alias in aliases or not stable_card_id:
                    raise ValueError(f"Missing or duplicate flat card alias: {alias!r}")
                aliases[alias] = stable_card_id
                card.update(
                    level=str(provenance.get("level") or ""),
                    evidence_family=str(provenance.get("evidence_family") or ""),
                )
                for name in ("selected_condition", "morgan_similarity", "assay_transfer_score"):
                    if provenance.get(name) not in (None, ""):
                        target = "transfer_likelihood" if name == "assay_transfer_score" else name
                        level = str(provenance.get("level") or "")
                        if (
                            prompt_version == CONTEXT_V8_PROMPT_VERSION
                            and name == "assay_transfer_score"
                        ):
                            card[target] = provenance[name]
                        elif prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS and name == "morgan_similarity":
                            _set_consistent(molecule_scores, target, provenance[name])
                        elif (
                            prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS
                            and name == "assay_transfer_score" and level == "L1"
                        ):
                            _set_consistent(molecule_scores, target, provenance[name])
                        elif not (
                            prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS
                            and prompt_version not in {*FINAL_FULL_FLAT_PROMPT_VERSIONS,
                                                       TDC_V2_MIXED_PROMPT_VERSION,
                                                       DILI_TDC_V2_MIXED_V3_PROMPT_VERSION,
                                                       DILI_TDC_V2_POLARIZED_PROMPT_VERSION}
                            and name == "assay_transfer_score" and level == "L5"
                        ):
                            card[target] = provenance[name]
                cards.append(card)
                references.append({
                    "unit_kind": "record",
                    "visible_id": alias,
                    "stable_id": stable_card_id,
                    "source_analog_id": parent_id,
                    "containing_unit_id": stable_molecule_id,
                    "first_visible_level": int(str(provenance.get("level") or "L0")[1:]),
                    "is_new": True,
                    "prompt_heading": (
                        alias if prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS
                        else f"Record {alias}"
                    ),
                })
            molecule = {
                "number": molecule_number,
                "canonical_smiles": str(neighbor.get("canonical_smiles") or ""),
                "evidence_cards": cards,
                **molecule_scores,
            }
            if (prompt_version in {
                CONTEXT_V5_SIX_TASKS_UPSTREAM_V2_PROMPT_VERSION,
                CONTEXT_V5_SIX_TASKS_UPSTREAM_V3_PROMPT_VERSION,
                FINAL_FULL_FLAT_PROMPT_VERSION,
                FINAL_FULL_FLAT_TDC_CARC_PROMPT_VERSION,
                FINAL_FULL_FLAT_TDC_CARC_STRONG_PROMPT_VERSION, FINAL_FULL_FLAT_TDC_CARC_POSITIVE_GATE_PROMPT_VERSION,
            } and molecule.get("morgan_similarity") is None):
                if not str(neighbor.get("similarity_metric") or "").startswith("Morgan "):
                    raise ValueError("Dual-score prompt requires Morgan-ranked neighbor similarity")
                molecule["morgan_similarity"] = round(float(neighbor["similarity"]), 4)
            if prompt_version in {
                CONTEXT_V5_SIX_TASKS_UPSTREAM_SCORES_PROMPT_VERSION,
                TDC_SKIN_LLNA_SCORES_PROMPT_VERSION,
            }:
                if not str(neighbor.get("similarity_metric") or "").startswith("Morgan "):
                    raise ValueError("Dual-score prompt requires Morgan-ranked neighbor similarity")
                molecule["morgan_similarity"] = round(float(neighbor["similarity"]), 4)
            if neighbor.get("molecule_description"):
                molecule["molecule_description"] = neighbor["molecule_description"]
            molecules.append(molecule)
            references.insert(len(references) - len(cards), {
                "unit_kind": "molecule",
                "visible_id": f"Molecule {molecule_number}",
                "stable_id": stable_molecule_id,
                "source_analog_id": parent_id,
                "containing_unit_id": "",
                "first_visible_level": min(
                    int(str((row.get("selection_provenance") or {}).get("level") or "L0")[1:])
                    for row in rows
                ),
                "is_new": True,
                "prompt_heading": f"## Molecule {molecule_number}",
            })
        sections.append({
            "heading": f"Level {section_level}" if section_level else "",
            "level": section_level or "",
            "molecules": molecules,
        })
    return sections, aliases, references


def _claim_errors(content: Mapping[str, Any], card_ids: set[str]) -> list[str]:
    claims = content.get("claims")
    if not isinstance(claims, list):
        return ["invalid_claims:not_a_list"]
    errors: list[str] = []
    for index, claim in enumerate(claims):
        if not isinstance(claim, Mapping):
            errors.append(f"invalid_claim:{index}:not_an_object")
            continue
        if set(claim) != {"claim", "card_ids"}:
            errors.append(f"invalid_claim:{index}:fields")
        if not str(claim.get("claim") or "").strip():
            errors.append(f"invalid_claim:{index}:empty_text")
        cited = claim.get("card_ids")
        if not isinstance(cited, list) or not cited:
            errors.append(f"invalid_claim:{index}:card_ids")
            continue
        if any(not isinstance(card_id, str) for card_id in cited):
            errors.append(f"invalid_claim:{index}:non_string_card_id")
            continue
        unknown = sorted(set(cited) - card_ids)
        if unknown:
            errors.append(f"unknown_card_ids:{index}:{','.join(unknown)}")
    return errors


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retrieve_flat_evidence(
    query_smiles: str,
    index: Mapping[str, Any],
    *,
    config: BranchRetrievalConfig,
    top_k_per_group: int,
    min_similarity: float,
    neighbor_identity_policy: str = NeighborIdentityPolicy.OPERATIONAL.value,
    neighbor_selector: str = SIMILARITY_SELECTOR,
    reranker: RetrievalReranker | None = None,
    assay_transfer_initial_morgan_filter: int = 100,
    assay_transfer_min_score: float | None = None,
    assay_transfer_diversity_mode: str = ASSAY_TRANSFER_DIVERSITY_NONE,
    assay_transfer_diversity_score_slack: float = 0.0,
    assay_transfer_selection_unit: str = ASSAY_TRANSFER_SELECTION_SCORED_RECORD,
    assay_transfer_records_per_molecule: int = ASSAY_TRANSFER_RECORDS_PER_MOLECULE_DEFAULT,
) -> dict[str, Any]:
    """Retrieve mechanism families, then collapse them into one group branch."""
    retrieval = retrieve_mechanism_evidence(
        query_smiles,
        index,
        config=config,
        mode="full_flat",
        top_k_per_group=top_k_per_group,
        min_similarity=min_similarity,
        neighbor_identity_policy=neighbor_identity_policy,
        neighbor_selector=neighbor_selector,
        reranker=reranker,
        assay_transfer_initial_morgan_filter=assay_transfer_initial_morgan_filter,
        assay_transfer_min_score=assay_transfer_min_score,
        assay_transfer_diversity_mode=assay_transfer_diversity_mode,
        assay_transfer_diversity_score_slack=assay_transfer_diversity_score_slack,
        assay_transfer_selection_unit=assay_transfer_selection_unit,
        assay_transfer_records_per_molecule=assay_transfer_records_per_molecule,
    )
    if retrieval.get("status") == "ok":
        retrieval["groups"] = [flatten_retrieval_groups(retrieval["groups"])]
        retrieval["coverage"] = retrieval_coverage(
            retrieval["groups"],
            min_similarity=min_similarity,
            top_k_per_group=top_k_per_group,
        )
    return retrieval


def cache_matched_flat_retrieval(
    query_id: str,
    query_smiles: str,
    molecules: list[dict[str, Any]],
    later_levels: Mapping[str, Mapping[str, Any]],
    *,
    task: str,
    reranking: str,
    query_audit: Mapping[str, Any],
    query_identity: Mapping[str, Any] | None = None,
    prompt_version: str = JOSEPH_V1_PROMPT_VERSION,
    selection_contract: str = "cache_matched_retrieval.v2",
    preserve_physical_records: bool = False,
    neighbor_identity_policy_by_level: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Flatten one cache-matched progressive selection without re-selecting it."""
    if query_identity:
        query_parent_smiles = str(query_identity.get("parent_smiles") or "")
        query_parent_id = str(query_identity.get("parent_id") or "")
        if not query_parent_smiles:
            raise ValueError(f"Incomplete cached query identity: {query_id}")
    else:
        identity = normalize_molecule_identity(query_smiles)
        if identity.status != "ok" or not identity.parent_smiles:
            raise ValueError(f"Unresolved query molecule: {query_id}")
        query_parent_smiles = identity.parent_smiles
        query_parent_id = identity.parent_inchi_key or ""

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for molecule in molecules:
        for row in molecule.get("l1_records") or []:
            selected.append((row, molecule))
    for level in sorted(later_levels, key=lambda value: int(value[1:])):
        for row in later_levels[level].get("records") or []:
            selected.append((row, {}))

    if task == "bbb_martins":
        excluded = prediction_only_uids({str(row["payload"].get("source_row_uid") or f"{index}:{row['record_id']}"): row
                                        for index, (row, _) in enumerate(selected)
                                        if str(row["payload"].get("progressive_level")).removeprefix("L") != "1"})
        if excluded:
            raise ValueError(f"Prediction-only BBB evidence is ineligible: {sorted(excluded)[:5]}")

    neighbors: dict[str, dict[str, Any]] = {}
    seen_records: set[tuple[str, str]] = set()
    level_identity_policy = dict(neighbor_identity_policy_by_level or {
        "L1": "scaffold_disjoint",
        **{level: "parent_disjoint" for level in later_levels},
    })
    for row, molecule in selected:
        record_id = str(row["record_id"])
        payload = dict(row["payload"])
        payload["progressive_level"] = "L" + str(payload["progressive_level"]).removeprefix("L")
        physical_uid = str(payload.get("source_row_uid") or "")
        if preserve_physical_records and not physical_uid:
            raise ValueError(f"Selected record lacks a physical source UID: {record_id}")
        dedup_id = physical_uid if preserve_physical_records else record_id
        context_id = ""
        if preserve_physical_records and payload["progressive_level"] == "L1":
            context_id = str(molecule.get("context_card_id") or molecule.get("_diagnostic_context_id") or "")
        member_key = (context_id, dedup_id)
        if member_key in seen_records:
            continue
        seen_records.add(member_key)
        parent_id = str(row["reference_molecule_id"])
        method = str(
            row.get("ranking_method") or molecule.get("ranking_method") or ""
        ).replace("_", "-")
        morgan_method = method in {MORGAN_VARIANT, "morgan-contrastive"}
        assay_method = method in {
            ASSAY_TRANSFER_VARIANT,
            "assay-transfer-contrastive",
            "assay-transfer-within-morgan",
        }
        if method == "assay-transfer-partial-hidden":
            morgan_method = True
        retrieval = {
            "record_id": record_id,
            "level": str(payload["progressive_level"]),
            "evidence_family": str(payload["family_key"]),
            "ranking_method": method,
        }
        if selection_contract in {
            "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
            CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
        }:
            retrieval["neighbor_identity_policy"] = level_identity_policy[
                retrieval["level"]
            ]
        if row.get("morgan_rank") is not None:
            retrieval["morgan_rank"] = int(row["morgan_rank"])
        if row.get("assay_rank") is not None:
            retrieval["assay_rank"] = int(row["assay_rank"])
        if morgan_method or method == JOINT_VARIANT or prompt_version in FINAL_FULL_FLAT_PROMPT_VERSIONS:
            retrieval["morgan_similarity"] = round(
                float(row.get("morgan_similarity", molecule.get("morgan_similarity"))), 4
            )
        transfer_score = row.get("transfer_likelihood", molecule.get("transfer_likelihood"))
        if prompt_version in FINAL_FULL_FLAT_PROMPT_VERSIONS and transfer_score is not None:
            retrieval["assay_transfer_score"] = round(float(transfer_score), 4)
        elif (assay_method or method == JOINT_VARIANT) and not (
            transfer_score is None and row.get("ranking_method") == "assay_transfer_partial_hidden"
        ):
            if transfer_score is None:
                raise ValueError(f"Selected assay-transfer record lacks a score: {record_id}")
            retrieval["assay_transfer_score"] = round(float(transfer_score), 4)
        if method == JOINT_VARIANT:
            retrieval["morgan_panel_rank"] = molecule.get("morgan_top5_rank")
            retrieval["assay_transfer_panel_rank"] = molecule.get(
                "assay_transfer_top5_rank"
            )
        if molecule.get("selected_condition"):
            retrieval["selected_condition"] = str(molecule["selected_condition"])
        for source, target in (
            ("_diagnostic_label", "diagnostic_label"),
            ("_diagnostic_context_id", "diagnostic_context_id"),
            ("_diagnostic_parent_id", "diagnostic_parent_id"),
        ):
            if molecule.get(source) is not None:
                retrieval[target] = molecule[source]

        neighbor_key = parent_id
        if str(payload["progressive_level"]) == "L1" and molecule.get("context_card_id"):
            neighbor_key = f"{parent_id}\0{molecule['context_card_id']}"
        neighbor = neighbors.setdefault(
            neighbor_key,
            {
                "rank": len(neighbors) + 1,
                "molecule_chembl_id": parent_id,
                "canonical_smiles": str(payload["canonical_smiles"]),
                "standard_inchi_key": parent_id,
                "similarity": round(float(row.get("morgan_similarity", molecule.get("morgan_similarity", 0.0))), 4),
                "similarity_bucket": _similarity_bucket(
                    float(row.get("morgan_similarity", molecule.get("morgan_similarity", 0.0)))
                ),
                "similarity_metric": "Morgan radius=2 bits=2048 Tanimoto",
                "molecule_relation": (
                    level_identity_policy[str(payload["progressive_level"])]
                    if selection_contract in {
                        "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
                        CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
                    }
                    else "scaffold_disjoint"
                ),
                "source_group_ids": [],
                "evidence_rows": [],
            },
        )
        if (
            selection_contract in {
                "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
                CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
            }
            and neighbor["molecule_relation"]
            != level_identity_policy[str(payload["progressive_level"])]
        ):
            neighbor["molecule_relation"] = "level_specific_disjoint"
        card_score: float | None = None
        if prompt_version == JOSEPH_PROMPT_VERSION:
            level = str(payload["progressive_level"])
            scores = neighbor.setdefault("_prompt_scores", {})
            if morgan_method or method == JOINT_VARIANT:
                _set_consistent(scores, "morgan_similarity", retrieval["morgan_similarity"])
            if level == "L1" and (assay_method or method == JOINT_VARIANT):
                _set_consistent(
                    scores,
                    "transfer_likelihood",
                    retrieval["assay_transfer_score"],
                )
            elif level != "L1" and (assay_method or method == JOINT_VARIANT):
                card_score = retrieval["assay_transfer_score"]
            if level == "L1" and method == JOINT_VARIANT:
                _set_consistent(
                    scores, "morgan_top5_rank", retrieval["morgan_panel_rank"]
                )
                _set_consistent(
                    scores,
                    "assay_transfer_top5_rank",
                    retrieval["assay_transfer_panel_rank"],
                )
        family = str(payload["family_key"])
        if family not in neighbor["source_group_ids"]:
            neighbor["source_group_ids"].append(family)
        if _source_semantic_prompt(prompt_version):
            card, accounting = _source_semantic_card(
                payload,
                card_id=f"C{len(seen_records):02d}",
                transfer_likelihood=card_score,
            )
            card["_molecule_scores"] = deepcopy(
                neighbor.get("_prompt_scores") or {}
            )
            neighbor["evidence_rows"].append(
                {
                    "evidence_id": dedup_id if preserve_physical_records else record_id,
                    "prompt_evidence": card,
                    "selection_provenance": deepcopy(retrieval),
                    "source_provenance": {
                        "source_id": str(payload["source_id"]),
                        "source_row_uid": str(payload.get("source_row_uid") or ""),
                        "measurement_kind": str(payload.get("measurement_kind") or ""),
                    },
                    "source_contract": deepcopy(payload["source_contract"]),
                    "source_fields": deepcopy(dict(payload.get("source_fields") or {})),
                    "field_accounting": accounting,
                }
            )
        else:
            neighbor["evidence_rows"].append(
                {
                    "evidence_id": dedup_id if preserve_physical_records else record_id,
                    "minimal_evidence": _minimal_selected_evidence(
                        payload, parent_id, retrieval
                    ),
                }
            )
        neighbor["n_evidence_rows"] = len(neighbor["evidence_rows"])

    group = {
        "group_id": "Flat.all_evidence",
        "tier": "Flat",
        "endpoint_group": "all_evidence",
        "source_group_ids": sorted(
            {family for neighbor in neighbors.values() for family in neighbor["source_group_ids"]}
        ),
        "n_candidate_molecules": len(neighbors),
        "neighbors": list(neighbors.values()),
    }
    selector = selector_metadata(SIMILARITY_SELECTOR)
    return {
        "status": "ok",
        "evidence_source": {
            "type": "starling_v10_cache_matched",
            "selection_contract": selection_contract,
        },
        "query": {
            "input_smiles": query_smiles,
            "canonical_smiles": query_parent_smiles,
            "standard_inchi_key": query_parent_id,
            "fingerprint": {"radius": 2, "bits": 2048},
        },
        "retrieval_policy": {
            **({"bbb_evidence_policy": BBB_EVIDENCE_POLICY} if task == "bbb_martins" else {}),
            "neighbor_selector": selector,
            "neighbor_identity_policy": (
                "level_specific_disjoint"
                if selection_contract in {
                    "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
                    CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
                }
                else "scaffold_disjoint"
            ),
            **(
                {"neighbor_identity_policy_by_level": level_identity_policy}
                if selection_contract in {
                    "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1",
                    CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT
                }
                else {}
            ),
            "similarity_floor": None,
        },
        "experiment": {
            "mode": "full_flat",
            "source": "starling_v10_cache_matched",
            "neighbor_selector": selector,
            # Replay validation must not mistake cache-matched selection for the
            # legacy branch reranker. Its real provenance is the contract below.
            "retrieval_reranker": {"name": "none"},
            "cache_matched_selection": {
                "version": selection_contract,
                "reranking": reranking,
                "query_id": query_id,
                "query_audit": deepcopy(dict(query_audit)),
            },
        },
        "groups": [group],
        "coverage": retrieval_coverage(
            [group], min_similarity=0.0, top_k_per_group=len(neighbors)
        ),
    }


def _minimal_selected_evidence(
    payload: Mapping[str, Any],
    parent_id: str,
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        key: value
        for key, value in dict(payload.get("source_fields") or {}).items()
        if key != "extra_details"
    }
    endpoint = _first_field(
        fields,
        "standard_type",
        "endpoint",
        "endpoint_type",
        "endpoint_name",
        "parameter_name",
        "metric_type",
        "bioavailability_report_type",
    ) or str(payload["family_key"])
    evidence_row = {
        "evidence_source": str(payload["source_id"]),
        "source_record_id": str(payload["record_id"]),
        "molecule_chembl_id": parent_id,
        "canonical_smiles": str(payload["canonical_smiles"]),
        "group_id": str(payload["family_key"]),
        "tier": str(payload["progressive_level"]),
        "endpoint_group": str(payload["family_key"]),
        "standard_type": endpoint,
        "standard_relation": _first_field(fields, "standard_relation", "relation"),
        "standard_value": _first_field(
            fields,
            "measurement_text",
            "standard_value",
            "reported_value",
            "parameter_value",
            "quant_value",
            "value",
        ),
        "standard_units": _first_field(
            fields, "standard_units", "reported_units", "parameter_units", "quant_units", "unit", "unit_text"
        ),
        "evidence_text": _first_field(
            fields, "support_text", "assay_description", "description", "activity_comment"
        ) or f"Source-contracted {payload['measurement_kind']} evidence record.",
        "context_text": "; ".join(
            str(value)
            for value in (
                _first_field(fields, "species_or_population", "organism", "species"),
                _first_field(fields, "qualifying_conditions", "assay_context", "assay_model", "context"),
            )
            if value not in (None, "")
        ),
        "source_record_examples": [
            {
                "source_contract": {
                    "contract_version": payload["source_contract"]["contract_version"],
                    "source_or_simply_cleaned": {name: True for name in fields},
                },
                "source_fields": fields,
            }
        ],
    }
    evidence = minimal_evidence_from_row(evidence_row)
    evidence.setdefault("provenance", {})["retrieval"] = deepcopy(dict(retrieval))
    return evidence


def _source_semantic_card(
    payload: Mapping[str, Any],
    *,
    card_id: str,
    transfer_likelihood: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project every approved nonempty source field or account for its exclusion."""
    fields = dict(payload.get("source_fields") or {})
    approved_flags = (payload.get("source_contract") or {}).get(
        "source_or_simply_cleaned"
    )
    if not isinstance(approved_flags, Mapping):
        raise ValueError(f"Source contract is missing for record {payload.get('record_id')}")
    approved = {name for name, allowed in approved_flags.items() if allowed is True}
    if set(fields) != approved:
        raise ValueError(
            f"Source-field contract mismatch for record {payload.get('record_id')}: "
            f"missing={sorted(approved - set(fields))}, extra={sorted(set(fields) - approved)}"
        )

    nonempty = {name: value for name, value in fields.items() if _is_nonempty(value)}
    core: dict[str, Any] = {}
    core_sources: dict[str, str] = {}
    for output_name, candidates in CORE_FIELD_SOURCES.items():
        for source_name in candidates:
            if source_name in nonempty:
                core[output_name] = deepcopy(nonempty[source_name])
                core_sources[output_name] = source_name
                break

    consumed = set(core_sources.values())
    excluded = {
        name: (
            "evidence_molecule_name"
            if name in MOLECULE_NAME_FIELDS
            else "nonsemantic_metadata_or_duplicate_structure"
        )
        for name in nonempty
        if name in MOLECULE_NAME_FIELDS or name in NONSEMANTIC_SOURCE_FIELDS
    }
    experimental = {
        name: deepcopy(value)
        for name, value in nonempty.items()
        if name not in consumed
        and name not in excluded
        and name != "extra_details"
    }
    accounted = consumed | set(experimental) | set(excluded)
    if "extra_details" in nonempty:
        accounted.add("extra_details")
    if accounted != set(nonempty):
        raise ValueError(
            f"Unaccounted source fields for record {payload.get('record_id')}: "
            f"{sorted(set(nonempty) - accounted)}"
        )

    card: dict[str, Any] = {"card_id": card_id}
    for name in FLAT_V2_CARD_FIELDS[1:7]:
        if name in core:
            card[name] = core[name]
    if experimental:
        card["experimental_details"] = experimental
    if "extra_details" in nonempty:
        card["extra_details"] = deepcopy(nonempty["extra_details"])
    if "support_text" in core:
        card["support_text"] = core["support_text"]
    if transfer_likelihood is not None:
        card["transfer_likelihood"] = round(float(transfer_likelihood), 4)

    accounting = {
        "approved_nonempty_fields": list(nonempty),
        "core_field_sources": core_sources,
        "experimental_detail_fields": list(experimental),
        "extra_details_visible": "extra_details" in nonempty,
        "excluded_fields": excluded,
    }
    return card, accounting


def _set_consistent(target: dict[str, Any], name: str, value: Any) -> None:
    if value is None:
        return
    if name in target and target[name] != value:
        raise ValueError(
            f"Conflicting molecule-scoped {name}: {target[name]!r} != {value!r}"
        )
    target[name] = value


def _is_nonempty(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _first_field(fields: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = fields.get(name)
        if value not in (None, "", [], {}):
            return value
    return ""


def _similarity_bucket(similarity: float) -> str:
    for floor, name in (
        (0.95, "very_close_analog"),
        (0.80, "close_analog"),
        (0.60, "moderate_analog"),
        (0.40, "weak_analog"),
        (0.20, "distant_analog"),
    ):
        if similarity >= floor:
            return name
    return "very_distant_analog"


def _load_preselected_uids(
    path: Path,
    *,
    task: str,
    subset: str,
    queries: Mapping[str, str],
    levels: set[str],
    limits: Mapping[str, int],
    cache_index: Path,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    """Validate an optimizer UID panel without trusting it for record content."""
    path = path.resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": PRESELECTED_UID_SCHEMA,
        "status": "complete",
        "task_id": task,
        "subset": subset,
    }
    if any(document.get(key) != value for key, value in required.items()):
        raise ValueError(f"Incompatible preselected UID manifest: {path}")
    query_ids = [str(value) for value in document.get("benchmark_row_ids") or []]
    if len(query_ids) != len(set(query_ids)) or set(query_ids) != set(queries):
        raise ValueError("Preselected UID queries differ from the requested queries")
    expected_limits = {level: int(limits[level]) for level in sorted(levels)}
    if document.get("k_per_level") != expected_limits:
        raise ValueError("Preselected UID level limits differ from the harness request")
    cache_index = cache_index.resolve()
    if document.get("release_index_sha256") != sha256_file(cache_index):
        raise ValueError("Preselected UID release index differs from the active cache")

    records = document.get("records") or {}
    relative = Path(str(records.get("path") or ""))
    records_path = (path.parent / relative).resolve()
    if not relative.name or not records_path.is_relative_to(path.parent.resolve()):
        raise ValueError("Preselected UID records path escapes its manifest")
    if sha256_file(records_path) != records.get("sha256"):
        raise ValueError("Preselected UID records hash mismatch")
    with records_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = [
            "benchmark_row_id", "level", "selection_rank", "source_row_uid"
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError("Preselected UID TSV has an incompatible schema")
        rows = list(reader)
    if len(rows) != int(records.get("row_count", -1)):
        raise ValueError("Preselected UID row count differs from its manifest")

    selected = {
        query_id: {level: [] for level in sorted(levels)} for query_id in queries
    }
    for row in rows:
        query_id, level = row["benchmark_row_id"], row["level"]
        if query_id not in selected or level not in levels:
            raise ValueError(f"Unexpected preselected UID row: {query_id}/{level}")
        try:
            rank = int(row["selection_rank"])
        except ValueError as error:
            raise ValueError("Preselected UID ranks must be integers") from error
        if rank != len(selected[query_id][level]) + 1:
            raise ValueError(f"Preselected UID ranks are not contiguous: {query_id}/{level}")
        selected[query_id][level].append(row["source_row_uid"])
    for query_id, by_level in selected.items():
        for level, uids in by_level.items():
            if len(uids) != expected_limits[level] or len(set(uids)) != len(uids):
                raise ValueError(
                    f"Preselected UID count is invalid: {query_id}/{level}"
                )
    return selected, {
        "path": str(path),
        "sha256": sha256_file(path),
        "records_path": str(records_path),
        "records_sha256": records["sha256"],
        "objective": deepcopy(document.get("objective") or {}),
    }


def _load_preselected_contexts(
    path: Path,
    *,
    task: str,
    queries: Mapping[str, str],
    budget: int,
    allow_subset: bool = False,
    benchmark: str | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Validate an ordered direct-card panel while leaving hydration to the cache."""
    path = path.resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") not in PRESELECTED_DIRECT_SCHEMAS
        or document.get("status") != "complete"
        or document.get("task_id") != task
        or int(document.get("budget", -1)) != budget
        or (benchmark is not None and document.get("benchmark") != benchmark)
    ):
        raise ValueError(f"Incompatible preselected direct manifest: {path}")
    query_ids = [str(value) for value in document.get("benchmark_row_ids") or []]
    valid_queries = (
        set(queries).issubset(query_ids) if allow_subset
        else set(query_ids) == set(queries)
    )
    if len(query_ids) != len(set(query_ids)) or not valid_queries:
        raise ValueError("Preselected direct queries differ from the requested queries")
    records = document.get("records") or {}
    relative = Path(str(records.get("path") or ""))
    records_path = (path.parent / relative).resolve()
    if not relative.name or not records_path.is_relative_to(path.parent.resolve()):
        raise ValueError("Preselected direct records path escapes its manifest")
    if sha256_file(records_path) != records.get("sha256"):
        raise ValueError("Preselected direct records hash mismatch")
    with records_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = [
            "benchmark_row_id", "selection_rank", "context_id", "parent_id", "gold_label"
        ]
        score_policy = document.get("score_policy")
        if score_policy:
            expected += ["assay_transfer_score", "score_key"]
        if reader.fieldnames != expected:
            raise ValueError("Preselected direct TSV has an incompatible schema")
        rows = list(reader)
    if len(rows) != int(records.get("row_count", -1)):
        raise ValueError("Preselected direct row count differs from its manifest")
    selected = {query_id: [] for query_id in query_ids}
    for row in rows:
        query_id = row["benchmark_row_id"]
        if query_id not in selected:
            raise ValueError(f"Unexpected preselected direct query: {query_id}")
        if int(row["selection_rank"]) != len(selected[query_id]) + 1:
            raise ValueError(f"Preselected direct ranks are not contiguous: {query_id}")
        selected[query_id].append(row["context_id"])
    for query_id, context_ids in selected.items():
        if len(context_ids) != budget or len(set(context_ids)) != budget:
            raise ValueError(f"Preselected direct count is invalid: {query_id}")
    receipt = {
        "path": str(path),
        "sha256": sha256_file(path),
        "records_path": str(records_path),
        "records_sha256": records["sha256"],
        "objective": deepcopy(document.get("objective") or {}),
    }
    if score_policy:
        if benchmark != "tdc" or score_policy.get("version") not in {
            "tdc_exact_direct_40_70", "available_scores.v1",
        }:
            raise ValueError("Exact-score direct selection requires TDC")
        overlay = Path(score_policy["overlay_manifest"]).resolve()
        if sha256_file(overlay) != score_policy["overlay_manifest_sha256"]:
            raise ValueError("Exact-score overlay manifest changed")
        for name in ("cache_bundle", "later_index"):
            source = Path(score_policy[name]).resolve()
            if sha256_file(source) != score_policy[f"{name}_sha256"]:
                raise ValueError(f"Exact-score {name} changed")
        overlay_manifest = json.loads(overlay.read_text(encoding="utf-8"))
        if (overlay_manifest.get("status") != "complete"
                or overlay_manifest.get("task") != task
                or overlay_manifest.get("database_sha256") != score_policy["database_sha256"]):
            raise ValueError("Exact-score overlay identity differs")
        coverage_entry = score_policy["coverage"]
        coverage_path = (path.parent / coverage_entry["path"]).resolve()
        if not coverage_path.is_relative_to(path.parent) or sha256_file(coverage_path) != coverage_entry["sha256"]:
            raise ValueError("Exact-score coverage table changed")
        with coverage_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            expected_fields = (["benchmark_row_id", "scored_candidates", "candidate_count",
                                "direct_scored", "indirect_scored"]
                               if score_policy["version"] == "available_scores.v1" else
                               ["benchmark_row_id", "scored_candidates", "candidate_count",
                                "coverage_band", "direct_scored", "indirect_scored"])
            if reader.fieldnames != expected_fields:
                raise ValueError("Exact-score coverage schema differs")
            coverage_rows = list(reader)
        if len(coverage_rows) != coverage_entry["row_count"]:
            raise ValueError("Exact-score coverage count differs")
        bands = {}
        for row in coverage_rows:
            query_id = row["benchmark_row_id"]
            scored, total = int(row["scored_candidates"]), int(row["candidate_count"])
            band = "high" if 10 * scored >= 7 * total else "low" if 5 * scored <= 2 * total else "middle"
            expected_band = "available" if score_policy["version"] == "available_scores.v1" else band
            if (query_id not in selected or query_id in bands or total < 100
                    or not 0 <= scored <= total
                    or int(row["direct_scored"]) + int(row["indirect_scored"]) != scored
                    or (score_policy["version"] != "available_scores.v1"
                        and row["coverage_band"] != expected_band)):
                raise ValueError(f"Exact-score coverage band differs: {query_id}")
            bands[query_id] = expected_band
        if set(bands) != set(selected):
            raise ValueError("Exact-score coverage omits queries")
        database = overlay.with_name(overlay_manifest["database"])
        scores = {query_id: {} for query_id in queries}
        with sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True) as connection:
            for row in rows:
                query_id, context_id, parent = row["benchmark_row_id"], row["context_id"], row["parent_id"]
                raw, key = row["assay_transfer_score"], row["score_key"]
                if bool(raw) != bool(key):
                    raise ValueError("Exact-score score/key mismatch")
                score = float(raw) if raw else None
                frozen = connection.execute(
                    "SELECT parent_id,score,score_key FROM scores WHERE query_id=? AND level='L1' AND item_id=?",
                    (query_id, context_id)).fetchone()
                if (frozen is None) != (score is None) or (frozen is not None and
                        (frozen[0] != parent or frozen[1] != score or frozen[2] != key)):
                    raise ValueError(f"Exact-score snapshot differs: {query_id}/{context_id}")
                if query_id in scores:
                    scores[query_id][("L1", parent)] = score
        receipt["score_visibility"] = {query_id: bands[query_id] for query_id in queries}
        receipt["_selected_scores"] = scores
    return {query_id: selected[query_id] for query_id in queries}, receipt


def _load_mixed_selection(
    path: Path, *, task: str, benchmark: str, subset: str,
    queries: Mapping[str, str], levels: set[str], cache_index: Path,
    cache_config: Path,
    allow_subset: bool = False,
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]], dict[str, Any]]:
    """Resolve a frozen ten-card/joint-UID selection; cache readers hydrate its IDs."""
    path = path.resolve()
    mixed = json.loads(path.read_text(encoding="utf-8"))
    if mixed.get("schema_version") in {TDC_V2_MIXED_SCHEMA, DILI_TDC_V2_MIXED_SCHEMA, TDC_V2_BBB_FILTERED_SCHEMA}:
        return _load_tdc_mixed_selection(
            path, mixed=mixed, task=task, benchmark=benchmark, subset=subset,
            queries=queries, levels=levels, cache_index=cache_index,
            cache_config=cache_config, allow_subset=allow_subset,
        )
    expected = {
        "schema_version": PRESELECTED_MIXED_SCHEMA, "status": "complete",
        "benchmark": f"{benchmark}_v1", "task_id": task,
        "subset": (mixed.get("subset") if benchmark == "gold" and subset == "valid"
                   and mixed.get("subset") in {"valid", "valid_small"} else subset),
        "direct_budget": 10,
    }
    budget = int(mixed.get("indirect_budget") or 0)
    if budget < 1 or any(mixed.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible mixed selection manifest: {path}")
    mixed_query_ids = list(map(str, mixed.get("benchmark_row_ids") or []))
    valid_queries = (
        set(queries).issubset(mixed_query_ids) if allow_subset
        else set(mixed_query_ids) == set(queries)
    )
    if len(mixed_query_ids) != len(set(mixed_query_ids)) or not valid_queries:
        raise ValueError("Mixed selection queries differ from the requested cohort")
    children = {}
    for kind in ("direct", "indirect"):
        entry = mixed.get(kind) or {}
        child_path = Path(str(entry.get("path") or "")).resolve()
        if not child_path.is_file() or sha256_file(child_path) != entry.get("sha256"):
            raise ValueError(f"Mixed {kind} manifest hash mismatch: {child_path}")
        child = json.loads(child_path.read_text(encoding="utf-8"))
        if any(child.get(key) != mixed.get(key) for key in (
            "benchmark", "task_id", "subset", "benchmark_row_ids"
        )) or child.get("budget") != mixed[f"{kind}_budget"]:
            raise ValueError(f"Mixed {kind} manifest disagrees with its parent")
        children[kind] = (child_path, child)
    if mixed.get("score_policy") == "query_mean_missing_scores.v1" and (
        benchmark != "gold" or task != "carcinogens" or any(
            child.get("objective", {}).get("assay_score_policy") != mixed["score_policy"]
            for _, child in children.values()
        )
    ):
        raise ValueError("Gold query-mean selection policy differs from its children")
    score_visibility = None
    coverage_entry = mixed.get("score_coverage")
    if coverage_entry:
        if (benchmark != "gold" or any(
            child.get("score_coverage") != coverage_entry for _, child in children.values()
        )):
            raise ValueError("Gold mixed score coverage differs from selections")
        coverage_path = Path(coverage_entry["path"])
        if sha256_file(coverage_path) != coverage_entry["sha256"]:
            raise ValueError("Gold mixed score coverage changed")
        direct_index = Path(children["direct"][1]["inputs"]["direct_cache"]["release_index"])
        coverage = load_coverage(
            coverage_path, task=task, queries=mixed_query_ids,
            direct_index=direct_index, later_index=cache_index,
            cache_bundle=cache_config,
        )
        score_visibility = {
            query_id: ("available" if mixed.get("score_policy") == "available_scores.v1"
                       else coverage["queries"][query_id]["coverage_band"])
            for query_id in queries
        }
    contexts, direct_receipt = _load_preselected_contexts(
        children["direct"][0], task=task, queries=queries, budget=10,
        allow_subset=allow_subset,
    )
    indirect_path, indirect = children["indirect"]
    if indirect.get("schema_version") != "gold_joint_indirect_uid_selection.v1":
        raise ValueError("Mixed indirect selection has an incompatible schema")
    inputs = indirect.get("inputs") or {}
    if (inputs.get("cache_bundle_sha256") != sha256_file(cache_config)
            or (inputs.get("cache") or {}).get("release_index_sha256") != sha256_file(cache_index)):
        raise ValueError("Mixed indirect selection differs from the active retrieval cache")
    records = indirect.get("records") or {}
    records_path = (indirect_path.parent / str(records.get("path") or "")).resolve()
    if (not records_path.is_relative_to(indirect_path.parent)
            or sha256_file(records_path) != records.get("sha256")):
        raise ValueError("Mixed indirect UID table hash mismatch")
    with records_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != [
            "benchmark_row_id", "selection_rank", "level", "source_row_uid"
        ]:
            raise ValueError("Mixed indirect UID table has an incompatible schema")
        rows = list(reader)
    if len(rows) != records.get("row_count"):
        raise ValueError("Mixed indirect UID table row count differs")
    selected = {query_id: {level: [] for level in levels} for query_id in mixed_query_ids}
    seen = {query_id: set() for query_id in mixed_query_ids}
    ranks = {query_id: 0 for query_id in mixed_query_ids}
    for row in rows:
        query_id, level, uid = row["benchmark_row_id"], row["level"], row["source_row_uid"]
        if query_id not in selected or level not in levels or not uid or uid in seen[query_id]:
            raise ValueError(f"Invalid mixed indirect UID: {query_id}/{level}/{uid}")
        ranks[query_id] += 1
        if int(row["selection_rank"]) != ranks[query_id]:
            raise ValueError(f"Mixed indirect selection ranks are not contiguous: {query_id}")
        selected[query_id][level].append(uid)
        seen[query_id].add(uid)
    if any(count != budget for count in ranks.values()):
        raise ValueError(f"Mixed indirect selection must contain {budget} UIDs per query")
    return contexts, {query_id: selected[query_id] for query_id in queries}, {
        "path": str(path), "sha256": sha256_file(path),
        "indirect_budget": budget,
        "score_policy": mixed.get("score_policy"),
        "direct": direct_receipt,
        "indirect": {"path": str(indirect_path), "sha256": sha256_file(indirect_path),
                     "records_path": str(records_path), "records_sha256": records["sha256"]},
        **({"score_visibility": score_visibility,
            "score_coverage": coverage_entry} if score_visibility is not None else {}),
    }


def _load_tdc_mixed_selection(
    path: Path, *, mixed: dict[str, Any], task: str, benchmark: str, subset: str,
    queries: Mapping[str, str], levels: set[str], cache_index: Path,
    cache_config: Path, allow_subset: bool,
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]], dict[str, Any]]:
    if (benchmark != "tdc" or mixed.get("status") != "complete"
            or mixed.get("benchmark") != "tdc" or mixed.get("task_id") != task
            or mixed.get("subset") != subset or mixed.get("direct_budget") != 10
            or type(mixed.get("indirect_budget")) is not int
            or not 1 <= mixed["indirect_budget"] <= 100
            or mixed.get("semantic_policy") not in {
                "selected_central_ledger_with_task_level_mean_fallback",
                "pinned_candidate_snapshot.v1"}):
        raise ValueError("Incompatible TDC mixed selection")
    excluded = set()
    if mixed["schema_version"] == TDC_V2_BBB_FILTERED_SCHEMA:
        if task != "bbb_martins":
            raise ValueError("Filtered TDC selection is BBB-specific")
        filter_entry = mixed.get("prediction_filter") or {}
        filter_path = Path(filter_entry.get("path") or "")
        if not filter_path.is_file() or sha256_file(filter_path) != filter_entry.get("sha256"):
            raise ValueError("BBB prediction filter receipt changed")
        filter_manifest = json.loads(filter_path.read_text(encoding="utf-8"))
        if (filter_manifest.get("schema_version") != "bbb_prediction_only_filter.v1"
                or filter_manifest.get("status") != "complete"
                or filter_manifest.get("subset") != subset
                or filter_manifest.get("excluded_count") != filter_entry.get("excluded_count")):
            raise ValueError("BBB prediction filter receipt differs")
        from predict.tasks.bbb_martins.prediction_filter import prediction_only
        if filter_manifest.get("predicate_sha256") != sha256_file(Path(prediction_only.__code__.co_filename)):
            raise ValueError("BBB prediction predicate changed")
        table = filter_path.with_name("eligibility.tsv")
        if sha256_file(table) != filter_manifest["files"]["eligibility.tsv"]["sha256"]:
            raise ValueError("BBB prediction filter table changed")
        with table.open(encoding="utf-8", newline="") as handle:
            decisions = list(csv.DictReader(handle, delimiter="\t"))
        if len(decisions) != filter_manifest["files"]["eligibility.tsv"]["rows"]:
            raise ValueError("BBB prediction filter table count differs")
        excluded = {row["source_row_uid"] for row in decisions if row["prediction_only"] == "1"}
        if len(excluded) != filter_entry["excluded_count"]:
            raise ValueError("BBB prediction filter exclusion count differs")
    query_ids = list(map(str, mixed.get("benchmark_row_ids") or []))
    if (len(query_ids) != len(set(query_ids))
            or (not set(queries).issubset(query_ids) if allow_subset else set(queries) != set(query_ids))):
        raise ValueError("TDC mixed selection queries differ")
    cohort = mixed.get("query_ids_file")
    if cohort:
        source = Path(cohort["path"])
        if (sha256_file(source) != cohort["sha256"]
                or source.read_text(encoding="utf-8").splitlines() != query_ids):
            raise ValueError("TDC frozen query cohort changed")
    inputs = {str(canonical_location(k)): v for k, v in (mixed.get("inputs") or {}).items()}
    from data.processing.gold_labels.conditioned_benchmark import tdc_split_path, tdc_manifest
    release_manifest = tdc_manifest(task, subset)
    query_path = tdc_split_path(task, subset)
    for source in (release_manifest, query_path):
        if inputs.get(str(source.resolve())) != sha256_file(source):
            raise ValueError(f"TDC original-split input changed: {source}")
    bundle = yaml.safe_load(cache_config.read_text(encoding="utf-8"))
    direct_index = (cache_config.parent / bundle["caches"][task]["L1"]).resolve()
    dili_successor = mixed["schema_version"] == DILI_TDC_V2_MIXED_SCHEMA
    overlay_entry = mixed.get("score_overlay") or {}
    overlay = (Path(overlay_entry["path"]) if overlay_entry else
               Path(__file__).resolve().parents[3] / (
                   f"data/caches/assay_reranking/active/flat_v5/tdc/{task}/scores/manifest.json"))
    if overlay_entry and sha256_file(overlay) != overlay_entry.get("sha256"):
        raise ValueError("TDC selected score overlay changed")
    if dili_successor:
        from optimization.selection.record_scores import load_scores as load_dili_scores
        tool = mixed.get("dili_tool_scores") or {}
        direct = mixed.get("direct_manifest") or {}
        if (task != "dili" or mixed.get("semantic_relevance_formula") != "sqrt(dili_tool_score * semantic_weight)"
                or sha256_file(overlay) != mixed["score_overlay"]["sha256"]
                or sha256_file(Path(direct["path"])) != direct["sha256"]):
            raise ValueError("DILI mixed successor provenance differs")
        _, audit = load_dili_scores(Path(tool["manifest"]))
        if (audit["manifest_sha256"] != tool["manifest_sha256"]
                or audit["release_index_sha256"] != sha256_file(cache_index)):
            raise ValueError("DILI tool score release differs")
        if mixed.get("semantic_policy") != "pinned_candidate_snapshot.v1":
            extension = mixed.get("semantic_extension") or {}
            extension_path = Path(extension.get("path") or "")
            if not extension_path.is_file() or sha256_file(extension_path) != extension.get("sha256"):
                raise ValueError("DILI semantic extension changed")
            extension_manifest = json.loads(extension_path.read_text(encoding="utf-8"))
            extension_table = extension_path.with_name(extension_manifest["projection"]["path"])
            if (extension_manifest.get("schema_version") != "semantic_retrieval_mean_weights.v2"
                    or sha256_file(extension_table) != extension_manifest["projection"]["sha256"]):
                raise ValueError("DILI semantic extension payload changed")
    for source in (cache_config.resolve(), cache_index.resolve(), direct_index, overlay.resolve()):
        if inputs.get(str(source)) != sha256_file(source):
            raise ValueError(f"TDC selection input changed: {source}")
    if dili_successor:
        cache_release = json.loads(cache_index.read_text(encoding="utf-8"))
        if (cache_release.get("parent_relation_filter") != "parent_disjoint"
                or cache_release.get("identity_normalizer_version") != "rdkit_fragment_parent.v2"):
            raise ValueError("DILI successor cache lacks the reviewed parent policy")
    if mixed["schema_version"] == TDC_V2_BBB_FILTERED_SCHEMA:
        for source, digest in filter_manifest["inputs"].items():
            if inputs.get(source) != digest:
                raise ValueError(f"BBB filter source differs from selection: {source}")
    overlay_manifest = json.loads(overlay.read_text(encoding="utf-8"))
    available = (mixed.get("score_policy") or {}).get("version") == "available_scores.v1"
    if available and overlay_manifest.get("score_count") != sum(overlay_manifest.get("counts", {}).values()):
        raise ValueError("TDC exact-score snapshot is incomplete")
    database = overlay.with_name(overlay_manifest["database"])
    if inputs.get(str(database.resolve())) != overlay_manifest.get("database_sha256"):
        raise ValueError("TDC selected score database differs")
    candidate_semantics = None
    if mixed["semantic_policy"] == "pinned_candidate_snapshot.v1":
        from optimization.semantic_buckets.inputs import load_skin_candidate_snapshot, load_dili_candidate_snapshot
        entry = mixed.get("semantic_snapshot") or {}
        snapshot = canonical_location(entry.get("path") or "")
        if (task not in {"skin_reaction", "dili"}
                or sha256_file(snapshot) != entry.get("sha256")):
            raise ValueError("TDC candidate snapshot changed")
        loader = load_dili_candidate_snapshot if task == "dili" else load_skin_candidate_snapshot
        candidate_semantics, pins = loader(snapshot)
        if any(inputs.get(source) != digest for source, digest in pins.items()):
            raise ValueError("TDC candidate lineage changed")
    else:
        from optimization.semantic_buckets.inputs import resolve_semantic_bucket_artifacts
        semantic_artifacts = resolve_semantic_bucket_artifacts(task, "CURRENT")
        semantic_path = semantic_artifacts.manifest.resolve()
        if sha256_file(semantic_path) != inputs.get(str(semantic_path)):
            raise ValueError("TDC selected semantic release changed")
        semantic_release = json.loads(semantic_path.read_text(encoding="utf-8"))
        from optimization.semantic_buckets.inputs import MEMBERSHIP_WEIGHT_RESOLUTION, load_reviewed_record_weights
        if semantic_release.get("semantic_weight_resolution") == MEMBERSHIP_WEIGHT_RESOLUTION:
            _, audit = load_reviewed_record_weights(task, "CURRENT")
            if any(inputs.get(source) != digest for source, digest in audit["inputs"].items()):
                raise ValueError("TDC selected semantic memberships or weights changed")
        else:
            if (semantic_release.get("central_ledger_generation") is None
                    or semantic_release.get("provisional_retrieval_weight_policy")
                    != "mean_of_reviewed_semantic_buckets_in_task_level"):
                raise ValueError("TDC selection lacks an official semantic ledger")
            for key in ("central_ledger", "semantic_bucket_weights", "record_relevance_rankings",
                        "weight_projection_manifest", "pending_uid_worklist"):
                entry = (semantic_release.get("files") or {}).get(key) or {}
                source = (semantic_path.parent / str(entry.get("path") or "")).resolve()
                if inputs.get(str(source)) != entry.get("sha256"):
                    raise ValueError(f"TDC selected semantic ledger differs: {key}")
    tables = {}
    fields = {
        "selected_contexts.tsv": ["benchmark_row_id", "selection_rank", "context_id", "parent_id", "gold_label", "assay_transfer_score", "score_key"],
        "selected_uids.tsv": ["benchmark_row_id", "selection_rank", "level", "source_row_uid", "parent_id", "assay_transfer_score", "score_key", "semantic_bucket_id", "semantic_weight", "semantic_weight_source"],
        "coverage.tsv": (["benchmark_row_id", "scored_candidates", "candidate_count", "direct_scored", "indirect_scored"]
                         if (mixed.get("score_policy") or {}).get("version") == "available_scores.v1" else
                         ["benchmark_row_id", "scored_candidates", "candidate_count", "coverage_band", "direct_scored", "indirect_scored"]),
    }
    for name, expected in fields.items():
        source = (path.parent / name).resolve()
        entry = (mixed.get("files") or {}).get(name) or {}
        if not source.is_relative_to(path.parent) or sha256_file(source) != entry.get("sha256"):
            raise ValueError(f"TDC selection table changed: {name}")
        with source.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != expected:
                raise ValueError(f"TDC selection table schema differs: {name}")
            tables[name] = list(reader)
        if len(tables[name]) != entry.get("rows"):
            raise ValueError(f"TDC selection table count differs: {name}")
    if mixed.get("direct_manifest"):
        direct_entry = mixed["direct_manifest"]
        direct_path = Path(direct_entry["path"])
        if sha256_file(direct_path) != direct_entry["sha256"]:
            raise ValueError("TDC frozen direct manifest changed")
        with direct_path.with_name("selected_contexts.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            frozen = [(row["benchmark_row_id"], row["selection_rank"], row["context_id"])
                      for row in csv.DictReader(handle, delimiter="\t")]
        selected = [(row["benchmark_row_id"], row["selection_rank"], row["context_id"])
                    for row in tables["selected_contexts.tsv"]]
        if frozen != selected:
            raise ValueError("TDC mixed selection changed the frozen direct panel")
    bands = {}
    for row in tables["coverage.tsv"]:
        query_id = row["benchmark_row_id"]
        scored, total = int(row["scored_candidates"]), int(row["candidate_count"])
        band = ("available" if available else
                "high" if 10 * scored >= 7 * total else "low" if 5 * scored <= 2 * total else "middle")
        if (query_id in bands or query_id not in query_ids or total < 100
                or not 0 <= scored <= total
                or (not available and row["coverage_band"] != band)
                or int(row["direct_scored"]) + int(row["indirect_scored"]) != scored):
            raise ValueError(f"TDC score band differs: {query_id}")
        bands[query_id] = band
    if set(bands) != set(query_ids):
        raise ValueError("TDC score bands omit queries")
    contexts = {query_id: [] for query_id in query_ids}
    uids = {query_id: {level: [] for level in levels} for query_id in query_ids}
    scores = {query_id: {} for query_id in query_ids}
    score_db = sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True)
    for name, budget in (("selected_contexts.tsv", 10),
                         ("selected_uids.tsv", mixed["indirect_budget"])):
        ranks = {query_id: 0 for query_id in query_ids}
        for row in tables[name]:
            query_id = row["benchmark_row_id"]
            if query_id not in ranks or int(row["selection_rank"]) != ranks[query_id] + 1:
                raise ValueError(f"TDC selection ranks differ: {query_id}")
            ranks[query_id] += 1
            level = "L1" if name == "selected_contexts.tsv" else row["level"]
            item = row["parent_id"] if level == "L1" else row["source_row_uid"]
            if level != "L1" and level not in levels:
                raise ValueError(f"TDC UID level differs: {level}")
            if level != "L1" and item in excluded:
                raise ValueError(f"Prediction-only UID entered BBB selection: {item}")
            if level != "L1":
                source = row["semantic_weight_source"]
                if (source not in {"reviewed_family_mean", "provisional_task_level_mean",
                                   "candidate_family_mean", "candidate_calibrated_v2"}
                        or bool(row["semantic_bucket_id"]) == (source == "provisional_task_level_mean")
                        or not 0 <= float(row["semantic_weight"]) <= 1):
                    raise ValueError(f"TDC semantic weight provenance differs: {query_id}/{item}")
                if candidate_semantics is not None:
                    frozen = candidate_semantics.get((item, level))
                    if (frozen is None
                            or frozen["semantic_bucket_id"] != row["semantic_bucket_id"]
                            or frozen["semantic_weight"] != float(row["semantic_weight"])):
                        raise ValueError(f"TDC candidate semantic weight differs: {query_id}/{item}")
            key = (level, item)
            if not item or key in scores[query_id]:
                raise ValueError(f"TDC selected unit repeats: {query_id}/{key}")
            raw_score, score_key = row["assay_transfer_score"], row["score_key"]
            if bool(raw_score) != bool(score_key):
                raise ValueError(f"TDC score/key mismatch: {query_id}/{key}")
            score = float(raw_score) if raw_score else None
            if score is not None and not 0 <= score <= 1:
                raise ValueError(f"TDC score out of range: {query_id}/{key}")
            lookup = row["context_id"] if level == "L1" else row["source_row_uid"]
            frozen = score_db.execute(
                "SELECT parent_id,score,score_key FROM scores WHERE query_id=? AND level=? AND item_id=?",
                (query_id, level, lookup),
            ).fetchone()
            if (frozen is None) != (score is None) or (frozen is not None and
                    (frozen[0] != row["parent_id"] or frozen[1] != score or frozen[2] != score_key)):
                raise ValueError(f"TDC selected score differs from frozen snapshot: {query_id}/{key}")
            scores[query_id][key] = score
            if level == "L1":
                contexts[query_id].append(row["context_id"])
            else:
                uids[query_id][level].append(item)
        if any(count != budget for count in ranks.values()):
            raise ValueError(f"TDC selection must contain {budget} {name} per query")
    score_db.close()
    return ({query_id: contexts[query_id] for query_id in queries},
            {query_id: uids[query_id] for query_id in queries},
            {"path": str(path), "sha256": sha256_file(path),
             "indirect_budget": mixed["indirect_budget"],
             "score_visibility": {query_id: bands[query_id] for query_id in queries},
             "_selected_scores": {query_id: scores[query_id] for query_id in queries}})


def _load_flat_context_candidates(
    args: argparse.Namespace,
    queries: Mapping[str, str],
    *,
    load_cache_policy: Any,
    load_candidates: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select cached UIDs, hydrate evidence rows, and compose the flat context."""
    selection_contract = (
        CONTEXT_V5_SELECTION_CONTRACT
        if args.prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS
        else CONTEXT_V4_SELECTION_CONTRACT
    )
    policy = load_cache_policy(
        args.assay_transfer_cache,
        args.task,
        args.evaluation_subset,
        args.reranking,
        args.max_level,
        benchmark=args.benchmark,
    )
    active_levels = {level for level in policy["stages"] if level != "L1"}
    all_later_records = policy.get("all_later_records") is True
    physical_records = policy.get("cache_set") is not None
    physical_selection = None
    manual_uids = getattr(args, "uids", None)
    manual_file = getattr(args, "uids_file", None)
    manual_receipt = None
    if manual_uids is not None or manual_file is not None:
        from predict.retrieval.selection import read_selection
        if manual_file is not None:
            physical_selection, manual_receipt = read_selection(
                manual_file, task=args.task, subset=args.evaluation_subset, queries=queries,
            )
        else:
            if len(queries) != 1 or not getattr(args, "uid_level", None):
                raise ValueError("--uids requires exactly one query and --uid-level")
            query_id = next(iter(queries))
            physical_selection = [dict(benchmark_row_id=query_id, level=args.uid_level,
                                       source_row_uid=uid) for uid in manual_uids]
            manual_receipt = {"uids": list(manual_uids), "level": args.uid_level}
        if policy.get("selection_contract") != "ranked_uid_retrieval.v1":
            raise ValueError("Manual physical UIDs require a frozen ranked UID cache")
        if manual_file is not None and "manifest" in manual_receipt:
            pins = manual_receipt["manifest"]["cache_manifests"]
            if args.benchmark == "tdc" and "L1" in pins:
                from data.processing.paths import REPO_ROOT
                from predict.retrieval.selection import indexed_manifests
                card_index = REPO_ROOT / tdc_config(args.task)["selection"]["direct_cache"]
                card_manifest = indexed_manifests(card_index, args.evaluation_subset, ("L1",))["L1"]
                if pins["L1"]["sha256"] == sha256_file(card_manifest):
                    policy = {**policy,
                              "stages": {**policy["stages"], "L1": "morgan"},
                              "cache_indexes": {**policy["cache_indexes"], "L1": str(card_index)},
                              "cache_manifests": {**policy["cache_manifests"], "L1": str(card_manifest)},
                              "inputs": {**policy["inputs"], str(card_index): sha256_file(card_index),
                                         str(card_manifest): sha256_file(card_manifest)}}
            if "L1" in pins:
                canonical_manifest = Path(pins["L1"]["path"])
                canonical_header = json.loads(canonical_manifest.read_text())
                if canonical_header.get("context_identity") == "canonical_context_molecule.v1":
                    canonical_index = canonical_manifest.parents[3] / "RELEASE_INDEX.json"
                    from predict.retrieval.selection import indexed_manifests
                    checked = indexed_manifests(canonical_index, args.evaluation_subset, ("L1",))["L1"]
                    if sha256_file(checked) != pins["L1"]["sha256"]:
                        raise ValueError("Canonical L1 index differs from the selected cards")
                    policy = {**policy, "stages": {**policy["stages"], "L1": "morgan"},
                              "cache_indexes": {**policy["cache_indexes"], "L1": str(canonical_index)},
                              "cache_manifests": {**policy["cache_manifests"], "L1": str(checked)}}
            for level, entry in pins.items():
                active = Path(policy["cache_manifests"][level])
                if sha256_file(active) != entry["sha256"]:
                    raise ValueError("Manual selection differs from the requested cache release")
    preselected_uids = None
    preselected_contexts = None
    preselection_receipt = None
    mixed_scores = None
    if manual_receipt and "manifest" in manual_receipt:
        uid_manifest = manual_receipt["manifest"]
        snapshot = uid_manifest.get("provenance", {}).get("optimizer_snapshot")
        if snapshot:
            canonical_path = Path(manual_receipt["path"])
            snapshot_path = (canonical_path.parent / snapshot["path"]).resolve()
            if sha256_file(snapshot_path) != snapshot["sha256"]:
                raise ValueError("Optimizer scientific provenance changed")
            source = json.loads(snapshot_path.read_text())
            if source["schema_version"] in {"context_selection.v1", "combined_context_selection.v1"}:
                from optimization.selection.search import selected_scores
                scientific_manifest = Path(manual_receipt.get("optimizer_manifest", {}).get("path") or snapshot_path.parent / "manifest.json")
                mixed_scores = selected_scores(scientific_manifest)
                if set(mixed_scores) != set(queries):
                    raise ValueError("Canonical selection score cohort differs")
                preselection_receipt = {"score_visibility": {query: "available" for query in queries}}
            elif source["schema_version"] in PRESELECTED_DIRECT_SCHEMAS:
                _, preselection_receipt = _load_preselected_contexts(
                    snapshot_path, task=args.task, queries=queries, budget=args.l1_molecules,
                    allow_subset=bool(args.indices),
                    benchmark="gold_v1" if args.benchmark == "gold" else "tdc",
                )
                mixed_scores = preselection_receipt.pop("_selected_scores", None)
            elif source["schema_version"] in {PRESELECTED_MIXED_SCHEMA, TDC_V2_MIXED_SCHEMA,
                                           DILI_TDC_V2_MIXED_SCHEMA, TDC_V2_BBB_FILTERED_SCHEMA}:
                _, _, preselection_receipt = _load_mixed_selection(
                    snapshot_path, task=args.task, benchmark=args.benchmark,
                    subset=args.evaluation_subset, queries=queries, levels=active_levels,
                    cache_index=Path(policy["cache_indexes"]["L2"]), cache_config=args.assay_transfer_cache,
                    allow_subset=False,
                )
                mixed_scores = preselection_receipt.pop("_selected_scores", None)
    if getattr(args, "flat_mixed_selection", None):
        preselected_contexts, preselected_uids, preselection_receipt = _load_mixed_selection(
            args.flat_mixed_selection,
            task=args.task, benchmark=args.benchmark, subset=args.evaluation_subset,
            queries=queries, levels=active_levels,
            cache_index=Path(policy["cache_indexes"]["L2"]),
            cache_config=args.assay_transfer_cache,
            allow_subset=bool(getattr(args, "flat_mixed_selection_subset", False)),
        )
        mixed_scores = preselection_receipt.pop("_selected_scores", None)
        if (preselection_receipt.get("score_policy") == "query_mean_missing_scores.v1"
                and args.evaluation_subset == "valid"):
            policy = {**policy, "allow_unscored_preselected": True}
    if getattr(args, "flat_preselected_uids", None):
        preselected_uids, preselection_receipt = _load_preselected_uids(
            args.flat_preselected_uids,
            task=args.task,
            subset=args.evaluation_subset,
            queries=queries,
            levels=active_levels,
            limits=args.record_limits_by_level,
            cache_index=Path(
                policy.get("cache_index")
                or next(iter(policy["cache_indexes"].values()))
            ),
        )
    if getattr(args, "flat_preselected_contexts", None):
        preselected_contexts, preselection_receipt = _load_preselected_contexts(
            args.flat_preselected_contexts,
            task=args.task,
            queries=queries,
            budget=args.l1_molecules,
            allow_subset=bool(args.indices),
            benchmark=("gold_v1" if args.benchmark == "gold" else
                       "tdc"),
        )
        mixed_scores = preselection_receipt.pop("_selected_scores", None)
    molecules, ranked_later, audit = load_candidates(
        queries,
        task=args.task,
        subset=args.evaluation_subset,
        policy=policy,
        molecule_limit=args.l1_molecules,
        l1_limit=args.l1_records_per_molecule,
        later_limit={
            level: (
                args.record_limits_by_level[level]
                if args.prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS else 100
            )
            for level in policy["stages"] if level != "L1"
        },
        tie_seed=args.ranking_tie_seed,
        min_contrast=args.l1_min_contrast,
        morgan_primary_parent_width=args.morgan_primary_parent_width,
        cache_pool="all",
        preselected_uids=preselected_uids,
        preselected_contexts=preselected_contexts,
        physical_selection=physical_selection,
        preselected_uid_budget=(
            preselection_receipt["indirect_budget"]
            if getattr(args, "flat_mixed_selection", None) else None
        ),
    )

    later: dict[str, Any] = {}
    query_audits: dict[str, Any] = {}
    for query_id in queries:
        l1_record_ids = {
            str(record["record_id"])
            for molecule in molecules[query_id]
            for record in molecule.get("l1_records") or []
        }
        later[query_id] = {}
        query_audits[query_id] = {
            "L1": deepcopy(audit["query_audits"][query_id]["L1"])
        }
        for level, group in ranked_later[query_id].items():
            limit = args.record_limits_by_level[level]
            records = (
                list(group["records"])
                if all_later_records or physical_records or physical_selection is not None or getattr(args, "flat_mixed_selection", None) else [
                    row for row in group["records"]
                    if str(row["record_id"]) not in l1_record_ids
                ][:limit]
            )
            if (not all_later_records and not physical_records and physical_selection is None and not getattr(args, "flat_mixed_selection", None)
                    and args.prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS
                    and len(records) != limit and not group.get("allow_shortfall")):
                raise ValueError(
                    f"Requested {limit} records but {query_id}/{level} has "
                    f"{len(records)} after L1 exclusion"
                )
            selected_molecules = len({
                str(row["reference_molecule_id"]) for row in records
            })
            later[query_id][level] = {
                **group,
                "records": records,
                "selected_records": len(records),
                "selected_molecules": selected_molecules,
                "shortfall": 0 if all_later_records or physical_selection is not None or getattr(args, "flat_mixed_selection", None)
                else max(0, limit - len(records)),
            }
            query_audits[query_id][level] = {
                key: value for key, value in later[query_id][level].items()
                if key not in {"records", "ranking_method", "allow_shortfall"}
            }

    later_content_ids = {
        level: content_id
        for level, content_id in audit.get("cache_content_ids", {}).items()
        if level != "L1"
    }
    return molecules, later, {
        "_tdc_selected_scores": mixed_scores,
        "_tdc_score_visibility": (
            preselection_receipt.get("score_visibility") if mixed_scores is not None else None
        ),
        "_gold_score_visibility": (
            preselection_receipt.get("score_visibility") if mixed_scores is None
            and preselection_receipt is not None else None
        ),
        "manual_uid_selection": manual_receipt,
        "selection_policy": selection_contract,
        "inputs": dict(audit.get("inputs") or {}),
        "contract": {
            **({"bbb_evidence_policy": BBB_EVIDENCE_POLICY} if args.task == "bbb_martins" else {}),
            "policy": {
                "selection_contract": selection_contract,
                "uid_retrieval": policy,
            },
            "molecule_limit": args.l1_molecules,
            "l1_limit": args.l1_records_per_molecule,
            "later_limits": {
                level: args.record_limits_by_level[level]
                for level in active_levels if level != "L1"
            },
            "all_later_records": all_later_records,
            "joint_uid_budget": (
                preselection_receipt["indirect_budget"]
                if getattr(args, "flat_mixed_selection", None) else None
            ),
            "min_contrast": args.l1_min_contrast,
            "preselected_uids": (
                preselection_receipt if preselected_uids is not None else None
            ),
            "preselected_contexts": (
                preselection_receipt if preselected_contexts is not None else None
            ),
        },
        "cache_pool": "all",
        "query_identities": audit["query_identities"],
        "query_audits": query_audits,
        "neighbor_identity_policy": audit["neighbor_identity_policy"],
        "neighbor_identity_policy_by_level": {
            level: value
            for level, value in audit["neighbor_identity_policy_by_level"].items()
            if level in policy["stages"]
        },
        "similarity_floor": None,
        "scaffold_overlap": 0,
        "parent_overlap": 0,
        "l1_cache": {
            "bundle": str(args.assay_transfer_cache),
            "selection_policy": audit["selection_policy"],
            "cache_index": audit.get("cache_index"),
            "content_id": audit.get("cache_content_ids", {}).get("L1"),
            "evidence_manifest": audit.get("evidence_manifest"),
        },
        "later_cache": (
            {
                "bundle": str(args.assay_transfer_cache),
                "selection_policy": audit["selection_policy"],
                "cache_index": audit.get("cache_index"),
                "content_ids": later_content_ids,
            }
            if later_content_ids else None
        ),
    }


def _load_molecule_descriptions(mode: str, version: str) -> tuple[dict[str, str], dict[str, Any]]:
    if mode == "none":
        return {}, {"mode": "none"}
    import pyarrow.parquet as pq

    column = MOLECULE_DESCRIPTION_COLUMNS[mode]
    path, expected_sha256 = MOLECULE_DESCRIPTION_ARTIFACTS[version]
    root = Path(os.environ.get("TXAGENT_MOLECULE_DESCRIPTION_ROOT", REPO_ROOT / "data/caches/molecule_descriptions"))
    path = (root.expanduser() / path).resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Molecule description cache hash mismatch: {path}")
    table = pq.read_table(path, columns=["canonical_smiles", column, "error"])
    values: dict[str, str] = {}
    for row in table.to_pylist():
        smiles = str(row.get("canonical_smiles") or "")
        description = row.get(column)
        if smiles in values:
            raise ValueError(f"Duplicate molecule description SMILES: {smiles}")
        if smiles and isinstance(description, str) and description.strip() and not row.get("error"):
            values[smiles] = description.strip()
    return values, {
        "mode": mode,
        "cache_version": version,
        "column": column,
        "path": str(path),
        "sha256": expected_sha256,
        "row_count": table.num_rows,
    }


def _attach_molecule_descriptions(
    retrieval: dict[str, Any], values: Mapping[str, str]
) -> list[str]:
    if not values:
        return []
    missing = []
    query = retrieval["query"]
    query_smiles = str(query.get("canonical_smiles") or "")
    if query_smiles in values:
        query["molecule_description"] = values[query_smiles]
    else:
        missing.append(query_smiles)
    for neighbor in retrieval["groups"][0]["neighbors"]:
        smiles = str(neighbor.get("canonical_smiles") or "")
        if smiles in values:
            neighbor["molecule_description"] = values[smiles]
        else:
            missing.append(smiles)
    return sorted(set(missing))


def _apply_tdc_scores(
    retrieval: dict[str, Any], selected: Mapping[tuple[str, str], float | None], band: str,
) -> None:
    if band not in {"low", "middle", "high", "available"}:
        raise ValueError("TDC selection has no valid score band")
    seen = set()
    for neighbor in retrieval["groups"][0]["neighbors"]:
        parent = str(neighbor["standard_inchi_key"])
        (neighbor.get("_prompt_scores") or {}).pop("transfer_likelihood", None)
        for row in neighbor["evidence_rows"]:
            provenance = row["selection_provenance"]
            level = str(provenance["level"])
            uid = str(row["source_provenance"]["source_row_uid"])
            card_key = ("L1", str(provenance.get("diagnostic_context_id") or ""))
            key = (card_key if card_key in selected else ("L1", parent)) if level == "L1" else (level, uid)
            if key not in selected:
                raise ValueError(f"TDC rendered an unselected unit: {key}")
            seen.add(key)
            provenance.pop("assay_transfer_score", None)
            provenance.pop("assay_rank", None)
            provenance.pop("assay_transfer_panel_rank", None)
            card = row["prompt_evidence"]
            card.pop("transfer_likelihood", None)
            (card.get("_molecule_scores") or {}).pop("transfer_likelihood", None)
            score = selected[key]
            if band in {"high", "available"} and score is not None:
                provenance["assay_transfer_score"] = round(score, 4)
    if seen != set(selected):
        raise ValueError(f"TDC selected units were not fully rendered: {len(seen)}/{len(selected)}")
    retrieval["tdc_score_visibility"] = band


def _apply_gold_score_visibility(retrieval: dict[str, Any], band: str) -> None:
    if band not in {"low", "middle", "high", "available"}:
        raise ValueError("Gold selection has no valid score band")
    if band in {"low", "middle"}:
        for neighbor in retrieval["groups"][0]["neighbors"]:
            (neighbor.get("_prompt_scores") or {}).pop("transfer_likelihood", None)
            for row in neighbor["evidence_rows"]:
                provenance = row["selection_provenance"]
                for key in ("assay_transfer_score", "assay_rank", "assay_transfer_panel_rank"):
                    provenance.pop(key, None)
                card = row["prompt_evidence"]
                card.pop("transfer_likelihood", None)
                (card.get("_molecule_scores") or {}).pop("transfer_likelihood", None)
    retrieval["score_visibility"] = band


def _materialize_cache_matched_retrievals(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    from predict.retrieval.assay_reranking.cache_matched import (
        load_cache_policy,
        load_candidates,
    )

    if args.prompt_version in FINAL_FULL_FLAT_PROMPT_VERSIONS:
        require_current_cache_bundle(args.assay_transfer_cache)
        selection_path = args.flat_mixed_selection or args.flat_preselected_contexts
        if selection_path is not None:
            selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
            policy = (selection.get("score_policy") or {}).get("version") if args.benchmark == "tdc" else (
                selection.get("score_policy") if args.flat_mixed_selection else
                (selection.get("objective") or {}).get("assay_score_policy")
            )
            if policy != "available_scores.v1" and not (
                args.benchmark == "gold" and args.task == "carcinogens"
                and policy == "query_mean_missing_scores.v1"
            ):
                raise ValueError("final_full_flat requires an available_scores.v1 selection")

    artifact_receipt = None
    if args.prompt_version in CONTEXT_NUMBERED_PROMPT_VERSIONS:
        artifact_benchmark = cache_task(args.assay_transfer_cache, args.task, args.benchmark)["benchmark"]
        artifact_manifest = Path(
            getattr(args, "flat_artifact_manifest", None)
            or bundle_manifest_path(args.task, artifact_benchmark, cache_set=args.assay_transfer_cache)
        ).resolve()
        artifact_receipt_path = Path(
            getattr(args, "flat_artifact_receipt", None)
            or bundle_receipt_path(args.task, artifact_benchmark, cache_set=args.assay_transfer_cache)
        ).resolve()
        try:
            artifact_receipt = require_receipt(
                artifact_manifest,
                receipt_path=artifact_receipt_path,
                verify=bool(getattr(args, "verify_flat_artifacts", False)),
                expected_cache_config=Path(args.assay_transfer_cache),
                expected_task=args.task,
                expected_benchmark=artifact_benchmark,
                expected_max_level=args.max_level,
            )
        except ArtifactBundleError as error:
            raise SystemExit(
                f"full-flat artifact preflight failed for {args.task}/{args.benchmark}: {error}\n"
                "Restore the bundle, or inventory and verify it with: "
                "python -m predict.retrieval.assay_reranking.artifact_bundle "
                f"inventory --task {args.task} --benchmark {artifact_benchmark} "
                f"--cache-config {args.assay_transfer_cache} "
                f"--output {artifact_manifest}"
            ) from error
        args.flat_artifact_manifest = artifact_manifest
        args.flat_artifact_receipt = artifact_receipt_path
    args.uids = getattr(args, "uids", None)
    args.uids_file = getattr(args, "uids_file", None)
    args.uid_level = getattr(args, "uid_level", None)
    args.flat_preselected_uids = getattr(args, "flat_preselected_uids", None)
    args.flat_preselected_contexts = getattr(args, "flat_preselected_contexts", None)
    args.flat_mixed_selection = getattr(args, "flat_mixed_selection", None)
    records = read_jsonl(args.input_jsonl)
    indices = _selected_indices(args, len(records))
    if args.uids_file and not args.indices and args.start == 0 and args.limit == 0:
        from predict.retrieval.selection import selection_query_ids
        requested = selection_query_ids(args.uids_file)
        by_id = {str(row["benchmark_row_id"]): index for index, row in enumerate(records)}
        if set(requested) - by_id.keys():
            raise ValueError("Manual UID query identities are absent from the declared input")
        indices = [by_id[query_id] for query_id in requested]
        args.indices = [str(index) for index in indices]
    preselected_manifest = (
        args.flat_mixed_selection or args.flat_preselected_uids or args.flat_preselected_contexts
    )
    if preselected_manifest and not args.indices and args.start == 0 and args.limit == 0:
        indices = _preselected_query_indices(
            preselected_manifest,
            records,
            task=args.task,
            subset=args.evaluation_subset,
        )
        args.indices = [str(index) for index in indices]
        if args.flat_mixed_selection and len(indices) < len(records):
            args.flat_mixed_selection_subset = True
    queries = {
        str(records[index]["benchmark_row_id"]): str(records[index]["drug"])
        for index in indices
    }
    policy = None if args.prompt_version in CONTEXT_PROMPT_VERSIONS else load_cache_policy(
        args.assay_transfer_cache,
        args.task,
        args.evaluation_subset,
        args.reranking,
        args.max_level,
        benchmark=args.benchmark,
    )
    batch_dir = canonical_location(args.batch_root / args.batch_id)
    source_batch = batch_dir / "cache_matched_retrieval"
    existing_path = source_batch / "manifest.json"
    if existing_path.exists():
        existing = read_record(existing_path)
        expected = {
            "organization": "flat",
            "prompt_version": args.prompt_version,
            "task": args.task,
            "evaluation_subset": args.evaluation_subset,
            "reranking": args.reranking,
            "record_pool": args.record_pool,
            "cache_pool": RECORD_POOLS[args.record_pool],
            "indices": indices,
            "input_jsonl": str(args.input_jsonl),
            "evidence_library": str(args.evidence_library),
            "level_mapper": str(args.level_mapper),
            "assay_transfer_cache": str(args.assay_transfer_cache),
            "gold_context_mapping": (
                str(args.gold_context_mapping) if args.task == "bbb_martins" else ""
            ),
            "allow_frozen_l1_vote_scores": args.allow_frozen_l1_vote_scores,
            "l1_molecules": args.l1_molecules,
            "l1_records_per_molecule": args.l1_records_per_molecule,
            "records_per_level": args.records_per_level,
            "record_limits_by_level": args.record_limits_by_level,
            "ranking_tie_seed": args.ranking_tie_seed,
            "max_level": args.max_level,
            "manual_uids": args.uids,
            "manual_uid_level": args.uid_level,
            "flat_preselected_uids": (
                str(args.flat_preselected_uids) if args.flat_preselected_uids else ""
            ),
            "flat_preselected_contexts": (
                str(args.flat_preselected_contexts) if args.flat_preselected_contexts else ""
            ),
            "flat_mixed_selection": str(args.flat_mixed_selection) if args.flat_mixed_selection else "",
        }
        if args.task == "bbb_martins":
            expected["bbb_evidence_policy"] = BBB_EVIDENCE_POLICY
        if artifact_receipt:
            expected.update(
                artifact_manifest=str(args.flat_artifact_manifest),
                artifact_manifest_sha256=artifact_receipt["manifest_sha256"],
                artifact_receipt=str(args.flat_artifact_receipt),
            )
        if args.prompt_version in CONTEXT_PROMPT_VERSIONS:
            expected.update(
                layout=args.layout,
                query_prior=args.query_prior,
                l1_min_contrast=args.l1_min_contrast,
                morgan_primary_parent_width=args.morgan_primary_parent_width,
                molecule_description_mode=args.molecule_description_mode,
                molecule_description_cache_version=args.molecule_description_cache_version,
            )
        if args.prompt_version == JOSEPH_PROMPT_VERSION:
            expected.update(
                evidence_projection=EVIDENCE_PROJECTION,
                extra_details_policy=EXTRA_DETAILS_POLICY,
                molecule_name_visible=False,
                group_tools_enabled=False,
            )
        wanted = scientific_settings(expected)
        mismatches = {key: {"expected": value, "observed": existing.get(key)}
                      for key, value in wanted.items() if existing.get(key) != value}
        require_query_ids(existing, records, indices)
        require_artifact_versions(existing, {key: expected.get(key) for key in
            ("input_jsonl", "evidence_library", "level_mapper", "assay_transfer_cache", "gold_context_mapping")})
        if existing.get("indices") != indices:
            mismatches["indices"] = "different query membership"
        automatic = None
        if args.uids_file or (preselected_manifest and args.prompt_version in CONTEXT_PROMPT_VERSIONS):
            automatic = _load_flat_context_candidates(
                args, queries, load_cache_policy=load_cache_policy, load_candidates=load_candidates,
            )
            if scientific_settings(automatic[2]["contract"]) != (existing.get("selection_audit") or {}).get("contract"):
                mismatches["selection_parameters"] = "different effective selection contract"
        for index in indices:
            retrieval_path = run_directory(source_batch, index) / "retrieval.json"
            if not retrieval_path.is_file():
                mismatches[f"retrieval[{index}]"] = "missing"
            elif automatic is not None:
                from predict.retrieval.selection import ordered_evidence
                molecules, later, audit = automatic
                query_id = str(records[index]["benchmark_row_id"])
                wanted_retrieval = cache_matched_flat_retrieval(
                    query_id, str(records[index]["drug"]), molecules[query_id], later[query_id],
                    task=args.task, reranking=args.reranking,
                    query_audit=audit["query_audits"][query_id],
                    query_identity=(audit.get("query_identities") or {}).get(query_id),
                    prompt_version=args.prompt_version, selection_contract=audit["selection_policy"],
                    neighbor_identity_policy_by_level=audit.get("neighbor_identity_policy_by_level"),
                    preserve_physical_records=bool(audit["contract"]["policy"].get("uid_retrieval", {}).get("cache_set")) or audit["contract"].get("all_later_records") is True or bool(args.flat_mixed_selection) or args.uids is not None or args.uids_file is not None,
                )
                if ordered_evidence(wanted_retrieval) != ordered_evidence(json.loads(retrieval_path.read_text())):
                    mismatches[f"selected_uids[{index}]"] = "different ordered physical selection"

        if mismatches:
            raise ValueError(
                "Existing flat selection is immutable and differs from this request: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return source_batch, existing

    if args.prompt_version in CONTEXT_PROMPT_VERSIONS:
        molecules, later, audit = _load_flat_context_candidates(
            args,
            queries,
            load_cache_policy=load_cache_policy,
            load_candidates=load_candidates,
        )
    else:
        molecules, later, audit = load_candidates(
            queries,
            task=args.task,
            subset=args.evaluation_subset,
            library=args.evidence_library,
            mapper=args.level_mapper,
            policy=policy,
            molecule_limit=args.l1_molecules,
            l1_limit=args.l1_records_per_molecule,
            later_limit=(
                {
                    level: args.record_limits_by_level[level]
                    for level in policy["stages"]
                    if level != "L1"
                }
                if policy.get("selection_contract") in {
                    "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1"
                }
                else args.records_per_level
            ),
            tie_seed=args.ranking_tie_seed,
            gold_context_mapping=(
                args.gold_context_mapping if args.task == "bbb_martins" else None
            ),
            allow_frozen_l1_vote_scores=args.allow_frozen_l1_vote_scores,
            cache_pool=RECORD_POOLS[args.record_pool],
        )
    run_root = source_batch / "runs"
    retrieval_hashes = {}
    molecule_descriptions, molecule_description_receipt = (
        _load_molecule_descriptions(
            args.molecule_description_mode,
            args.molecule_description_cache_version,
        )
        if args.prompt_version in CONTEXT_PROMPT_VERSIONS
        else ({}, {"mode": "none"})
    )
    description_missing: set[str] = set()
    for index in indices:
        record = records[index]
        query_id = str(record["benchmark_row_id"])
        retrieval = cache_matched_flat_retrieval(
            query_id,
            str(record["drug"]),
            molecules[query_id],
            later[query_id],
            task=args.task,
            reranking=args.reranking,
            query_audit=audit["query_audits"][query_id],
            query_identity=(audit.get("query_identities") or {}).get(query_id),
            prompt_version=args.prompt_version,
            selection_contract=audit["selection_policy"],
            neighbor_identity_policy_by_level=audit.get("neighbor_identity_policy_by_level"),
            preserve_physical_records=(
                bool(audit["contract"]["policy"].get("uid_retrieval", {}).get("cache_set"))
                or audit["contract"].get("all_later_records") is True
                or bool(args.flat_mixed_selection)
                or args.uids is not None or args.uids_file is not None
            ),
        )
        if args.prompt_version in FINAL_FULL_FLAT_PROMPT_VERSIONS:
            retrieval["benchmark"] = "tdc" if args.benchmark == "tdc" else "gold_v1"
        if audit.get("_tdc_selected_scores") is not None:
            _apply_tdc_scores(
                retrieval, audit["_tdc_selected_scores"][query_id],
                audit["_tdc_score_visibility"][query_id],
            )
        if audit.get("_gold_score_visibility") is not None:
            _apply_gold_score_visibility(
                retrieval, audit["_gold_score_visibility"][query_id],
            )
        description_missing.update(
            _attach_molecule_descriptions(retrieval, molecule_descriptions)
        )
        run_dir = run_root / f"{source_batch.name}_idx{index:05d}"
        path = run_dir / "retrieval.json"
        write_json_atomic(path, retrieval)

    contract = deepcopy(audit["contract"])
    contract.pop("inputs", None)
    contract["policy"] = deepcopy(contract["policy"])
    contract["policy"].pop("inputs", None)
    compact_audit = {
        "selection_policy": audit["selection_policy"],
        "inputs": audit["inputs"],
        "contract": contract,
        "query_audits": audit["query_audits"],
        "cache_pool": audit["cache_pool"],
        "neighbor_identity_policy": audit.get("neighbor_identity_policy"),
        "neighbor_identity_policy_by_level": audit.get(
            "neighbor_identity_policy_by_level"
        ),
        "similarity_floor": audit["similarity_floor"],
        "scaffold_overlap": audit["scaffold_overlap"],
        "parent_overlap": audit["parent_overlap"],
    }
    if audit["selection_policy"] in {CONTEXT_V4_SELECTION_CONTRACT, CONTEXT_V5_SELECTION_CONTRACT}:
        compact_audit.update(
            query_identities=audit["query_identities"],
            l1_cache=audit["l1_cache"],
            later_cache=audit["later_cache"],
        )
    elif audit["selection_policy"] in SQLITE_SELECTION_CONTRACTS:
        compact_audit.update(cache_capacities=audit["cache_capacities"],
                             query_identities=audit["query_identities"])
        if audit["selection_policy"] in {
            "ranked_level_retrieval.v2", "ranked_uid_retrieval.v1"
        }:
            compact_audit.update(
                cache_index=audit["cache_index"],
                cache_content_ids=audit["cache_content_ids"],
            )
        else:
            compact_audit.update(
                cache_version=audit["cache_version"],
                cache_content_id=audit["cache_content_id"],
                cache_assignment_counts=audit["cache_assignment_counts"],
                cache_record_count=audit["cache_record_count"],
            )
    else:
        compact_audit.update(
            v9=audit["v9"],
            l1_pool_size=audit["l1_pool_size"],
            gold_context_mapping=audit["gold_context_mapping"],
            library_compatibility=audit["library_compatibility"],
            cache_versions=[
                {
                    "path": path,
                    "schema_version": version.get("schema_version"),
                    "status": version.get("status"),
                    "morgan_pool_size": version.get("morgan_pool_size"),
                    "models": version.get("models"),
                }
                for path, version in audit["cache_versions"].items()
            ],
            mapped_level_counts=audit["mapped_level_counts"],
            unmapped_record_count=len(audit["unmapped_record_ids"]),
            unmapped_record_ids_sha256=hashlib.sha256(
                json.dumps(sorted(audit["unmapped_record_ids"])).encode("utf-8")
            ).hexdigest(),
        )
    manifest = {
        "organization": "flat",
        **({"bbb_evidence_policy": BBB_EVIDENCE_POLICY} if args.task == "bbb_martins" else {}),
        "run_name": getattr(args, "run_name", ""),
        "benchmark": args.benchmark,
        "prompt_version": args.prompt_version,
        "task": args.task,
        "evaluation_subset": args.evaluation_subset,
        "reranking": args.reranking,
        "record_pool": args.record_pool,
        "cache_pool": RECORD_POOLS[args.record_pool],
        "indices": indices,
        "input_jsonl": str(args.input_jsonl),
        "evidence_library": str(args.evidence_library),
        "level_mapper": str(args.level_mapper),
        "assay_transfer_cache": str(args.assay_transfer_cache),
        "gold_context_mapping": (
            str(args.gold_context_mapping) if args.task == "bbb_martins" else ""
        ),
        "allow_frozen_l1_vote_scores": args.allow_frozen_l1_vote_scores,
        "l1_molecules": args.l1_molecules,
        "l1_records_per_molecule": args.l1_records_per_molecule,
        "records_per_level": args.records_per_level,
        "record_limits_by_level": args.record_limits_by_level,
        "ranking_tie_seed": args.ranking_tie_seed,
        "max_level": args.max_level,
        "manual_uids": args.uids,
        "manual_uid_level": args.uid_level,
        "flat_preselected_uids": (
            str(args.flat_preselected_uids) if args.flat_preselected_uids else ""
        ),
        "flat_preselected_contexts": (
            str(args.flat_preselected_contexts) if args.flat_preselected_contexts else ""
        ),
        "flat_mixed_selection": str(args.flat_mixed_selection) if args.flat_mixed_selection else "",
        "selection_audit": compact_audit,
        "cache_set": cache_task(args.assay_transfer_cache, args.task, args.benchmark)["cache_set"],
        "benchmark_release": cache_task(args.assay_transfer_cache, args.task, args.benchmark)["benchmark"],
        "filtering_policy": compact_audit["neighbor_identity_policy_by_level"],
        "selection_method": "physical_uids" if args.uids is not None or args.uids_file is not None else "cached_ranking",
    }
    if artifact_receipt:
        manifest.update(
            artifact_manifest=str(args.flat_artifact_manifest),
            artifact_manifest_sha256=artifact_receipt["manifest_sha256"],
            artifact_receipt=str(args.flat_artifact_receipt),
        )
    if args.prompt_version in CONTEXT_PROMPT_VERSIONS:
        molecule_description_receipt.update(
            missing_policy="omit",
            missing_selected_count=len(description_missing),
            missing_selected_smiles=sorted(description_missing),
        )
        manifest.update(
            layout=args.layout,
            query_prior=args.query_prior,
            l1_min_contrast=args.l1_min_contrast,
            morgan_primary_parent_width=args.morgan_primary_parent_width,
            molecule_description_mode=args.molecule_description_mode,
            molecule_description_cache_version=args.molecule_description_cache_version,
            molecule_description=molecule_description_receipt,
            evidence_projection=EVIDENCE_PROJECTION,
            extra_details_policy=EXTRA_DETAILS_POLICY,
            molecule_name_visible=False,
            group_tools_enabled=False,
        )
    if args.prompt_version == JOSEPH_PROMPT_VERSION:
        manifest.update(
            evidence_projection=EVIDENCE_PROJECTION,
            extra_details_policy=EXTRA_DETAILS_POLICY,
            molecule_name_visible=False,
            group_tools_enabled=False,
        )
    write_batch(source_batch / "manifest.json", manifest,
                records=[(index, records[index]) for index in indices])
    return source_batch, manifest


def _selected_indices(args: argparse.Namespace, size: int) -> list[int]:
    if args.indices:
        output = []
        for token in args.indices:
            if "-" in token:
                start, end = token.split("-", 1)
                output.extend(range(int(start), int(end) + 1))
            else:
                output.append(int(token))
    else:
        end = size if args.limit == 0 else min(size, args.start + args.limit)
        output = list(range(args.start, end))
    output = sorted(dict.fromkeys(output))
    if not output or any(index < 0 or index >= size for index in output):
        raise SystemExit(f"Query indices must be within 0..{size - 1}")
    return output


def _preselected_query_indices(
    path: Path,
    records: list[Mapping[str, Any]],
    *,
    task: str,
    subset: str,
) -> list[int]:
    """Resolve an optimizer cohort to stable indices in the canonical split."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") not in {
            PRESELECTED_UID_SCHEMA, PRESELECTED_MIXED_SCHEMA, TDC_V2_MIXED_SCHEMA,
            DILI_TDC_V2_MIXED_SCHEMA,
            TDC_V2_BBB_FILTERED_SCHEMA,
            *PRESELECTED_DIRECT_SCHEMAS
        }
        or document.get("status") != "complete"
        or document.get("task_id") != task
        or document.get("subset") not in {subset, "valid_small"}
    ):
        raise ValueError(f"Incompatible preselected UID manifest: {path}")
    requested = [str(value) for value in document.get("benchmark_row_ids") or []]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Preselected UID manifest has invalid benchmark query IDs")
    positions = {
        str(record["benchmark_row_id"]): index for index, record in enumerate(records)
    }
    if len(positions) != len(records):
        raise ValueError("Canonical split repeats benchmark query IDs")
    missing = [query_id for query_id in requested if query_id not in positions]
    if missing:
        raise ValueError(f"Preselected UID queries are absent from the split: {missing[:5]}")
    return sorted(positions[query_id] for query_id in requested)


def _validate_query_prior_batch(
    prior_batch: Path,
    records: list[Mapping[str, Any]],
    indices: list[int],
) -> None:
    """Bind cached prior artifacts to current stable query identities."""
    overlay_path = prior_batch / "manifest.json"
    overlay = read_record(overlay_path) if overlay_path.is_file() else {}
    if overlay.get("queries") is not None:
        require_query_ids(overlay, records, indices)
    overlay_sources = (
        overlay.get("sources") or []
        if overlay.get("schema_version") == "branch_query_prior_overlay.v1"
        else None
    )
    if overlay_sources is not None:
        if (
            len(overlay_sources) != len(records)
        ):
            raise ValueError(f"Cached query-prior overlay differs from its input: {prior_batch}")
    for index in indices:
        run_dir = (
            Path(overlay_sources[index]["run_dir"])
            if overlay_sources is not None
            else run_directory(prior_batch, index)
        )
        if overlay_sources is not None:
            source = overlay_sources[index]
            key = (
                str(records[index].get("molecule_identity_key") or ""),
                str(records[index].get("condition_group") or ""),
            )
            if key != (source.get("molecule_identity_key"), source.get("condition_group")):
                raise ValueError(f"Cached query-prior overlay identity differs at index {index}")
        retrieval_path = run_dir / "retrieval.json"
        single_path = run_dir / "single_molecule_reasoning_output.json"
        if not retrieval_path.is_file() or not single_path.is_file():
            raise ValueError(f"Cached query prior is incomplete: {run_dir}")
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        current_smiles = str(records[index].get("drug") or "")
        cached_smiles = str((retrieval.get("query") or {}).get("input_smiles") or "")
        if cached_smiles != current_smiles:
            raise ValueError(
                f"Cached query prior identity differs at index {index}: "
                f"{cached_smiles!r} != {current_smiles!r}"
            )
        validate_query_prior_output(json.loads(single_path.read_text(encoding="utf-8")))


def mode_main(mode: str, argv: list[str] | None = None, *, launch_command: list[str] | None = None) -> int:
    """Load the shared launcher lazily so batch code can import flat contracts."""
    from predict.harnesses.branches.runner import mode_main as run_mode

    return run_mode(mode, argv, launch_command=launch_command)


def run(argv: list[str] | None = None) -> int:
    from predict.retrieval.assay_reranking.cache_matched import DEFAULT_GOLD_CONTEXT_MAPPING

    parser = argparse.ArgumentParser(description="Predict from frozen caches, explicit selections and prompt artifacts.", allow_abbrev=False)
    parser.add_argument("--prompt", dest="prompt_version", choices=sorted(FINAL_FULL_FLAT_PROMPT_VERSIONS), default=FINAL_FULL_FLAT_PROMPT_VERSION)
    parser.add_argument("--task", required=True, choices=tuple(TASKS))
    parser.add_argument("--benchmark", choices=("gold", "tdc"), default="gold")
    parser.add_argument("--evaluation-subset", choices=("valid", "test"), default="valid")
    parser.add_argument("--retrieval-cache", dest="assay_transfer_cache", type=Path, metavar="CACHE_SET")
    parser.add_argument("--reranking", choices=(MORGAN_VARIANT, ASSAY_TRANSFER_VARIANT), default=ASSAY_TRANSFER_VARIANT)
    parser.add_argument("--layout", choices=CONTEXT_V4_LAYOUTS, default="global")
    parser.add_argument("--query-prior", choices=("cached", "none"), default="cached")
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--provider-pool-config", type=Path, default=DEFAULT_PROVIDER_CONFIG)
    parser.add_argument("--flat-artifact-manifest", type=Path)
    parser.add_argument("--flat-artifact-receipt", type=Path)
    parser.add_argument("--verify-flat-artifacts", action="store_true")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--evidence-library", type=Path)
    parser.add_argument("--level-mapper", type=Path, default=Path("data/evidence_libraries/level_mappings.v1.json"))
    parser.add_argument("--gold-context-mapping", type=Path, default=DEFAULT_GOLD_CONTEXT_MAPPING)
    parser.add_argument("--allow-frozen-l1-vote-scores", action="store_true")
    parser.add_argument("--l1-molecules", type=int, default=10)
    parser.add_argument("--l1-records-per-molecule", type=int, default=10)
    parser.add_argument("--records-per-level", type=int, default=50)
    parser.add_argument("--level-record-limit", action="append", default=[], metavar="LEVEL=K")
    parser.add_argument("--ranking-tie-seed", type=int, default=0)
    parser.add_argument("--max-level", type=int, default=0)
    manual = parser.add_mutually_exclusive_group()
    manual.add_argument("--uids", nargs="+")
    manual.add_argument("--uids-file", type=Path)
    parser.add_argument("--uid-level")
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=PREDICTION_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--batch-id")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices", nargs="*")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--molecule-description-mode", choices=("none", *MOLECULE_DESCRIPTION_COLUMNS), default="none")
    parser.add_argument("--molecule-description-cache-version", choices=tuple(MOLECULE_DESCRIPTION_ARTIFACTS), default="v1")
    raw = list(sys.argv[1:] if argv is None else argv)
    args, remaining = parser.parse_known_args(raw)
    if args.batch_root is not None:
        args.batch_root = canonical_location(args.batch_root)
        if args.run_name:
            parser.error("--batch-root is an exact legacy override; use --output-root with --run-name")
    elif not args.run_name:
        parser.error("New prediction runs require --run-name")
    if args.run_name:
        campaign = prediction_run_root(args.output_root, "flat", args.prompt_version, args.run_name)
        batch = prediction_batch_directory(campaign, args.benchmark, args.task, args.evaluation_subset)
        args.batch_root, args.batch_id = batch.parent, batch.name
    supplied = {value.split("=", 1)[0] for value in raw}
    saved = None
    if args.batch_root is not None and args.batch_id:
        previous = canonical_location(args.batch_root / args.batch_id / "cache_matched_retrieval/manifest.json")
        if previous.exists():
            from predict.harnesses.branches.manifests import resolve_artifact
            saved = read_record(previous)
            if "--retrieval-cache" not in supplied:
                pinned = resolve_artifact(saved, "assay_transfer_cache", previous.parent)
                if pinned is None or not pinned.is_file():
                    parser.error("Saved cache identity is unavailable; use a new run identity")
                args.assay_transfer_cache = pinned
            for key, flag in (("prompt_version", "--prompt"), ("benchmark", "--benchmark"),
                              ("evaluation_subset", "--evaluation-subset"), ("reranking", "--reranking"),
                              ("layout", "--layout"), ("query_prior", "--query-prior"),
                              ("max_level", "--max-level"), ("l1_molecules", "--l1-molecules"),
                              ("l1_records_per_molecule", "--l1-records-per-molecule"),
                              ("records_per_level", "--records-per-level"), ("ranking_tie_seed", "--ranking-tie-seed")):
                if flag not in supplied and key in saved:
                    setattr(args, key, saved[key])
            if not supplied.intersection({"--indices", "--start", "--limit"}):
                args.indices = list(map(str, saved["indices"]))
            if not supplied.intersection({"--uids", "--uids-file"}) and saved.get("manual_uids"):
                args.uids, args.uid_level = saved["manual_uids"], saved["manual_uid_level"]
    forbidden = {"--harness-version", "--organization", "--assay-transfer-cache", "--flat-prompt-version",
                 "--experiment-mode", "--retrieval-strategy", "--retrieval-source", "--index",
                 "--neighbor-identity-policy", "--neighbor-selector", "--morgan-neighbor-selector",
                 "--flat-preselected-uids", "--flat-preselected-contexts", "--flat-mixed-selection", "--record_pool", "-record_pool"}
    rejected = sorted({token.split("=", 1)[0] for token in remaining} & forbidden)
    if rejected:
        parser.error("Use --prompt, --retrieval-cache and --uids-file; retired selectors: " + ", ".join(rejected))
    if args.uid_level and not args.uids:
        parser.error("--uid-level applies only to --uids")
    if bool(args.flat_artifact_manifest) != bool(args.flat_artifact_receipt):
        parser.error("pass both --flat-artifact-manifest and --flat-artifact-receipt")
    if min(args.l1_molecules, args.l1_records_per_molecule, args.records_per_level) < 1 or min(args.start, args.limit, args.max_level) < 0:
        parser.error("selection budgets must be positive; start, limit and max-level must be non-negative")
    try:
        args.assay_transfer_cache = resolve_cache_set(args.task, args.benchmark, args.assay_transfer_cache)
        task_max_level = cache_level_limit(args.assay_transfer_cache, args.task, args.benchmark)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.max_level > task_max_level:
        parser.error(f"{args.task} cache ends at L{task_max_level}")
    args.max_level = args.max_level or task_max_level
    overrides = dict(saved.get("record_limits_by_level", {})) if saved and "--level-record-limit" not in supplied else {}
    for value in args.level_record_limit:
        level, separator, raw = value.partition("=")
        try:
            limit = int(raw)
        except ValueError:
            parser.error("--level-record-limit requires LEVEL=K")
        if not separator or level not in {f"L{i}" for i in range(2, args.max_level + 1)} or level in overrides or limit < 1:
            parser.error("--level-record-limit requires unique supported levels and positive budgets")
        overrides[level] = limit
    args.record_limits_by_level = {f"L{i}": overrides.get(f"L{i}", args.records_per_level) for i in range(2, args.max_level + 1)}
    canonical = (tdc_split_path(args.task, args.evaluation_subset) if args.benchmark == "tdc" else split_path(args.task, args.evaluation_subset)).with_name(f"{args.evaluation_subset}_molecule_condition_labels.jsonl").resolve()
    if args.input_jsonl is not None and args.input_jsonl.resolve() != canonical:
        parser.error("Prediction accepts only the declared benchmark split")
    args.input_jsonl = canonical
    spec = cache_task(args.assay_transfer_cache, args.task, args.benchmark)
    from data.processing.paths import REPO_ROOT
    index_path = REPO_ROOT / spec["retrieval"]["later"]
    index = json.loads(index_path.read_text())
    evidence = (index_path.parent / index["evidence"]["manifest"]).resolve()
    library = next(parent for parent in evidence.parents if parent.parent.name == args.task)
    args.evidence_library = (args.evidence_library or library).resolve()
    args.level_mapper = args.level_mapper.resolve()
    args.gold_context_mapping = args.gold_context_mapping.resolve()
    args.prior_root = (args.prior_root or (
        default_query_prior_root(args.prompt_version, args.benchmark, args.evaluation_subset)
        if args.query_prior == "cached" else DEFAULT_QUERY_PRIOR_ROOT
    )).resolve()
    if args.run_name:
        if "--trace-root" not in supplied:
            remaining += ["--trace-root", str(campaign / "traces")]
    else:
        args.batch_root = args.batch_root.resolve()
        args.batch_id = args.batch_id or "default"
    args.record_pool, args.l1_min_contrast, args.morgan_primary_parent_width = "all", 0, 100
    args.flat_preselected_uids = args.flat_preselected_contexts = args.flat_mixed_selection = None
    source_batch, manifest = _materialize_cache_matched_retrievals(args)
    prior_batch = None
    if args.query_prior == "cached":
        prior_batch = query_prior_batch(args.prior_root, args.task, args.evaluation_subset, args.benchmark)
        _validate_query_prior_batch(prior_batch, read_jsonl(args.input_jsonl), manifest["indices"])
    if args.prepare_only:
        print(json.dumps({"retrieval_source": str(source_batch), "manifest": manifest}, indent=2))
        return 0
    if "--max-tokens" not in {token.split("=", 1)[0] for token in remaining}:
        parser.error("Prediction requires an explicit reviewed --max-tokens")
    forwarded = [*remaining, "--provider-pool-config", str(args.provider_pool_config),
                 "--task", args.task, "--input-jsonl", str(args.input_jsonl),
                 "--batch-root", str(args.batch_root), "--batch-id", args.batch_id,
                 "--retrieval-replay-source-batch", str(source_batch),
                 "--flat-selection-manifest", str(source_batch / "manifest.json"),
                 "--flat-prompt-version", args.prompt_version, "--flat-reranking", args.reranking,
                 "--flat-layout", args.layout, "--flat-query-prior", args.query_prior,
                 "--retrieval-strategy", "morgan_fingerprint", "--neighbor-identity-policy", "scaffold_disjoint",
                 "--min-similarity", "0", "--disable-flat-tools"]
    if args.run_name:
        forwarded += ["--run-name", args.run_name]
    if prior_batch is not None:
        forwarded += ["--single-analysis-source-batch", str(prior_batch)]
    forwarded += ["--indices", *args.indices] if args.indices else ["--start", str(args.start), "--limit", str(args.limit)]
    return mode_main("full_flat", forwarded, launch_command=[
        sys.executable, "-m", "predict.harnesses.branches",
        *(sys.argv[1:] if argv is None else argv),
    ])
