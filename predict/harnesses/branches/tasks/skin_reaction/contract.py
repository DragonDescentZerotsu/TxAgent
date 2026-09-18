"""Define Skin branch retrieval/context policy and shared-runner configuration."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from predict.retrieval.assay_reranking.v9 import (
    default_cache_paths,
    model_profile,
    preflight_cache_coverage,
)
from predict.harnesses.branches.runner import BatchConfig, main as run_batch
from predict.harnesses.branches.exact_context import (
    DEFAULT_CHEMBL_SQLITE,
    QUERY_ACTIVITY_FIELDS,
    attach_shared_assay_context,
    exact_evidence_rows,
    load_query_activities_for_assays,
    lookup_exact_molecule,
    enrich_retrieval_with_chembl_context as enrich_common_retrieval_with_chembl_context,
)
from predict.harnesses.branches.retrieval import BranchDefinition, BranchRetrievalConfig
from predict.tasks.skin_reaction.prompts import (
    DEFAULT_SKIN_PROMPT_PROFILE,
    HISTORICAL_SKIN_PROMPT_PROFILE,
    SKIN_PROMPT_PROFILES,
)


CHEMBL = BranchRetrievalConfig(
    source_name="chembl",
    direct_groups=(
        BranchDefinition(
            "Direct.skin_reaction", "Tier 1", "direct_skin_reaction",
            source_group_prefixes=("Tier 1.",),
            exclude_source_groups=("Tier 1.context_dependent",),
        ),
    ),
    mechanism_groups=tuple(
        BranchDefinition(
            f"Mechanism.tier_{tier}", f"Tier {tier}",
            {1: "direct_skin_reaction", 2: "sensitisation_aop", 3: "phototoxicity_irritation_local_damage", 4: "skin_exposure"}[tier],
            source_group_prefixes=(f"Tier {tier}.",),
        )
        for tier in range(1, 5)
    ),
)

STARLING = BranchRetrievalConfig(
    source_name="starling",
    direct_groups=(
        BranchDefinition("Direct.skin_reaction", "Tier 1", "direct_skin_reaction", source_groups=("Direct.skin_reaction",)),
    ),
    mechanism_groups=(
        BranchDefinition("Mechanism.tier_1", "Tier 1", "direct_skin_reaction", source_groups=("Direct.skin_reaction",)),
        BranchDefinition("Mechanism.tier_2", "Tier 2", "sensitisation_aop", source_groups=("Mechanism.sensitization_aop",)),
    ),
)

SOURCES = {"chembl": CHEMBL, "starling": STARLING}


def get_source_config(source: str) -> BranchRetrievalConfig:
    return SOURCES[source]


def enrich_retrieval_with_chembl_context(
    retrieval: dict[str, Any],
    index: dict[str, Any],
    *,
    chembl_sqlite: str | Path = DEFAULT_CHEMBL_SQLITE,
    max_exact_skin_reaction_rows: int = 30,
    max_shared_per_neighbor: int = 8,
) -> dict[str, Any]:
    return enrich_common_retrieval_with_chembl_context(
        retrieval,
        index,
        relevant_evidence_key="skin_reaction_relevant_evidence_rows",
        context_label="Skin_Reaction",
        chembl_sqlite=chembl_sqlite,
        max_exact_rows=max_exact_skin_reaction_rows,
        max_shared_per_neighbor=max_shared_per_neighbor,
    )


def exact_skin_reaction_evidence_rows(
    index: dict[str, Any], molecule_chembl_id: str, *, max_rows: int = 30
) -> list[dict[str, Any]]:
    return exact_evidence_rows(index, molecule_chembl_id, max_rows=max_rows)


CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input="data/gold_labels/legacy/processed/Skin_Reaction/test.jsonl",
    default_batch_root="outputs/chembl_tool/tasks/skin_reaction/reasoning/batches",
    default_index="outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_neighbor_index.pkl",
    default_model="deepseek-v4-pro",
    batch_id_prefix="skin_reaction_batch",
    pipeline_module="predict.harnesses.branches.tasks.skin_reaction.pipeline",
    log_prefix="skin_reaction_reasoning_batch",
    report_title="Skin_Reaction Batch Report",
    prediction_field="skin_reaction_prediction",
    canonical_positive="risk",
    canonical_negative="no_risk",
    positive_predictions=frozenset({"risk", "positive", "skin_reaction_positive", "sensitizer", "irritant", "phototoxic", "1"}),
    negative_predictions=frozenset({"no_risk", "negative", "skin_reaction_negative", "non_sensitizer", "non_irritant", "0"}),
    rerank_preflight=partial(preflight_cache_coverage, task_id="skin_reaction"),
    supports_assay_transfer_scores=True,
    supports_retrieval_strategy=True,
    supports_analogous_reasoning_only=True,
    analogous_reasoning_modes=("full_flat",),
    group_prompt_formats=("legacy", "assay_transfer_tool"),
    default_group_prompt_format="legacy",
    assay_transfer_profile_default="v9_direct_gold",
    rerank_catalog_default=default_cache_paths("skin_reaction")["catalog"],
    rerank_cache_default=default_cache_paths("skin_reaction")["cache"],
    rerank_candidate_manifest_default=default_cache_paths("skin_reaction")[
        "candidate_manifest"
    ],
    rerank_version_manifest_default=default_cache_paths("skin_reaction")["version"],
    assay_transfer_model_default=model_profile("skin_reaction")["model"],
    assay_transfer_model_revision_default=model_profile("skin_reaction")["revision"],
    assay_transfer_template_profile_default="v9_context_conditioned",
    v9_rerank_catalog_default=default_cache_paths("skin_reaction")["catalog"],
    v9_rerank_cache_default=default_cache_paths("skin_reaction")["cache"],
    v9_rerank_candidate_manifest_default=default_cache_paths("skin_reaction")[
        "candidate_manifest"
    ],
    v9_rerank_version_manifest_default=default_cache_paths("skin_reaction")["version"],
    v9_assay_transfer_model_default=model_profile("skin_reaction")["model"],
    v9_assay_transfer_model_revision_default=model_profile("skin_reaction")["revision"],
    v9_index_default=(
        "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
        "starling_normalized_v7/08_neighbor_index/scaffold"
    ),
    prompt_profile_option="--skin-prompt-profile",
    prompt_profile_choices=SKIN_PROMPT_PROFILES,
    default_prompt_profile=DEFAULT_SKIN_PROMPT_PROFILE,
    historical_prompt_profile=HISTORICAL_SKIN_PROMPT_PROFILE,
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
