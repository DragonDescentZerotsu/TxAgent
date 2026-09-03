"""Run Skin_Reaction reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

from functools import partial

from predict.retrieval.assay_reranking.v9 import (
    default_cache_paths,
    model_profile,
    preflight_cache_coverage,
)
from predict.harnesses.branches.batch import BatchConfig, main as run_batch
from predict.tasks.skin_reaction.prompts import (
    DEFAULT_SKIN_PROMPT_PROFILE,
    HISTORICAL_SKIN_PROMPT_PROFILE,
    SKIN_PROMPT_PROFILES,
)


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
