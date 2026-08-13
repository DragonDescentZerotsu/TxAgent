"""Run Skin_Reaction reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

from functools import partial

from tools.chembl_tool.common.assay_reranking.v11 import (
    default_cache_paths,
    model_profile,
    preflight_cache_coverage,
)
from tools.chembl_tool.common.task_workflows.reasoning_batch import BatchConfig, main as run_batch
from tools.chembl_tool.tasks.skin_reaction.prompt_profiles import (
    DEFAULT_SKIN_PROMPT_PROFILE,
    HISTORICAL_SKIN_PROMPT_PROFILE,
    SKIN_PROMPT_PROFILES,
)


CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input="data/processed/Skin_Reaction/test.jsonl",
    default_batch_root="outputs/chembl_tool/tasks/skin_reaction/reasoning/batches",
    default_index="outputs/chembl_tool/tasks/skin_reaction/evidence_library/skin_reaction_neighbor_index.pkl",
    default_model="deepseek-v4-pro",
    batch_id_prefix="skin_reaction_batch",
    pipeline_module="tools.chembl_tool.tasks.skin_reaction.run_reasoning_pipeline",
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
    group_prompt_formats=("legacy", "assay_transfer_tool"),
    default_group_prompt_format="legacy",
    assay_transfer_profile_default="v11_with_categorical",
    rerank_catalog_default=default_cache_paths("skin_reaction")["catalog"],
    rerank_cache_default=default_cache_paths("skin_reaction")["cache"],
    rerank_candidate_manifest_default=default_cache_paths("skin_reaction")[
        "candidate_manifest"
    ],
    rerank_version_manifest_default=default_cache_paths("skin_reaction")["version"],
    assay_transfer_model_default=model_profile("skin_reaction")["model"],
    assay_transfer_model_revision_default=model_profile("skin_reaction")["revision"],
    assay_transfer_template_profile_default="v11_query_context_copy",
    v11_rerank_catalog_default=default_cache_paths("skin_reaction")["catalog"],
    v11_rerank_cache_default=default_cache_paths("skin_reaction")["cache"],
    v11_rerank_candidate_manifest_default=default_cache_paths("skin_reaction")[
        "candidate_manifest"
    ],
    v11_rerank_version_manifest_default=default_cache_paths("skin_reaction")["version"],
    v11_assay_transfer_model_default=model_profile("skin_reaction")["model"],
    v11_assay_transfer_model_revision_default=model_profile("skin_reaction")["revision"],
    v11_index_default=(
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
