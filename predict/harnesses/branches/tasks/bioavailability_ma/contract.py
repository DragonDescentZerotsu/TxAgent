"""Task profiles and shared FINAL batch configuration."""

from __future__ import annotations

from functools import partial

from predict.harnesses.branches.runner import BatchConfig, main as run_batch

from predict.retrieval.assay_reranking.v9 import default_cache_paths, model_profile, preflight_cache_coverage

from predict.prompts import task_profiles

_PROFILE_DATA = task_profiles("bioavailability_ma")
DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE = _PROFILE_DATA["default"]
HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE = _PROFILE_DATA["historical"]
BIOAVAILABILITY_PROMPT_PROFILES = tuple(_PROFILE_DATA["profiles"])

CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input="data/gold_labels/legacy/processed/Bioavailability_Ma/test.jsonl",
    default_batch_root="outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches",
    default_index="outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl",
    default_model="deepseek-v4-pro",
    batch_id_prefix="bioavailability_batch",
    pipeline_module="predict.harnesses.branches.tasks.bioavailability_ma.pipeline",
    log_prefix="bioavailability_reasoning_batch",
    report_title="Bioavailability_Ma Batch Report",
    prediction_field="bioavailability_prediction",
    canonical_positive="high",
    canonical_negative="low",
    positive_predictions=frozenset({"high", "pass", "positive", "bioavailability_positive", "1"}),
    negative_predictions=frozenset({"low", "fail", "negative", "bioavailability_negative", "0"}),
    rerank_preflight=partial(preflight_cache_coverage, task_id="bioavailability_ma"),
    supports_assay_transfer_scores=True,
    supports_retrieval_strategy=True,
    assay_transfer_profile_default="v9_direct_gold",
    rerank_catalog_default=default_cache_paths("bioavailability_ma")["catalog"],
    rerank_cache_default=default_cache_paths("bioavailability_ma")["cache"],
    rerank_candidate_manifest_default=default_cache_paths("bioavailability_ma")[
        "candidate_manifest"
    ],
    rerank_version_manifest_default=default_cache_paths("bioavailability_ma")["version"],
    assay_transfer_model_default=model_profile("bioavailability_ma")["model"],
    assay_transfer_model_revision_default=model_profile("bioavailability_ma")["revision"],
    assay_transfer_template_profile_default="v9_context_conditioned",
    v9_rerank_catalog_default=default_cache_paths("bioavailability_ma")["catalog"],
    v9_rerank_cache_default=default_cache_paths("bioavailability_ma")["cache"],
    v9_rerank_candidate_manifest_default=default_cache_paths("bioavailability_ma")[
        "candidate_manifest"
    ],
    v9_rerank_version_manifest_default=default_cache_paths("bioavailability_ma")["version"],
    v9_assay_transfer_model_default=model_profile("bioavailability_ma")["model"],
    v9_assay_transfer_model_revision_default=model_profile("bioavailability_ma")["revision"],
    v9_index_default=(
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
        "starling_normalized_v7/08_neighbor_index/scaffold"
    ),
    prompt_profile_option="--bioavailability-prompt-profile",
    prompt_profile_choices=BIOAVAILABILITY_PROMPT_PROFILES,
    default_prompt_profile=DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    historical_prompt_profile=HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE,
)

def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
