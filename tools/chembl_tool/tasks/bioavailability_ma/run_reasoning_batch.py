"""Run Bioavailability_Ma reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

from functools import partial

from tools.chembl_tool.common.task_workflows.reasoning_batch import BatchConfig, main as run_batch
from tools.chembl_tool.common.assay_reranking.v11 import (
    default_cache_paths,
    model_profile,
    preflight_cache_coverage as preflight_v11_cache_coverage,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.precompute_assay_transfer_rerank import (
    preflight_cache_coverage as preflight_legacy_cache_coverage,
)
from tools.chembl_tool.tasks.bioavailability_ma.group_prompt_render import (
    DEFAULT_GROUP_PROMPT_VERSION,
    GROUP_PROMPT_VERSIONS,
    final_prompt_provenance,
    group_prompt_provenance,
)


LEGACY_PREFLIGHT = preflight_legacy_cache_coverage
V11_PREFLIGHT = partial(preflight_v11_cache_coverage, task_id="bioavailability_ma")


def rerank_preflight(**kwargs):
    profile = kwargs.pop("assay_transfer_profile", "legacy_bio")
    return (V11_PREFLIGHT if profile == "v11_with_categorical" else LEGACY_PREFLIGHT)(
        **kwargs
    )


from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    BIOAVAILABILITY_PROMPT_PROFILES,
    DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE,
)


CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input="data/processed/Bioavailability_Ma/test.jsonl",
    default_batch_root="outputs/chembl_tool/tasks/bioavailability_ma/reasoning/batches",
    default_index="outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/bioavailability_neighbor_index.pkl",
    default_model="deepseek-v4-pro",
    batch_id_prefix="bioavailability_batch",
    pipeline_module="tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_pipeline",
    log_prefix="bioavailability_reasoning_batch",
    report_title="Bioavailability_Ma Batch Report",
    prediction_field="bioavailability_prediction",
    canonical_positive="high",
    canonical_negative="low",
    positive_predictions=frozenset({"high", "pass", "positive", "bioavailability_positive", "1"}),
    negative_predictions=frozenset({"low", "fail", "negative", "bioavailability_negative", "0"}),
    rerank_preflight=rerank_preflight,
    supports_assay_transfer_scores=True,
    supports_retrieval_strategy=True,
    supports_analogous_reasoning_only=True,
    analogous_reasoning_modes=("full_flat", "full_mechanism"),
    final_prompt_provenance=final_prompt_provenance,
    group_prompt_formats=("legacy", "morganfingerprint", "assay_transfer_tool"),
    default_group_prompt_format="legacy",
    group_output_schemas=("legacy", "assay-transfer"),
    default_group_output_schema="legacy",
    group_prompt_versions=GROUP_PROMPT_VERSIONS,
    default_group_prompt_version=DEFAULT_GROUP_PROMPT_VERSION,
    group_prompt_provenance=group_prompt_provenance,
    v11_rerank_catalog_default=default_cache_paths("bioavailability_ma")["catalog"],
    v11_rerank_cache_default=default_cache_paths("bioavailability_ma")["cache"],
    v11_rerank_candidate_manifest_default=default_cache_paths("bioavailability_ma")[
        "candidate_manifest"
    ],
    v11_rerank_version_manifest_default=default_cache_paths("bioavailability_ma")["version"],
    v11_assay_transfer_model_default=model_profile("bioavailability_ma")["model"],
    v11_assay_transfer_model_revision_default=model_profile("bioavailability_ma")["revision"],
    v11_index_default=(
        "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
        "starling_normalized_v7/08_neighbor_index/scaffold"
    ),
    supports_final_decision_profiles=True,
    prompt_profile_option="--bioavailability-prompt-profile",
    prompt_profile_choices=BIOAVAILABILITY_PROMPT_PROFILES,
    default_prompt_profile=DEFAULT_BIOAVAILABILITY_PROMPT_PROFILE,
    historical_prompt_profile=HISTORICAL_BIOAVAILABILITY_PROMPT_PROFILE,
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
