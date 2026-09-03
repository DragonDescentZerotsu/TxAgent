"""Run BBB Martins reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

from functools import partial

from predict.retrieval.assay_reranking.v9 import (
    default_cache_paths,
    model_profile,
    preflight_cache_coverage,
)
from predict.retrieval.assay_reranking.v19_1 import (
    GROUP_IDS as V19_1_GROUP_IDS,
    default_cache_paths as v19_1_default_cache_paths,
    model_profile as v19_1_model_profile,
)
from predict.harnesses.branches.batch import (
    BatchConfig,
    compute_metrics as _compute_metrics,
    main as run_batch,
    prediction_to_label as _prediction_to_label,
)
from predict.harnesses.branches.reasoning.final_decision import (
    GENERAL_FINAL_DECISION_PROFILES,
)
from predict.tasks.bbb_martins.prompts import (
    BBB_PROMPT_PROFILES,
    DEFAULT_BBB_PROMPT_PROFILE,
    HISTORICAL_BBB_PROMPT_PROFILE,
)


CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input="data/gold_labels/legacy/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl",
    default_batch_root="outputs/chembl_tool/tasks/bbb_martins/reasoning/batches",
    default_index="outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl",
    default_model="deepseek-v4-pro",
    batch_id_prefix="bbb_batch",
    pipeline_module="predict.harnesses.branches.tasks.bbb_martins.pipeline",
    log_prefix="bbb_reasoning_batch",
    report_title="BBB Martins Batch Report",
    prediction_field="bbb_prediction",
    canonical_positive="pass",
    canonical_negative="fail",
    positive_predictions=frozenset({"pass", "positive", "bbb+", "bbb_positive", "1"}),
    negative_predictions=frozenset({"fail", "negative", "bbb-", "bbb_negative", "0"}),
    rerank_preflight=partial(preflight_cache_coverage, task_id="bbb_martins"),
    supports_assay_transfer_scores=True,
    supports_retrieval_strategy=True,
    supports_analogous_reasoning_only=True,
    analogous_reasoning_modes=("full_flat",),
    group_prompt_formats=("legacy", "assay_transfer_tool"),
    default_group_prompt_format="legacy",
    assay_transfer_profile_default="v9_direct_gold",
    rerank_catalog_default=default_cache_paths("bbb_martins")["catalog"],
    rerank_cache_default=default_cache_paths("bbb_martins")["cache"],
    rerank_candidate_manifest_default=default_cache_paths("bbb_martins")[
        "candidate_manifest"
    ],
    rerank_version_manifest_default=default_cache_paths("bbb_martins")["version"],
    assay_transfer_model_default=model_profile("bbb_martins")["model"],
    assay_transfer_model_revision_default=model_profile("bbb_martins")["revision"],
    v9_rerank_catalog_default=default_cache_paths("bbb_martins")["catalog"],
    v9_rerank_cache_default=default_cache_paths("bbb_martins")["cache"],
    v9_rerank_candidate_manifest_default=default_cache_paths("bbb_martins")[
        "candidate_manifest"
    ],
    v9_rerank_version_manifest_default=default_cache_paths("bbb_martins")["version"],
    v9_assay_transfer_model_default=model_profile("bbb_martins")["model"],
    v9_assay_transfer_model_revision_default=model_profile("bbb_martins")["revision"],
    v9_index_default=(
        "outputs/paper/molecular_evidence_agent_starling_scaffold_"
        "experimental_meaningful_cns_access_v3/evidence/"
        "bbb_starling_v7/08_neighbor_index"
    ),
    v19_1_rerank_cache_default=v19_1_default_cache_paths("bbb_martins")["cache"],
    v19_1_rerank_version_manifest_default=v19_1_default_cache_paths("bbb_martins")[
        "version"
    ],
    v19_1_assay_transfer_model_default=v19_1_model_profile("bbb_martins")["model"],
    v19_1_assay_transfer_model_revision_default=v19_1_model_profile("bbb_martins")[
        "revision"
    ],
    v19_1_index_default=(
        "outputs/paper/molecular_evidence_agent_starling_scaffold_conditioned_benchmark/"
        "evidence/bbb_starling_v7/08_neighbor_index"
    ),
    v19_1_groups=V19_1_GROUP_IDS,
    supports_final_decision_profiles=True,
    final_decision_profile_choices=GENERAL_FINAL_DECISION_PROFILES,
    prompt_profile_option="--bbb-prompt-profile",
    prompt_profile_choices=BBB_PROMPT_PROFILES,
    default_prompt_profile=DEFAULT_BBB_PROMPT_PROFILE,
    historical_prompt_profile=HISTORICAL_BBB_PROMPT_PROFILE,
)


def compute_metrics(rows: list[dict]) -> dict:
    return _compute_metrics(CONFIG, rows)


def prediction_to_label(prediction: str | None) -> int | None:
    return _prediction_to_label(CONFIG, prediction)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
