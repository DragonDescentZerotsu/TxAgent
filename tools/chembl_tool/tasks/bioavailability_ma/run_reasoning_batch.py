"""Run Bioavailability_Ma reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.reasoning_batch import BatchConfig, main as run_batch
from tools.chembl_tool.tasks.bioavailability_ma.reranking.precompute_assay_transfer_rerank import (
    preflight_cache_coverage,
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
    rerank_preflight=preflight_cache_coverage,
    supports_assay_transfer_scores=True,
    supports_retrieval_strategy=True,
    supports_nondirect_bioavailability_filter=True,
    group_prompt_formats=("legacy", "morganfingerprint", "assay_transfer_tool"),
    default_group_prompt_format="legacy",
    group_output_schemas=("legacy", "assay-transfer"),
    default_group_output_schema="legacy",
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
