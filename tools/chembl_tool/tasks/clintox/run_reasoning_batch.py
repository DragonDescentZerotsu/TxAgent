"""Run ClinTox reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.reasoning_batch import BatchConfig, main as run_batch


CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input="data/processed/ClinTox/test.jsonl",
    default_batch_root="outputs/chembl_tool/tasks/clintox/reasoning/batches",
    default_index="outputs/chembl_tool/tasks/clintox/evidence_library/clintox_neighbor_index.pkl",
    default_model="deepseek-v4-pro",
    batch_id_prefix="clintox_batch",
    pipeline_module="tools.chembl_tool.tasks.clintox.run_reasoning_pipeline",
    log_prefix="clintox_reasoning_batch",
    report_title="ClinTox Batch Report",
    prediction_field="clintox_prediction",
    canonical_positive="toxic",
    canonical_negative="non_toxic",
    positive_predictions=frozenset({"toxic", "tox", "positive", "clintox_positive", "ct_tox", "1"}),
    negative_predictions=frozenset({"non_toxic", "nontoxic", "non-toxic", "negative", "clintox_negative", "0"}),
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
