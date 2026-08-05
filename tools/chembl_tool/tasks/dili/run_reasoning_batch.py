"""Run DILI reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    BatchConfig,
    compute_metrics as _compute_metrics,
    main as run_batch,
    prediction_to_label as _prediction_to_label,
)


CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input="data/processed/DILI/test.jsonl",
    default_batch_root="outputs/chembl_tool/tasks/dili/reasoning/batches",
    default_index="outputs/chembl_tool/tasks/dili/evidence_library/dili_neighbor_index.pkl",
    default_model="deepseek-v4-pro",
    batch_id_prefix="dili_batch",
    pipeline_module="tools.chembl_tool.tasks.dili.run_reasoning_pipeline",
    log_prefix="dili_reasoning_batch",
    report_title="DILI Batch Report",
    prediction_field="dili_prediction",
    canonical_positive="dili_risk",
    canonical_negative="no_dili_risk",
    positive_predictions=frozenset({"dili_risk", "positive", "dili_positive", "hepatotoxic", "hepatotoxicity", "1"}),
    negative_predictions=frozenset({"no_dili_risk", "negative", "dili_negative", "non_hepatotoxic", "non-hepatotoxic", "0"}),
    supports_shared_retrieval_contract=False,
)


def compute_metrics(rows: list[dict]) -> dict:
    return _compute_metrics(CONFIG, rows)


def prediction_to_label(prediction: str | None) -> int | None:
    return _prediction_to_label(CONFIG, prediction)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
