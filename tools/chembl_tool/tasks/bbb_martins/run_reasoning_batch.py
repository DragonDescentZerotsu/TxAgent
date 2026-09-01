"""Run BBB Martins reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.reasoning_batch import (
    BatchConfig,
    compute_metrics as _compute_metrics,
    main as run_batch,
    prediction_to_label as _prediction_to_label,
)
from tools.chembl_tool.tasks.bbb_martins.prompt_profiles import (
    BBB_PROMPT_PROFILES,
    DEFAULT_BBB_PROMPT_PROFILE,
    HISTORICAL_BBB_PROMPT_PROFILE,
)


CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input="data/processed/BBB_Martins/B3DB_cleaned/test/test_efflux.jsonl",
    default_batch_root="outputs/chembl_tool/tasks/bbb_martins/reasoning/batches",
    default_index="outputs/chembl_tool/tasks/bbb_martins/evidence_library/bbb_neighbor_index.pkl",
    default_model="deepseek-v4-pro",
    batch_id_prefix="bbb_batch",
    pipeline_module="tools.chembl_tool.tasks.bbb_martins.run_reasoning_pipeline",
    log_prefix="bbb_reasoning_batch",
    report_title="BBB Martins Batch Report",
    prediction_field="bbb_prediction",
    canonical_positive="pass",
    canonical_negative="fail",
    positive_predictions=frozenset({"pass", "positive", "bbb+", "bbb_positive", "1"}),
    negative_predictions=frozenset({"fail", "negative", "bbb-", "bbb_negative", "0"}),
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
