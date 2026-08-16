"""Run ClinTox reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.reasoning_batch import BatchConfig, main as run_batch
from tools.chembl_tool.tasks.clintox.prompt_profiles import (
    CLINTOX_PROMPT_PROFILES,
    DEFAULT_CLINTOX_PROMPT_PROFILE,
)


CONFIG = BatchConfig(
    description=__doc__ or "",
    default_input=(
        "data/processed_clintox_clinical_trial_failure_v1/ClinTox/scaffold/test.jsonl"
    ),
    default_batch_root=(
        "outputs/chembl_tool/tasks/clintox/reasoning/clinical_trial_failure_v1/batches"
    ),
    default_index=(
        "outputs/paper/molecular_evidence_agent_starling_scaffold_"
        "clinical_trial_failure_v1/evidence/clintox_starling_full/"
        "starling_clintox_neighbor_index.pkl"
    ),
    default_model="deepseek-ai/DeepSeek-V4-Flash-0731",
    batch_id_prefix="clintox_batch",
    pipeline_module="tools.chembl_tool.tasks.clintox.run_reasoning_pipeline",
    log_prefix="clintox_reasoning_batch",
    report_title="ClinTox Batch Report",
    prediction_field="clintox_prediction",
    canonical_positive="toxic",
    canonical_negative="non_toxic",
    positive_predictions=frozenset({"toxic", "tox", "positive", "clintox_positive", "ct_tox", "1"}),
    negative_predictions=frozenset({"non_toxic", "nontoxic", "non-toxic", "negative", "clintox_negative", "0"}),
    prompt_profile_option="--clintox-prompt-profile",
    prompt_profile_choices=CLINTOX_PROMPT_PROFILES,
    default_prompt_profile=DEFAULT_CLINTOX_PROMPT_PROFILE,
    default_api_key_env="CLINTOX_LOCAL_API_KEY",
    default_base_url="http://127.0.0.1:50001/v1",
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
