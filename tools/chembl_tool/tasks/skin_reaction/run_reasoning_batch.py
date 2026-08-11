"""Run Skin_Reaction reasoning for many query molecules and summarize metrics."""

from __future__ import annotations

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
    prompt_profile_option="--skin-prompt-profile",
    prompt_profile_choices=SKIN_PROMPT_PROFILES,
    default_prompt_profile=DEFAULT_SKIN_PROMPT_PROFILE,
    historical_prompt_profile=HISTORICAL_SKIN_PROMPT_PROFILE,
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
