"""Ames configuration for the shared flat-only reasoning harness."""

from predict.harnesses.branches.runner import BatchConfig, main as run_batch


CONFIG = BatchConfig(
    description=__doc__ or "", default_input="", default_batch_root="outputs/ames",
    default_index="/dev/null", default_model="deepseek-ai/DeepSeek-V4-Flash-0731",
    batch_id_prefix="ames_batch", pipeline_module="predict.harnesses.branches.tasks.ames.pipeline",
    log_prefix="ames_reasoning_batch", report_title="Ames Batch Report",
    prediction_field="final_prediction", canonical_positive="positive",
    canonical_negative="negative", positive_predictions=frozenset({"positive", "1"}),
    negative_predictions=frozenset({"negative", "0"}), supports_retrieval_strategy=True,
    group_prompt_formats=("legacy",), default_group_prompt_format="legacy",
    prompt_profile_option="--ames-prompt-profile",
    prompt_profile_choices=("ames_gold_v1", "ames_bacterial_reverse_mutation.v1"),
    default_prompt_profile="ames_gold_v1", historical_prompt_profile="ames_gold_v1",
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
