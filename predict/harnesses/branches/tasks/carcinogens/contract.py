"""Carcinogens configuration for the shared flat-only reasoning harness."""

from predict.harnesses.branches.runner import BatchConfig, main as run_batch


CONFIG = BatchConfig(
    description=__doc__ or "", default_input="", default_batch_root="outputs/carcinogens",
    default_index="/dev/null", default_model="deepseek-ai/DeepSeek-V4-Flash-0731",
    batch_id_prefix="carcinogens_batch",
    pipeline_module="predict.harnesses.branches.tasks.carcinogens.pipeline",
    log_prefix="carcinogens_reasoning_batch", report_title="Carcinogens Batch Report",
    prediction_field="final_prediction", canonical_positive="positive",
    canonical_negative="negative", positive_predictions=frozenset({"positive", "1", "pass"}),
    negative_predictions=frozenset({"negative", "0", "fail"}), supports_retrieval_strategy=True,
    group_prompt_formats=("legacy",), default_group_prompt_format="legacy",
    prompt_profile_option="--carcinogens-prompt-profile",
    prompt_profile_choices=(
        "carcinogens_gold_v1",
        "carcinogens_starling_only_five_organism_groups.v1",
    ),
    default_prompt_profile="carcinogens_gold_v1",
    historical_prompt_profile="carcinogens_gold_v1",
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
