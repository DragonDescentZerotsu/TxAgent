"""DILI configuration for the shared flat-only reasoning harness."""

from predict.harnesses.branches.runner import BatchConfig, main as run_batch


CONFIG = BatchConfig(
    description=__doc__ or "", default_input="", default_batch_root="outputs/dili",
    default_index="/dev/null", default_model="deepseek-ai/DeepSeek-V4-Flash-0731",
    batch_id_prefix="dili_batch", pipeline_module="predict.harnesses.branches.tasks.dili.pipeline",
    log_prefix="dili_reasoning_batch", report_title="DILI Batch Report",
    prediction_field="final_prediction", canonical_positive="dili_risk",
    canonical_negative="no_dili_risk", positive_predictions=frozenset({"dili_risk", "1"}),
    negative_predictions=frozenset({"no_dili_risk", "0"}), supports_retrieval_strategy=True,
    group_prompt_formats=("legacy",), default_group_prompt_format="legacy",
    prompt_profile_option="--dili-prompt-profile",
    prompt_profile_choices=("dili_gold_v1", "dili_conditioned_or_source_molecule_outcome.v1"),
    default_prompt_profile="dili_gold_v1", historical_prompt_profile="dili_gold_v1",
)


def main(argv: list[str] | None = None) -> int:
    return run_batch(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
