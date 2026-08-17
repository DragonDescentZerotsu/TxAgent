"""Frozen task contracts for the matched train-label neighbor experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.chembl_tool.tasks.bioavailability_ma.prompt_profiles import (
    LEGACY_BIOAVAILABILITY_V1,
)


SCHEMA_VERSION = "matched_train_label_direct_agent.v1"
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/paper/matched_train_label_direct_agent_v1_scaffold_valid_gpt_oss_120b"
)


@dataclass(frozen=True)
class TaskSpec:
    task: str
    data_name: str
    batch_module: str
    condition: str
    input_jsonl: Path
    knn_predictions: Path
    retrieval_index: Path
    single_source_batch: Path
    group_id: str
    tier: str
    endpoint_group: str
    source_group_id: str
    endpoint_name: str
    negative_label_text: str
    positive_label_text: str
    full_pool_direct_batch: Path | None = None
    prompt_profile_args: tuple[str, ...] = ()
    retrieval_feature: str = "morgan_fingerprint"
    retrieval_similarity: str = "tanimoto"
    retrieval_similarity_metric: str = "Morgan Tanimoto radius=2 n_bits=2048"
    retrieval_neighbor_set_contract: str = "exactly_the_formal_morgan_knn_top3"

    def label_text(self, label: int) -> str:
        if label == 0:
            return self.negative_label_text
        if label == 1:
            return self.positive_label_text
        raise ValueError(f"{self.task} received non-binary label {label}")


BBB_PAPER_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2"
)
BBB_RUN_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_experimental_meaningful_cns_access_v2_valid_gpt_oss_120b/"
    "runs_identity_blind_parent_disjoint"
)
RECORD_SUPPORTED_PAPER_ROOT = Path(
    "outputs/paper/molecular_evidence_agent_starling_scaffold_record_supported_v2"
)
RECORD_SUPPORTED_RUN_ROOT = Path(
    "outputs/paper/"
    "molecular_evidence_agent_starling_scaffold_record_supported_v2_valid_gpt_oss_120b_bugfix_v2/"
    "runs_identity_blind_parent_disjoint"
)


TASK_SPECS: dict[str, TaskSpec] = {
    "bbb_martins": TaskSpec(
        task="bbb_martins",
        data_name="BBB_Martins",
        batch_module="tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch",
        condition="bbb_martins__matched_train_label_direct",
        input_jsonl=Path(
            "data/processed_starling_experimental_meaningful_cns_access_v2/"
            "BBB_Martins/scaffold/valid.jsonl"
        ),
        knn_predictions=Path(
            "outputs/baselines/structure_knn_experimental_meaningful_cns_access_v2/"
            "BBB_Martins/scaffold/valid_predictions.jsonl"
        ),
        retrieval_index=BBB_PAPER_ROOT
        / "evidence/bbb_starling_full/starling_bbb_neighbor_index.pkl",
        single_source_batch=BBB_RUN_ROOT / "bbb_martins/bbb_martins__none",
        group_id="Direct.bbb",
        tier="Direct",
        endpoint_group="direct_bbb",
        source_group_id="TrainLabel.meaningful_cns_access",
        endpoint_name="frozen benchmark training label for meaningful CNS access",
        negative_label_text=(
            "Y=0: restricted or poor meaningful CNS access after systemic administration"
        ),
        positive_label_text=(
            "Y=1: meaningful or adequate CNS access after systemic administration"
        ),
        full_pool_direct_batch=BBB_RUN_ROOT / "bbb_martins/bbb_martins__starling_direct",
    ),
    "bioavailability_ma": TaskSpec(
        task="bioavailability_ma",
        data_name="Bioavailability_Ma",
        batch_module="tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        condition="bioavailability_ma__matched_train_label_direct",
        input_jsonl=Path(
            "data/processed_starling_record_supported_v2/"
            "Bioavailability_Ma/scaffold/valid.jsonl"
        ),
        knn_predictions=Path(
            "outputs/baselines/structure_knn_starling_record_supported_v2/"
            "Bioavailability_Ma/scaffold/valid_predictions.jsonl"
        ),
        retrieval_index=RECORD_SUPPORTED_PAPER_ROOT
        / "evidence/bioavailability_starling_full/starling_factor_neighbor_index.pkl",
        single_source_batch=RECORD_SUPPORTED_RUN_ROOT
        / "bioavailability_ma/bioavailability_ma__none",
        group_id="Observed.direct_oral_bioavailability",
        tier="Observed",
        endpoint_group="direct_oral_bioavailability",
        source_group_id="TrainLabel.direct_oral_bioavailability",
        endpoint_name="frozen benchmark training label for oral bioavailability",
        negative_label_text="Y=0: low oral bioavailability, F < 20%",
        positive_label_text="Y=1: high oral bioavailability, F >= 20%",
        full_pool_direct_batch=RECORD_SUPPORTED_RUN_ROOT
        / "bioavailability_ma/bioavailability_ma__starling_direct_full",
        prompt_profile_args=(
            "--bioavailability-prompt-profile",
            LEGACY_BIOAVAILABILITY_V1,
        ),
    ),
    "skin_reaction": TaskSpec(
        task="skin_reaction",
        data_name="Skin_Reaction",
        batch_module="tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch",
        condition="skin_reaction__matched_train_label_direct",
        input_jsonl=Path(
            "data/processed_starling_record_supported_v2/"
            "Skin_Reaction/scaffold/valid.jsonl"
        ),
        knn_predictions=Path(
            "outputs/baselines/structure_knn_starling_record_supported_v2/"
            "Skin_Reaction/scaffold/valid_predictions.jsonl"
        ),
        retrieval_index=RECORD_SUPPORTED_PAPER_ROOT
        / "evidence/skin_reaction_starling_full/starling_skin_reaction_neighbor_index.pkl",
        single_source_batch=RECORD_SUPPORTED_RUN_ROOT
        / "skin_reaction/skin_reaction__none",
        group_id="Direct.skin_reaction",
        tier="Direct",
        endpoint_group="direct_skin_reaction",
        source_group_id="TrainLabel.skin_sensitization_contact_allergy",
        endpoint_name="frozen benchmark training label for skin sensitization/contact allergy",
        negative_label_text="Y=0: non-sensitizer/contact-allergy negative",
        positive_label_text="Y=1: sensitizer/contact-allergy positive",
        full_pool_direct_batch=RECORD_SUPPORTED_RUN_ROOT
        / "skin_reaction/skin_reaction__starling_direct",
    ),
}


def selected_specs(tasks: list[str] | tuple[str, ...] | None = None) -> list[TaskSpec]:
    names = list(tasks or TASK_SPECS)
    missing = sorted(set(names) - set(TASK_SPECS))
    if missing:
        raise ValueError(f"Unknown tasks: {', '.join(missing)}")
    return [TASK_SPECS[name] for name in names]


def replay_batch(output_root: str | Path, spec: TaskSpec) -> Path:
    return Path(output_root) / "retrieval_replay" / spec.task / spec.condition


def agent_batch(output_root: str | Path, spec: TaskSpec) -> Path:
    return (
        Path(output_root)
        / "runs_identity_blind_parent_disjoint"
        / spec.task
        / spec.condition
    )
