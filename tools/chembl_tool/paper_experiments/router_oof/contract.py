"""Frozen task and artifact contracts for the first direct-agent router pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FOLD_SCHEMA_VERSION = "router_oof_folds.scaffold_or_parent.v1"
FEATURE_SCHEMA_VERSION = "router_oof_features.direct.v2"
ROUTER_SCHEMA_VERSION = "task_local_knn_agent_router.v2"
ROUTER_ARTIFACT_DIR = "router_v2"
ROUTER_FEATURE_STEM = "router_features_v2"
DEFAULT_SPLIT = "scaffold"
DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 20260804
DEFAULT_OUTPUT_ROOT = Path("outputs/paper/router_oof/gpt_oss_120b/scaffold")


@dataclass(frozen=True)
class TaskSpec:
    task: str
    data_name: str
    batch_module: str
    direct_condition: str
    direct_index_name: str
    source_index: str
    evidence_filename: str
    index_filename: str
    meta_filename: str


TASK_SPECS: dict[str, TaskSpec] = {
    "bbb_martins": TaskSpec(
        task="bbb_martins",
        data_name="BBB_Martins",
        batch_module="tools.chembl_tool.tasks.bbb_martins.run_reasoning_batch",
        direct_condition="bbb_martins__starling_direct",
        direct_index_name="bbb_starling_direct",
        source_index=(
            "outputs/paper/molecular_evidence_agent/evidence/bbb_starling/all/"
            "starling_bbb_neighbor_index.pkl"
        ),
        evidence_filename="starling_bbb_evidence.jsonl",
        index_filename="starling_bbb_neighbor_index.pkl",
        meta_filename="starling_bbb_neighbor_index.meta.json",
    ),
    "bioavailability_ma": TaskSpec(
        task="bioavailability_ma",
        data_name="Bioavailability_Ma",
        batch_module="tools.chembl_tool.tasks.bioavailability_ma.run_reasoning_batch",
        direct_condition="bioavailability_ma__starling_direct_full",
        direct_index_name="bioavailability_starling_full",
        source_index=(
            "outputs/paper/molecular_evidence_agent/evidence/"
            "bioavailability_starling_full_v2/starling_factor_neighbor_index.pkl"
        ),
        evidence_filename="starling_factor_evidence.jsonl",
        index_filename="starling_factor_neighbor_index.pkl",
        meta_filename="starling_factor_neighbor_index.meta.json",
    ),
    "skin_reaction": TaskSpec(
        task="skin_reaction",
        data_name="Skin_Reaction",
        batch_module="tools.chembl_tool.tasks.skin_reaction.run_reasoning_batch",
        direct_condition="skin_reaction__starling_direct",
        direct_index_name="skin_reaction_starling_full",
        source_index=(
            "outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_full/"
            "starling_skin_reaction_neighbor_index.pkl"
        ),
        evidence_filename="starling_skin_reaction_evidence.jsonl",
        index_filename="starling_skin_reaction_neighbor_index.pkl",
        meta_filename="starling_skin_reaction_neighbor_index.meta.json",
    ),
}


def selected_task_specs(tasks: list[str] | tuple[str, ...] | None = None) -> list[TaskSpec]:
    names = list(tasks or TASK_SPECS)
    missing = sorted(set(names) - set(TASK_SPECS))
    if missing:
        raise ValueError(f"Unknown router tasks: {', '.join(missing)}")
    return [TASK_SPECS[name] for name in names]


def task_root(output_root: str | Path, task: str) -> Path:
    return Path(output_root) / task


def fold_root(output_root: str | Path, task: str, fold: int) -> Path:
    return task_root(output_root, task) / f"fold_{fold:02d}"


def direct_index_path(output_root: str | Path, spec: TaskSpec, fold: int) -> Path:
    return (
        fold_root(output_root, spec.task, fold)
        / "evidence"
        / spec.direct_index_name
        / spec.index_filename
    )


def direct_batch_root(output_root: str | Path, spec: TaskSpec, fold: int) -> Path:
    return (
        fold_root(output_root, spec.task, fold)
        / "agent"
        / "runs_identity_blind_parent_disjoint"
        / spec.task
        / spec.direct_condition
    )
