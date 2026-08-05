"""Stable paths and names for the MiniMol agent-retrieval experiment."""

from __future__ import annotations

from pathlib import Path


MINIMOL_RETRIEVAL_FEATURE = "minimol"
MORGAN_RETRIEVAL_FEATURE = "morgan"
RETRIEVAL_FEATURES = (MORGAN_RETRIEVAL_FEATURE, MINIMOL_RETRIEVAL_FEATURE)
DEFAULT_FEATURE_ROOT = Path(
    "outputs/paper/minimol_retrieval_features_v7_record_agreement70_split811_v1"
)


def descriptor_path_for_experiment(
    split: str,
    experiment_name: str,
    *,
    output_root: Path = DEFAULT_FEATURE_ROOT,
) -> Path:
    return output_root / split / "descriptors" / f"{experiment_name}.json"


def paper_root_for_minimol_retrieval(split: str) -> Path:
    return (
        Path("outputs/paper")
        / f"molecular_evidence_agent_starling_{split}_record_agreement70_split811_v1_minimol_retrieval"
    )
