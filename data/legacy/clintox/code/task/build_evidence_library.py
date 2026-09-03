"""Build a molecule-level ClinTox evidence library and neighbor index."""

from __future__ import annotations

from tools.chembl_tool.common.task_workflows.evidence_library import (
    DEFAULT_CHEMBL_FPS,
    FP_BITS,
    FP_RADIUS,
    EvidenceLibraryConfig,
    build_evidence_rows as build_common_evidence_rows,
    build_neighbor_index as build_common_neighbor_index,
    main as run_builder,
    standardize_smiles,
    standardize_smiles_and_fp,
)
from tools.chembl_tool.tasks.clintox.endpoint_groups import assign_endpoint_group


DEFAULT_ASSAYS_CSV = "outputs/chembl_tool/tasks/clintox/assay_screening/v6/clintox_assay_candidates.csv"
DEFAULT_ACTIVITIES_CSV = "outputs/chembl_tool/tasks/clintox/assay_screening/v6/clintox_activity_evidence.csv"
DEFAULT_OUT_DIR = "outputs/chembl_tool/tasks/clintox/evidence_library"

CONFIG = EvidenceLibraryConfig(
    description=__doc__ or "",
    default_assays_csv=DEFAULT_ASSAYS_CSV,
    default_activities_csv=DEFAULT_ACTIVITIES_CSV,
    default_out_dir=DEFAULT_OUT_DIR,
    evidence_filename="clintox_molecule_evidence.jsonl",
    index_filename="clintox_neighbor_index.pkl",
    meta_filename="clintox_neighbor_index.meta.json",
    index_version="clintox_neighbor_index.v6",
    assign_endpoint_group=assign_endpoint_group,
)


def build_evidence_rows(*args, **kwargs):
    return build_common_evidence_rows(*args, assign_endpoint_group=assign_endpoint_group, **kwargs)


def build_neighbor_index(*args, **kwargs):
    return build_common_neighbor_index(*args, index_version=CONFIG.index_version, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return run_builder(CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main())
