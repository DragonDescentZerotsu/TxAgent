"""Build a molecule-level oral bioavailability evidence library and neighbor index."""

from __future__ import annotations

from tools.chembl_tool.tasks.bioavailability_ma.endpoint_groups import assign_endpoint_group
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


DEFAULT_ASSAYS_CSV = "outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/v4/bioavailability_assay_candidates.csv"
DEFAULT_ACTIVITIES_CSV = "outputs/chembl_tool/tasks/bioavailability_ma/assay_screening/v4/bioavailability_activity_evidence.csv"
DEFAULT_OUT_DIR = "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library"

CONFIG = EvidenceLibraryConfig(
    description=__doc__ or "",
    default_assays_csv=DEFAULT_ASSAYS_CSV,
    default_activities_csv=DEFAULT_ACTIVITIES_CSV,
    default_out_dir=DEFAULT_OUT_DIR,
    evidence_filename="bioavailability_molecule_evidence.jsonl",
    index_filename="bioavailability_neighbor_index.pkl",
    meta_filename="bioavailability_neighbor_index.meta.json",
    index_version="bioavailability_ma_neighbor_index.v1",
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
