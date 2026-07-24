"""Run a bounded strict-hop availability census for ClinTox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.chembl_tool.common.hop_availability_census import (
    CandidateSpec,
    CensusConfig,
    run_census,
)
from tools.chembl_tool.tasks.clintox.experiment_config import CHEMBL


CANDIDATES = (
    CandidateSpec(
        family_id="keap1_nrf2_interaction",
        display_name="Direct KEAP1-NRF2 interaction",
        declared_level="H1",
        parent_c_family_id="Mechanism.tier_5",
        measured_node="keap1_nrf2_interaction",
        admissible_path=("keap1_nrf2_interaction", "nrf2_cellular_stress_response"),
        selector_kind="target",
        selector_values=("CHEMBL3038498", "CHEMBL2069156"),
        assay_mode="direct_ppi",
        citations=("https://pubmed.ncbi.nlm.nih.gov/23562243/",),
        mechanism_status="task_label_too_heterogeneous_for_directional_mapping",
        same_molecule_status="requires_toxic_exposure_context",
        scope_status="composite_clintox_label_not_directionally_mapped",
        note="A valid NRF2 edge is not a directional ClinTox-label mapping without exposure and injury context.",
    ),
    CandidateSpec(
        family_id="gsk3b_activity",
        display_name="GSK3B functional activity",
        declared_level="H1",
        parent_c_family_id="Mechanism.tier_5",
        measured_node="gsk3b_activity",
        admissible_path=("gsk3b_activity", "nrf2_cellular_stress_response"),
        selector_kind="target",
        selector_values=("CHEMBL262",),
        assay_mode="functional_target",
        citations=("https://pubmed.ncbi.nlm.nih.gov/22964642/",),
        mechanism_status="task_label_too_heterogeneous_for_directional_mapping",
        same_molecule_status="requires_toxic_exposure_context",
        scope_status="composite_clintox_label_not_directionally_mapped",
        note="Target modulation can be protective or harmful and cannot transfer to the composite ClinTox label by itself.",
    ),
    CandidateSpec(
        family_id="mitochondrial_complex_i_activity",
        display_name="Mitochondrial complex-I functional activity",
        declared_level="H1",
        parent_c_family_id="Mechanism.tier_3",
        measured_node="mitochondrial_complex_i_activity",
        admissible_path=("mitochondrial_complex_i_activity", "mitochondrial_dysfunction"),
        selector_kind="target",
        selector_values=("CHEMBL2363065",),
        assay_mode="functional_target",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/17241123/",
            "https://pubmed.ncbi.nlm.nih.gov/28659801/",
        ),
        mechanism_status="requires_organ_exposure_and_injury_context",
        same_molecule_status="requires_toxic_exposure_context",
        scope_status="organ_and_exposure_dependent",
        note="Complex-I inhibition is mechanistically relevant but not sufficient for the heterogeneous clinical-toxicity label.",
    ),
    CandidateSpec(
        family_id="cul3_activity",
        display_name="CUL3 ubiquitin-ligase activity",
        declared_level="H2",
        parent_c_family_id="Mechanism.tier_5",
        measured_node="cul3_activity",
        admissible_path=("cul3_activity", "keap1_nrf2_interaction", "nrf2_cellular_stress_response"),
        selector_kind="target",
        selector_values=("CHEMBL6067538",),
        assay_mode="functional_target",
        citations=("https://pubmed.ncbi.nlm.nih.gov/15601839/",),
        mechanism_status="task_label_too_heterogeneous_for_directional_mapping",
        same_molecule_status="requires_toxic_exposure_context",
        scope_status="composite_clintox_label_not_directionally_mapped",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chembl-sqlite",
        type=Path,
        default=Path("tools/chembl_tool/chembl_data/chembl_36_sqlite/chembl_36.db"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/chembl_tool/tasks/clintox/distance_expansion/analysis/hop_availability_census"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_census(
        CensusConfig(
            task_name="clintox",
            chembl_sqlite=args.chembl_sqlite,
            base_index=Path(
                "outputs/chembl_tool/tasks/clintox/evidence_library/clintox_neighbor_index.pkl"
            ),
            input_jsonl=Path("data/processed/ClinTox/test.jsonl"),
            output_dir=args.out_dir,
            source_config=CHEMBL,
            candidates=CANDIDATES,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
