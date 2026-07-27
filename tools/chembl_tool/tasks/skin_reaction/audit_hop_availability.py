"""Run the strict H1/H2 availability census for Skin_Reaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.chembl_tool.common.hop_availability_census import (
    CandidateSpec,
    CensusConfig,
    run_census,
)
from tools.chembl_tool.tasks.skin_reaction.experiment_config import CHEMBL


CANDIDATES = (
    CandidateSpec(
        family_id="keap1_nrf2_interaction",
        display_name="Direct KEAP1-NRF2 interaction",
        declared_level="H1",
        parent_c_family_id="Mechanism.tier_2",
        measured_node="keap1_nrf2_interaction",
        admissible_path=("keap1_nrf2_interaction", "nrf2_are_keratinocyte_activation"),
        selector_kind="target",
        selector_values=("CHEMBL3038498", "CHEMBL2069156"),
        assay_mode="direct_ppi",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/23562243/",
            "https://pubmed.ncbi.nlm.nih.gov/32866756/",
        ),
        scope_status="nrf2_activation_not_skin_sensitizer_specific",
        note="Direct PPI disruption stabilizes/activates NRF2; generic KEAP1 binding without an interaction readout is excluded.",
    ),
    CandidateSpec(
        family_id="gsk3b_activity",
        display_name="GSK3B functional activity",
        declared_level="H1",
        parent_c_family_id="Mechanism.tier_2",
        measured_node="gsk3b_activity",
        admissible_path=("gsk3b_activity", "nrf2_are_keratinocyte_activation"),
        selector_kind="target",
        selector_values=("CHEMBL262",),
        assay_mode="functional_target",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/22964642/",
            "https://pubmed.ncbi.nlm.nih.gov/25937177/",
        ),
        scope_status="nrf2_activation_not_skin_sensitizer_specific",
        note="GSK3 phosphorylation creates an NRF2 phosphodegron and changes NRF2 stability.",
    ),
    CandidateSpec(
        family_id="cul3_activity",
        display_name="CUL3 ubiquitin-ligase activity",
        declared_level="H2",
        parent_c_family_id="Mechanism.tier_2",
        measured_node="cul3_activity",
        admissible_path=(
            "cul3_activity",
            "keap1_nrf2_interaction",
            "nrf2_are_keratinocyte_activation",
        ),
        selector_kind="target",
        selector_values=("CHEMBL6067538",),
        assay_mode="functional_target",
        citations=(
            "https://pubmed.ncbi.nlm.nih.gov/15282312/",
            "https://pubmed.ncbi.nlm.nih.gov/15601839/",
        ),
        scope_status="nrf2_activation_not_skin_sensitizer_specific",
        note="Mechanistically defensible H2 candidate, expected to be data-limited in ChEMBL.",
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
            "outputs/chembl_tool/tasks/skin_reaction/distance_expansion/analysis/hop_availability_census"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_census(
        CensusConfig(
            task_name="skin_reaction",
            chembl_sqlite=args.chembl_sqlite,
            base_index=Path(
                "outputs/chembl_tool/tasks/skin_reaction/evidence_library/"
                "skin_reaction_neighbor_index.pkl"
            ),
            input_jsonl=Path("data/processed/Skin_Reaction/test.jsonl"),
            output_dir=args.out_dir,
            source_config=CHEMBL,
            candidates=CANDIDATES,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
