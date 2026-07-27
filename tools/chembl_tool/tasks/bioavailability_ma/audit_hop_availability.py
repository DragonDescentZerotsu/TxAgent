"""Run the strict H1/H2 availability census for Bioavailability_Ma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.chembl_tool.common.hop_availability_census import (
    CandidateSpec,
    CensusConfig,
    run_census,
)
from tools.chembl_tool.tasks.bioavailability_ma.experiment_config import CHEMBL


CANDIDATES = (
    CandidateSpec(
        family_id="ionization_pka",
        display_name="Measured pKa",
        declared_level="H1",
        parent_c_family_id="Fa.absorption_solubility_permeability",
        measured_node="molecular_ionization_pka",
        admissible_path=("molecular_ionization_pka", "solubility_or_membrane_permeability"),
        selector_kind="standard_type",
        selector_values=("pKa", "pKA", "pka", "PKa", "Pka"),
        assay_mode="property_measurement",
        citations=("https://pubmed.ncbi.nlm.nih.gov/14659483/",),
        scope_status="overlaps_query_molecule_property_tool_semantics",
        note="Ionization directly conditions aqueous solubility and passive membrane permeability.",
    ),
    CandidateSpec(
        family_id="lipophilicity_logd_logp",
        display_name="Measured LogD/LogP",
        declared_level="H1",
        parent_c_family_id="Fa.absorption_solubility_permeability",
        measured_node="molecular_lipophilicity",
        admissible_path=("molecular_lipophilicity", "solubility_or_membrane_permeability"),
        selector_kind="standard_type",
        selector_values=("LogD", "logD", "LogP", "logP"),
        assay_mode="property_measurement",
        citations=("https://pubmed.ncbi.nlm.nih.gov/30395703/",),
        scope_status="overlaps_query_molecule_property_tool_semantics",
        note="Measured lipophilicity directly conditions the permeability-solubility balance.",
    ),
    CandidateSpec(
        family_id="plasma_unbound_fraction",
        display_name="Measured plasma protein binding/unbound fraction",
        declared_level="H1",
        parent_c_family_id="Fh.hepatic_clearance_metabolic_stability",
        measured_node="plasma_unbound_fraction",
        admissible_path=("plasma_unbound_fraction", "hepatic_clearance"),
        selector_kind="standard_type",
        selector_values=("PPB", "Fu"),
        assay_mode="property_measurement",
        citations=("https://pubmed.ncbi.nlm.nih.gov/7229915/",),
        scope_status="supports_clearance_not_direct_absolute_bioavailability",
        note="The unbound fraction is an explicit input to hepatic-clearance models; it does not by itself establish absolute F.",
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
            "outputs/chembl_tool/tasks/bioavailability_ma/distance_expansion/analysis/"
            "hop_availability_census"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_census(
        CensusConfig(
            task_name="bioavailability_ma",
            chembl_sqlite=args.chembl_sqlite,
            base_index=Path(
                "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/"
                "bioavailability_neighbor_index.pkl"
            ),
            input_jsonl=Path("data/processed/Bioavailability_Ma/test.jsonl"),
            output_dir=args.out_dir,
            source_config=CHEMBL,
            candidates=CANDIDATES,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
