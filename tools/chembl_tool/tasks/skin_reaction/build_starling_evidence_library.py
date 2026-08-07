"""Build the paper-facing Skin_Reaction Starling four-family index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.chembl_tool.common.starling import (
    StarlingSourceProfile,
    build_and_write_starling_index,
    build_starling_parquet_evidence_rows,
)


DEFAULT_STARLING_DATA_DIR = "data/starling_data/skin_reaction"
DEFAULT_OUT_DIR = "outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_full"
EVIDENCE_FILENAME = "starling_skin_reaction_evidence.jsonl"
INDEX_FILENAME = "starling_skin_reaction_neighbor_index.pkl"
META_FILENAME = "starling_skin_reaction_neighbor_index.meta.json"
INDEX_VERSION = "skin_reaction_starling_full_neighbor_index.v1"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    evidence_rows, source_stats = build_starling_parquet_evidence_rows(
        skin_reaction_profiles(Path(args.starling_data_dir), max_rows=args.max_rows_per_source),
        max_record_examples=args.max_record_examples,
        min_confidence=args.min_confidence,
    )
    source = {
        "type": "starling_profile_index",
        "dataset": "Starling Skin_Reaction task-specific literature acquisitions",
        "families": [
            "direct_skin_reaction",
            "sensitization_aop",
            "phototoxicity_irritation_local_damage",
            "skin_exposure",
        ],
        "exact_query_exclusion": True,
    }
    meta = build_and_write_starling_index(
        evidence_rows,
        out_dir=args.out_dir,
        index_version=INDEX_VERSION,
        source=source,
        source_stats=source_stats,
        evidence_filename=EVIDENCE_FILENAME,
        index_filename=INDEX_FILENAME,
        meta_filename=META_FILENAME,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


def skin_reaction_profiles(data_dir: Path, *, max_rows: int = 0) -> list[StarlingSourceProfile]:
    return [
        StarlingSourceProfile(
            source_id="skin_direct_reaction",
            path=str(data_dir / "direct_skin_reaction" / "extractions.parquet"),
            group_id="Direct.skin_reaction",
            assay_tier="Tier 1",
            endpoint_group="direct_skin_reaction",
            evidence_source="Starling/Skin_Reaction/direct_skin_reaction",
            endpoint_field="outcome_label",
            smiles_field="SMILES",
            context_fields=(
                "reaction_type",
                "assay_or_test",
                "species_or_population",
                "dose_or_concentration",
                "positive_count",
                "total_tested",
                "effect_metric",
                "extra_details",
            ),
            scope_fields=("assay_or_test", "species_or_population", "dose_or_concentration"),
            name_fields=(),
            target_pref_name="direct skin reaction outcome",
            evidence_role="direct_outcome",
            standard_type_prefix="direct skin reaction",
            include_endpoint_values=("positive", "negative", "inconclusive"),
            extra_example_fields=("positive_count", "total_tested", "effect_metric", "pmid"),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="skin_sensitization_aop",
            path=str(data_dir / "sensitization_aop" / "extractions.parquet"),
            group_id="Mechanism.sensitization_aop",
            assay_tier="Tier 2",
            endpoint_group="sensitization_aop",
            evidence_source="Starling/Skin_Reaction/sensitization_aop",
            endpoint_field="aop_event",
            smiles_field="SMILES",
            value_field="result_value",
            unit_field="result_unit",
            context_fields=(
                "assay_type",
                "endpoint_or_target",
                "result_label",
                "experimental_conditions",
                "qualifying_conditions",
                "extra_details",
            ),
            scope_fields=("assay_type", "experimental_conditions", "qualifying_conditions"),
            target_pref_name="skin sensitization adverse-outcome pathway",
            evidence_role="mechanistic_factor",
            standard_type_prefix="skin sensitization AOP",
            include_endpoint_values=(
                "MIE_protein_binding",
                "KE2_keratinocyte_activation",
                "KE3_dendritic_cell_activation",
                "KE4_T_cell_activation",
                "adverse_outcome_skin_sensitization",
                "integrated_or_unspecified",
            ),
            extra_example_fields=("assay_type", "result_label", "endpoint_or_target", "pmid"),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="skin_phototoxicity_irritation_local_damage",
            path=str(data_dir / "phototoxicity_irritation_local_damage" / "extractions.parquet"),
            group_id="Mechanism.phototoxicity_irritation_local_damage",
            assay_tier="Tier 3",
            endpoint_group="phototoxicity_irritation_local_damage",
            evidence_source="Starling/Skin_Reaction/phototoxicity_irritation_local_damage",
            endpoint_field="evidence_endpoint",
            smiles_field="SMILES",
            context_fields=(
                "evidence_system",
                "assay_method",
                "result_label",
                "observed_effect",
                "experimental_conditions",
                "light_conditions",
                "qualifying_conditions",
                "extra_details",
            ),
            scope_fields=("evidence_system", "assay_method", "qualifying_conditions"),
            target_pref_name="phototoxicity, irritation, corrosion, and local skin damage",
            evidence_role="mechanistic_factor",
            standard_type_prefix="skin local damage",
            include_endpoint_values=(
                "phototoxicity_or_photosensitivity",
                "light_dependent_ros",
                "light_dependent_cytotoxicity",
                "uv_dependent_skin_response",
                "skin_irritation",
                "skin_corrosion",
                "keratinocyte_damage",
                "inflammatory_response",
                "local_tissue_injury",
            ),
            context_filter_fields=(
                "support_text",
                "assay_method",
                "observed_effect",
                "experimental_conditions",
                "light_conditions",
                "qualifying_conditions",
                "extra_details",
            ),
            required_context_patterns_by_endpoint=(
                (
                    "inflammatory_response",
                    (
                        r"\bskin\b",
                        r"dermal",
                        r"cutaneous",
                        r"epiderm",
                        r"keratin",
                        r"topical",
                        r"contact dermatitis",
                        r"rash",
                        r"urticar",
                        r"erythema",
                        r"edema",
                        r"blister",
                        r"vesicat",
                        r"photo",
                        r"ultraviolet",
                        r"\buv[ab]?\b",
                    ),
                ),
                (
                    "local_tissue_injury",
                    (
                        r"\bskin\b",
                        r"dermal",
                        r"cutaneous",
                        r"epiderm",
                        r"keratin",
                        r"topical",
                        r"contact dermatitis",
                        r"rash",
                        r"urticar",
                        r"erythema",
                        r"edema",
                        r"blister",
                        r"vesicat",
                        r"corrosion",
                        r"irrit",
                        r"photo",
                        r"ultraviolet",
                        r"\buv[ab]?\b",
                    ),
                ),
            ),
            extra_example_fields=("result_label", "observed_effect", "light_conditions", "pmid"),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="skin_exposure",
            path=str(data_dir / "skin_exposure" / "extractions.parquet"),
            group_id="Mechanism.skin_exposure",
            assay_tier="Tier 4",
            endpoint_group="skin_exposure",
            evidence_source="Starling/Skin_Reaction/skin_exposure",
            endpoint_field="evidence_type",
            smiles_field="SMILES",
            value_field="result_value",
            unit_field="result_unit",
            context_fields=(
                "study_design",
                "skin_source",
                "formulation_vehicle",
                "exposure_time",
                "qualifying_conditions",
                "extra_details",
            ),
            scope_fields=("study_design", "skin_source", "formulation_vehicle", "qualifying_conditions"),
            target_pref_name="dermal exposure and skin penetration",
            evidence_role="context_modifier",
            standard_type_prefix="skin exposure",
            include_endpoint_values=(
                "permeability_coefficient",
                "flux",
                "lag_time",
                "cumulative_permeated_amount",
                "dermal_absorption",
                "skin_retention",
                "tape_strip_result",
                "skin_exposure_measurement",
                "dermally_applied_dose",
                "relative_penetration",
                "qualitative_penetration",
            ),
            extra_example_fields=("result_value", "result_unit", "skin_source", "pmid"),
            max_rows=max_rows,
        ),
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starling-data-dir", default=DEFAULT_STARLING_DATA_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-record-examples", type=int, default=6)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
