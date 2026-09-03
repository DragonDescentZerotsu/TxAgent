"""Build versioned paper-facing Skin_Reaction Starling evidence indices."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

from tools.chembl_tool.common.json_utils import sha256_file
from data.processing.evidence_library.evidence_library import (
    StarlingSourceProfile,
    build_and_write_starling_index,
    build_starling_parquet_evidence_rows,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.canonical_starling_source import (
    CANONICAL_SOURCE_DIR,
    CANONICAL_VERSION,
    MANIFEST_PATH,
)
from data.processing.evidence_library.versions.v7.tasks.skin_reaction.starling_benchmark import (
    is_tdc_skin_sensitization_scope,
)


DEFAULT_STARLING_DATA_DIR = "data/raw/starling/skin_reaction"
HISTORICAL_OUT_DIR = "outputs/paper/molecular_evidence_agent/evidence/skin_reaction_starling_full"
ALIGNED_V2_OUT_DIR = (
    "outputs/paper/molecular_evidence_agent/evidence/"
    "skin_reaction_starling_sensitization_v2"
)
DEFAULT_OUT_DIR = (
    "outputs/paper/molecular_evidence_agent/evidence/"
    "skin_reaction_starling_sensitization_canonical_v3"
)
EVIDENCE_FILENAME = "starling_skin_reaction_evidence.jsonl"
INDEX_FILENAME = "starling_skin_reaction_neighbor_index.pkl"
META_FILENAME = "starling_skin_reaction_neighbor_index.meta.json"
CANONICAL_HELDOUT_INDEX_NAME = "skin_reaction_starling_sensitization_canonical_v3"
DEFAULT_EVIDENCE_PATH = Path(DEFAULT_OUT_DIR) / EVIDENCE_FILENAME
DEFAULT_INDEX_PATH = Path(DEFAULT_OUT_DIR) / INDEX_FILENAME

SOURCE_PROFILE_BROAD_V1 = "broad_skin_reaction_v1"
SOURCE_PROFILE_SENSITIZATION_V2 = "sensitization_contact_allergy_v2"
SOURCE_PROFILE_CANONICAL_V3 = "sensitization_direct_aop_canonical_v3"
DEFAULT_SOURCE_PROFILE = SOURCE_PROFILE_CANONICAL_V3
SOURCE_PROFILES = (
    SOURCE_PROFILE_BROAD_V1,
    SOURCE_PROFILE_SENSITIZATION_V2,
    SOURCE_PROFILE_CANONICAL_V3,
)
OUT_DIRS = {
    SOURCE_PROFILE_BROAD_V1: HISTORICAL_OUT_DIR,
    SOURCE_PROFILE_SENSITIZATION_V2: ALIGNED_V2_OUT_DIR,
    SOURCE_PROFILE_CANONICAL_V3: DEFAULT_OUT_DIR,
}
INDEX_VERSIONS = {
    SOURCE_PROFILE_BROAD_V1: "skin_reaction_starling_full_neighbor_index.v1",
    SOURCE_PROFILE_SENSITIZATION_V2: "skin_reaction_starling_sensitization_neighbor_index.v2",
    SOURCE_PROFILE_CANONICAL_V3: "skin_reaction_starling_sensitization_direct_aop_index.v3",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = _resolve_out_dir(args)
    data_dir = Path(args.starling_data_dir)
    if args.source_profile == SOURCE_PROFILE_CANONICAL_V3:
        _validate_canonical_source()
    profiles = skin_reaction_profiles(
        data_dir,
        max_rows=args.max_rows_per_source,
        source_profile=args.source_profile,
    )
    evidence_rows, source_stats = build_starling_parquet_evidence_rows(
        profiles,
        max_record_examples=args.max_record_examples,
        min_confidence=args.min_confidence,
    )
    source = {
        "type": "starling_profile_index",
        "dataset": "Starling Skin_Reaction task-specific literature acquisitions",
        "families": [profile.endpoint_group for profile in profiles],
        "exact_query_exclusion": True,
        "source_profile": args.source_profile,
    }
    meta = build_and_write_starling_index(
        evidence_rows,
        out_dir=out_dir,
        index_version=INDEX_VERSIONS[args.source_profile],
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


def skin_reaction_profiles(
    data_dir: Path,
    *,
    max_rows: int = 0,
    source_profile: str = DEFAULT_SOURCE_PROFILE,
) -> list[StarlingSourceProfile]:
    if source_profile not in SOURCE_PROFILES:
        raise ValueError(f"Unknown Skin source profile: {source_profile}")
    if source_profile == SOURCE_PROFILE_CANONICAL_V3:
        return _canonical_v3_profiles(data_dir, max_rows=max_rows)
    aligned_scope = source_profile == SOURCE_PROFILE_SENSITIZATION_V2
    return [
        StarlingSourceProfile(
            source_id="skin_direct_reaction",
            path=str(data_dir / "direct_skin_reaction" / "extractions.parquet"),
            group_id="Direct.skin_reaction",
            assay_tier="Tier 1",
            endpoint_group="direct_skin_reaction",
            evidence_source=(
                "Starling/Skin_Reaction/direct_skin_sensitization"
                if aligned_scope
                else "Starling/Skin_Reaction/direct_skin_reaction"
            ),
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
            target_pref_name=(
                "direct skin sensitization/contact-allergy outcome"
                if aligned_scope
                else "direct skin reaction outcome"
            ),
            evidence_role="direct_outcome",
            standard_type_prefix=(
                "direct skin sensitization/contact allergy"
                if aligned_scope
                else "direct skin reaction"
            ),
            include_endpoint_values=("positive", "negative", "inconclusive"),
            record_filter=(
                _is_skin_sensitization_record if aligned_scope else None
            ),
            record_filter_name=(
                "is_tdc_skin_sensitization_scope.v1" if aligned_scope else ""
            ),
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


def _canonical_v3_profiles(
    data_dir: Path,
    *,
    max_rows: int,
) -> list[StarlingSourceProfile]:
    del data_dir
    canonical_dir = CANONICAL_SOURCE_DIR
    return [
        StarlingSourceProfile(
            source_id="skin_direct_sensitization_canonical_v3",
            path=str(canonical_dir / "direct_records.parquet"),
            group_id="Direct.skin_reaction",
            assay_tier="Tier 1",
            endpoint_group="direct_skin_reaction",
            evidence_source="Starling/Skin_Reaction/direct_sensitization_canonical_v3",
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
            target_pref_name="direct skin sensitization/contact-allergy outcome",
            evidence_role="direct_outcome",
            standard_type_prefix="direct skin sensitization/contact allergy",
            include_endpoint_values=("positive", "negative", "inconclusive"),
            extra_example_fields=(
                "positive_count",
                "total_tested",
                "effect_metric",
                "pmid",
                "source_partition",
                "source_record_id",
            ),
            max_rows=max_rows,
        ),
        StarlingSourceProfile(
            source_id="skin_sensitization_aop_canonical_v3",
            path=str(canonical_dir / "aop_records.parquet"),
            group_id="Mechanism.sensitization_aop",
            assay_tier="Tier 2",
            endpoint_group="sensitization_aop",
            evidence_source="Starling/Skin_Reaction/sensitization_aop_canonical_v3",
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
            target_pref_name="skin sensitization AOP key events",
            evidence_role="mechanistic_factor",
            standard_type_prefix="skin sensitization AOP key event",
            include_endpoint_values=(
                "MIE_protein_binding",
                "KE2_keratinocyte_activation",
                "KE3_dendritic_cell_activation",
                "KE4_T_cell_activation",
            ),
            extra_example_fields=(
                "assay_type",
                "result_label",
                "endpoint_or_target",
                "pmid",
                "source_partition",
                "source_record_id",
            ),
            max_rows=max_rows,
        ),
    ]


def _is_skin_sensitization_record(record: Mapping[str, object]) -> bool:
    """Keep direct evidence in exact parity with the benchmark label scope."""
    return is_tdc_skin_sensitization_scope(record.get("reaction_type"))


def _validate_canonical_source(data_dir: Path | None = None) -> None:
    canonical_dir = (
        data_dir / "canonical_sensitization_v3"
        if data_dir is not None
        else CANONICAL_SOURCE_DIR
    )
    manifest_path = canonical_dir / MANIFEST_PATH.name
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing canonical Skin source manifest: {manifest_path}. "
            "Run build_canonical_starling_source first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != CANONICAL_VERSION:
        raise ValueError(f"Unexpected canonical Skin contract in {manifest_path}")
    stats = manifest.get("stats") or {}
    if not stats.get("partition_reconciles"):
        raise ValueError("Canonical Skin partition does not reconcile to raw sources")
    if int(stats.get("direct_aop_source_record_overlap", -1)) != 0:
        raise ValueError("Canonical Skin direct/AOP partitions overlap")

    manifest_paths = manifest.get("paths") or {}
    for name in ("direct_records", "aop_records"):
        path = canonical_dir / f"{name}.parquet"
        expected = str(manifest_paths.get(f"{name}_sha256") or "")
        if not path.is_file() or not expected or sha256_file(path) != expected:
            raise ValueError(f"Canonical Skin artifact failed hash validation: {path}")


def _resolve_out_dir(args: argparse.Namespace) -> str:
    if args.max_rows_per_source and not args.out_dir:
        raise SystemExit(
            "--max-rows-per-source requires an explicit non-canonical --out-dir"
        )
    return str(args.out_dir or OUT_DIRS[args.source_profile])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starling-data-dir", default=DEFAULT_STARLING_DATA_DIR)
    parser.add_argument(
        "--out-dir",
        default="",
        help="Defaults to a versioned directory for the selected source profile.",
    )
    parser.add_argument(
        "--source-profile",
        "--direct-scope",
        dest="source_profile",
        choices=SOURCE_PROFILES,
        default=DEFAULT_SOURCE_PROFILE,
        help="Versioned Skin source selection; --direct-scope is a compatibility alias.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-record-examples", type=int, default=6)
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
