"""Pinned source contracts for the layered Skin_Reaction normalizer.

Row counts and file digests are the ones frozen in
``data/starling_data/skin_reaction/SOURCE_MANIFEST.json``.  Unlike
Bioavailability_Ma there is no separate HuggingFace direct source: all four
mechanism families are Starling extraction parquets that carry their own
structure column, which is spelled ``SMILES`` (uppercase) in every file.
"""

from __future__ import annotations

from pathlib import Path

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.normalization.contracts import NormalizedSourceProfile


EXPECTED_SOURCE_ROWS = {
    "direct_skin_reaction": 66_597,
    "sensitization_aop": 45_985,
    "phototoxicity_irritation_local_damage": 382_726,
    "skin_exposure": 311_834,
}

EXPECTED_SOURCE_SHA256 = {
    "direct_skin_reaction": "e7c4819c98af47eae4dc0563f40dc39f8b09eb8e03c9626daee603147125f9c1",
    "sensitization_aop": "3eca3f70d1243795bc37f06e31795566295ce8ea3bf03e810efd3807152cf2c8",
    "phototoxicity_irritation_local_damage": "c942e956be13c80d3f6075922dcc3731ffaea298dd824b2cf8f09598238501db",
    "skin_exposure": "14f9ba586b6744e7412867110080ad9bf7b51b2ae39be0d12572f9ce66fd7713",
}


def source_profiles(data_dir: Path) -> list[NormalizedSourceProfile]:
    return [
        NormalizedSourceProfile(
            source_id="direct_skin_reaction",
            source_name="starling-labs/skin_reaction/Direct_Skin_Reaction",
            source_path=str(data_dir / "direct_skin_reaction" / "extractions.parquet"),
            endpoint_field="reaction_type",
            measurement_field="effect_metric",
            embedded_unit=True,
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=(),
            context_fields=(
                "outcome_label",
                "assay_or_test",
                "species_or_population",
                "dose_or_concentration",
                "positive_count",
                "total_tested",
                "extra_details",
            ),
        ),
        NormalizedSourceProfile(
            source_id="sensitization_aop",
            source_name="starling-labs/skin_reaction/Sensitization_AOP",
            source_path=str(data_dir / "sensitization_aop" / "extractions.parquet"),
            endpoint_field="endpoint_or_target",
            measurement_field="result_value",
            unit_field="result_unit",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=("global_identifier",),
            context_fields=(
                "assay_type",
                "aop_event",
                "result_label",
                "experimental_conditions",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        NormalizedSourceProfile(
            source_id="phototoxicity_irritation_local_damage",
            source_name="starling-labs/skin_reaction/Phototoxicity_Irritation_Local_Damage",
            source_path=str(
                data_dir / "phototoxicity_irritation_local_damage" / "extractions.parquet"
            ),
            endpoint_field="evidence_endpoint",
            measurement_field="observed_effect",
            embedded_unit=True,
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=("global_identifier",),
            context_fields=(
                "evidence_system",
                "assay_method",
                "result_label",
                "experimental_conditions",
                "light_conditions",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
        NormalizedSourceProfile(
            source_id="skin_exposure",
            source_name="starling-labs/skin_reaction/Skin_Exposure",
            source_path=str(data_dir / "skin_exposure" / "extractions.parquet"),
            endpoint_field="evidence_type",
            measurement_field="result_value",
            unit_field="result_unit",
            smiles_field="SMILES",
            structure_mode="direct",
            name_fields=("global_identifier",),
            context_fields=(
                "study_design",
                "skin_source",
                "formulation_vehicle",
                "exposure_time",
                "qualifying_conditions",
                "extra_details",
            ),
        ),
    ]


def validate_source_digest(source_id: str, source_parquet: Path) -> str:
    """Assert one source file still matches its pinned manifest digest."""
    expected = EXPECTED_SOURCE_SHA256.get(source_id)
    if expected is None:
        raise ValueError(f"no pinned digest for source {source_id!r}")
    actual = file_sha256(source_parquet)
    if actual != expected:
        raise ValueError(
            f"source digest drift for {source_id}: expected {expected}, found {actual}"
        )
    return actual


__all__ = [
    "EXPECTED_SOURCE_ROWS",
    "EXPECTED_SOURCE_SHA256",
    "source_profiles",
    "validate_source_digest",
]
