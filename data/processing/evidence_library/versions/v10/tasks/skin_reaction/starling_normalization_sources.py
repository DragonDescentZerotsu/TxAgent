"""Pinned source contracts for the layered Skin_Reaction normalizer.

Row counts and file digests are the ones frozen in
``data/raw/starling/skin_reaction/SOURCE_MANIFEST.json``.  Unlike
Bioavailability_Ma there is no separate HuggingFace direct source: all four
mechanism families are Starling extraction parquets that carry their own
structure column, which is spelled ``SMILES`` (uppercase) in every file.
"""

from __future__ import annotations

from pathlib import Path

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.normalization.contracts import NormalizedSourceProfile


EXPECTED_SOURCE_ROWS = {
    "direct_skin_reaction": 66_597,
    "sensitization_aop": 45_985,
    "phototoxicity_irritation_local_damage": 382_726,
    "skin_exposure": 311_834,
}

EXPECTED_SOURCE_SHA256 = {
    "direct_skin_reaction": "ce8aca4ebb4bbbd9cdf1f138a6f1b65d527f591b8acac81d69bdfc9f49bbb27a",
    "sensitization_aop": "821b35a0a56ddbac849af97ebf622e38d2331a81a99f69306a7ddd4c39bd9a12",
    "phototoxicity_irritation_local_damage": "fe6d272c5b0a3a7e244f123e1c810ef56c21c919f40d3ed44a6563c7da53875a",
    "skin_exposure": "e2ba14d68a2235e984b3ee84b05e7aeb03920a46a4cd34122280385267399e15",
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
