import pandas as pd
import pyarrow.parquet as pq

from tools.chembl_tool.tasks.bioavailability_ma.build_nondirect_assay_context import (
    SOURCE_GROUP,
    build_overlay,
    exposure_context,
    nondirect_assay_context,
    population_context,
)


def test_context_taxonomy_is_bounded_and_animal_first():
    assert population_context("male Sprague-Dawley rats") == "rat"
    assert population_context("healthy male volunteers") == "healthy human"
    assert exposure_context("SNEDDS oral formulation") == "lipid or self-emulsifying formulation"
    assert exposure_context("oral tablet") == "tablet"
    assert nondirect_assay_context(
        report_type="relative_comparison",
        species_or_population="patients",
        oral_exposure_mode="tablet",
    ) == (
        "literature oral bioavailability; report=relative comparison; "
        "population=human clinical; exposure=tablet"
    )


def test_overlay_changes_only_missing_target_contexts(tmp_path):
    input_path = tmp_path / "records.parquet"
    output_dir = tmp_path / "overlay"
    pd.DataFrame(
        [
            {
                "canonical_assay_context": None,
                "canonical_bioavailability_report_type": "relative_comparison",
                "group_id": SOURCE_GROUP,
                "oral_exposure_mode": "tablet",
                "retrieval_eligible": True,
                "source_id": "hf_bioavailability",
                "species_or_population": "healthy volunteers",
                "payload": "target",
            },
            {
                "canonical_assay_context": "existing context",
                "canonical_bioavailability_report_type": "__unknown__",
                "group_id": SOURCE_GROUP,
                "oral_exposure_mode": None,
                "retrieval_eligible": True,
                "source_id": "hf_bioavailability",
                "species_or_population": None,
                "payload": "existing",
            },
            {
                "canonical_assay_context": None,
                "canonical_bioavailability_report_type": "__unknown__",
                "group_id": "Fa.absorption_solubility_permeability",
                "oral_exposure_mode": None,
                "retrieval_eligible": True,
                "source_id": "fa",
                "species_or_population": None,
                "payload": "other",
            },
        ]
    ).to_parquet(input_path, index=False)

    manifest = build_overlay(input_path, output_dir)
    output = pq.read_table(output_dir / "records.parquet").to_pandas()

    assert manifest["n_rows_enriched"] == 1
    assert manifest["n_contexts_added"] == 1
    assert output.loc[0, "canonical_assay_context"].startswith(
        "literature oral bioavailability;"
    )
    assert output.loc[1, "canonical_assay_context"] == "existing context"
    assert pd.isna(output.loc[2, "canonical_assay_context"])
    assert output["payload"].tolist() == ["target", "existing", "other"]
