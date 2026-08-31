"""End-to-end bounded build of the Skin_Reaction normalized library.

This exercises the shared staged builder through the Skin_Reaction policy on a
small slice of the real sources, so it catches contract drift between the two
without a full multi-hundred-thousand-row build.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.chembl_tool.common.starling.categorical_response import ENCODED_UNITS
from tools.chembl_tool.tasks.skin_reaction.starling_categorical_response import (
    UNINFORMATIVE_LABELS,
)

from tools.chembl_tool.tasks.skin_reaction import (
    build_normalized_starling_evidence_library as builder,
)
from tools.chembl_tool.tasks.skin_reaction.build_starling_pair_bucket_sidecar import (
    build_sidecar,
)
from tools.chembl_tool.tasks.skin_reaction.starling_normalization_sources import (
    source_profiles,
)

DATA_DIR = Path("data/starling_data/skin_reaction")
QUALITATIVE_SOURCES = {
    "direct_skin_reaction",
    "phototoxicity_irritation_local_damage",
}
SCALAR_SOURCES = {"sensitization_aop", "skin_exposure"}


@pytest.fixture(scope="module")
def bounded_build(tmp_path_factory):
    if any(not Path(p.source_path).exists() for p in source_profiles(DATA_DIR)):
        pytest.skip("Starling skin_reaction sources are not staged")
    out_dir = tmp_path_factory.mktemp("skin_v6")
    assert (
        builder.main(
            [
                "--out-dir",
                str(out_dir),
                "--max-rows-per-source",
                "400",
                "--no-strict-endpoint-inventory",
                "--allow-partial-measurement-resolution",
                "--workers",
                "2",
            ]
        )
        == 0
    )
    return out_dir


def test_core_build_publishes_only_three_stage_artifacts(bounded_build):
    for relative in (
        "01_cleaned/records.parquet",
        "02_canonicalized/records.parquet",
        "03_pair_buckets/records.parquet",
        "03_pair_buckets/pair_bucket_records.parquet",
        "manifest.json",
    ):
        assert (bounded_build / relative).exists(), relative
    assert not (bounded_build / "04_pair_buckets").exists()
    assert not (bounded_build / "03_records").exists()


def test_normalized_children_preserve_cleaned_record_lineage(bounded_build):
    cleaned = pd.read_parquet(bounded_build / "01_cleaned/records.parquet")
    normalized = pd.read_parquet(bounded_build / "02_canonicalized/records.parquet")
    assert len(normalized) >= len(cleaned)
    assert 0 < len(cleaned) < 1600
    assert set(cleaned["cleaned_record_id"]) == set(
        normalized["measurement_resolution_parent_cleaned_record_id"]
    )


def test_categorical_sources_get_a_scalar_only_through_a_declared_encoder(
    bounded_build,
):
    """These sources report outcomes, not measurements.

    Any scalar they carry must come from a named categorical encoder — never
    from a parsed source measurement, which would mean the pipeline had
    silently invented a number out of an outcome label.
    """
    records = pd.read_parquet(bounded_build / "03_pair_buckets/records.parquet")
    qualitative = records[records["source_id"].isin(QUALITATIVE_SOURCES)]
    assert len(qualitative) > 0
    with_scalar = qualitative[qualitative["finite_scalar_value"].notna()]
    assert len(with_scalar) > 0, "the encoders unlocked nothing"
    assert set(with_scalar["measurement_kind"]) <= {
        "continuous",
        "binary",
        "ordinal",
    }
    categorical = with_scalar[
        with_scalar["measurement_kind"].isin({"binary", "ordinal"})
    ]
    assert (
        categorical["canonical_measurement_scale_id"].fillna("").astype(str) != ""
    ).all()
    assert categorical["canonical_category_id"].notna().all()
    assert categorical["canonical_category_rank"].notna().all()
    assert categorical["canonical_unit_text"].isin(ENCODED_UNITS).all()
    # Records the encoders decline stay evidence, not silently dropped.
    assert qualitative["finite_scalar_value"].isna().any()
    assert "retrieval_eligible" not in qualitative


def test_uninformative_outcomes_are_never_encoded(bounded_build):
    """"not classified" is absence of information, not evidence of no effect."""
    records = pd.read_parquet(bounded_build / "03_pair_buckets/records.parquet")
    encoded = records[
        records["canonical_measurement_scale_id"].fillna("").astype(str) != ""
    ]
    labels = encoded["result_label"].fillna("").astype(str).str.casefold()
    assert not labels.isin(UNINFORMATIVE_LABELS).any()


def test_every_encoded_value_is_finite(bounded_build):
    """An unshrunk rate of 0 or 1 would have an infinite logit."""
    records = pd.read_parquet(bounded_build / "03_pair_buckets/records.parquet")
    encoded = records[
        records["canonical_measurement_scale_id"].fillna("").astype(str) != ""
    ]
    values = encoded["finite_scalar_value"].dropna()
    assert len(values) > 0
    assert np.isfinite(values.to_numpy(dtype=float)).all()


def test_no_bucket_mixes_units_encoders_or_sources(bounded_build, tmp_path):
    """Scales that are not comparable must never share a comparison stratum."""
    records_path = _records_with_test_auxiliary_context(bounded_build, tmp_path)
    build_sidecar(
        records_path=records_path,
        out_dir=tmp_path / "06_pair_buckets",
    )
    buckets = pd.read_parquet(tmp_path / "06_pair_buckets/pair_bucket_records.parquet")
    joined = buckets[buckets["pair_bucket_key"].notna()]
    grouped = joined.groupby("pair_bucket_key")
    assert grouped["canonical_unit_text"].nunique().max() == 1
    assert grouped["source_id"].nunique().max() == 1
    # Incidence and single-subject outcomes share the fraction unit, so the
    # measurement scale remains part of the pair key.
    assert grouped["canonical_measurement_scale_id"].nunique().max() == 1


def test_scalar_sources_produce_scalars_with_canonical_units(bounded_build):
    records = pd.read_parquet(bounded_build / "03_pair_buckets/records.parquet")
    scalar = records[records["source_id"].isin(SCALAR_SOURCES)]
    with_value = scalar[scalar["finite_scalar_value"].notna()]
    assert len(with_value) > 0
    assert with_value["canonical_unit_text"].notna().all()


def test_manifest_records_task_scoped_versions(bounded_build):
    manifest = json.loads((bounded_build / "manifest.json").read_text(encoding="utf-8"))
    assert "index_version" not in manifest
    assert "compact_artifact_version" not in manifest
    assert manifest["source_inventory"]["dataset"] == "starling-labs/Skin_Reaction"
    assert manifest["categorical_response_version"].startswith("skin_reaction")


def test_auxiliary_mapping_is_complete_before_stage_three(bounded_build):
    stage = json.loads(
        (bounded_build / "02_canonicalized/manifest.json").read_text(encoding="utf-8")
    )
    assert stage["validations"]["globally_reconciled_auxiliary_coverage"] is True


def test_every_source_can_now_reach_a_bucket(bounded_build, tmp_path):
    """Before the encoders only the two measured sources could be compared."""
    records_path = _records_with_test_auxiliary_context(bounded_build, tmp_path)
    build_sidecar(
        records_path=records_path,
        out_dir=tmp_path / "06_pair_buckets",
    )
    buckets = pd.read_parquet(tmp_path / "06_pair_buckets/pair_bucket_records.parquet")
    bucketed = buckets[buckets["pair_bucket_key"].notna()]
    assert set(bucketed["source_id"]) == QUALITATIVE_SOURCES | SCALAR_SOURCES


def test_sensitization_uses_reviewed_endpoint_concept(
    bounded_build, tmp_path
):
    records_path = _records_with_test_auxiliary_context(bounded_build, tmp_path)
    build_sidecar(
        records_path=records_path,
        out_dir=tmp_path / "endpoint_pair_buckets",
    )
    buckets = pd.read_parquet(
        tmp_path / "endpoint_pair_buckets/pair_bucket_records.parquet"
    )
    sensitization = buckets[buckets["source_id"] == "sensitization_aop"]
    eligible = sensitization[sensitization["pair_bucket_key"].notna()]
    assert len(eligible) > 0
    metadata = json.loads(
        (tmp_path / "endpoint_pair_buckets/pair_bucket_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["bucket_endpoint_field_by_source"]["sensitization_aop"] == (
        "canonical_endpoint_concept"
    )
    records = pd.read_parquet(records_path).set_index("canonical_record_id")
    concept_by_id = records["canonical_endpoint_concept"].astype(str)
    endpoint_by_id = records["canonical_endpoint_name"].astype(str)
    assert all(
        json.loads(row.pair_bucket_key)[1] == concept_by_id[row.canonical_record_id]
        for row in eligible.itertuples()
    )
    assert all(
        row.canonical_endpoint_name == endpoint_by_id[row.canonical_record_id]
        for row in eligible.itertuples()
    )


def _records_with_test_auxiliary_context(bounded_build, tmp_path):
    records = pd.read_parquet(bounded_build / "02_canonicalized/records.parquet")
    records["canonical_assay_or_test"] = records["source_id"].map(
        {
            "direct_skin_reaction": "patch test",
        }
    )
    records["canonical_species_or_population"] = records["source_id"].map(
        {
            "direct_skin_reaction": "human",
        }
    )
    records["canonical_assay_type"] = records["source_id"].map(
        {"sensitization_aop": "in vitro sensitization assay"}
    )
    records["canonical_species_context"] = records["source_id"].map(
        {"sensitization_aop": "human cell", "skin_exposure": "human"}
    )
    records["canonical_assay_method"] = records["source_id"].map(
        {"phototoxicity_irritation_local_damage": "3t3 nru phototoxicity"}
    )
    records["canonical_evidence_system"] = records["source_id"].map(
        {"phototoxicity_irritation_local_damage": "mouse fibroblast"}
    )
    records["canonical_study_design"] = records["source_id"].map(
        {"skin_exposure": "ex vivo skin permeation"}
    )
    sensitization = records["source_id"] == "sensitization_aop"
    records.loc[sensitization, "canonical_endpoint_name"] = (
        records.loc[sensitization, "endpoint_name"]
        .fillna("sensitization endpoint")
        .astype(str)
        .str.casefold()
    )
    output = tmp_path / "records_with_auxiliary.parquet"
    records.to_parquet(output, index=False)
    return output
