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
                "--allow-missing-auxiliary-mapping",
                "--workers",
                "2",
            ]
        )
        == 0
    )
    return out_dir


def test_pending_mapping_build_publishes_only_complete_record_stages(bounded_build):
    for relative in (
        "01_cleaned/records.parquet",
        "02_normalized/records.parquet",
        "03_records/records.parquet",
        "manifest.json",
    ):
        assert (bounded_build / relative).exists(), relative
    assert not (bounded_build / "04_pair_buckets").exists()


def test_one_normalized_record_per_cleaned_record(bounded_build):
    cleaned = pd.read_parquet(bounded_build / "01_cleaned/records.parquet")
    normalized = pd.read_parquet(bounded_build / "02_normalized/records.parquet")
    assert len(cleaned) == len(normalized) == 1600
    assert set(cleaned["cleaned_record_id"]) == set(normalized["cleaned_record_id"])


def test_categorical_sources_get_a_scalar_only_through_a_declared_encoder(
    bounded_build,
):
    """These sources report outcomes, not measurements.

    Any scalar they carry must come from a named categorical encoder — never
    from a parsed source measurement, which would mean the pipeline had
    silently invented a number out of an outcome label.
    """
    records = pd.read_parquet(bounded_build / "03_records/records.parquet")
    qualitative = records[records["source_id"].isin(QUALITATIVE_SOURCES)]
    assert len(qualitative) > 0
    with_scalar = qualitative[qualitative["finite_scalar_value"].notna()]
    assert len(with_scalar) > 0, "the encoders unlocked nothing"
    assert (
        with_scalar["categorical_encoder_id"].fillna("").astype(str) != ""
    ).all()
    assert with_scalar["is_absolute_and_continuous"].all()
    assert with_scalar["canonical_unit"].isin(ENCODED_UNITS).all()
    # Records the encoders decline stay evidence, not silently dropped.
    assert qualitative["retrieval_eligible"].any()


def test_uninformative_outcomes_are_never_encoded(bounded_build):
    """"not classified" is absence of information, not evidence of no effect."""
    records = pd.read_parquet(bounded_build / "03_records/records.parquet")
    encoded = records[
        records["categorical_encoder_id"].fillna("").astype(str) != ""
    ]
    labels = encoded["result_label"].fillna("").astype(str).str.casefold()
    assert not labels.isin(UNINFORMATIVE_LABELS).any()


def test_every_encoded_value_is_finite(bounded_build):
    """An unshrunk rate of 0 or 1 would have an infinite logit."""
    records = pd.read_parquet(bounded_build / "03_records/records.parquet")
    encoded = records[
        records["categorical_encoder_id"].fillna("").astype(str) != ""
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
    records = pd.read_parquet(
        bounded_build / "03_records/records.parquet",
        columns=["normalized_record_id", "categorical_encoder_id"],
    )
    joined = buckets[buckets["pair_bucket_key"].notna()].merge(
        records, on="normalized_record_id"
    )
    grouped = joined.groupby("pair_bucket_key")
    assert grouped["canonical_unit"].nunique().max() == 1
    assert grouped["source_id"].nunique().max() == 1
    # count_logit, single_subject_logit and percent_positive_logit share the
    # logit_response unit, so the encoder id is what keeps them apart.
    assert grouped["categorical_encoder_id"].nunique().max() == 1


def test_scalar_sources_produce_scalars_with_canonical_units(bounded_build):
    records = pd.read_parquet(bounded_build / "03_records/records.parquet")
    scalar = records[records["source_id"].isin(SCALAR_SOURCES)]
    with_value = scalar[scalar["finite_scalar_value"].notna()]
    assert len(with_value) > 0
    assert with_value["canonical_unit"].notna().all()


def test_manifest_records_task_scoped_versions(bounded_build):
    manifest = json.loads((bounded_build / "manifest.json").read_text(encoding="utf-8"))
    assert "index_version" not in manifest
    assert "compact_artifact_version" not in manifest
    assert manifest["source_inventory"]["dataset"] == "starling-labs/Skin_Reaction"
    assert manifest["categorical_response_version"].startswith("skin_reaction")


def test_incomplete_auxiliary_mapping_fails_its_coverage_validation(bounded_build):
    stage = json.loads(
        (bounded_build / "02_normalized/manifest.json").read_text(encoding="utf-8")
    )
    assert stage["validations"]["globally_reconciled_auxiliary_coverage"] is False


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


def test_sensitization_uses_reconciled_endpoint_without_rewriting_record(
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
        "global_endpoint_context"
    )
    records = pd.read_parquet(records_path).set_index("normalized_record_id")
    endpoint_by_id = records["global_endpoint_context"].astype(str)
    assert all(
        json.loads(row.pair_bucket_key)[1] == endpoint_by_id[row.normalized_record_id]
        for row in eligible.itertuples()
    )
    assert any(
        row.canonical_endpoint != endpoint_by_id[row.normalized_record_id]
        for row in eligible.itertuples()
    )


def _records_with_test_auxiliary_context(bounded_build, tmp_path):
    records = pd.read_parquet(bounded_build / "03_records/records.parquet")
    records["global_context"] = records["source_id"].map(
        {
            "direct_skin_reaction": "patch test",
            "sensitization_aop": "in vitro sensitization assay",
            "phototoxicity_irritation_local_damage": "3t3 nru phototoxicity",
            "skin_exposure": "ex vivo skin permeation",
        }
    )
    records["global_species_context"] = records["source_id"].map(
        {
            "direct_skin_reaction": "human",
            "sensitization_aop": "human cell",
            "phototoxicity_irritation_local_damage": "mouse fibroblast",
            "skin_exposure": "human",
        }
    )
    sensitization = records["source_id"] == "sensitization_aop"
    records.loc[sensitization, "global_endpoint_context"] = (
        records.loc[sensitization, "endpoint_or_target"]
        .fillna("sensitization endpoint")
        .astype(str)
        .str.casefold()
    )
    output = tmp_path / "records_with_auxiliary.parquet"
    records.to_parquet(output, index=False)
    return output
