import json

import pandas as pd
import pytest

from tools.chembl_tool.common.starling.normalization.cleaning import file_sha256
from tools.chembl_tool.common.starling.pair_buckets import (
    UNKNOWN_TOKEN,
    materialize_pair_buckets,
)
from tools.chembl_tool.tasks.bioavailability_ma.build_starling_pair_bucket_sidecar import (
    build_sidecar,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_pair_buckets import (
    BIOAVAILABILITY_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)


def _record(
    number,
    *,
    source="fa",
    endpoint="absorption",
    unit="%",
    status="valid",
    smiles="CCO",
    **canonical_fields,
):
    return {
        "normalized_record_id": f"record-{number}",
        "source_id": source,
        "canonical_endpoint": endpoint,
        "canonical_unit": unit,
        "normalization_validity_status": status,
        "canonical_smiles": smiles,
        "molecule_id": f"molecule-{smiles}",
        **canonical_fields,
    }


def _materialize(records):
    return materialize_pair_buckets(
        records,
        source_required_fields=SOURCE_PAIR_FIELDS,
        contract_version=BIOAVAILABILITY_PAIR_BUCKET_VERSION,
    )


def test_bioavailability_source_field_mapping_uses_only_persisted_canonical_fields():
    assert SOURCE_PAIR_FIELDS == {
        "direct_hf": ("canonical_bioavailability_report_type",),
        "oral_exposure": (),
        "fa": ("global_context", "global_species_context"),
        "fg": ("global_context", "global_species_context"),
        "fh": ("global_context", "global_species_context"),
    }


def test_materializer_maps_persisted_values_without_recanonicalizing_them():
    records = [
        _record(1, global_context="Caco-2|EXACT", global_species_context="human"),
        _record(2, smiles="CCN", global_context="caco_2|exact", global_species_context="human"),
    ]
    rows, metadata = _materialize(records)
    assert rows[0]["pair_bucket_key"] != rows[1]["pair_bucket_key"]
    assert metadata["stats"]["buckets"] == 2


def test_unknown_fields_match_unknown_and_sources_remain_distinct():
    records = [
        _record(
            1,
            source="fh",
            endpoint="intrinsic_clearance",
            unit="mL/min/kg",
            global_species_context=None,
            global_context="liver microsomes",
        ),
        _record(
            2,
            source="fh",
            endpoint="intrinsic_clearance",
            unit="mL/min/kg",
            smiles="CCN",
            global_species_context=None,
            global_context="liver microsomes",
        ),
        _record(
            3,
            source="fh",
            endpoint="intrinsic_clearance",
            unit="mL/min/kg",
            smiles="CCC",
            global_species_context="rat",
            global_context="liver microsomes",
        ),
    ]
    rows, metadata = _materialize(records)
    fields = json.loads(rows[0]["canonical_pair_fields_json"])
    assert fields["global_species_context"] == UNKNOWN_TOKEN
    assert rows[0]["pair_bucket_key"] == rows[1]["pair_bucket_key"]
    assert rows[0]["pair_bucket_key"] != rows[2]["pair_bucket_key"]
    assert metadata["unknown_field_rates"]["global_species_context"] == pytest.approx(2 / 3)
    assert all(metadata["validations"].values())


def test_oral_exposure_dose_is_metadata_not_a_bucket_boundary():
    records = [
        _record(
            1,
            source="oral_exposure",
            endpoint="auc",
            unit="ng·h/mL",
            canonical_dose_quantity_kind="mass",
            canonical_dose_basis="per_kg",
            canonical_dose_bin="log2:3",
            canonical_dose_regimen="single",
        ),
        _record(
            2,
            source="oral_exposure",
            endpoint="auc",
            unit="ng·h/mL",
            smiles="CCN",
            canonical_dose_quantity_kind="molar",
            canonical_dose_basis="absolute",
            canonical_dose_bin="log2:10",
            canonical_dose_regimen="repeated",
        ),
    ]
    rows, metadata = _materialize(records)
    assert json.loads(rows[0]["pair_bucket_key"]) == [
        "oral_exposure",
        "auc",
        "ng·h/mL",
    ]
    assert rows[0]["pair_bucket_key"] == rows[1]["pair_bucket_key"]
    assert metadata["stats"]["buckets"] == 1


@pytest.mark.parametrize(
    "status",
    [
        "unresolved_structure",
        "non_scalar_measurement",
        "outside_bounded_percentage_domain",
        "incompatible_canonical_unit",
    ],
)
def test_normalization_validity_excludes_without_sidecar_recomputation(status):
    rows, metadata = _materialize(
        [_record(1, status=status, global_context="caco_2")]
    )
    assert rows[0]["bucket_eligible"] is False
    assert rows[0]["pair_bucket_key"] is None
    assert rows[0]["bucket_exclusion_reason"] == status
    assert metadata["validations"]["ineligible_records_have_no_bucket"] is True


def test_materializer_rejects_unmapped_sources():
    with pytest.raises(ValueError, match="no pair-bucket field mapping"):
        _materialize(
            [_record(1, source="new_source", global_context="aqueous")]
        )


def test_standalone_builder_writes_exactly_two_files_without_rewriting_v5(tmp_path):
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            _record(1, global_context="caco_2"),
            _record(
                2,
                smiles="CCN",
                global_context="caco_2",
            ),
        ]
    ).to_parquet(records_path, index=False)
    original_hash = file_sha256(records_path)
    output = tmp_path / "pair_buckets"
    output.mkdir()
    for legacy in (
        "endpoint_pair_registry.json",
        "pair_bucket_audit.json",
        "manifest.json",
    ):
        (output / legacy).write_text("legacy", encoding="utf-8")
    metadata = build_sidecar(records_path=records_path, out_dir=output)
    assert file_sha256(records_path) == original_hash
    assert metadata["stats"] == {
        "input_records": 2,
        "sidecar_records": 2,
        "eligible_records": 2,
        "excluded_records": 0,
        "buckets": 1,
        "pairable_buckets": 1,
        "singleton_bucket_rate": 0.0,
    }
    assert {
        "pair_bucket_records.parquet",
        "pair_bucket_metadata.json",
    } == {path.name for path in output.iterdir()}
    sidecar = pd.read_parquet(output / "pair_bucket_records.parquet")
    assert list(sidecar.columns) == [
        "normalized_record_id",
        "source_id",
        "canonical_endpoint",
        "canonical_unit",
        "canonical_pair_fields_json",
        "pair_bucket_key",
        "bucket_eligible",
        "bucket_exclusion_reason",
    ]
