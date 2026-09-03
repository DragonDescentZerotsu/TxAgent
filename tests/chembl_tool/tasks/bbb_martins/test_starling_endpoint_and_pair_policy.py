import json

import pandas as pd

from data.processing.evidence_library.shared.v1.pair_buckets import materialize_pair_buckets
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.build_starling_pair_bucket_sidecar import (
    build_sidecar,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_endpoint_normalization import (
    context_fields,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_normalization_sources import (
    source_profiles,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_pair_buckets import (
    BBB_MARTINS_PAIR_BUCKET_VERSION,
    BBB_MARTINS_V7_PAIR_BUCKET_VERSION,
    SOURCE_PAIR_FIELDS,
)
from data.processing.evidence_library.versions.v7.tasks.bbb_martins.starling_source_column_contracts import (
    normalized_column_contract,
)


def test_v7_pair_bucket_uses_reviewed_endpoint_concept(tmp_path):
    records = [
        {
            "canonical_record_id": f"record-{number}",
            "source_id": "direct_bbb",
            "canonical_endpoint_name": endpoint,
            "canonical_endpoint_concept": "log_ps",
            "canonical_unit_text": "log10(cm/s)",
            "canonicalization_status": "valid",
            "retrieval_eligible": True,
            "canonical_smiles": smiles,
            "canonical_measurement_scale_id": None,
            "canonical_assay_context": "passive permeability",
            "canonical_species_context": "rat",
            "canonical_reference_scope": "absolute",
            "canonical_reference_basis": "not_applicable",
            "measurement_kind": "continuous",
            "finite_scalar_value": float(number),
        }
        for number, (endpoint, smiles) in enumerate(
            (("log_ps", "CCO"), ("logps", "CCN")), start=1
        )
    ]
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(records).to_parquet(records_path, index=False)

    metadata = build_sidecar(records_path=records_path, out_dir=tmp_path / "pairs")
    sidecar = pd.read_parquet(tmp_path / "pairs/pair_bucket_records.parquet")

    assert metadata["contract_version"] == BBB_MARTINS_V7_PAIR_BUCKET_VERSION
    assert metadata["bucket_endpoint_field_by_source"]["direct_bbb"] == (
        "canonical_endpoint_concept"
    )
    assert sidecar["pair_bucket_key"].nunique() == 1
    assert json.loads(sidecar.iloc[0]["pair_bucket_key"])[1] == "log_ps"


def test_source_profiles_promote_measurement_endpoints_and_retain_policy_context(tmp_path):
    profiles = {profile.source_id: profile for profile in source_profiles(tmp_path)}
    assert profiles["direct_bbb"].endpoint_field == "quant_metric"
    assert profiles["passive_permeability"].endpoint_field == "metric_name"
    assert profiles["efflux_transport"].endpoint_field == "quantitative_metric"
    assert profiles["influx_transport"].endpoint_field == "transport_endpoint"
    assert "assay_type" in profiles["passive_permeability"].context_fields
    assert "evidence_type" in profiles["efflux_transport"].context_fields
    assert "transport_mechanism" in profiles["influx_transport"].context_fields


def test_context_fields_are_canonical_and_source_specific():
    passive = context_fields(
        {"source_id": "passive_permeability", "assay_type": "PAMPA-BBB"}
    )
    efflux = context_fields(
        {"source_id": "efflux_transport", "evidence_type": "qualitative transpor_claim"}
    )
    influx = context_fields(
        {"source_id": "influx_transport", "transport_mechanism": "carrier‑mediated influx"}
    )
    assert passive["canonical_assay_type"] == "pampa_bbb"
    assert efflux["canonical_evidence_type"] == "qualitative_transporter_claim"
    assert influx["canonical_transport_mechanism"] == "carrier_mediated_influx"


def _record(record_id, source, endpoint, unit, **fields):
    return {
        "normalized_record_id": record_id,
        "source_id": source,
        "canonical_endpoint": endpoint,
        "canonical_unit": unit,
        "normalization_validity_status": "valid" if unit else "missing_canonical_unit",
        "molecule_id": f"molecule-{record_id}",
        **fields,
    }


def test_pair_bucket_tuple_shapes_match_the_source_contract():
    records = [
        _record("d", "direct_bbb", "logbb", "dimensionless", categorical_encoder_id=None, global_context="in vivo", global_species_context="rat"),
        _record("p", "passive_permeability", "papp_a_to_b", "cm/s", categorical_encoder_id=None, canonical_assay_type="mdck", global_context="mdck", global_species_context="human"),
        _record("e", "efflux_transport", "efflux_ratio", "ratio", categorical_encoder_id=None, transporter_identifier="ABCB1", canonical_evidence_type="bidirectional_transport", global_context="mdck-mdr1", global_species_context="human"),
        _record("i", "influx_transport", "brain_uptake", "ratio", canonical_transport_mechanism="carrier_mediated_influx"),
    ]
    rows, audit = materialize_pair_buckets(
        records,
        source_required_fields=SOURCE_PAIR_FIELDS,
        contract_version=BBB_MARTINS_PAIR_BUCKET_VERSION,
    )
    assert all(row["bucket_eligible"] for row in rows)
    decoded = {row["source_id"]: json.loads(row["pair_bucket_key"]) for row in rows}
    assert decoded["direct_bbb"] == ["direct_bbb", "logbb", "dimensionless", "__unknown__", "in vivo", "rat"]
    assert decoded["passive_permeability"] == ["passive_permeability", "papp_a_to_b", "cm/s", "__unknown__", "mdck", "mdck", "human"]
    assert decoded["efflux_transport"] == ["efflux_transport", "efflux_ratio", "ratio", "__unknown__", "ABCB1", "bidirectional_transport", "mdck-mdr1", "human"]
    assert decoded["influx_transport"] == ["influx_transport", "brain_uptake", "ratio", "carrier_mediated_influx"]
    assert audit["validations"]["no_bucket_spans_sources"] is True


def test_unresolved_unit_is_retained_in_sidecar_but_not_bucketed():
    records = [
        _record(
            "p",
            "passive_permeability",
            "papp_a_to_b",
            None,
            categorical_encoder_id=None,
            canonical_assay_type="mdck",
            global_context="mdck",
            global_species_context="human",
            retrieval_eligible=True,
        )
    ]
    rows, _ = materialize_pair_buckets(records, source_required_fields=SOURCE_PAIR_FIELDS)
    assert len(rows) == 1
    assert rows[0]["bucket_eligible"] is False
    assert rows[0]["pair_bucket_key"] is not None
    assert rows[0]["bucket_exclusion_reason"] == "missing_canonical_unit"


def test_unresolved_endpoint_sentinel_is_not_bucketed():
    records = [
        _record(
            "d",
            "direct_bbb",
            "missing_endpoint",
            "binary_outcome_class",
            categorical_encoder_id="bbb_permeability_binary.v1",
            global_context="in vivo",
            global_species_context="rat",
            retrieval_eligible=True,
        )
    ]

    rows, _ = materialize_pair_buckets(
        records, source_required_fields=SOURCE_PAIR_FIELDS
    )

    assert rows[0]["bucket_eligible"] is False
    assert rows[0]["pair_bucket_key"] is not None
    assert rows[0]["bucket_exclusion_reason"] == "unresolved_endpoint"


def test_source_alias_provenance_is_fail_closed_per_source():
    columns = [
        "confidence",
        "measurement_text",
        "unit_text",
        "canonical_assay_type",
        "canonical_evidence_type",
        "canonical_transport_mechanism",
        "canonical_unit_resolution_source",
    ]
    direct = normalized_column_contract("direct_bbb", columns)
    efflux = normalized_column_contract("efflux_transport", columns)
    influx = normalized_column_contract("influx_transport", columns)
    assert direct["confidence"] is False
    assert efflux["measurement_text"] is True
    assert efflux["unit_text"] is False
    assert influx["measurement_text"] is False
    assert influx["unit_text"] is False
    for contract in (direct, efflux, influx):
        assert contract["canonical_assay_type"] is False
        assert contract["canonical_evidence_type"] is False
        assert contract["canonical_transport_mechanism"] is False
        assert contract["canonical_unit_resolution_source"] is False
