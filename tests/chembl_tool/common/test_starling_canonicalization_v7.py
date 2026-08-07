from __future__ import annotations

import json

import pytest

from tools.chembl_tool.common.starling.canonicalization_v7 import (
    CanonicalDimensionSpec,
    CanonicalProducerSpec,
    PairBucketSpec,
    SourceProfile,
    StarlingRecordContract,
)
from tools.chembl_tool.common.starling.pair_buckets import materialize_pair_buckets
from tools.chembl_tool.common.starling.normalization.cleaning import (
    resolve_structure_value,
    stable_id,
)
from tools.chembl_tool.common.starling.normalization.organization import (
    deduplicate_within_source,
)
from tools.chembl_tool.tasks.bbb_martins.starling_schema import (
    RECORD_CONTRACT as BBB_CONTRACT,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_schema import (
    RECORD_CONTRACT as BIO_CONTRACT,
)
from tools.chembl_tool.tasks.skin_reaction.starling_schema import (
    RECORD_CONTRACT as SKIN_CONTRACT,
)


def test_one_cleaned_input_can_feed_multiple_canonical_dimensions() -> None:
    profile = SKIN_CONTRACT.source("sensitization_aop")
    assay = next(
        item
        for item in profile.canonical_dimensions
        if item.output_field == "canonical_assay_type"
    )
    species = next(
        item
        for item in profile.canonical_dimensions
        if item.output_field == "canonical_species_context"
    )
    assert assay.input_fields == ("assay_type",)
    assert species.input_fields == ("assay_type",)
    assert "species_context" not in profile.cleaned_source_fields


def test_mapped_structure_keeps_raw_source_smiles_in_cleaned_view() -> None:
    profile = BIO_CONTRACT.source("fa")
    row = {
        "source_id": "fa",
        "endpoint_name": "Fa",
        "measurement_text": "0.7",
        "unit_text": None,
        "source_smiles": "MAPPED",
        "canonical_smiles": "CCO",
        "source_payload_json": json.dumps(
            {
                **{field: None for field in profile.source_columns},
                "endpoint_category": "Fa",
                "reported_value": "0.7",
                "smiles": "RAW-SOURCE",
                "global_identifier": "compound-1",
            }
        ),
    }
    cleaned = BIO_CONTRACT.clean_projection(row)
    assert cleaned["smiles"] == "RAW-SOURCE"
    assert "source_smiles" not in cleaned
    assert "endpoint_category" not in cleaned


def test_dataset_constants_are_not_claimed_as_source_visible() -> None:
    profile = BIO_CONTRACT.source("hf_bioavailability")
    assert "endpoint_name" in profile.cleaned_source_fields
    assert "unit_text" in profile.cleaned_source_fields
    assert "endpoint_name" not in profile.source_visible_fields
    # The unified source uses embedded units because nondirect rows may carry
    # percent/fold/ratio text; unit_text is no longer a dataset constant.
    assert "unit_text" in profile.source_visible_fields


def test_all_pair_identity_fields_are_canonical() -> None:
    for contract in (BBB_CONTRACT, BIO_CONTRACT, SKIN_CONTRACT):
        for source_id, spec in contract.pair_buckets.items():
            assert spec.canonical_dimensions
            assert all(
                field.startswith("canonical_")
                for field in spec.canonical_dimensions
            )
            canonical_inputs = {
                field
                for dimension in contract.source(source_id).canonical_dimensions
                if (
                    dimension.method != "controlled_encoder"
                    and not dimension.classification_evidence
                )
                for field in dimension.input_fields
            }
            canonical_inputs.update(
                field
            for dimension in contract.source(source_id).canonical_dimensions
            if not dimension.classification_evidence
            for producer in dimension.producer_variants
                if producer.method != "controlled_encoder"
                for field in producer.input_fields
            )
            assert not canonical_inputs & set(spec.variance_candidates)


def test_controlled_only_input_can_remain_a_continuous_variance_candidate() -> None:
    scale = BIO_CONTRACT.measurement_scales["fg_substrate_status_binary.v1"]
    candidates = set(BIO_CONTRACT.pair_buckets["fg"].variance_candidates)
    assert {"substrate_status", "transporter_or_enzyme"} <= set(
        scale.input_fields
    )
    assert {"substrate_status", "transporter_or_enzyme"} <= candidates


def test_bbb_scalar_and_categorical_producers_are_both_declared() -> None:
    for source_id in ("direct_bbb", "passive_permeability", "efflux_transport"):
        profile = BBB_CONTRACT.source(source_id)
        measurement = next(
            item
            for item in profile.canonical_dimensions
            if item.output_field == "canonical_measurement_text"
        )
        unit = next(
            item
            for item in profile.canonical_dimensions
            if item.output_field == "canonical_unit_text"
        )
        assert measurement.input_fields == (
            "endpoint_name",
            "measurement_text",
            "unit_text",
        )
        assert unit.input_fields == measurement.input_fields
        assert measurement.producer_id_field == "canonical_pair_producer_id"
        assert unit.producer_id_field == "canonical_pair_producer_id"
        assert {item.producer_id for item in measurement.producer_variants}


def test_canonical_producer_rejects_undeclared_runtime_selection() -> None:
    producers = (
        CanonicalProducerSpec(
            "categorical.v1",
            ("category",),
            "controlled_encoder",
            "test.v1",
        ),
    )
    dimensions = (
        CanonicalDimensionSpec(
            "canonical_endpoint_name",
            "endpoint",
            ("endpoint_name",),
            "deterministic_rule",
            "test.v1",
            producer_id="endpoint.v1",
            producer_id_field="canonical_endpoint_producer_id",
            legacy_value_field="canonical_endpoint",
        ),
        CanonicalDimensionSpec(
            "canonical_measurement_text",
            "measurement",
            ("endpoint_name", "measurement_text", "unit_text"),
            "deterministic_rule",
            "test.v1",
            atomic_group="pair",
            producer_id="scalar.v1",
            producer_id_field="canonical_pair_producer_id",
            producer_variants=producers,
            legacy_value_field="canonical_measurement",
        ),
        CanonicalDimensionSpec(
            "canonical_unit_text",
            "unit",
            ("endpoint_name", "measurement_text", "unit_text"),
            "deterministic_rule",
            "test.v1",
            atomic_group="pair",
            producer_id="scalar.v1",
            producer_id_field="canonical_pair_producer_id",
            producer_variants=producers,
            legacy_value_field="canonical_unit",
        ),
        CanonicalDimensionSpec(
            "canonical_smiles",
            "structure",
            ("smiles",),
            "deterministic_rule",
            "test.v1",
            legacy_value_field="canonical_smiles",
        ),
    )
    profile = SourceProfile(
        source_id="source",
        source_columns=("raw_endpoint", "raw_measurement", "raw_unit", "raw_smiles", "category"),
        endpoint_field="raw_endpoint",
        measurement_field="raw_measurement",
        unit_field="raw_unit",
        smiles_field="raw_smiles",
        canonical_dimensions=dimensions,
    )
    contract = StarlingRecordContract(
        task_id="task",
        sources={"source": profile},
        pair_buckets={
            "source": PairBucketSpec(
                "source", ("canonical_endpoint_name", "canonical_unit_text"), ()
            )
        },
    )
    row = {
        "source_id": "source",
        "endpoint_name": "endpoint",
        "measurement_text": "1",
        "unit_text": "%",
        "source_payload_json": json.dumps(
            {
                "raw_endpoint": "endpoint",
                "raw_measurement": "1",
                "raw_unit": "%",
                "raw_smiles": "CCO",
                "category": None,
            }
        ),
        "canonical_endpoint": "endpoint",
        "canonical_measurement": "1",
        "canonical_unit": "%",
        "canonical_smiles": "CCO",
        "normalized_record_id": "r1",
        "normalization_validity_status": "valid",
        "finite_scalar_value": 1.0,
        "canonical_endpoint_producer_id": "endpoint.v1",
        "canonical_pair_producer_id": "not_declared",
    }
    with pytest.raises(ValueError, match="undeclared producer"):
        contract.canonical_projection(row)


def test_pair_spec_rejects_a_raw_identity_field() -> None:
    dimension = CanonicalDimensionSpec(
        "canonical_endpoint_name",
        "endpoint_name",
        ("endpoint_name",),
        "deterministic_rule",
        "test.v1",
    )
    profile = SourceProfile(
        source_id="source",
        source_columns=("raw_endpoint", "raw_measurement", "raw_smiles", "assay"),
        endpoint_field="raw_endpoint",
        measurement_field="raw_measurement",
        smiles_field="raw_smiles",
        canonical_dimensions=(dimension,),
    )
    with pytest.raises(ValueError, match="lack canonical declarations"):
        StarlingRecordContract(
            task_id="task",
            sources={"source": profile},
            pair_buckets={
                "source": PairBucketSpec("source", ("assay",), ())
            },
        )


def test_pair_spec_rejects_canonicalization_input_as_variance_candidate() -> None:
    dimension = CanonicalDimensionSpec(
        "canonical_endpoint_name",
        "endpoint_name",
        ("endpoint_name",),
        "deterministic_rule",
        "test.v1",
    )
    profile = SourceProfile(
        source_id="source",
        source_columns=("raw_endpoint", "raw_measurement", "raw_smiles"),
        endpoint_field="raw_endpoint",
        measurement_field="raw_measurement",
        smiles_field="raw_smiles",
        canonical_dimensions=(dimension,),
    )
    with pytest.raises(ValueError, match="already used for canonicalization"):
        StarlingRecordContract(
            task_id="task",
            sources={"source": profile},
            pair_buckets={
                "source": PairBucketSpec(
                    "source",
                    ("canonical_endpoint_name",),
                    ("endpoint_name",),
                )
            },
        )


def test_v7_pair_bucket_uses_v7_names_and_canonical_dimensions() -> None:
    record = {
        "canonical_record_id": "r1",
        "source_id": "source",
        "canonical_endpoint_name": "endpoint",
        "canonical_unit_text": "%",
        "canonical_assay_context": "in vivo",
        "canonicalization_status": "valid",
        "canonical_smiles": "CCO",
        "finite_scalar_value": 1.0,
    }
    rows, audit = materialize_pair_buckets(
        [record],
        source_required_fields={"source": ("canonical_assay_context",)},
    )
    assert rows[0]["canonical_record_id"] == "r1"
    assert rows[0]["canonical_endpoint_name"] == "endpoint"
    assert "normalized_record_id" not in rows[0]
    assert audit["validations"]["one_sidecar_row_per_input_record"]


def test_controlled_categorical_scale_persists_kind_and_category_identity() -> None:
    scale = SKIN_CONTRACT.measurement_scales["single_subject_logit"]
    category = scale.category_for_value(1.0986122886681098)
    assert scale.kind == "binary"
    assert category is not None
    assert category.category_id == "response"
    assert category.rank == 1


def test_controlled_measurement_registry_is_source_and_input_scoped() -> None:
    scale = SKIN_CONTRACT.measurement_scales["signed_direction"]
    assert scale.source_id == "phototoxicity_irritation_local_damage"
    assert scale.input_fields == ("result_label",)
    assert scale.kind == "ordinal"


def test_resume_context_identity_preserves_fresh_deduplication() -> None:
    base = {
        "source_id": "source",
        "source_smiles": "CCO",
        "canonical_smiles": "CCO",
        "group_id": "group",
        "endpoint_name": "endpoint",
        "canonical_measurement": "1",
        "canonical_unit": "%",
        "support_text": "same support",
    }
    fresh = [
        {**base, "source_record_id": "a", "source_row_number": 1, "evidence_context_json": '{"dose":"1 mg"}'},
        {**base, "source_record_id": "b", "source_row_number": 2, "evidence_context_json": '{"dose":"2 mg"}'},
    ]
    persisted = [
        {
            **row,
            "deduplication_context_id": stable_id(
                "source_context", row["source_id"], row["evidence_context_json"]
            ),
            "evidence_context_json": "{}",
        }
        for row in fresh
    ]
    fresh_kept, _ = deduplicate_within_source(fresh)
    resumed_kept, _ = deduplicate_within_source(persisted)
    assert len(fresh_kept) == len(resumed_kept) == 2
    assert [row["duplicate_group_id"] for row in fresh_kept] == [
        row["duplicate_group_id"] for row in resumed_kept
    ]


def test_structure_status_restoration_matches_fresh_resolution() -> None:
    assert resolve_structure_value(None, structure_mode="mapped") == (
        None,
        "unresolved_mapped_structure",
    )
    assert resolve_structure_value(None, structure_mode="direct") == (
        None,
        "missing_structure",
    )
    assert resolve_structure_value("not-smiles", structure_mode="direct") == (
        None,
        "invalid_structure",
    )
