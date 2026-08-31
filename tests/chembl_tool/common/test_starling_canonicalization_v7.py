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
from tools.chembl_tool.tasks.clintox.starling_schema import (
    RECORD_CONTRACT as CLINTOX_CONTRACT,
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
    assert species.input_fields == (
        "assay_type",
        "experimental_conditions",
        "support_text",
    )
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
        "molecule_name": "sibling-source-only",
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
    assert cleaned["source_smiles"] == "MAPPED"
    assert cleaned["canonical_smiles"] == "CCO"
    assert "molecule_name" not in cleaned
    assert "endpoint_category" not in cleaned

    resumed = BIO_CONTRACT.inflate_cleaned(cleaned)
    assert resumed["smiles"] == "RAW-SOURCE"
    assert resumed["source_smiles"] == "MAPPED"
    assert resumed["canonical_smiles"] == "CCO"


def test_dataset_constants_are_not_claimed_as_source_visible() -> None:
    profile = BIO_CONTRACT.source("hf_bioavailability")
    assert "endpoint_name" in profile.cleaned_source_fields
    assert "unit_text" in profile.cleaned_source_fields
    assert "endpoint_name" not in profile.source_visible_fields
    # The unified source uses embedded units because nondirect rows may carry
    # percent/fold/ratio text; unit_text is no longer a dataset constant.
    assert "unit_text" in profile.source_visible_fields


def test_inflate_cleaned_restores_sparse_source_shape_after_parquet_resume() -> None:
    resumed = {
        "source_id": "human_clinical_toxicity",
        "endpoint_name": "human_clinical_toxicity",
        "measurement_text": "toxicity_absent",
        "unit_text": None,
        "smiles": "CCO",
        "clinical_context": "trial",
        "assay_context": None,
        "result_metric": None,
    }
    inflated = CLINTOX_CONTRACT.inflate_cleaned(resumed)
    assert "assay_context" not in inflated
    assert "result_metric" not in inflated
    assert inflated["clinical_context"] == "trial"

    with pytest.raises(ValueError, match="populated foreign source fields"):
        CLINTOX_CONTRACT.inflate_cleaned({**resumed, "assay_context": "in vitro"})


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


def test_controlled_only_inputs_do_not_become_continuous_variance_candidates() -> None:
    scale = BIO_CONTRACT.measurement_scales["fg_substrate_status_binary.v1"]
    candidates = set(BIO_CONTRACT.pair_buckets["fg"].variance_candidates)
    assert {"substrate_status", "transporter_or_enzyme"} <= set(
        scale.input_fields
    )
    assert not candidates


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


def test_required_known_compatibility_argument_does_not_remove_descriptive_bucket() -> None:
    record = {
        "canonical_record_id": "r1",
        "source_id": "sensitization_aop",
        "canonical_endpoint_name": "stimulation_index",
        "canonical_unit_text": "fold",
        "measurement_kind": "continuous",
        "finite_scalar_value": 1.0,
        "canonical_reference_scope": "absolute",
        "canonical_species_context": None,
        "canonicalization_status": "valid",
        "canonical_smiles": "CCO",
    }
    rows, audit = materialize_pair_buckets(
        [record],
        source_required_fields={
            "sensitization_aop": ("canonical_species_context",)
        },
        required_known_fields_by_source={
            "sensitization_aop": ("canonical_species_context",)
        },
    )
    assert len(rows) == 1
    assert rows[0]["bucket_eligible"] is True
    assert rows[0]["pair_bucket_key"]
    assert rows[0]["bucket_exclusion_reason"] is None
    assert audit["unknown_field_rates"]["canonical_species_context"] == 1.0


def test_semantic_pair_bucket_uses_canonical_unit_without_enabling_transfer() -> None:
    records = [
        {
            "canonical_record_id": "free-text",
            "source_id": "source",
            "canonical_endpoint_name": "outcome",
            "canonical_unit_text": "free-text",
            "canonicalization_status": "non_scalar_measurement",
            "canonical_smiles": "CCO",
            "retrieval_eligible": True,
            "canonical_measurement_text": "reported response",
        },
        {
            "canonical_record_id": "relative",
            "source_id": "source",
            "canonical_endpoint_name": "outcome",
            "canonical_unit_text": "relative-scalar",
            "canonicalization_status": "missing_canonical_unit",
            "canonical_smiles": "CCN",
            "retrieval_eligible": True,
            "canonical_reference_scope": "comparator_relative",
        },
        {
            "canonical_record_id": "unresolved",
            "source_id": "source",
            "canonical_endpoint_name": "outcome",
            "canonical_unit_text": "unresolved-scalar",
            "canonicalization_status": "missing_canonical_unit",
            "canonical_smiles": "CCC",
            "retrieval_eligible": True,
            "measurement_parse_kind": "point",
        },
        {
            "canonical_record_id": "no-endpoint",
            "source_id": "source",
            "canonical_endpoint_name": None,
            "canonical_unit_text": "free-text",
            "canonicalization_status": "non_scalar_measurement",
            "canonical_smiles": "CCCC",
            "retrieval_eligible": False,
        },
        {
            "canonical_record_id": "unresolved-mechanism",
            "source_id": "source",
            "canonical_endpoint_name": "outcome",
            "canonical_unit_text": "%",
            "canonicalization_status": "valid",
            "canonical_smiles": "CCCCC",
            "retrieval_eligible": False,
            "organization_status": "unresolved_mechanism_family",
        },
        {
            "canonical_record_id": "direct-vote-without-endpoint",
            "source_id": "source",
            "canonical_endpoint_name": None,
            "canonical_unit_text": "free-text",
            "canonicalization_status": "non_scalar_measurement",
            "canonical_smiles": "CCCCCC",
            "retrieval_eligible": True,
        },
    ]
    rows, audit = materialize_pair_buckets(
        records,
        source_required_fields={"source": ()},
        semantic_pair_bucket_sources=("source",),
    )
    assert [row["canonical_unit_text"] for row in rows] == [
        "free-text",
        "relative-scalar",
        "unresolved-scalar",
        "free-text",
        "%",
        "free-text",
    ]
    assert all(row["pair_bucket_key"] for row in rows[:3])
    assert all(not row["assay_transfer_eligible"] for row in rows)
    assert all(row["pair_bucket_key"] for row in rows)
    assert all(audit["validations"].values())


def test_controlled_categorical_scale_persists_kind_and_category_identity() -> None:
    scale = SKIN_CONTRACT.measurement_scales["single_subject_fraction.v1"]
    category = scale.category_for_value(1.0)
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


def test_frozen_extraction_is_a_declarable_canonicalization_method() -> None:
    """A per-row extraction is a distinct method from a per-value mapping.

    ``frozen_mapping`` is keyed by a distinct source value and reused across
    every row sharing it; ``frozen_extraction`` is keyed by one record and never
    reused.  Recording that difference is what lets the manifest state how a
    canonical value was produced.
    """
    dimension = CanonicalDimensionSpec(
        "canonical_measurement_text",
        "measurement",
        ("measurement_text",),
        "frozen_extraction",
        "bbb_martins_measurement_resolution.v1",
    )
    assert dimension.manifest()["method"] == "frozen_extraction"

    with pytest.raises(ValueError, match="unsupported canonicalization method"):
        CanonicalDimensionSpec(
            "canonical_measurement_text",
            "measurement",
            ("measurement_text",),
            "llm_freeform",
            "v1",
        )


def test_atomic_group_members_must_record_the_same_producer() -> None:
    """An atomic group is only verifiable when its members share a producer field.

    ``_validate_canonical_projection`` can compare selected producers across an
    atomic group only for dimensions that persist one.  A group mixing a
    producer-recording dimension with a silent one would be declared atomic but
    never checked, so the disagreement is refused at declaration time.
    """

    def dimension(output_field: str, producer_field: str | None) -> CanonicalDimensionSpec:
        return CanonicalDimensionSpec(
            output_field,
            "measurement",
            ("measurement_text",),
            "deterministic_rule",
            "v1",
            atomic_group="canonical_measurement_unit_pair",
            producer_id="pair.v1" if producer_field else None,
            producer_id_field=producer_field,
        )

    with pytest.raises(ValueError, match="spans different producer fields"):
        SourceProfile(
            source_id="s",
            source_columns=("m", "smiles"),
            endpoint_constant="e",
            measurement_field="m",
            canonical_dimensions=(
                dimension("canonical_measurement_text", "canonical_pair_producer_id"),
                dimension("canonical_unit_text", None),
            ),
        )

    # Agreeing declarations remain legal, with or without a producer field.
    for producer_field in ("canonical_pair_producer_id", None):
        SourceProfile(
            source_id="s",
            source_columns=("m", "smiles"),
            endpoint_constant="e",
            measurement_field="m",
            canonical_dimensions=(
                dimension("canonical_measurement_text", producer_field),
                dimension("canonical_unit_text", producer_field),
            ),
        )


def test_influx_reports_its_quantity_through_the_measurement_role() -> None:
    """Influx's value lives in ``reported_result``; it must be the measurement.

    While the source declared no measurement field, every influx row cleaned to
    a null ``measurement_text`` and could only ever resolve to a non-scalar, so
    the whole source was absent from scalar evidence.  Promoting the column also
    has to remove it from the cleaned schema, or the projection would carry the
    same value twice under two names.
    """
    from tools.chembl_tool.tasks.bbb_martins.starling_schema import SOURCES

    profile = SOURCES["influx_transport"]
    assert profile.measurement_field == "reported_result"
    assert "reported_result" in profile.source_columns
    assert "reported_result" not in profile.cleaned_source_fields
    assert "measurement_text" in profile.source_visible_fields
