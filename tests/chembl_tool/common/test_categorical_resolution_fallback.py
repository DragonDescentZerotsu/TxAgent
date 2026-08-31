from tools.chembl_tool.common.starling.categorical_response import (
    BINARY_OUTCOME_UNIT,
    CategoricalEncoding,
)
from tools.chembl_tool.common.starling.normalization.measurements import (
    normalize_cleaned_records,
)
from tools.chembl_tool.common.starling.normalization.audit import (
    validate_measurement_pairs,
)


def test_failed_exact_resolution_keeps_the_categorical_fallback() -> None:
    encoding = CategoricalEncoding(
        encoder_id="binary.v1",
        value=1.0,
        unit=BINARY_OUTCOME_UNIT,
        measurement_text="1",
        inputs={"label": "positive"},
    )
    records = normalize_cleaned_records(
        [
            {
                "source_id": "source",
                "cleaned_record_id": "row",
                "endpoint_name": "endpoint",
                "measurement_resolution_status": "unsure",
                "measurement_resolution_route": "extract",
                "measurement_resolution_active": True,
            }
        ],
        endpoint_normalizer=lambda _source, endpoint: endpoint,
        family_resolver=lambda _source, _endpoint, _record: None,
        record_enricher=lambda _record: encoding.record_fields(version="test"),
    )
    assert records[0]["finite_scalar_value"] == 1.0
    assert records[0]["canonical_unit"] == BINARY_OUTCOME_UNIT
    assert records[0]["categorical_encoder_id"] == "binary.v1"
    assert validate_measurement_pairs(records) == []


def test_unresolved_exact_rows_may_use_the_non_scalar_unit_sentinel() -> None:
    record = {
        "normalized_record_id": "row",
        "measurement_resolution_status": "unsure",
        "measurement_unit_mapping_status": "",
        "measurement_unit_status": "measurement_resolution_unsure",
        "canonical_measurement": None,
        "canonical_unit": "free-text",
        "finite_scalar_value": None,
    }
    assert validate_measurement_pairs([record]) == []


def test_rejected_routes_may_use_the_free_text_unit_sentinel() -> None:
    record = {
        "normalized_record_id": "row",
        "measurement_resolution_status": "not_extracted",
        "measurement_resolution_route": "reject",
        "measurement_resolution_active": True,
        "canonical_measurement": None,
        "canonical_unit": "free-text",
        "finite_scalar_value": None,
    }
    assert validate_measurement_pairs([record]) == []
