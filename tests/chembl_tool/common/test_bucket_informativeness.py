from __future__ import annotations

import pytest

from tools.chembl_tool.common.starling.bucket_informativeness import (
    render_batch,
    select_records,
    validate_response,
    visible_record,
)


TEST_SOURCE_COLUMNS = {"source": ("support_text", "assay_model", "molecule_name")}


def _record(index: int) -> dict[str, object]:
    return {
        "canonical_record_id": f"r{index:02d}",
        "source_id": "source",
        "canonical_smiles": f"C{'C' * index}",
        "measurement_text": str(index),
        "unit_text": "nmol/g",
        "canonical_measurement_text": str(index),
        "finite_scalar_value": float(index),
        "canonical_reference_scope": "absolute",
        "canonical_reference_basis": "brain_tissue",
        "support_text": f"Secret {index} had measurement {index}",
        "assay_model": f"model {index % 3}",
        "molecule_name": f"secret {index}",
    }


def test_sampling_is_stable_bounded_and_covers_continuous_range() -> None:
    records = [_record(index) for index in range(20)]
    selected = select_records(
        records,
        measurement_kind="continuous",
        source_columns=TEST_SOURCE_COLUMNS,
    )
    reversed_selected = select_records(
        list(reversed(records)),
        measurement_kind="continuous",
        source_columns=TEST_SOURCE_COLUMNS,
    )
    assert len(selected) == 12
    assert [row["canonical_record_id"] for row in selected] == [
        row["canonical_record_id"] for row in reversed_selected
    ]
    assert {row["finite_scalar_value"] for row in selected} >= {0.0, 19.0}


def test_visible_records_and_prompt_are_identity_blind() -> None:
    visible = visible_record(_record(1), TEST_SOURCE_COLUMNS)
    assert visible["support_text"] == "[molecule] had measurement 1"
    assert "molecule_name" not in visible
    item = {
        "payload": {"sampled_records": [visible]},
        "pair_bucket_key": "bucket",
        "payload_sha256": "hash",
    }
    prompt = render_batch([item], "meaningful CNS access")
    assert "measurement 1" in prompt
    assert "secret 1" not in prompt
    assert "0.01 resolution" in prompt


def test_response_is_strict_and_rounded_to_hundredths() -> None:
    result = validate_response(
        {
            "items": [
                {
                    "id": "0",
                    "informativeness_score": 0.337,
                    "rationale": "This endpoint is a meaningful but incomplete CNS-access proxy.",
                }
            ]
        },
        1,
    )
    assert result[0]["informativeness_score"] == 0.34
    with pytest.raises(ValueError, match="outside"):
        validate_response(
            {
                "items": [
                    {
                        "id": "0",
                        "informativeness_score": 1.01,
                        "rationale": "Invalid score.",
                    }
                ]
            },
            1,
        )
