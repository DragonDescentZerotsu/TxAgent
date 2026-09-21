import json
from pathlib import Path

import pytest

from data.processing.evidence_library.versions.v10.tasks.ames.semantic_display import (
    semantic_display_fields,
    semantic_prompt_payload,
)


FIXTURE = Path(__file__).with_name("fixtures") / "ames_semantic_display.v1.json"


def test_reviewed_ames_semantic_display_fixture():
    for scale, category, measurement, unit in json.loads(FIXTURE.read_text())["cases"]:
        record = {"measurement_kind": "ordinal", "canonical_measurement_scale_id": scale,
                  "canonical_category_id": category, "canonical_measurement_text": "1",
                  "canonical_unit_text": "ordinal_outcome_class"}
        assert semantic_display_fields(record) == {
            "measurement_text": measurement, "unit_text": unit}


def test_ames_projection_consumes_encoder_input_without_mutating_record():
    record = {"measurement_kind": "binary",
              "canonical_measurement_scale_id": "ames_mechanism_detection.v1",
              "canonical_category_id": "detected", "canonical_measurement_text": "1",
              "canonical_unit_text": "binary_outcome_class"}
    payload = {"result_direction": "detected", "canonical_assay_context": "DNA repair"}
    projected = semantic_prompt_payload(record, payload)
    assert projected == {"canonical_assay_context": "DNA repair", "measurement_text": "detected",
                         "unit_text": "mechanistic endpoint detection status"}
    assert record["canonical_measurement_text"] == "1"


def test_unknown_ames_categorical_display_fails_closed():
    with pytest.raises(ValueError, match="unsupported Ames display scale"):
        semantic_display_fields({"measurement_kind": "ordinal"})
