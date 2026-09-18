from __future__ import annotations

import json

import pytest

from data.processing.evidence_library.versions.v10.tasks.ames.starling_candidate_resolution import (
    CANDIDATE_CONTRACT_VERSION,
    SELECTION_COMPILER_VERSION,
    candidate_set_sha256,
    compile_selection,
    compile_selections,
)


def _candidate_row(record_id: str = "row-1") -> dict[str, object]:
    candidates = [
        {
            "candidate_id": "1",
            "measurement": "6",
            "unit": "10^-5 mutation frequency",
            "rule_id": "scientific_named_readout.v1",
            "evidence": ["support_text: 6 x 10^-5 mutation frequency"],
            "hints": ["scientific_scale"],
        },
        {
            "candidate_id": "2",
            "measurement": "1",
            "unit": "resistant mutant",
            "rule_id": "named_event.v1",
            "evidence": ["support_text: produced 1 resistant mutant"],
            "hints": ["event_count"],
        },
    ]
    payload = json.dumps(candidates)
    return {
        "cleaned_record_id": record_id,
        "source_row_uid": f"source-{record_id}",
        "source_id": "fixed_mutation",
        "canonical_endpoint_name": "microbial_forward_mutation",
        "measurement_candidates_json": payload,
        "candidate_set_sha256": candidate_set_sha256(payload),
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
    }


def _selection(status: str = "ok", candidate_id: str = "1") -> dict[str, object]:
    measurements = (
        [{"measurement": candidate_id, "unit": "candidate_id"}]
        if status == "ok"
        else []
    )
    reason = {
        "ok": "absolute",
        "relative": "experiment_relative",
        "unsure": "missing_unit",
        "unavailable": "no_eligible_numeric",
    }[status]
    raw_candidate = candidate_id if status == "ok" else None
    return {
        "cleaned_record_id": "row-1",
        "source_row_uid": "source-row-1",
        "source_id": "fixed_mutation",
        "status": status,
        "measurements_json": json.dumps(measurements),
        "quantity_count": len(measurements),
        "assignment_method": "model_single_pass",
        "raw_response_json": json.dumps(
            {
                "id": "r1",
                "reason": reason,
                "candidate_id": raw_candidate,
                "status": status,
                "measurements": measurements,
            }
        ),
    }


def test_compile_selection_materializes_only_the_frozen_pair():
    result = compile_selection(_selection(), _candidate_row())
    assert json.loads(result["measurements_json"]) == [
        {"measurement": "6", "unit": "10^-5 mutation frequency"}
    ]
    assert result["selected_candidate_id"] == "1"
    assert result["assignment_method"] == "model_single_pass"
    assert result["selection_compiler_version"] == SELECTION_COMPILER_VERSION


@pytest.mark.parametrize("status", ["relative", "unsure", "unavailable"])
def test_compile_selection_preserves_non_scalar_terminal_status(status):
    result = compile_selection(_selection(status), _candidate_row())
    assert result["status"] == status
    assert result["quantity_count"] == 0
    assert json.loads(result["measurements_json"]) == []


def test_compile_selection_rejects_an_unenumerated_id():
    with pytest.raises(ValueError, match="selected candidate ID is absent"):
        compile_selection(_selection(candidate_id="9"), _candidate_row())


def test_compile_selection_rejects_reason_status_disagreement():
    row = _selection("unsure")
    raw = json.loads(row["raw_response_json"])
    raw["reason"] = "bound_or_range"
    raw["status"] = "relative"
    row["raw_response_json"] = json.dumps(raw)
    with pytest.raises(ValueError, match="reason and status disagree"):
        compile_selection(row, _candidate_row())


def test_compile_selection_rejects_hidden_non_ok_measurement():
    row = _selection("unsure")
    row["measurements_json"] = '[{"measurement":"1","unit":"candidate_id"}]'
    raw = json.loads(row["raw_response_json"])
    raw["measurements"] = json.loads(row["measurements_json"])
    row["raw_response_json"] = json.dumps(raw)
    with pytest.raises(ValueError, match="non-ok candidate selection carries"):
        compile_selection(row, _candidate_row())


def test_candidate_ids_must_be_canonical_integers():
    row = _candidate_row()
    candidates = json.loads(row["measurement_candidates_json"])
    candidates[0]["candidate_id"] = "01"
    row["measurement_candidates_json"] = json.dumps(candidates)
    with pytest.raises(ValueError, match="positive integer"):
        compile_selection(_selection(), row)


def test_candidate_units_cannot_use_the_transport_sentinel():
    row = _candidate_row()
    candidates = json.loads(row["measurement_candidates_json"])
    candidates[0]["unit"] = "candidate_id"
    row["measurement_candidates_json"] = json.dumps(candidates)
    with pytest.raises(ValueError, match="reserved selection sentinel"):
        compile_selection(_selection(), row)


@pytest.mark.parametrize("measurement", ["NaN", "Infinity", "1e-3", "1,000"])
def test_candidate_measurements_must_be_finite_plain_decimals(measurement):
    row = _candidate_row()
    candidates = json.loads(row["measurement_candidates_json"])
    candidates[0]["measurement"] = measurement
    row["measurement_candidates_json"] = json.dumps(candidates)
    with pytest.raises(ValueError, match="finite plain decimal"):
        compile_selection(_selection(), row)


def test_raw_response_requires_exact_fields_and_canonical_id():
    row = _selection()
    raw = json.loads(row["raw_response_json"])
    raw["extra"] = "not part of the contract"
    row["raw_response_json"] = json.dumps(raw)
    with pytest.raises(ValueError, match="fields disagree"):
        compile_selection(row, _candidate_row())

    row = _selection(candidate_id="01")
    with pytest.raises(ValueError, match="not canonical"):
        compile_selection(row, _candidate_row())


def test_quantity_count_must_match_the_transport_shape():
    row = _selection()
    row["quantity_count"] = 0
    with pytest.raises(ValueError, match="quantity count disagrees"):
        compile_selection(row, _candidate_row())


def test_compile_selections_requires_exact_uid_and_row_coverage():
    with pytest.raises(ValueError, match="coverage differs"):
        compile_selections([_selection()], [_candidate_row(), _candidate_row("row-2")])
    wrong_uid = _selection()
    wrong_uid["source_row_uid"] = "wrong"
    with pytest.raises(ValueError, match="source UID mismatch"):
        compile_selections([wrong_uid], [_candidate_row()])
