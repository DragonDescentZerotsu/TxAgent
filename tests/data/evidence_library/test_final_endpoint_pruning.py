import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.processing.evidence_library.shared.v1.normalization.audit import write_parquet
from data.processing.evidence_library.shared.v1.normalization.cleaning import (
    file_sha256,
)
from data.processing.evidence_library.versions.v7.final_endpoint_pruning import (
    DECISIONS_FILENAME,
    SCHEMA_VERSION,
    VERSION,
    _candidate_buckets,
    _compact_semantic_row,
    _is_context_limit_rejection,
    _run_review,
    _validate_candidate_review,
    _value_spread_rows,
    load_reviewed_record_ineligibility,
    log10_approval_gate,
    render_review_prompt,
    review_decisions,
    semantic_review_columns,
    split_review_candidate,
    supported_gap,
    tail_outliers,
    validate_review_response,
)


def test_supported_gap_is_inclusive_100x_not_threefold():
    assert supported_gap([1, 1.1, 1.2, 3.6, 3.7, 3.8]) is None
    assert supported_gap([1, 1.1, 1.2, 118.8, 130, 140]) is None
    gap = supported_gap([1, 1.1, 1.2, 120, 130, 140])
    assert gap is not None
    assert gap["gap_lower_value"] == 1.2
    assert gap["gap_upper_value"] == 120
    assert gap["gap_decades"] == pytest.approx(2.0)


def test_supported_gap_uses_logged_units_without_double_logging():
    values = [-8.2, -8.1, -8.0, -6.0, -5.9, -5.8]
    gap = supported_gap(values, unit_text="log10(cm/s)")
    assert gap is not None
    assert gap["gap_lower_value"] == -8.0
    assert gap["gap_upper_value"] == -6.0
    assert gap["gap_decades"] == pytest.approx(2.0)
    assert gap["gap_geometry"] == "canonical_log10"


def test_supported_gap_handles_negative_log_and_natural_log_units():
    negative_log = [2.0, 2.1, 2.2, 4.2, 4.3, 4.4]
    assert supported_gap(negative_log, unit_text="-log10(cm/s)") is not None
    natural_log = [
        value * 2.302585092994046 for value in (-4.2, -4.1, -4.0, -2.0, -1.9, -1.8)
    ]
    gap = supported_gap(natural_log, unit_text="ln(ng/mL)")
    assert gap is not None
    assert gap["gap_decades"] == pytest.approx(2.0)


@pytest.mark.parametrize("unit", ["log(cm/s)", "log2", "logit_response"])
def test_supported_gap_skips_ambiguous_log_geometry(unit):
    assert supported_gap([-4.2, -4.1, -4.0, -2.0, -1.9, -1.8], unit_text=unit) is None


def test_tail_detector_applies_to_three_record_buckets() -> None:
    tails = tail_outliers([1.0, 1.1, 10.0], unit_text="ng/mL")
    assert [tail["record_index"] for tail in tails] == [2]
    assert tails[0]["tail_trigger_reasons"] == ["leave_one_out_5sd"]


def test_tail_detector_reviews_a_100x_tail_even_below_five_sd() -> None:
    tails = tail_outliers([0.001, 1.0, 1.0, 1.0, 1000.0], unit_text="ng/mL")
    assert any(
        tail["record_index"] == 4
        and "leave_one_out_2decades" in tail["tail_trigger_reasons"]
        and "leave_one_out_5sd" not in tail["tail_trigger_reasons"]
        for tail in tails
    )


def test_tail_detector_handles_zero_reference_sd_and_small_buckets() -> None:
    assert tail_outliers([1.0, 1000.0], unit_text="ng/mL") == []
    [tail] = tail_outliers([1.0, 1.0, 1000.0], unit_text="ng/mL")
    assert tail["tail_zero_reference_sd"] is True
    assert tail["tail_sigma_distance"] is None
    assert set(tail["tail_trigger_reasons"]) == {
        "leave_one_out_5sd",
        "leave_one_out_2decades",
    }


def test_tail_detector_uses_native_declared_log10_geometry() -> None:
    [tail] = tail_outliers([0.0, 0.0, 2.0], unit_text="log10(ng/mL)")
    assert tail["record_index"] == 2
    assert tail["tail_distance_decades"] == pytest.approx(2.0)


def test_log10_gate_requires_verified_raw_to_log10_normalization() -> None:
    logged = np.random.default_rng(0).normal(size=80).tolist()
    raw = [10**value for value in logged]

    result = log10_approval_gate(
        raw,
        logged,
        ["log10.v1"] * len(raw),
        unit_text="log10(ng/mL)",
    )

    assert result is not None
    assert result["raw_normality_rejected"] is True
    assert result["approved"] is True
    assert (
        log10_approval_gate(
            raw,
            logged,
            ["raw.v1"] * len(raw),
            unit_text="log10(ng/mL)",
        )
        is None
    )


def test_log10_gate_does_not_require_logged_normality() -> None:
    logged = [-1.0] * 30 + [0.0] * 20 + [3.0] * 30
    raw = [10**value for value in logged]

    result = log10_approval_gate(
        raw,
        logged,
        ["log10.v1"] * len(raw),
        unit_text="log10(ng/mL)",
    )

    assert result is not None
    assert result["raw_normality_rejected"] is True
    assert result["approved"] is True


def test_log10_gate_rejects_an_already_normal_raw_distribution() -> None:
    raw = np.random.default_rng(0).normal(loc=10.0, scale=1.0, size=80).tolist()
    logged = np.log10(raw).tolist()

    result = log10_approval_gate(
        raw,
        logged,
        ["log10.v1"] * len(raw),
        unit_text="log10(ng/mL)",
    )

    assert result is not None
    assert result["raw_normality_rejected"] is False
    assert result["approved"] is False


@pytest.mark.parametrize(
    ("values", "unit"),
    [
        ([0.0, 0.0, 100.0], "%"),
        ([-4.0, -4.0, 4.0], "log(response)"),
    ],
)
def test_tail_detector_falls_back_to_native_sd_for_non_loggable_buckets(
    values, unit
) -> None:
    [tail] = tail_outliers(values, unit_text=unit)
    assert tail["record_index"] == 2
    assert tail["tail_geometry"] == "native_scale_sd_only"
    assert tail["tail_distance_decades"] is None
    assert tail["tail_trigger_reasons"] == ["leave_one_out_5sd"]


def test_candidate_detection_uses_pair_bucket_eligibility_not_retrieval(
    tmp_path: Path,
) -> None:
    records = pd.DataFrame(
        [
            {
                "canonical_record_id": record_id,
                "canonical_smiles": "CCO",
                "retrieval_eligible": False,
                "measurement_kind": "continuous",
                "finite_scalar_value": value,
                "assay_transfer_pretransform_scalar_value": value,
                "assay_transfer_transform_id": "raw.v1",
            }
            for record_id, value in (
                [(f"r{i:02d}", 1.0) for i in range(19)] + [("r19", 1000.0)]
            )
        ]
    )
    records["canonical_smiles"] = [
        f"parent-{min(index, 15):02d}" for index in range(20)
    ]
    buckets = pd.DataFrame(
        [
            {
                "canonical_record_id": record_id,
                "pair_bucket_key": '["source","endpoint","mg/L"]',
                "canonical_pair_fields_json": "{}",
                "canonical_endpoint_name": "endpoint",
                "canonical_unit_text": "mg/L",
                "assay_transfer_eligible": True,
            }
            for record_id in records["canonical_record_id"]
        ]
    )
    records_path = tmp_path / "records.parquet"
    buckets_path = tmp_path / "buckets.parquet"
    records.to_parquet(records_path, index=False)
    buckets.to_parquet(buckets_path, index=False)

    candidates, _ = _candidate_buckets("test", records_path, buckets_path)

    assert len(candidates) == 1
    assert candidates[0]["target_record_ids"] == ["r19"]
    assert len(candidates[0]["context_record_ids"]) == 5

    records.loc[records["canonical_smiles"] == "parent-15", "canonical_smiles"] = (
        "parent-00"
    )
    records.to_parquet(records_path, index=False)
    assert _candidate_buckets("test", records_path, buckets_path)[0] == []


def test_log10_gate_does_not_bypass_a_five_sd_tail(
    tmp_path: Path,
) -> None:
    logged = [0.0] * 18 + [1.0, 3.0]
    raw = [10**value for value in logged]
    records = pd.DataFrame(
        [
            {
                "canonical_record_id": f"r{index:03d}",
                "canonical_smiles": f"C{index}",
                "measurement_kind": "continuous",
                "finite_scalar_value": logged_value,
                "assay_transfer_pretransform_scalar_value": raw_value,
                "assay_transfer_transform_id": "log10.v1",
            }
            for index, (raw_value, logged_value) in enumerate(
                zip(raw, logged, strict=True)
            )
        ]
    )
    buckets = pd.DataFrame(
        [
            {
                "canonical_record_id": f"r{index:03d}",
                "pair_bucket_key": '["source","endpoint","log10(ng/mL)"]',
                "canonical_pair_fields_json": "{}",
                "canonical_endpoint_name": "endpoint",
                "canonical_unit_text": "log10(ng/mL)",
                "assay_transfer_eligible": True,
            }
            for index in range(len(raw))
        ]
    )
    records_path = tmp_path / "records.parquet"
    buckets_path = tmp_path / "buckets.parquet"
    records.to_parquet(records_path, index=False)
    buckets.to_parquet(buckets_path, index=False)

    candidates, summary = _candidate_buckets("test", records_path, buckets_path)

    assert len(candidates) == 1
    assert candidates[0]["target_record_ids"] == ["r019"]
    assert summary["log10_gate_approved_buckets"] == 1
    assert len(summary["log10_gate_approvals"]) == 1


def test_log10_gate_bypasses_a_two_decade_only_tail(tmp_path: Path) -> None:
    logged = [-2.0, -1.0, 0.0, 1.0, 2.0] * 4
    raw = [10**value for value in logged]
    records = pd.DataFrame(
        [
            {
                "canonical_record_id": f"r{index:03d}",
                "canonical_smiles": f"parent-{index:03d}",
                "measurement_kind": "continuous",
                "finite_scalar_value": logged_value,
                "assay_transfer_pretransform_scalar_value": raw_value,
                "assay_transfer_transform_id": "log10.v1",
            }
            for index, (raw_value, logged_value) in enumerate(
                zip(raw, logged, strict=True)
            )
        ]
    )
    buckets = pd.DataFrame(
        [
            {
                "canonical_record_id": f"r{index:03d}",
                "pair_bucket_key": "bucket",
                "canonical_pair_fields_json": "{}",
                "canonical_endpoint_name": "endpoint",
                "canonical_unit_text": "log10(ng/mL)",
                "assay_transfer_eligible": True,
            }
            for index in range(20)
        ]
    )
    records_path = tmp_path / "records.parquet"
    buckets_path = tmp_path / "buckets.parquet"
    records.to_parquet(records_path, index=False)
    buckets.to_parquet(buckets_path, index=False)

    candidates, summary = _candidate_buckets("test", records_path, buckets_path)

    assert candidates == []
    assert summary["log10_gate_approved_review_rows"] > 0


def test_value_spread_context_is_deterministic_and_covers_range() -> None:
    rows = [
        {"canonical_record_id": f"r{index:02d}", "finite_scalar_value": index}
        for index in range(9)
    ]

    selected = _value_spread_rows(list(reversed(rows)))

    assert [row["finite_scalar_value"] for row in selected] == [0, 2, 4, 6, 8]
    assert _value_spread_rows(rows[:4]) == rows[:4]


def test_semantic_projection_excludes_identity_provenance_labels_and_policy():
    columns = semantic_review_columns(
        [
            "canonical_record_id",
            "canonical_smiles",
            "source_id",
            "source_record_id",
            "pmid",
            "outcome_label",
            "retrieval_eligible",
            "measurement_resolution_rule_id",
            "measurement_resolution_origin",
            "measurement_resolution_input_measurement",
            "measurement_resolution_input_unit",
            "canonical_endpoint_name",
            "finite_scalar_value",
            "assay_or_test",
            "species_or_population",
            "qualifying_conditions",
            "support_text",
        ]
    )
    assert set(columns) == {
        "canonical_record_id",
        "measurement_resolution_origin",
        "measurement_resolution_input_measurement",
        "measurement_resolution_input_unit",
        "finite_scalar_value",
        "assay_or_test",
        "species_or_population",
        "qualifying_conditions",
        "support_text",
    }


def test_semantic_row_names_the_llm_refined_extraction():
    row = _compact_semantic_row(
        {
            "canonical_record_id": "r1",
            "measurement_text": "5.6 x 10^-3",
            "unit_text": "cm/h",
            "measurement_resolution_origin": "llm",
            "measurement_resolution_input_measurement": "5.6",
            "measurement_resolution_input_unit": "10^-3 cm/h",
            "canonical_measurement_text": "0.0000015555555556",
            "canonical_unit_text": "cm/s",
            "finite_scalar_value": 1.5555555556e-6,
        }
    )
    assert row["llm_refined_extraction"] == {
        "measurement": "5.6",
        "unit": "10^-3 cm/h",
    }
    assert not any(key.startswith("measurement_resolution_") for key in row)


def test_review_contract_and_single_vote_decision():
    response = {
        "schema_version": SCHEMA_VERSION,
        "rows": [
            {
                "canonical_record_id": "r1",
                "decision": "drop",
                "reason_code": "wrong_unit_or_scale",
                "reason": "The raw unit denotes a different scale.",
            },
            {
                "canonical_record_id": "r2",
                "decision": "keep",
                "reason_code": "valid_extreme",
                "reason": "The source describes the same quantity.",
            },
        ],
    }
    validated = validate_review_response(response, record_ids={"r1", "r2"})
    candidate = {
        "task_id": "test",
        "bucket_id": "bucket",
        "pair_bucket_key": "hidden-key",
        "target_rows": [
            {"canonical_record_id": "r1", "finite_scalar_value": 1.0},
            {"canonical_record_id": "r2", "finite_scalar_value": 100.0},
        ],
    }
    decisions = review_decisions(candidate, {"response": validated})
    assert [row["review_decision"] for row in decisions] == ["drop", "keep"]
    assert decisions[0]["review_reason_code"] == "wrong_unit_or_scale"


def test_single_valid_review_makes_one_model_call() -> None:
    class Client:
        calls = 0

        def chat_json(self, _messages):
            self.calls += 1
            return {
                "content": {
                    "schema_version": SCHEMA_VERSION,
                    "rows": [
                        {
                            "row_id": "r0001",
                            "decision": "keep",
                            "reason_code": "valid_extreme",
                            "reason": "The support describes the same quantity.",
                        }
                    ],
                },
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "model": "model",
            }

    client = Client()
    review = _run_review(
        client,
        {
            "bucket_id": "bucket",
            "target_rows": [{"canonical_record_id": "record"}],
        },
        "prompt",
        "model",
        "https://example.test/v1",
        3,
    )

    assert client.calls == 1
    assert review["reviewer"] == "single"
    assert review["response"]["rows"][0]["canonical_record_id"] == "record"


def test_prompt_contains_compact_semantics_not_bucket_key():
    prompt = render_review_prompt(
        {
            "task_id": "test",
            "pair_bucket_key": "secret-key",
            "canonical_bucket": {
                "canonical_endpoint_name": "permeability",
                "canonical_unit_text": "cm/s",
                "canonical_pair_fields": {
                    "canonical_direct_condition_group": "no_reported_external_condition",
                    "canonical_reference_scope": "absolute",
                },
            },
            "distribution": {
                "record_count": 2,
                "finite_record_count": 2,
                "review_scope": "tail_records_only",
                "supported_gap": None,
            },
            "target_rows": [
                {
                    "canonical_record_id": "r1",
                    "measurement_text": "1.0",
                    "unit_text": "cm/s",
                    "finite_scalar_value": 1.0,
                    "endpoint_name": "Papp",
                    "embedded_unit": False,
                    "measurement_kind": "continuous",
                    "canonical_measurement_source": "source_exact",
                    "canonical_reference_scope": "absolute",
                    "assay_model": "MDCK-MDR1 monolayer",
                    "support_text": "measured permeability",
                }
            ],
            "context_rows": [
                {
                    "canonical_record_id": "r2",
                    "measurement_text": "2.0",
                    "unit_text": "cm/s",
                    "finite_scalar_value": 2.0,
                    "support_text": "comparison permeability",
                }
            ],
        }
    )
    assert "secret-key" not in prompt
    assert "measured permeability" in prompt
    assert "comparison permeability" in prompt
    assert "MDCK-MDR1 monolayer" in prompt
    assert '"rows"' in prompt
    assert '"canonical_bucket"' not in prompt
    for removed in (
        "Endpoint name:",
        "Embedded unit:",
        "Measurement kind:",
        "Measurement source:",
        "Reference scope:",
        "Direct condition group:",
        "Finite records:",
        "Review scope:",
    ):
        assert removed not in prompt


def test_bucket_uses_short_ids_and_maps_them_back():
    rows = [
        {"canonical_record_id": f"canonical-{index}", "finite_scalar_value": index}
        for index in range(2)
    ]
    candidate = {
        "task_id": "test",
        "canonical_bucket": {},
        "distribution": {
            "finite_record_count": 2,
            "review_scope": "tail_records_only",
            "supported_gap": None,
        },
        "target_rows": rows,
        "context_rows": [],
    }
    prompt = render_review_prompt(candidate)
    assert "canonical-0" not in prompt
    assert '"row_id": "r0001"' in prompt
    response = {
        "schema_version": SCHEMA_VERSION,
        "rows": [
            {
                "row_id": f"r{index:04d}",
                "decision": "keep",
                "reason_code": "fits_bucket",
                "reason": "Same scientific quantity.",
            }
            for index in range(1, 3)
        ],
    }
    normalized = _validate_candidate_review(response, candidate)
    assert normalized["rows"][0]["canonical_record_id"] == "canonical-0"
    assert normalized["rows"][-1]["canonical_record_id"] == "canonical-1"


def test_oversized_bucket_is_split_without_losing_rows():
    candidate = {
        "task_id": "test",
        "bucket_id": "bucket",
        "canonical_bucket": {},
        "distribution": {
            "finite_record_count": 8,
            "review_record_count": 8,
            "review_scope": "complete_supported_gap_bucket",
            "supported_gap": None,
        },
        "target_rows": [
            {
                "canonical_record_id": f"r{index}",
                "finite_scalar_value": float(index),
                "support_text": "x" * 400,
            }
            for index in range(8)
        ],
        "context_rows": [],
    }
    one_row_bytes = len(
        render_review_prompt(
            {
                **candidate,
                "target_rows": candidate["target_rows"][:1],
                "context_rows": candidate["target_rows"][1:6],
            }
        ).encode("utf-8")
    )
    chunks = split_review_candidate(candidate, max_prompt_bytes=one_row_bytes + 10)

    assert len(chunks) == 8
    assert {
        row["canonical_record_id"] for chunk in chunks for row in chunk["target_rows"]
    } == {
        "r0",
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
        "r6",
        "r7",
    }
    assert all(
        len(render_review_prompt(chunk).encode("utf-8")) <= one_row_bytes + 10
        for chunk in chunks
    )
    assert [chunk["parent_bucket_id"] for chunk in chunks] == ["bucket"] * 8
    assert all(
        {row["canonical_record_id"] for row in chunk["target_rows"]}.isdisjoint(
            row["canonical_record_id"] for row in chunk["context_rows"]
        )
        for chunk in chunks
    )


def test_context_limit_rejection_is_detected_for_reservation_release():
    assert _is_context_limit_rejection(
        RuntimeError(
            "code: context_length_exceeded; Input tokens exceed the configured limit"
        )
    )
    assert not _is_context_limit_rejection(RuntimeError("connection reset"))


def test_stage3_loader_returns_only_reviewed_drops(tmp_path: Path) -> None:
    canonical_records = tmp_path / "canonical.parquet"
    write_parquet(canonical_records, [{"canonical_record_id": "r1"}])
    decisions = tmp_path / DECISIONS_FILENAME
    write_parquet(
        decisions,
        [
            {
                "canonical_record_id": "r1",
                "review_decision": "drop",
                "review_reason_code": "wrong_unit_or_scale",
            },
            {
                "canonical_record_id": "r2",
                "review_decision": "keep",
                "review_reason_code": "valid_extreme",
            },
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": VERSION,
                "task_id": "test",
                "inputs": {
                    "canonical_records": {"sha256": file_sha256(canonical_records)}
                },
                "files": {DECISIONS_FILENAME: file_sha256(decisions)},
                "summary": {"dropped_rows": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dropped, audit = load_reviewed_record_ineligibility(
        manifest_path,
        task_id="test",
        canonical_records_path=canonical_records,
    )

    assert dropped == {"r1": "llm_review_drop:wrong_unit_or_scale"}
    assert audit["reviewed_rows"] == 2


def _record(record_id: str, value: float) -> dict:
    return {
        "canonical_record_id": record_id,
        "source_id": "source",
        "source_record_id": record_id,
        "canonical_smiles": "CCO",
        "retrieval_eligible": True,
        "measurement_kind": "continuous",
        "finite_scalar_value": value,
        "variation_value": None,
        "canonical_endpoint_name": "test_endpoint",
        "canonical_measurement_text": str(value),
        "canonical_unit_text": "mg/L",
        "support_text": "same quantity",
        "group_id": "Observed.indirect",
    }


def _bucket(record_id: str) -> dict:
    return {
        "canonical_record_id": record_id,
        "pair_bucket_key": "bucket",
        "canonical_unit_text": "mg/L",
        "canonical_pair_fields_json": "{}",
        "assay_transfer_eligible": True,
        "assay_transfer_ineligibility_reason": None,
    }
