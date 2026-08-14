from __future__ import annotations

import json
from dataclasses import replace

from tools.chembl_tool.common.starling.build_reference_semantics_mapping import (
    MAX_EPOCH_TOKENS,
    RequestBatch,
    SubmissionCache,
    TokenLedger,
    _batches,
    _query_batch,
    _reconcile_cached_assignment,
)
from tools.chembl_tool.common.starling.pair_buckets import materialize_pair_buckets
from tools.chembl_tool.common.starling.reference_semantics import (
    REFERENCE_SCOPE_ABSOLUTE,
    REFERENCE_SCOPE_COMPARATOR,
    ReferenceEligibilitySpec,
    deterministic_default_assignment,
    reference_exclusion_reason,
)
from tools.chembl_tool.tasks.bioavailability_ma.starling_reference_semantics import (
    REFERENCE_SEMANTICS_CONFIG,
)
from tools.chembl_tool.tasks.bbb_martins.starling_reference_semantics import (
    REFERENCE_SEMANTICS_CONFIG as BBB_REFERENCE_SEMANTICS_CONFIG,
    generation_no_call_assignment,
)
from tools.chembl_tool.tasks.skin_reaction.starling_reference_semantics import (
    REFERENCE_SEMANTICS_CONFIG as SKIN_REFERENCE_SEMANTICS_CONFIG,
)


def test_default_assignments_are_fail_closed() -> None:
    encoded = deterministic_default_assignment(
        {
            "finite_scalar_value": 1.0,
            "canonical_measurement_scale_id": "binary.v1",
        },
        output_basis=True,
    )
    assert encoded is not None
    assert encoded.scope == "not_applicable"
    assert encoded.basis == "none"

    non_scalar = deterministic_default_assignment(
        {"finite_scalar_value": None}, output_basis=False
    )
    assert non_scalar is not None
    assert non_scalar.scope == "unknown"


def test_reference_eligibility_prefers_precision() -> None:
    spec = ReferenceEligibilitySpec((REFERENCE_SCOPE_ABSOLUTE,), False)
    assert (
        reference_exclusion_reason(
            {"canonical_reference_scope": REFERENCE_SCOPE_ABSOLUTE}, spec
        )
        is None
    )
    assert reference_exclusion_reason(
        {"canonical_reference_scope": REFERENCE_SCOPE_COMPARATOR}, spec
    ) == "reference_scope_comparator_relative"
    ratio_spec = ReferenceEligibilitySpec(("endpoint_defined_ratio",), True)
    assert (
        reference_exclusion_reason(
            {
                "canonical_reference_scope": "endpoint_defined_ratio",
                "canonical_reference_basis": "other_explicit",
            },
            ratio_spec,
        )
        is None
    )


def test_pair_materialization_excludes_relative_rows_before_calibration() -> None:
    base = {
        "canonicalization_status": "valid",
        "source_id": "oral_exposure",
        "canonical_endpoint_name": "auc",
        "canonical_unit_text": "ng*h/mL",
        "canonical_smiles": "CCO",
    }
    rows, audit = materialize_pair_buckets(
        [
            {
                **base,
                "canonical_record_id": "absolute",
                "canonical_reference_scope": "absolute",
            },
            {
                **base,
                "canonical_record_id": "relative",
                "canonical_reference_scope": "comparator_relative",
            },
        ],
        source_required_fields={
            "oral_exposure": ("canonical_reference_scope",)
        },
        reference_eligibility_by_source={
            "oral_exposure": ReferenceEligibilitySpec(("absolute",), False)
        },
    )
    assert rows[0]["bucket_eligible"] is True
    assert rows[1]["bucket_eligible"] is False
    assert rows[1]["bucket_exclusion_reason"] == (
        "reference_scope_comparator_relative"
    )
    assert audit["stats"]["eligible_records"] == 1


def test_bio_generator_batches_fifty_unique_rows_and_never_resubmits(tmp_path) -> None:
    rows = [
        {"id": f"row-{index:02d}", "source_id": "fa"}
        for index in range(53)
    ]
    batches = _batches(
        rows,
        config=REFERENCE_SEMANTICS_CONFIG,
        prompt="classify",
        attempted=set(),
    )
    assert [len(batch.rows) for batch in batches] == [50, 3]
    assert len({row_id for batch in batches for row_id in batch.row_ids}) == 53

    cache = SubmissionCache(tmp_path / "requests.jsonl")
    cache.submit(batches[0], epoch="key-1")
    remaining = _batches(
        rows,
        config=REFERENCE_SEMANTICS_CONFIG,
        prompt="classify",
        attempted=cache.attempted,
    )
    assert not set(batches[0].row_ids) & {
        row_id for batch in remaining for row_id in batch.row_ids
    }


def test_batch_uses_one_api_submission_and_no_internal_retry() -> None:
    calls = []

    def fake_llm(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return {
            "content": json.dumps(
                {
                    "rows": [
                        {
                            "id": "row-1",
                            "reference_scope": "absolute",
                            "evidence_field": "support_text",
                            "evidence_quote": "standalone AUC",
                        }
                    ]
                }
            ),
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }

    batch = RequestBatch(
        request_id="request-1",
        source_id="oral_exposure",
        rows=(
            {
                "id": "row-1",
                "source_id": "oral_exposure",
                "support_text": "standalone AUC",
            },
        ),
        prompt="classify",
        max_completion_tokens=8192,
    )
    assignments, usage, status = _query_batch(
        batch,
        config=REFERENCE_SEMANTICS_CONFIG,
        llm=fake_llm,
    )
    assert len(calls) == 1
    assert calls[0][1]["max_retries"] == 1
    assert calls[0][1]["reasoning_effort"] == "low"
    assert status == "valid"
    assert assignments[0]["reference_scope"] == "absolute"
    assert usage == {"input_tokens": 12, "output_tokens": 8}


def test_bio_batches_show_only_value_and_support_and_keep_source_order() -> None:
    rows = [
        {
            "id": f"fa-{index:02d}",
            "source_id": "fa",
            "measurement_text": str(index),
            "support_text": f"The measured value was {index} mg/mL.",
            "canonical_endpoint_name": "must_not_be_visible",
        }
        for index in range(51)
    ] + [
        {
            "id": "oral-00",
            "source_id": "oral_exposure",
            "measurement_text": "40%",
            "support_text": "Exposure increased by 40% relative to baseline.",
        }
    ]
    batches = _batches(
        rows,
        config=REFERENCE_SEMANTICS_CONFIG,
        prompt="classify",
        attempted=set(),
    )
    assert [batch.source_id for batch in batches] == ["fa", "fa", "oral_exposure"]
    assert [len(batch.rows) for batch in batches] == [50, 1, 1]
    assert batches[0].api_rows[0] == {
        "id": "0",
        "measurement_text": "0",
        "support_text": "The measured value was 0 mg/mL.",
    }
    assert "canonical_endpoint_name" not in batches[0].api_rows[0]


def test_bio_labels_only_response_maps_local_id_to_full_record() -> None:
    def fake_llm(prompt, **kwargs):
        assert json.loads(prompt["user"])["rows"][0]["id"] == "0"
        return {
            "content": json.dumps(
                {"rows": [{"id": "0", "reference_scope": "comparator_relative"}]}
            ),
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        }

    batch = RequestBatch(
        "request-bio",
        "oral_exposure",
        ({"id": "full-bio-id", "source_id": "oral_exposure"},),
        "classify",
        8192,
        payload_rows=(
            {
                "id": "0",
                "measurement_text": "40%",
                "support_text": "Exposure increased by 40% relative to baseline.",
            },
        ),
    )
    assignments, _, status = _query_batch(
        batch, config=REFERENCE_SEMANTICS_CONFIG, llm=fake_llm
    )
    assert status == "valid"
    assert assignments[0]["cleaned_record_id"] == "full-bio-id"
    assert assignments[0]["reference_scope"] == "comparator_relative"
    assert assignments[0]["assignment_method"] == (
        "gpt_5_4_mini_single_pass:labels_only"
    )
    assert assignments[0]["evidence_field"] is None
    assert assignments[0]["evidence_quote"] is None


def test_model_evidence_text_is_normalized_back_to_unique_field_name() -> None:
    support = "Compound 49 exhibited a brain/plasma ratio of 0.7."

    def fake_llm(prompt, **kwargs):
        return {
            "content": json.dumps(
                {
                    "rows": [
                        {
                            "id": "row-1",
                            "reference_scope": "endpoint_defined_ratio",
                            "reference_basis": "plasma",
                            "evidence_field": support,
                            "evidence_quote": "brain/plasma ratio of 0.7",
                        }
                    ]
                }
            ),
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }

    assignments, _, status = _query_batch(
        RequestBatch(
            "request-1",
            "direct_bbb",
            ({"id": "row-1", "source_id": "direct_bbb", "support_text": support},),
            "classify",
            8192,
        ),
        config=replace(BBB_REFERENCE_SEMANTICS_CONFIG, labels_only_output=False),
        llm=fake_llm,
    )
    assert status == "valid"
    assert assignments[0]["evidence_field"] == "support_text"
    assert assignments[0]["assignment_method"].endswith(
        ":normalized_evidence_field"
    )


def test_contradictory_scope_and_basis_are_rejected_without_override() -> None:
    support = "Carbamazepine had a logBB value of 0.00."

    def fake_llm(prompt, **kwargs):
        return {
            "content": json.dumps(
                {
                    "rows": [
                        {
                            "id": "row-1",
                            "reference_scope": "standardized_control_ratio",
                            "reference_basis": "blood",
                            "evidence_field": "support_text",
                            "evidence_quote": "logBB value of 0.00",
                        }
                    ]
                }
            ),
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }

    assignments, _, status = _query_batch(
        RequestBatch(
            "request-1",
            "direct_bbb",
            ({"id": "row-1", "source_id": "direct_bbb", "support_text": support},),
            "classify",
            8192,
        ),
        config=replace(BBB_REFERENCE_SEMANTICS_CONFIG, labels_only_output=False),
        llm=fake_llm,
    )
    assert status == "partial_invalid"
    assert assignments[0]["reference_scope"] == "unknown"
    assert assignments[0]["reference_basis"] == "unknown"
    assert assignments[0]["assignment_method"] == (
        "invalid_row_response:standard_control_basis_mismatch"
    )


def test_exact_quote_can_be_rerouted_to_one_supplied_field() -> None:
    support = "The compound showed adequate brain penetration."

    def fake_llm(prompt, **kwargs):
        return {
            "content": json.dumps(
                {
                    "rows": [
                        {
                            "id": "row-1",
                            "reference_scope": "endpoint_defined_ratio",
                            "reference_basis": "plasma",
                            "evidence_field": support,
                            "evidence_quote": "AUC_brain/AUC_plasma",
                        }
                    ]
                }
            ),
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }

    assignments, _, status = _query_batch(
        RequestBatch(
            "request-1",
            "direct_bbb",
            (
                {
                    "id": "row-1",
                    "source_id": "direct_bbb",
                    "support_text": support,
                    "endpoint_name": "AUC_brain/AUC_plasma",
                },
            ),
            "classify",
            8192,
        ),
        config=replace(BBB_REFERENCE_SEMANTICS_CONFIG, labels_only_output=False),
        llm=fake_llm,
    )
    assert status == "valid"
    assert assignments[0]["evidence_field"] == "endpoint_name"
    assert assignments[0]["evidence_quote"] == "AUC_brain/AUC_plasma"
    assert assignments[0]["assignment_method"].endswith(
        ":rerouted_exact_evidence_quote"
    )


def test_bbb_batches_fifty_rows_with_only_value_and_support() -> None:
    rows = [
        {
            "id": f"record-{index:03d}",
            "source_id": "direct_bbb" if index < 60 else "efflux_transport",
            "measurement_text": str(index / 10),
            "support_text": f"The reported target value was {index / 10}.",
            "canonical_endpoint_name": "must_not_be_visible",
            "canonical_unit_text": "must_not_be_visible",
            "_generation_no_call_assignment": None,
        }
        for index in range(103)
    ]
    batches = _batches(
        rows,
        config=BBB_REFERENCE_SEMANTICS_CONFIG,
        prompt="classify",
        attempted=set(),
    )
    assert [len(batch.rows) for batch in batches] == [50, 50, 3]
    assert all(batch.source_id == "mixed" for batch in batches)
    assert batches[0].api_rows[0] == {
        "id": "0",
        "measurement_text": "0.0",
        "support_text": "The reported target value was 0.0.",
    }
    assert set(batches[0].api_rows[-1]) == {
        "id",
        "measurement_text",
        "support_text",
    }


def test_bbb_labels_only_response_maps_local_ids_back_to_records() -> None:
    def fake_llm(prompt, **kwargs):
        assert json.loads(prompt["user"])["rows"] == [
            {
                "id": "0",
                "measurement_text": "0.7",
                "support_text": "The brain/plasma ratio was 0.7.",
            }
        ]
        return {
            "content": json.dumps(
                {
                    "rows": [
                        {
                            "id": "0",
                            "reference_scope": "endpoint_defined_ratio",
                            "reference_basis": "plasma",
                        }
                    ]
                }
            ),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    batch = RequestBatch(
        "request-1",
        "mixed",
        (
            {
                "id": "full-record-id",
                "source_id": "direct_bbb",
                "measurement_text": "0.7",
                "support_text": "The brain/plasma ratio was 0.7.",
            },
        ),
        "classify",
        8192,
        payload_rows=(
            {
                "id": "0",
                "measurement_text": "0.7",
                "support_text": "The brain/plasma ratio was 0.7.",
            },
        ),
    )
    assignments, _, status = _query_batch(
        batch, config=BBB_REFERENCE_SEMANTICS_CONFIG, llm=fake_llm
    )
    assert status == "valid"
    assert assignments == [
        {
            "cleaned_record_id": "full-record-id",
            "source_id": "direct_bbb",
            "reference_scope": "endpoint_defined_ratio",
            "assignment_method": "gpt_5_4_mini_single_pass:labels_only",
            "evidence_field": None,
            "evidence_quote": None,
            "reference_basis": "plasma",
        }
    ]


def test_skin_batches_fifty_rows_with_only_value_and_support() -> None:
    rows = [
        {
            "id": f"skin-{index:03d}",
            "source_id": "skin_exposure",
            "measurement_text": f"{index}%",
            "support_text": f"Recovery was {index}% of the applied dose.",
            "canonical_endpoint_name": "must_not_be_visible",
        }
        for index in range(53)
    ]
    batches = _batches(
        rows,
        config=SKIN_REFERENCE_SEMANTICS_CONFIG,
        prompt="classify",
        attempted=set(),
    )
    assert [len(batch.rows) for batch in batches] == [50, 3]
    assert all(batch.source_id == "skin_exposure" for batch in batches)
    assert batches[0].api_rows[0] == {
        "id": "0",
        "measurement_text": "0%",
        "support_text": "Recovery was 0% of the applied dose.",
    }


def test_skin_labels_only_response_requires_and_preserves_basis() -> None:
    def fake_llm(prompt, **kwargs):
        return {
            "content": json.dumps(
                {
                    "rows": [
                        {
                            "id": "0",
                            "reference_scope": "endpoint_defined_ratio",
                            "reference_basis": "applied_dose",
                        }
                    ]
                }
            ),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    batch = RequestBatch(
        "request-skin",
        "skin_exposure",
        ({"id": "full-skin-id", "source_id": "skin_exposure"},),
        "classify",
        8192,
        payload_rows=(
            {
                "id": "0",
                "measurement_text": "2.3%",
                "support_text": "Recovery was 2.3% of the applied dose.",
            },
        ),
    )
    assignments, _, status = _query_batch(
        batch, config=SKIN_REFERENCE_SEMANTICS_CONFIG, llm=fake_llm
    )
    assert status == "valid"
    assert assignments[0]["cleaned_record_id"] == "full-skin-id"
    assert assignments[0]["reference_scope"] == "endpoint_defined_ratio"
    assert assignments[0]["reference_basis"] == "applied_dose"
    assert assignments[0]["assignment_method"] == (
        "gpt_5_4_mini_single_pass:labels_only"
    )


def test_bbb_no_call_gates_are_narrow_and_fail_closed() -> None:
    physical = generation_no_call_assignment(
        {
            "canonical_semantics_status": "approved",
            "canonical_quantity_kind": "concentration",
            "canonical_unit_text": "ng/mL",
            "support_text": "The concentration increased to 2 ng/mL versus control.",
        }
    )
    assert physical is not None
    assert (physical.scope, physical.basis) == ("absolute", "none")
    assert generation_no_call_assignment(
        {
            "canonical_semantics_status": "approved",
            "canonical_quantity_kind": "concentration",
            "canonical_unit_text": "%ID/g",
        }
    ) is None

    forward = generation_no_call_assignment(
        {
            "canonical_semantics_status": "approved",
            "canonical_quantity_kind": "ratio",
            "canonical_endpoint_name": "brain_to_plasma_auc_ratio",
        }
    )
    reverse = generation_no_call_assignment(
        {
            "canonical_semantics_status": "approved",
            "canonical_quantity_kind": "ratio",
            "canonical_endpoint_name": "plasma_to_brain_ratio",
        }
    )
    assert forward is not None and forward.basis == "plasma"
    assert reverse is not None and reverse.basis == "brain"
    assert generation_no_call_assignment(
        {
            "canonical_semantics_status": "approved",
            "canonical_quantity_kind": "ratio",
            "canonical_endpoint_name": "knockout_to_wildtype_ratio",
        }
    ) is None

    fold = generation_no_call_assignment(
        {
            "canonical_semantics_status": "approved",
            "canonical_quantity_kind": "fold_change",
            "canonical_unit_text": "fold",
        }
    )
    assert fold is not None
    assert (fold.scope, fold.basis) == ("comparator_relative", "unknown")


def test_prior_safe_gate_conflict_is_preserved_but_excluded() -> None:
    candidate = {
        "id": "row-1",
        "source_id": "direct_bbb",
        "_generation_no_call_assignment": generation_no_call_assignment(
            {
                "canonical_semantics_status": "approved",
                "canonical_quantity_kind": "ratio",
                "canonical_endpoint_name": "brain_to_plasma_ratio",
            }
        ),
    }
    prior = {
        "cleaned_record_id": "row-1",
        "source_id": "direct_bbb",
        "reference_scope": "endpoint_defined_ratio",
        "reference_basis": "brain",
        "assignment_method": "gpt_5_4_mini_single_pass",
    }
    reconciled = _reconcile_cached_assignment(
        candidate, prior, config=BBB_REFERENCE_SEMANTICS_CONFIG
    )
    assert reconciled["reference_scope"] == "unknown"
    assert reconciled["reference_basis"] == "unknown"
    assert reconciled["assignment_method"].startswith("safe_gate_conflict:")
    assert json.loads(reconciled["prior_assignment_json"])["reference_basis"] == "brain"


def test_token_ledger_stops_before_nine_million(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger = TokenLedger(ledger_path, epoch="key-1", start_new_epoch=True)
    assert ledger.reserve("first", MAX_EPOCH_TOKENS - 1)
    ledger.complete("first", None)
    assert not ledger.reserve("second", 2)
    ledger.mark_exhausted()
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert payload["epochs"]["key-1"]["status"] == "budget_exhausted"


def test_token_ledger_supports_a_distinct_custom_epoch_limit(tmp_path) -> None:
    ledger_path = tmp_path / "bio-ledger.json"
    ledger = TokenLedger(
        ledger_path,
        epoch="refreshed-key",
        start_new_epoch=True,
        max_tokens=9_500_000,
    )
    assert ledger.reserve("first", 9_499_999)
    ledger.complete("first", None)
    assert not ledger.reserve("second", 2)
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert payload["epochs"]["refreshed-key"]["max_tokens"] == 9_500_000
