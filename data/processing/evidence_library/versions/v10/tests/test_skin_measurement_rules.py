"""Behavior checks for the Skin Reaction v9 Stage-01 and Stage-02 contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.evidence_library.versions.v10.measurement_routing import (
    attach_stage1_routes,
)
from data.processing.evidence_library.versions.v10.tasks.skin_reaction.starling_measurement_resolution import (
    BATCH_SIZE,
    DEFAULT_BASE_MAPPING_PATH,
    MAPPING_VERSION,
    MAX_MEASUREMENTS_PER_ROW,
    PROMPT_VERSION,
    SOURCE_IDS,
    STRATIFY_BATCHES,
    EXACT_UNIT_MAPPING_PATH,
    EXACT_UNIT_MAPPING_VERSION,
    ROUTING_EXACT_UNIT_MAPPING_PATH,
    prompt_manifest,
    prompt_row_fields,
    route_measurement,
)


def test_skin_expanded_exact_units_use_a_successor_mapping() -> None:
    from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
        load_exact_unit_mapping,
    )

    legacy_path = ROUTING_EXACT_UNIT_MAPPING_PATH.with_name(
        "exact_measurement_unit_map.v2.json"
    )
    legacy = load_exact_unit_mapping(legacy_path)
    current = load_exact_unit_mapping(EXACT_UNIT_MAPPING_PATH)
    legacy_key = ("skin_reaction", "adaptive tolerance", "%")
    wildcard_key = ("skin_reaction", "*", "%")
    assert legacy_key not in legacy
    assert wildcard_key in current
    assert json.loads(EXACT_UNIT_MAPPING_PATH.read_text())["version"] == EXACT_UNIT_MAPPING_VERSION


def test_skin_unit_reconciliation_is_conservative_and_coefficient_aware() -> None:
    from data.processing.evidence_library.versions.v10.tasks.skin_reaction.data_processing import (
        build_unit_reconciliation,
    )

    decision = build_unit_reconciliation._decision
    assert decision("EC3")[:4] == ("map", "%", "1", "nonnegative")
    assert decision("positive ear responses out of 10")[:4] == (
        "map",
        "fraction",
        "0.1",
        "nonnegative",
    )
    assert decision("DILL value")[:4] == ("map", "DILL value", "1", "any")
    assert decision("patient positive patch test to cobalt chloride")[0] == "exclude"


def _route(measurement, unit=None, **fields):
    return route_measurement(
        {"measurement_text": measurement, "unit_text": unit, **fields}
    )


def test_structured_incidence_precedes_ambiguous_measurement_text() -> None:
    decision = _route(
        "28% (8/29) subjects",
        source_id="direct_skin_reaction",
        canonical_endpoint_name="sensitization",
        positive_count=8,
        total_tested=29,
    )
    assert decision.rule_id == "structured_positive_count_fraction.v1"
    assert decision.measurement_text == "0.2758620689655172413793103448"
    assert decision.unit_text == "fraction"


def test_separate_points_defer_but_reviewed_embedded_units_copy() -> None:
    assert _route("0", "cm/s").bucket == "extract"
    assert _route("-6.36", "cm/s").bucket == "extract"
    assert _route("2.5 mg").unit_text == "mg"
    spread = _route("12.26 ± 3.22 ng/cm^2")
    assert spread.measurement_text == "12.26"
    assert spread.unit_text == "ng/cm^2"
    assert _route("2.5 mg", "cm/s").bucket == "extract"
    assert _route("2.5 invented_unit").bucket == "extract"
    scaled = _route("2.73 ± 0.29 × 10^-6", "cm/h")
    assert (scaled.measurement_text, scaled.unit_text) == (
        "2.73",
        "10^-6 cm/h",
    )


def test_percent_and_relative_uncertainty_forms_are_exact() -> None:
    percent = _route("15.4% ±2.7%")
    assert (percent.measurement_text, percent.unit_text) == ("15.4", "%")
    relative = _route("10±11%", "cm/h × 10^5")
    assert (relative.measurement_text, relative.unit_text) == (
        "10",
        "cm/h × 10^5",
    )
    assert _route("15.4%", "invented unit").bucket == "extract"


def test_direct_outcome_percent_is_source_and_endpoint_restricted() -> None:
    accepted = _route(
        "3.8% positivity rate",
        source_id="direct_skin_reaction",
        canonical_endpoint_name="sensitization",
    )
    assert accepted.rule_id == "direct_outcome_percent.v1"
    assert _route(
        "3.8% positivity rate",
        source_id="skin_exposure",
        canonical_endpoint_name="relative_penetration",
    ).bucket == "extract"
    assert _route(
        "approximately 3.8% positivity rate",
        source_id="direct_skin_reaction",
        canonical_endpoint_name="sensitization",
    ).bucket == "extract"


def test_fold_score_and_support_inferred_si_use_narrow_contexts() -> None:
    fold = _route(
        "218 times more permeable",
        source_id="skin_exposure",
        canonical_endpoint_name="relative_penetration",
    )
    assert (fold.measurement_text, fold.unit_text) == ("218", "fold")
    assert _route(
        "approximately 4-fold",
        source_id="skin_exposure",
        canonical_endpoint_name="relative_penetration",
    ).bucket == "extract"

    score = _route(
        "mean response score 0.9",
        source_id="direct_skin_reaction",
        canonical_endpoint_name="sensitization",
    )
    assert (score.measurement_text, score.unit_text) == ("0.9", "score")
    assert _route(
        "24 h response score 0.9",
        source_id="direct_skin_reaction",
        canonical_endpoint_name="sensitization",
    ).bucket != "accept"

    si = _route(
        "1.2",
        source_id="sensitization_aop",
        canonical_endpoint_name="stimulation index",
        support_text="The Stimulation Index (SI) was 1.2.",
    )
    assert (si.measurement_text, si.unit_text) == ("1.2", "SI")
    assert _route(
        "1.2",
        source_id="sensitization_aop",
        canonical_endpoint_name="sensitization",
        support_text="The Stimulation Index (SI) was 1.2.",
    ).bucket == "extract"


def test_nonpoints_reject_but_potency_and_plain_counts_defer() -> None:
    assert _route("<5", "ng/cm^2").rule_id == "hard_bound_no_point.v1"
    assert _route("7-12", "%").rule_id == "explicit_range_no_point.v1"
    assert _route("plasma 54 ng/mL; skin 94 ng/mL").rule_id == (
        "multiple_numeric_candidates.v1"
    )
    assert _route("EC50 = 2.1 µM").bucket == "extract"
    assert _route("12 subjects").bucket == "extract"
    assert _route("12", "subjects").bucket == "extract"
    assert _route("12 ± 2", "subjects").bucket == "extract"


def test_stage1_attaches_rules_without_running_later_stages() -> None:
    [record] = attach_stage1_routes(
        [
            {
                "source_id": "skin_exposure",
                "canonical_endpoint_name": "flux",
                "measurement_text": "2.5",
                "unit_text": "ng/cm^2",
            }
        ],
        task="skin_reaction",
    )
    assert record["measurement_resolution_route"] == "extract"
    assert record["measurement_resolution_rule_id"] is None
    assert "finite_scalar_value" not in record


def test_stage2_is_a_fresh_one_value_skin_extraction() -> None:
    manifest = prompt_manifest()
    assert PROMPT_VERSION == "skin_reaction_measurement_resolution_prompt.v12"
    assert MAPPING_VERSION == "skin_reaction_measurement_resolution.v8"
    assert BATCH_SIZE == 20
    assert MAX_MEASUREMENTS_PER_ROW == 1
    assert DEFAULT_BASE_MAPPING_PATH is None
    assert STRATIFY_BATCHES is True
    assert manifest["maximum_measurements_per_row"] == 1
    assert manifest["stratified_batch_order"] is True
    assert "unit_text" not in prompt_row_fields("direct_skin_reaction")
    assert "unit_text" in prompt_row_fields("sensitization_aop")


def _candidate(source_id: str, index: int, endpoint: str = "endpoint") -> dict:
    row = {
        "id": f"{source_id}-{index:03d}",
        "source_id": source_id,
        "canonical_endpoint_name": endpoint,
        "endpoint_name": endpoint,
        "measurement_text": f"result {index}",
        "support_text": "support",
    }
    if "unit_text" in prompt_row_fields(source_id):
        row["unit_text"] = "unit"
    return row


def test_stage2_batches_interleave_sources_and_repack_unattempted_rows() -> None:
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        TaskConfig,
        plan_batches,
    )

    config = TaskConfig("skin_reaction")
    candidates = [
        _candidate(source_id, index, endpoint=f"endpoint-{index % 2}")
        for source_id in SOURCE_IDS
        for index in range(60)
    ]
    batches = plan_batches(candidates, config, attempted=set())
    assert {batch.source_id for batch in batches[: len(SOURCE_IDS)]} == set(SOURCE_IDS)

    one_source = [_candidate("skin_exposure", index) for index in range(21)]
    attempted = {one_source[0]["id"]}
    resumed = plan_batches(one_source, config, attempted=attempted)
    resumed_ids = [row_id for batch in resumed for row_id in batch.row_ids]
    assert len(resumed_ids) == 20
    assert attempted.isdisjoint(resumed_ids)


def test_stage2_validation_rejects_more_than_one_measurement() -> None:
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        validate_row,
    )

    returned = {
        "id": "row-1",
        "status": "ok",
        "measurements": [
            {"measurement": "1", "unit": "%"},
            {"measurement": "2", "unit": "%"},
        ],
    }
    assignment, reason = validate_row(
        returned,
        {"id": "row-1", "source_id": "direct_skin_reaction"},
        task="skin_reaction",
        max_measurements=MAX_MEASUREMENTS_PER_ROW,
    )
    assert reason == "too_many_measurements:2>1"
    assert assignment["status"] == "unsure"
    assert assignment["quantity_count"] == 0


def test_stage2_preserves_an_unmapped_deterministic_source_unit(tmp_path: Path) -> None:
    from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import (
        apply_measurement_resolution,
    )
    from data.processing.evidence_library.versions.v10.measurement_routing import (
        MEASUREMENT_ROUTING_VERSION,
    )

    mapping_path = tmp_path / "mapping.parquet"
    pq.write_table(
        pa.table(
            {
                "cleaned_record_id": pa.array([], type=pa.string()),
                "status": pa.array([], type=pa.string()),
                "measurements_json": pa.array([], type=pa.string()),
            }
        ),
        mapping_path,
    )
    unit_path = tmp_path / "units.json"
    unit_path.write_text(
        '{"version":"starling_exact_measurement_units.v2","entries":[]}',
        encoding="utf-8",
    )
    record = {
        "cleaned_record_id": "row-1",
        "measurement_routing_version": MEASUREMENT_ROUTING_VERSION,
        "measurement_resolution_route": "accept",
        "measurement_resolution_exact_measurement": "17",
        "measurement_resolution_exact_unit": "% sensitization",
        "measurement_resolution_exact_unit_is_canonical": False,
        "measurement_text": "17% sensitization",
        "unit_text": None,
        "canonical_endpoint_name": "sensitization",
    }
    with pytest.raises(ValueError, match="has no rule"):
        apply_measurement_resolution(
            [dict(record)],
            mapping_path=mapping_path,
            task="skin_reaction",
            unit_mapping_path=unit_path,
            expected_routing_version=MEASUREMENT_ROUTING_VERSION,
        )
    records = [dict(record)]
    audit = apply_measurement_resolution(
        records,
        mapping_path=mapping_path,
        task="skin_reaction",
        unit_mapping_path=unit_path,
        expected_routing_version=MEASUREMENT_ROUTING_VERSION,
        allow_unmapped_source_exact_units=True,
    )
    assert records[0]["measurement_resolution_status"] == "ok"
    assert records[0]["measurement_resolution_origin"] == "source_exact"
    assert records[0]["measurement_unit_mapping_status"] == "mapped"
    assert records[0]["measurement_unit_mapping_rule_source"] == (
        "source_exact_identity"
    )
    assert records[0]["resolved_measurement_text"] == "17"
    assert records[0]["resolved_unit_text"] == "% sensitization"
    assert audit["unit_mapping_status_counts"] == {"mapped": 1}
    assert audit["source_exact_identity_unit_rows"] == 1


def test_openrouter_gpt_adapter_uses_supported_gpt5_parameters() -> None:
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        openai_compatible_llm,
    )

    call = {}

    class Completions:
        def create(self, **kwargs):
            call.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"rows": []}'))
                ],
                usage=None,
            )

    client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat=SimpleNamespace(completions=Completions()),
    )
    openai_compatible_llm(client)(
        {"system": "system", "user": "user"},
        model="openai/gpt-5.6-luna",
        max_tokens=8192,
        temperature=1.0,
        reasoning_effort="low",
    )
    assert call["max_completion_tokens"] == 8192
    assert "max_tokens" not in call
    assert "temperature" not in call
    assert call["extra_body"] == {"provider": {"require_parameters": True}}


def test_openrouter_adapter_retries_a_rate_limit_envelope(monkeypatch) -> None:
    from data.processing.evidence_library.versions.v10 import (
        build_measurement_resolution_mapping as generator,
    )

    calls = 0
    delays = []

    class Completions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return SimpleNamespace(
                    choices=None,
                    error={"code": 429, "message": "temporarily rate-limited"},
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"rows": []}'))
                ],
                usage=None,
            )

    monkeypatch.setattr(generator.time, "sleep", delays.append)
    client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat=SimpleNamespace(completions=Completions()),
    )
    result = generator.openai_compatible_llm(client)(
        {"system": "system", "user": "user"},
        model="openai/gpt-5.6-luna",
        max_tokens=8192,
        temperature=1.0,
        reasoning_effort="low",
    )
    assert calls == 2
    assert delays == [1]
    assert result["content"] == '{"rows": []}'


def test_materialized_rows_keep_per_submission_provider_provenance(
    tmp_path: Path,
) -> None:
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        TaskConfig,
        materialize,
    )

    def assignment(record_id: str) -> dict:
        return {
            "cleaned_record_id": record_id,
            "source_id": "skin_exposure",
            "status": "ok",
            "measurements_json": '[{"measurement":"1","unit":"%"}]',
            "quantity_count": 1,
            "assignment_method": "model_single_pass",
            "rejected_response_json": None,
            "raw_response_json": "{}",
        }

    records_path = tmp_path / "records.parquet"
    profile_path = tmp_path / "profile.json"
    records_path.write_text("records", encoding="utf-8")
    profile_path.write_text("profile", encoding="utf-8")
    mapping_path = tmp_path / "measurement_resolution.parquet"
    cache = SimpleNamespace(
        assignments={"openai": assignment("openai"), "luna": assignment("luna")},
        attempted={"openai", "luna"},
        provenance={
            "openai": {
                "inference_model": "gpt-5.4-mini",
                "inference_base_url": "https://api.openai.com/v1",
                "inference_credential_env": "OPENAI_API_KEY",
            },
            "luna": {
                "inference_model": "openai/gpt-5.6-luna",
                "inference_base_url": "https://openrouter.ai/api/v1",
                "inference_credential_env": "OPEN_ROUTER_KEY",
            },
        },
    )
    manifest = materialize(
        [
            {
                "id": "openai",
                "source_id": "skin_exposure",
                "canonical_endpoint_name": "dermal_absorption",
            },
            {
                "id": "luna",
                "source_id": "skin_exposure",
                "canonical_endpoint_name": "dermal_absorption",
            },
        ],
        cache,
        TaskConfig("skin_reaction"),
        mapping_path=mapping_path,
        records_path=records_path,
        profile_path=profile_path,
        model="openai/gpt-5.6-luna",
        api_base_url="https://openrouter.ai/api/v1",
        max_completion_tokens=8192,
        reasoning_mode="low",
    )
    rows = {
        row["cleaned_record_id"]: row
        for row in pq.read_table(mapping_path).to_pylist()
    }
    assert rows["openai"]["inference_model"] == "gpt-5.4-mini"
    assert rows["luna"]["inference_model"] == "openai/gpt-5.6-luna"
    assert manifest["model"] == "mixed"
    assert manifest["api_base_url"] == "mixed"
    assert manifest["inference_model_counts"] == {
        "gpt-5.4-mini": 1,
        "openai/gpt-5.6-luna": 1,
    }


def test_materialize_preserves_unmapped_unit_for_later_reconciliation(
    tmp_path: Path,
) -> None:
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        TaskConfig,
        materialize,
    )

    assignment = {
        "cleaned_record_id": "row-1",
        "source_id": "skin_exposure",
        "status": "ok",
        "measurements_json": '[{"measurement":"1","unit":"invented unit"}]',
        "quantity_count": 1,
        "assignment_method": "model_single_pass",
        "rejected_response_json": None,
        "raw_response_json": '{"status":"ok"}',
    }
    records_path = tmp_path / "records.parquet"
    profile_path = tmp_path / "profile.json"
    records_path.write_text("records", encoding="utf-8")
    profile_path.write_text("profile", encoding="utf-8")
    mapping_path = tmp_path / "measurement_resolution.parquet"
    manifest = materialize(
        [
            {
                "id": "row-1",
                "source_id": "skin_exposure",
                "canonical_endpoint_name": "flux",
            }
        ],
        SimpleNamespace(
            assignments={"row-1": assignment},
            attempted={"row-1"},
            provenance={},
        ),
        TaskConfig("skin_reaction"),
        mapping_path=mapping_path,
        records_path=records_path,
        profile_path=profile_path,
        model="openai/gpt-5.6-luna",
        api_base_url="https://openrouter.ai/api/v1",
        max_completion_tokens=8192,
        reasoning_mode="low",
    )
    [row] = pq.read_table(mapping_path).to_pylist()
    assert row["status"] == "ok"
    assert row["assignment_method"] == "model_single_pass"
    assert json.loads(row["measurements_json"])[0]["unit"] == "invented unit"
    assert row["rejected_response_json"] is None
    assert row["raw_response_json"] == '{"status":"ok"}'
    assert manifest["exact_unit_coverage"] is None


def test_materialize_collapses_percentage_descriptor_before_exact_unit_mapping(
    tmp_path: Path,
) -> None:
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        TaskConfig,
        materialize,
    )

    raw_response = (
        '{"id":"r1","status":"ok","measurements":'
        '[{"measurement":"20","unit":"% sensitization rate"}]}'
    )
    assignment = {
        "cleaned_record_id": "row-1",
        "source_id": "direct_skin_reaction",
        "status": "ok",
        "measurements_json": '[{"measurement":"20","unit":"% sensitization rate"}]',
        "quantity_count": 1,
        "assignment_method": "model_single_pass",
        "rejected_response_json": None,
        "raw_response_json": raw_response,
    }
    records_path = tmp_path / "records.parquet"
    profile_path = tmp_path / "profile.json"
    records_path.write_text("records", encoding="utf-8")
    profile_path.write_text("profile", encoding="utf-8")
    mapping_path = tmp_path / "measurement_resolution.parquet"
    manifest = materialize(
        [{
            "id": "row-1",
            "source_id": "direct_skin_reaction",
            "canonical_endpoint_name": "sensitization",
        }],
        SimpleNamespace(
            assignments={"row-1": assignment},
            attempted={"row-1"},
            provenance={},
        ),
        TaskConfig("skin_reaction"),
        mapping_path=mapping_path,
        records_path=records_path,
        profile_path=profile_path,
        model="gpt-5.4-mini",
        api_base_url="https://api.openai.com/v1",
        max_completion_tokens=8192,
        reasoning_mode="low",
    )
    [row] = pq.read_table(mapping_path).to_pylist()
    assert json.loads(row["measurements_json"]) == [
        {"measurement": "20", "unit": "%"}
    ]
    assert row["raw_response_json"] == raw_response
    assert row["unit_postprocess_rule_id"] == "skin_percent_unit_postprocess.v1"
    assert row["rejected_response_json"] is None
    assert manifest["post_extraction_processing"]["raw_response_preserved"] is True


def test_terminal_failure_can_be_retried_without_restoring_it_after_a_crash(
    tmp_path: Path,
) -> None:
    from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import (
        retryable_assignment_ids,
    )
    from data.processing.evidence_library.versions.v10.build_reference_semantics_mapping import (
        RequestBatch,
        SubmissionCache,
    )

    path = tmp_path / "requests.jsonl"
    row = {"id": "row-1", "source_id": "skin_exposure"}
    paid_batch = RequestBatch("paid", "skin_exposure", (row,), "prompt", 8192)
    fallback_batch = RequestBatch(
        "fallback", "skin_exposure", (row,), "prompt", 8192
    )
    failed = {
        "cleaned_record_id": "row-1",
        "source_id": "skin_exposure",
        "status": "unsure",
        "measurements_json": "[]",
        "quantity_count": 0,
        "assignment_method": "api_failure",
        "rejected_response_json": '{"reason":"api_failure"}',
        "raw_response_json": "{}",
    }

    cache = SubmissionCache(path)
    cache.submit(
        paid_batch,
        epoch="paid",
        model="gpt-5.4-mini",
        base_url="https://api.openai.com/v1",
    )
    cache.terminal(
        paid_batch,
        assignments=[failed],
        usage=None,
        response_status="api_failure",
    )
    cache = SubmissionCache(path)
    retry_ids = retryable_assignment_ids(cache)
    assert retry_ids == {"row-1"}
    cache.allow_retry(retry_ids)
    cache.submit(
        fallback_batch,
        epoch="unmetered",
        model="openai/gpt-5.6-luna",
        base_url="https://openrouter.ai/api/v1",
    )

    interrupted = SubmissionCache(path)
    assert "row-1" in interrupted.attempted
    assert "row-1" not in interrupted.assignments
    assert interrupted.provenance["row-1"]["inference_model"] == (
        "openai/gpt-5.6-luna"
    )
    assert retryable_assignment_ids(interrupted) == {"row-1"}
    assert retryable_assignment_ids(
        interrupted,
        retry_model="openai/gpt-5.6-luna",
        max_attempts=1,
    ) == set()


def test_unreviewed_axis_prunes_only_the_record_from_transfer() -> None:
    from data.processing.evidence_library.shared.v2.assay_transfer_measurements import (
        finalize_assay_transfer_measurement,
    )

    working = {
        "canonical_record_id": "record-1",
        "normalization_validity_status": "valid",
    }
    projected = {
        "canonical_record_id": "record-1",
        "source_id": "source",
        "canonical_endpoint_name": "newly extracted endpoint",
        "canonical_measurement_text": "1",
        "canonical_unit_text": "ng/mL",
        "canonicalization_status": "valid",
        "canonical_reference_scope": "absolute",
        "measurement_kind": "continuous",
        "finite_scalar_value": 1.0,
        "variation_value": None,
        "context": "context",
    }
    policy = {
        "policy_version": "frozen.test.v1",
        "bucket_decisions": {},
        "axis_decisions": {"unrelated": {"transform": "log10"}},
        "axis_decision_required_sources": ["source"],
    }
    contract = SimpleNamespace(
        pair_buckets={"source": SimpleNamespace(additional_dimensions=("context",))}
    )

    updated, persisted = finalize_assay_transfer_measurement(
        working,
        projected,
        record_contract=contract,
        policy=policy,
        prune_unreviewed_record=True,
    )

    reason = "measurement_axis_absent_from_frozen_transfer_policy"
    assert updated["assay_transfer_ineligibility_reason"] == reason
    assert persisted["assay_transfer_ineligibility_reason"] == reason
    assert persisted["finite_scalar_value"] == 1.0
    assert persisted["assay_transfer_transform_id"] == "raw.v1"


def test_pair_bucket_survives_unreviewed_axis_record_pruning() -> None:
    from data.processing.evidence_library.shared.v2.pair_buckets import (
        materialize_pair_buckets,
    )

    reason = "measurement_axis_absent_from_frozen_transfer_policy"
    rows, audit = materialize_pair_buckets(
        [
            {
                "canonical_record_id": "record-1",
                "source_id": "source",
                "canonical_endpoint_name": "newly extracted endpoint",
                "canonical_unit_text": "ng/mL",
                "canonicalization_status": "valid",
                "canonical_smiles": "CCC",
                "molecule_id": "molecule-1",
                "context": "context",
                "measurement_kind": "continuous",
                "finite_scalar_value": 1.0,
                "canonical_reference_scope": "absolute",
                "assay_transfer_ineligibility_reason": reason,
            }
        ],
        source_required_fields={"source": ("context",)},
    )

    assert rows[0]["pair_bucket_key"]
    assert rows[0]["assay_transfer_eligible"] is False
    assert rows[0]["assay_transfer_ineligibility_reason"] == reason
    assert audit["stats"]["ineligible_records"] == 1


def test_new_nonpositive_log_value_is_pruned_at_record_level() -> None:
    from data.processing.evidence_library.shared.v2.assay_transfer_measurements import (
        assay_transfer_axis_key,
        finalize_assay_transfer_measurement,
    )

    working = {
        "canonical_record_id": "record-1",
        "normalization_validity_status": "valid",
    }
    projected = {
        "canonical_record_id": "record-1",
        "source_id": "source",
        "canonical_endpoint_name": "endpoint",
        "canonical_measurement_text": "0",
        "canonical_unit_text": "participants",
        "canonicalization_status": "valid",
        "canonical_reference_scope": "unknown",
        "measurement_kind": "continuous",
        "finite_scalar_value": 0.0,
        "variation_value": None,
        "context": "context",
    }
    policy = {
        "policy_version": "frozen.test.v1",
        "bucket_decisions": {},
        "axis_decisions": {
            assay_transfer_axis_key(projected): {"transform": "log10"}
        },
        "axis_decision_required_sources": ["source"],
        "record_ineligibility": {},
    }
    contract = SimpleNamespace(
        pair_buckets={"source": SimpleNamespace(additional_dimensions=("context",))}
    )

    updated, persisted = finalize_assay_transfer_measurement(
        working,
        projected,
        record_contract=contract,
        policy=policy,
        prune_unreviewed_record=True,
    )

    reason = "nonpositive_measurement_on_frozen_log10_axis"
    assert updated["assay_transfer_ineligibility_reason"] == reason
    assert persisted["assay_transfer_ineligibility_reason"] == reason
    assert persisted["finite_scalar_value"] == 0.0
    assert persisted["assay_transfer_transform_id"] == "raw.v1"
