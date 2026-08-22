"""Response validation and candidate selection for the measurement-resolution pass.

The generator's job is to be boring and fail closed: every refusal publishes the row
as ``unsure`` with the reason kept, so a malformed answer degrades one row instead of
poisoning the artifact or aborting the run.  These tests pin that, plus the candidate
selection that decides what gets paid for.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
    MODEL,
    REASONING_EFFORT,
    STATUSES,
    TaskConfig,
    _stratified_slice,
    openai_compatible_llm,
    plan_batches,
    query_batch,
    validate_row,
)
from tools.chembl_tool.tasks.bbb_martins.starling_measurement_resolution import (
    BATCH_SIZE,
)


ROW = {"id": "rec-1", "source_id": "direct_bbb"}


def _ok(**overrides):
    payload = {
        "id": "rec-1",
        "status": "ok",
        "measurements": [{"measurement": "11.5", "unit": "10^-6 cm/s"}],
    }
    payload.update(overrides)
    return payload


def _request_batch():
    config = TaskConfig("bbb_martins")
    return plan_batches(
        [
            {
                "id": "rec-1",
                "source_id": "direct_bbb",
                "canonical_endpoint_name": "effective_permeability",
                "endpoint_name": "Pe",
                "measurement_text": "11.5",
                "unit_text": "10^-6 cm/s",
                "support_text": "x",
            }
        ],
        config,
        attempted=set(),
    )[0]


# --------------------------------------------------------------------------- #
# accepting a well-formed answer
# --------------------------------------------------------------------------- #


def test_a_well_formed_ok_row_is_accepted() -> None:
    assignment, reason = validate_row(_ok(), ROW, task="bbb_martins")
    assert reason is None
    assert assignment["status"] == "ok"
    assert assignment["quantity_count"] == 1
    assert json.loads(assignment["measurements_json"]) == [
        {"measurement": "11.5", "unit": "10^-6 cm/s"}
    ]
    assert assignment["rejected_response_json"] is None


@pytest.mark.parametrize("status", ["unsure", "relative", "unavailable"])
def test_the_non_ok_statuses_are_accepted_and_carry_nothing(status) -> None:
    """``unsure`` and ``relative`` are answers, not failures."""
    assignment, reason = validate_row(
        {"id": "rec-1", "status": status, "measurements": []}, ROW, task="bbb_martins"
    )
    assert reason is None
    assert assignment["status"] == status
    assert assignment["quantity_count"] == 0
    assert json.loads(assignment["measurements_json"]) == []


def test_several_quantities_with_different_units_are_kept_in_order() -> None:
    assignment, reason = validate_row(
        _ok(
            measurements=[
                {"measurement": "0.026", "unit": "ml·min^-1·g^-1"},
                {"measurement": "0.067", "unit": "min^-1"},
            ]
        ),
        ROW,
        task="bbb_martins",
    )
    assert reason is None
    assert assignment["quantity_count"] == 2
    assert [e["unit"] for e in json.loads(assignment["measurements_json"])] == [
        "ml·min^-1·g^-1",
        "min^-1",
    ]


def test_bioavailability_retains_repeated_measurement_units_for_explosion() -> None:
    assignment, reason = validate_row(
        _ok(
            measurements=[
                {"measurement": "18", "unit": "%"},
                {"measurement": "21", "unit": "%"},
            ]
        ),
        {"id": "rec-1", "source_id": "hf_bioavailability"},
        task="bioavailability_ma",
    )
    assert reason is None
    assert assignment["status"] == "ok"
    assert assignment["quantity_count"] == 2


def test_a_missing_measurements_key_is_read_as_empty() -> None:
    assignment, reason = validate_row(
        {"id": "rec-1", "status": "unavailable"}, ROW, task="bbb_martins"
    )
    assert reason is None and assignment["quantity_count"] == 0


# --------------------------------------------------------------------------- #
# every refusal is fail-closed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "returned,expected_reason",
    [
        ("not an object", "not_an_object"),
        ({}, "id_mismatch"),
        ({"id": "someone-else", "status": "ok"}, "id_mismatch"),
        ({"id": "rec-1", "status": "definitely"}, "unsupported_status"),
        ({"id": "rec-1", "status": "ok", "measurements": {}}, "measurements_not_a_list"),
        ({"id": "rec-1", "status": "ok", "measurements": []}, "ok_without_measurements"),
        (
            {"id": "rec-1", "status": "relative",
             "measurements": [{"measurement": "2.5", "unit": "fold"}]},
            "non_ok_carries_measurements",
        ),
        ({"id": "rec-1", "status": "ok", "measurements": ["11.5"]}, "entry_not_an_object"),
        (
            {"id": "rec-1", "status": "ok", "measurements": [{"unit": "cm/s"}]},
            "empty_measurement",
        ),
        (
            {"id": "rec-1", "status": "ok", "measurements": [{"measurement": "11.5"}]},
            "entry_without_a_unit",
        ),
    ],
)
def test_a_malformed_answer_degrades_one_row_rather_than_the_run(
    returned, expected_reason
) -> None:
    assignment, reason = validate_row(returned, ROW, task="bbb_martins")
    assert reason is not None and expected_reason in reason
    assert assignment["status"] == "unsure"
    assert assignment["quantity_count"] == 0
    assert assignment["cleaned_record_id"] == "rec-1"


def test_an_id_belonging_to_another_row_is_refused() -> None:
    """The one guard left against cross-row bleed.

    The schema carries no verbatim quote, so a value attributed to the *right* id
    cannot be checked.  A value attributed to the *wrong* id still can, and must be.
    """
    assignment, reason = validate_row(_ok(id="rec-2"), ROW, task="bbb_martins")
    assert reason == "id_mismatch"
    assert assignment["status"] == "unsure"


def test_an_entry_must_be_self_contained() -> None:
    """A unit-less entry could not become an exploded row on its own."""
    _, reason = validate_row(
        _ok(measurements=[{"measurement": "0.03", "unit": "10^-6 cm/s"},
                          {"measurement": "0.1"}]),
        ROW,
        task="bbb_martins",
    )
    assert reason == "entry_without_a_unit"


def test_the_status_vocabulary_is_exactly_the_four_documented_values() -> None:
    assert STATUSES == ("ok", "unsure", "relative", "unavailable")


# --------------------------------------------------------------------------- #
# candidate selection
# --------------------------------------------------------------------------- #


def test_a_pilot_slice_spreads_across_sources() -> None:
    """A head slice of a source-sorted list is one source, so it would never
    exercise the prompt variant used by sources without a unit column."""
    candidates = (
        [{"id": f"a{i}", "source_id": "direct_bbb"} for i in range(500)]
        + [{"id": f"b{i}", "source_id": "efflux_transport"} for i in range(500)]
        + [{"id": f"c{i}", "source_id": "passive_permeability"} for i in range(500)]
    )
    sliced = _stratified_slice(candidates, 300)
    assert len(sliced) == 300
    counts = {s: sum(1 for r in sliced if r["source_id"] == s) for s in
              {r["source_id"] for r in sliced}}
    assert len(counts) == 3
    assert all(count == 100 for count in counts.values()), counts


def test_a_pilot_slice_is_reproducible() -> None:
    candidates = [{"id": f"a{i}", "source_id": "s"} for i in range(100)]
    assert _stratified_slice(candidates, 10) == _stratified_slice(candidates, 10)


def test_a_slice_larger_than_the_pool_returns_everything() -> None:
    candidates = [{"id": f"a{i}", "source_id": "s"} for i in range(7)]
    assert len(_stratified_slice(candidates, 50)) == 7


def test_batches_never_mix_sources() -> None:
    """The prompt differs per source, so a batch spanning two would carry the wrong
    instructions for half its rows."""
    config = TaskConfig("bbb_martins")
    candidates = [
        {"id": f"a{i}", "source_id": "direct_bbb",
         "canonical_endpoint_name": "effective_permeability", "endpoint_name": "Pe",
         "measurement_text": "1", "unit_text": "cm/s", "support_text": "x"}
        for i in range(5)
    ] + [
        {"id": f"b{i}", "source_id": "efflux_transport",
         "canonical_endpoint_name": "efflux_ratio", "endpoint_name": "ER",
         "measurement_text": "2", "support_text": "y"}
        for i in range(5)
    ]
    batches = plan_batches(candidates, config, attempted=set())
    assert batches
    for batch in batches:
        assert len({row["source_id"] for row in batch.rows}) == 1
    # and the two sources get different prompts
    prompts = {batch.source_id: batch.prompt for batch in batches}
    assert prompts["direct_bbb"] != prompts["efflux_transport"]


def test_already_attempted_rows_are_never_re_batched() -> None:
    """The submission cache is what makes a resumed run safe; batching must honour it."""
    config = TaskConfig("bbb_martins")
    candidates = [
        {"id": f"a{i}", "source_id": "direct_bbb",
         "canonical_endpoint_name": "effective_permeability", "endpoint_name": "Pe",
         "measurement_text": "1", "unit_text": "cm/s", "support_text": "x"}
        for i in range(40)
    ]
    original = plan_batches(candidates, config, attempted=set())
    batches = plan_batches(candidates, config, attempted=set(original[0].row_ids))
    remaining = [row["id"] for batch in batches for row in batch.rows]
    expected = [row_id for batch in original[1:] for row_id in batch.row_ids]
    assert sorted(remaining) == sorted(expected)


def test_the_payload_hides_the_routing_source_id() -> None:
    """The model is told the fields, not the bookkeeping column it must not condition on."""
    config = TaskConfig("bbb_martins")
    candidates = [
        {"id": "a1", "source_id": "direct_bbb",
         "canonical_endpoint_name": "effective_permeability", "endpoint_name": "Pe",
         "measurement_text": "1", "unit_text": "cm/s", "support_text": "x"}
    ]
    batch = plan_batches(candidates, config, attempted=set())[0]
    assert "source_id" not in batch.api_rows[0]
    assert batch.api_rows[0]["canonical_endpoint_name"] == "effective_permeability"
    assert batch.api_rows[0]["id"] == "a1"


def test_prompt_uses_endpoint_aware_reference_semantics() -> None:
    prompt = TaskConfig("bbb_martins").render_prompt("direct_bbb")
    assert BATCH_SIZE == 10
    assert "Solve concisely." in prompt
    assert "concentration_to_plasma_ratio" in prompt
    assert "csf_concentration_increase" in prompt
    assert "inhibition_potency" in prompt
    assert "k1_MeG/k1_FDG" in prompt
    assert "`around 1` -> `1`" in prompt
    assert "Never turn a bound, range, or fraction into a point" in prompt
    assert "shortest complete unit explicitly stated" in prompt
    assert "same-unit comparator values in support do not make" in prompt
    assert "Do not exclude an outcome merely because" in prompt
    assert "102.9% inhibition" in prompt
    assert "analgesic `%MPE`" in prompt
    assert "do not replace the conflicting linear unit" in prompt
    assert "If the row explicitly reports its value with a power-of-ten scale" in prompt
    assert "Never add a power-of-ten scale" in prompt
    assert "Never infer a scale" in prompt
    assert 'measurement `3.16`, unit `10^-6 cm/s`' in prompt
    assert "Return that implied unit rather than null" in TaskConfig(
        "bbb_martins"
    ).render_prompt("passive_permeability")


def test_skin_prompt_selects_one_value_and_treats_external_references_conservatively() -> None:
    from tools.chembl_tool.tasks.skin_reaction.starling_measurement_resolution import (
        PROMPT_VERSION,
        SOURCE_IDS,
        prompt_row_fields,
        render_prompt,
    )

    prompt = render_prompt("sensitization_aop")
    assert PROMPT_VERSION == "skin_reaction_measurement_resolution_prompt.v7"
    assert set(SOURCE_IDS) == {
        "direct_skin_reaction",
        "sensitization_aop",
        "skin_exposure",
        "phototoxicity_irritation_local_damage",
    }
    assert "unit_text" not in prompt_row_fields("direct_skin_reaction")
    assert "unit_text" not in prompt_row_fields(
        "phototoxicity_irritation_local_damage"
    )
    assert "unit_text" in prompt_row_fields("sensitization_aop")
    assert "This source has no unit column" in render_prompt("direct_skin_reaction")
    assert "`measurement_text` is the primary value selector" in prompt
    assert "Stage 02 will explode multiple returned quantities" in prompt
    assert "exact unit map decides mapping or exclusion" in prompt
    assert "never add a support-only number" in prompt
    assert "A separate comparator in support does not make" in prompt
    assert "With `missing_endpoint`" in prompt
    assert "ALN cell proliferation" in prompt
    assert "percent of an exchangeable ion pool" in prompt
    assert "KeratinoSens Imax uses `fold`" in prompt


def test_small_endpoints_are_co_packed_but_remain_contiguous() -> None:
    config = TaskConfig("bbb_martins")
    candidates = [
        {
            "id": f"{endpoint}-{index}",
            "source_id": "efflux_transport",
            "canonical_endpoint_name": endpoint,
            "endpoint_name": endpoint,
            "measurement_text": "2",
            "support_text": "x",
        }
        for endpoint in ("alpha", "beta")
        for index in range(3)
    ]
    batches = plan_batches(candidates, config, attempted=set())
    assert len(batches) == 1
    assert [row["canonical_endpoint_name"] for row in batches[0].rows] == [
        "alpha",
        "alpha",
        "alpha",
        "beta",
        "beta",
        "beta",
    ]
    assert "[endpoint: alpha - no reliable deterministic summary]" in batches[0].prompt
    assert "[endpoint: beta - no reliable deterministic summary]" in batches[0].prompt


def test_endpoint_context_excludes_rows_in_the_active_request() -> None:
    config = TaskConfig("bbb_martins")
    candidates = [
        {
            "id": "active",
            "source_id": "direct_bbb",
            "canonical_endpoint_name": "effective_permeability",
            "endpoint_name": "Pe",
            "measurement_text": "11.5",
            "unit_text": "x 10^6 cm/s",
            "support_text": "target row",
        }
    ]
    unit = {
        "unit": "cm/s",
        "n": 10,
        "median": 1e-5,
        "p10": 1e-6,
        "p90": 5e-5,
        "example_candidates": [
            {
                "cleaned_record_id": "active",
                "measurement_text": "11.5",
                "unit_text": "x 10^6 cm/s",
                "folded_value": 1.15e-5,
                "canonical_unit": "cm/s",
                "support_text": "must not be shown",
            },
            {
                "cleaned_record_id": "reference",
                "measurement_text": "9.2e-6",
                "unit_text": "cm/s",
                "folded_value": 9.2e-6,
                "canonical_unit": "cm/s",
                "support_text": "independent deterministic example",
            },
        ],
    }
    profile = {
        "endpoints": {
            "direct_bbb|effective_permeability": {
                "source_id": "direct_bbb",
                "canonical_endpoint_name": "effective_permeability",
                "resolved_rows": 10,
                "units": [unit],
            }
        }
    }
    batch = plan_batches(
        candidates,
        config,
        attempted=set(),
        endpoint_profile=profile,
        profile_digest="profile-a",
    )[0]
    assert "independent deterministic example" in batch.prompt
    assert "must not be shown" not in batch.prompt
    assert "median 1e-05" in batch.prompt


def test_request_identity_covers_the_endpoint_profile() -> None:
    config = TaskConfig("bbb_martins")
    candidates = [
        {
            "id": "a1",
            "source_id": "direct_bbb",
            "canonical_endpoint_name": "effective_permeability",
            "endpoint_name": "Pe",
            "measurement_text": "1",
            "unit_text": "cm/s",
            "support_text": "x",
        }
    ]
    first = plan_batches(
        candidates, config, attempted=set(), profile_digest="profile-a"
    )[0]
    repeated = plan_batches(
        candidates, config, attempted=set(), profile_digest="profile-a"
    )[0]
    second = plan_batches(
        candidates, config, attempted=set(), profile_digest="profile-b"
    )[0]
    assert first.request_id == repeated.request_id
    assert first.prompt == repeated.prompt
    assert first.request_id != second.request_id
    assert first.request_id != plan_batches(
        candidates, config, attempted=set(), model="another-model"
    )[0].request_id


def test_extraction_call_is_frozen_to_gpt_5_4_mini_low_reasoning() -> None:
    config = TaskConfig("bbb_martins")
    candidates = [
        {
            "id": "a1",
            "source_id": "direct_bbb",
            "canonical_endpoint_name": "effective_permeability",
            "endpoint_name": "Pe",
            "measurement_text": "1",
            "unit_text": "cm/s",
            "support_text": "x",
        }
    ]
    batch = plan_batches(candidates, config, attempted=set())[0]
    call = {}

    def fake_llm(messages, **kwargs):
        call.update(kwargs)
        return {
            "content": json.dumps(
                {
                    "rows": [
                        {
                            "id": "a1",
                            "status": "ok",
                            "measurements": [
                                {"measurement": "1", "unit": "cm/s"}
                            ],
                        }
                    ]
                }
            ),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    rows, _, status = query_batch(
        batch, llm=fake_llm, model=MODEL, task="bbb_martins"
    )
    assert status == "valid" and rows[0]["status"] == "ok"
    assert MODEL == "gpt-5.4-mini"
    assert REASONING_EFFORT == "low"
    assert call["model"] == MODEL
    assert call["reasoning_effort"] == REASONING_EFFORT


@pytest.mark.parametrize(
    "first_content",
    [
        "{broken json",
        json.dumps({"rows": [{"id": "rec-1", "status": "definitely"}]}),
    ],
)
def test_query_retries_malformed_json_or_row_schema_once(first_content) -> None:
    contents = iter([first_content, json.dumps({"rows": [_ok()]})])
    calls = 0

    def fake_llm(messages, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "content": next(contents),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    rows, usage, status = query_batch(
        _request_batch(), llm=fake_llm, model=MODEL, task="bbb_martins"
    )
    assert calls == 2
    assert status == "valid_after_structural_retry"
    assert rows[0]["status"] == "ok"
    assert rows[0]["assignment_method"] == "model_structural_retry"
    assert usage == {"input_tokens": 20, "output_tokens": 10}


@pytest.mark.parametrize(
    "first_rows",
    [
        [],
        [_ok(id="foreign")],
        [_ok(), _ok()],
        [_ok(), _ok(id="foreign")],
    ],
)
def test_query_retries_missing_mismatched_duplicate_or_foreign_ids(first_rows) -> None:
    contents = iter(
        [json.dumps({"rows": first_rows}), json.dumps({"rows": [_ok()]})]
    )
    calls = 0

    def fake_llm(messages, **kwargs):
        nonlocal calls
        calls += 1
        return {"content": next(contents)}

    rows, _, status = query_batch(
        _request_batch(), llm=fake_llm, model=MODEL, task="bbb_martins"
    )
    assert calls == 2
    assert status == "valid_after_structural_retry"
    assert rows[0]["assignment_method"] == "model_structural_retry"


def test_query_does_not_retry_a_measurement_that_is_not_a_plain_decimal() -> None:
    calls = 0

    def fake_llm(messages, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "content": json.dumps(
                {
                    "rows": [
                        _ok(measurements=[{"measurement": "<0.1", "unit": "%"}])
                    ]
                }
            )
        }

    rows, _, status = query_batch(
        _request_batch(), llm=fake_llm, model=MODEL, task="bbb_martins"
    )
    assert calls == 1
    assert status == "partial_invalid"
    assert rows[0]["assignment_method"] == (
        "invalid_row_response:measurement_is_not_a_plain_decimal"
    )


def test_query_does_not_retry_an_api_exception() -> None:
    calls = 0

    def fake_llm(messages, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("service unavailable")

    rows, usage, status = query_batch(
        _request_batch(), llm=fake_llm, model=MODEL, task="bbb_martins"
    )
    assert calls == 1
    assert status == "api_failure"
    assert usage is None
    assert rows[0]["assignment_method"] == "api_failure"


def test_openai_compatible_adapter_uses_chat_json_mode() -> None:
    call = {}

    class Completions:
        def create(self, **kwargs):
            call.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content='{"rows": []}', reasoning_content="reasoning"
                ))],
                usage=SimpleNamespace(model_dump=lambda: {"prompt_tokens": 3}),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = openai_compatible_llm(client)(
        {"system": "system", "user": "user"},
        model="local-model",
        max_tokens=8192,
        temperature=1.0,
    )
    assert call["response_format"] == {"type": "json_object"}
    assert call["max_tokens"] == 8192
    assert result["content"] == '{"rows": []}'
    assert result["reasoning"] == "reasoning"


def test_resume_refuses_a_partially_attempted_request() -> None:
    config = TaskConfig("bbb_martins")
    candidates = [
        {
            "id": f"a{i}",
            "source_id": "direct_bbb",
            "canonical_endpoint_name": "effective_permeability",
            "endpoint_name": "Pe",
            "measurement_text": "1",
            "unit_text": "cm/s",
            "support_text": "x",
        }
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="only part of deterministic request"):
        plan_batches(candidates, config, attempted={"a0"})


# --------------------------------------------------------------------------- #
# gold replay: the pilot must run the rows we can actually score
# --------------------------------------------------------------------------- #


def test_gold_replay_targets_only_rows_that_reach_the_model() -> None:
    """``accept`` and ``reject`` gold cases are settled by rule and never sent.

    Replaying them would pay for rows whose answer is already asserted
    deterministically, and would produce no new information.
    """
    from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
        gold_extract_ids,
    )
    from tools.chembl_tool.common.starling.measurement_routing import route
    import json as _json
    from pathlib import Path as _Path

    ids = gold_extract_ids()
    fixture = _Path("tests/chembl_tool/common/fixtures/measurement_resolution_gold.jsonl")
    cases = [
        _json.loads(line)
        for line in fixture.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    config = TaskConfig("bbb_martins")
    extract = {
        case["audit_case_id"].split(":", 1)[1]
        for case in cases
        if route(
            case["input"],
            config.rules[case["source_id"]],
            task=config.task_id,
        ).bucket
        == "extract"
    }
    assert set(ids) == extract
    assert len(ids) == 392
    assert len(cases) - len(ids) == 108


def test_gold_replay_routes_a_historical_fixture_with_current_rules() -> None:
    import json as _json

    from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
        GOLD_FIXTURE,
        gold_extract_ids,
    )

    lines = GOLD_FIXTURE.read_text(encoding="utf-8").splitlines()
    manifest = _json.loads(lines[0])
    manifest["routing_version"] = "starling_measurement_routing.v0"
    tmp = GOLD_FIXTURE.parent / "_stale_gold_for_test.jsonl"
    tmp.write_text("\n".join([_json.dumps(manifest)] + lines[1:]) + "\n", encoding="utf-8")
    try:
        assert gold_extract_ids(tmp) == gold_extract_ids()
    finally:
        tmp.unlink()


def test_a_pilot_that_scores_nothing_is_the_failure_mode_this_avoids() -> None:
    """The reason ``--gold-replay`` exists rather than a fresh random sample.

    392 labelled rows sit in a pool of ~178k candidates, so an independent 300-row
    draw expects fewer than one -- the pilot would spend tokens and yield no accuracy
    number at all.  Replaying the labelled ids instead makes every response
    scoreable, and costs fewer requests.
    """
    from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
        _stratified_slice,
        gold_extract_ids,
    )

    gold = set(gold_extract_ids())
    pool = [
        {"id": f"filler-{i}", "source_id": "direct_bbb"} for i in range(178_000)
    ] + [{"id": row_id, "source_id": "direct_bbb"} for row_id in gold]
    drawn = {row["id"] for row in _stratified_slice(pool, 300)}
    assert len(drawn & gold) < 10, "a random draw should almost never hit the gold rows"
    # whereas a replay hits every labelled extraction row.
    assert len(gold) == 392


def test_an_ok_row_without_a_plain_decimal_is_degraded_to_unsure() -> None:
    for measurement in ("<0.1", "0.25 - 0.4", "up to 7", ">10", "20–50%", "~ 0.2"):
        assignment, reason = validate_row(
            _ok(measurements=[{"measurement": measurement, "unit": "%"}]),
            ROW,
            task="bbb_martins",
        )
        assert reason is not None and "not_a_plain_decimal" in reason, measurement
        assert assignment["status"] == "unsure"
        assert assignment["quantity_count"] == 0


def test_plain_decimal_validation_does_not_parse_or_normalize_units() -> None:
    from tools.chembl_tool.common.starling.build_measurement_resolution_mapping import (
        is_plain_decimal,
    )

    assert is_plain_decimal("11.5")
    assert is_plain_decimal("-.5")
    assert not is_plain_decimal("8 ± 1.5")
    assert not is_plain_decimal("<0.1")
    assert not is_plain_decimal("0.25 - 0.4")
    assert not is_plain_decimal("1e-6")
    assert not is_plain_decimal("1_000")


def test_the_raw_answer_is_retained_so_a_rule_change_can_be_rescored() -> None:
    """Re-validating a frozen artifact must not require re-spending tokens.

    With a partial budget that matters: the submission cache never re-asks a row,
    so without the raw answer a tightened rule could only be applied to rows bought
    again.
    """
    accepted, _ = validate_row(_ok(), ROW, task="bbb_martins")
    assert json.loads(accepted["raw_response_json"])["status"] == "ok"
    refused, reason = validate_row(
        _ok(measurements=[{"measurement": "<0.1", "unit": "%"}]),
        ROW,
        task="bbb_martins",
    )
    assert reason is not None
    assert json.loads(refused["raw_response_json"])["measurements"][0][
        "measurement"
    ] == "<0.1"
