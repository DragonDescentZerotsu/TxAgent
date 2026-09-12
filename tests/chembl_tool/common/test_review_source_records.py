import copy
import asyncio
import json
from types import SimpleNamespace

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from tools.chembl_tool.common.starling import review_source_records as runner
from tools.chembl_tool.common.starling.review_source_records import (
    messages, read_prior, validate_verdict,
)


RAW = {"source_row_uid": "source1", "pmid": "123", "molecule_name": "Drug A",
       "causal_status": "not_supported",
       "support_text": "In this trial, 100 patients received Drug A. No liver injury was observed."}


def verdict():
    return {"endpoint_alignment": "target_outcome", "review_status": "supported", "reason": "observed_conditioned_negative",
            "notes": "Negative within this trial, not universal safety.",
            "specimen": {"name": "Drug A", "attribution": "exact_named_entity", "quote": "Drug A"},
            "claims": [{"negative_evidence": "observed_absence_of_target", "Y": 0, "evidence_quote": "No liver injury was observed.",
                        "evidence_basis": "primary_human",
                        "study_scope": "single_identified_original",
                        "study": {"kind": "reporting_pmid", "reference": "123", "quote": "In this trial"},
                        "conditions": [{"axis": "population", "value": "100 patients", "quote": "100 patients"}],
                        "limitations": "Only observed trial participants."}]}


def test_supported_claim_retains_unresolved_study_without_inventing_vote():
    value = verdict()
    value["claims"][0]["study"] = {"kind": "unresolved", "reference": "", "quote": ""}
    assert validate_verdict(value, RAW) == value


def test_pmid_prefix_is_normalized_without_rejecting_valid_study():
    value = verdict()
    value["claims"][0]["study"]["reference"] = "PMID: 123"
    assert validate_verdict(value, RAW)["claims"][0]["study"]["reference"] == "123"


def test_secondary_reporting_pmid_and_wrong_endpoint_are_not_valid_labels():
    value = verdict()
    value["claims"][0]["evidence_basis"] = "identifiable_secondary"
    with pytest.raises(ValueError, match="secondary/unresolved"):
        validate_verdict(value, RAW)
    value = verdict()
    value["endpoint_alignment"] = "narrower_or_different_outcome"
    with pytest.raises(ValueError, match="endpoint alignment"):
        validate_verdict(value, RAW)


def test_reporting_pmid_cannot_be_relabelled_as_a_cited_original_study():
    value = verdict()
    value["claims"][0]["study"]["kind"] = "cited_pmid"
    with pytest.raises(ValueError, match="different explicitly cited"):
        validate_verdict(value, RAW)


@pytest.mark.parametrize("change", ["hallucinated_quote", "invented_pmid", "invented_condition", "boolean_label", "extractor_status_as_evidence"])
def test_invalid_evidence_fails_validation(change):
    value = verdict()
    claim = value["claims"][0]
    if change == "hallucinated_quote":
        claim["evidence_quote"] = "This medicine is always safe."
    elif change == "invented_pmid":
        claim["study"]["reference"] = "999"
    elif change == "invented_condition":
        claim["conditions"][0]["quote"] = "children aged 5"
    elif change == "boolean_label":
        claim["Y"] = False
    else:
        claim["evidence_quote"] = "not_supported"
    with pytest.raises(ValueError):
        validate_verdict(value, RAW)


def test_distinct_opposite_arms_are_not_forced_into_one_label():
    raw = {**RAW, "extra_details": "Liver injury occurred after overdose."}
    value = verdict()
    positive = copy.deepcopy(value["claims"][0])
    positive.update(Y=1, negative_evidence="not_negative", evidence_quote=raw["extra_details"], conditions=[
        {"axis": "exposure", "value": "overdose", "quote": "after overdose"}])
    value["claims"].append(positive)
    assert [c["Y"] for c in validate_verdict(value, raw)["claims"]] == [0, 1]


def test_source_prompt_preserves_all_fields_without_truncation():
    raw = {**RAW, "extra_details": "x" * 30000, "needs_more_context": True}
    assert json.loads(messages("dili", raw)[1]["content"]) == raw


def test_resume_does_not_turn_error_into_rejection_and_repairs_partial_tail(tmp_path):
    path = tmp_path / "round1.jsonl"
    base = {"task": "dili", "review_contract_sha256": "c", "execution_sha256": "e",
            "source_row_uid": "u", "source_payload_sha256": "p", "status": "ok"}
    complete = json.dumps(base) + "\n" + json.dumps({**base, "source_row_uid": "failed", "status": "error"}) + "\n"
    path.write_text(complete + "{\"partial\":")
    assert read_prior(path, {"dili": "c"}, "e") == {("dili", "u", "p")}
    assert path.read_text() == complete
    with pytest.raises(ValueError, match="stale"):
        read_prior(path, {"dili": "changed"}, "e")


def test_complete_corrupt_line_is_not_silently_skipped(tmp_path):
    path = tmp_path / "round1.jsonl"
    path.write_text("invalid\n")
    with pytest.raises(json.JSONDecodeError):
        read_prior(path, {"dili": "c"}, "e")


def test_full_census_two_independent_passes_and_resume(tmp_path, monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def async_chat_json(self, request):
            raw = json.loads(request[1]["content"])
            calls.append(raw["source_row_uid"])
            return {"content": verdict(), "raw_content": json.dumps(verdict()),
                    "reasoning_content": "test", "usage": {}, "model": "fake", "id": "test"}

        async def aclose(self):
            pass

    monkeypatch.setattr(runner, "OpenAICompatibleClient", FakeClient)
    manifest = {"sources": {}}
    for task in runner.TASKS:
        rows = [{**RAW, "source_row_uid": f"{task}_{i}", "causal_status": label,
                 "carcinogenicity_conclusion": label, "SMILES": "invalid", "needs_more_context": True}
                for i, label in enumerate(("not_supported" if task == "dili" else "negative", "positive", None))]
        path = tmp_path / f"{task}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        manifest["sources"][task] = {"path": str(path)}
    args = SimpleNamespace(root=tmp_path, workers=2, base_url="unused", model="fake", timeout=1,
                           max_tokens=1024, reasoning_effort="medium", attempts=1, limit=0)
    for number in (1, 2):
        result = asyncio.run(runner.run_pass(args, manifest, number, "exec"))
        assert result["ok"] == 6
        assert result["error"] == 0
    assert len(calls) == 12
    assert set(calls) == {f"{task}_{i}" for task in runner.TASKS for i in range(3)}
    result = asyncio.run(runner.run_pass(args, manifest, 1, "exec"))
    assert result["scheduled"] == 0
    assert len(calls) == 12


def test_live_concurrency_reduction_does_not_cancel_inflight_review():
    async def scenario():
        limiter = runner.ReviewConcurrency(2)
        await limiter.__aenter__()
        await limiter.__aenter__()
        await limiter.resize(1)
        waiting = asyncio.create_task(limiter.__aenter__())
        await asyncio.sleep(0)
        assert limiter.active == 2 and not waiting.done()
        await limiter.__aexit__()
        await asyncio.sleep(0)
        assert limiter.active == 1 and not waiting.done()
        await limiter.__aexit__()
        await asyncio.wait_for(waiting, timeout=1)
        assert limiter.active == 1
        await limiter.__aexit__()
    asyncio.run(scenario())


def test_retry_receives_invalid_answer_and_preserves_earlier_validation_errors(monkeypatch):
    requests = []

    class Client:
        async def async_chat_json(self, request):
            requests.append(request)
            value = verdict()
            if len(requests) == 1:
                value["claims"][0]["evidence_quote"] = "Invented text"
            elif len(requests) == 2:
                value["claims"][0]["conditions"][0]["axis"] = "invented_axis"
            return {"content": value, "raw_content": json.dumps(value),
                    "reasoning_content": "test", "usage": {}, "model": "fake", "id": "test"}

    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(runner.asyncio, "sleep", no_sleep)
    result = asyncio.run(runner.review_one(Client(), "dili", RAW, 3))
    assert result["status"] == "ok" and result["attempt_count"] == 3
    assert requests[2][-2]["role"] == "assistant"
    assert "invented_axis" in requests[2][-2]["content"]
    assert "claim quote not in source" in requests[2][-1]["content"]
    assert "invalid condition axis" in requests[2][-1]["content"]


def test_unresolved_first_pass_errors_do_not_suppress_second_pass(tmp_path, monkeypatch):
    calls = []

    async def run_pass(args, manifest, number, execution_hash):
        calls.append(number)
        return {"error": 1 if number == 1 else 0}

    monkeypatch.setattr(runner, "prepare", lambda root: {"n_records": 1})
    monkeypatch.setattr(runner, "run_pass", run_pass)
    monkeypatch.setattr(runner, "read_prior", lambda path, *rest: set() if path.name == "round1.jsonl" else {"ok"})
    monkeypatch.setattr("sys.argv", ["review", "run", "--root", str(tmp_path), "--error-recovery-passes", "0"])
    with pytest.raises(SystemExit, match="coverage mismatch"):
        runner.main()
    assert calls == [1, 2]
    assert json.loads((tmp_path / "completion.json").read_text())["status"] == "incomplete_review_errors"
