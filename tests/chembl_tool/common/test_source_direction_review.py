import asyncio
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.chembl_tool.common.json_utils import sha256_file
from tools.chembl_tool.common.starling import source_direction_review as r


def raw(**overrides):
    return {"source_row_uid": "u1", "paragraph_idx": 1, "extraction_id": "e1", "pmid": "123",
            "molecule_name": "Drug A", "SMILES": "CCO", "agent_type": "small_molecule_or_chemical",
            "causal_status": "not_supported", "support_text": "Drug A was excluded as the cause in this patient.",
            "qualifying_conditions": None, "extra_details": None, **overrides}


@pytest.mark.parametrize("updates", [
    {"human_evidence_basis": "individual_case"},
    {"support_text": "Drug A showed no significant increase in liver enzyme elevations."},
    {"support_text": "No serious or fatal liver disease was attributed to Drug A."},
    {"needs_more_context": True, "human_evidence_basis": "review_or_reference_assertion"},
])
def test_retain_reported_negative_without_old_scope_or_condition_veto(updates):
    assert r.route("dili", raw(**updates), None) == ("explicit_source_label_retained", 0)


def test_uncertain_and_disagreeing_directions_are_reviewed_not_forced_binary():
    assert r.route("dili", raw(causal_status="possible"), None) == ("nonbinary_source_review", None)
    assert r.route("dili", raw(), {"directions": [1]}) == ("direction_conflict_review", 0)
    assert r.route("dili", raw(), {"directions": [0, 1]}) == ("direction_conflict_review", 0)


def test_previous_supported_direction_can_be_reused_without_old_rejections():
    assert r.route("dili", raw(causal_status="possible"), {"directions": [1]}) == ("prior_supported_direction_reused", 1)


def test_identity_hold_preserves_original_negative_instead_of_relabeling():
    assert r.route("dili", raw(agent_type="fixed_combination"), None) == ("specimen_or_material_hold", 0)


def test_exact_dedup_preserves_different_studies_conditions_and_direction():
    base = raw()
    key = r.review_key("dili", base)
    assert key == r.review_key("dili", raw(source_row_uid="u2", paragraph_idx=5, extraction_id="e2"))
    for changed in [raw(pmid="456"), raw(qualifying_conditions="overdose"), raw(causal_status="possible")]:
        assert key != r.review_key("dili", changed)


def test_quote_and_focal_attribution_are_validated_without_universal_hazard_requirement():
    v = {"direction": "negative", "attribution": "focal", "scope": "This patient's causal attribution",
         "quote": "Drug A was excluded as the cause", "reason": "Explicit patient-specific negative"}
    assert r.validate(v, raw()) == v
    with pytest.raises(ValueError, match="another entity"):
        r.validate({**v, "attribution": "other_entity"}, raw())
    with pytest.raises(ValueError, match="literal"):
        r.validate({**v, "quote": "No person can develop injury"}, raw())


@pytest.mark.parametrize("workers,pools", [(1, 8), (3, 2), (11, 4)])
def test_queue_visits_both_tasks_and_resume_skips_completed_sources(tmp_path, monkeypatch, workers, pools):
    calls = []
    clients = []
    class Client:
        def __init__(self, **kwargs):
            self.capacity = kwargs["async_max_connections"]
            self.active = 0
            self.closed = False
            clients.append(self)
        async def async_chat_json(self, messages):
            self.active += 1
            assert self.active <= self.capacity
            source = json.loads(messages[1]["content"]); calls.append(source["source_row_uid"])
            await asyncio.sleep(0)
            self.active -= 1
            value = {"direction": "negative", "attribution": "focal", "scope": "case-specific",
                     "quote": source["support_text"], "reason": "Source conclusion"}
            return {"content": value, "raw_content": json.dumps(value), "reasoning_content": "test",
                    "usage": {}, "model": "fake", "id": "test"}
        async def aclose(self): self.closed = True
    monkeypatch.setattr(r, "OpenAICompatibleClient", Client)
    manifest = {"tasks": {}, "prompt_sha256": "test", "counts": {"unique_queued": 4}}
    for task in r.TASKS:
        path = tmp_path / f"{task}.parquet"
        pq.write_table(pa.Table.from_pylist([raw(source_row_uid=f"{task}{i}", pmid=str(123+i)) for i in range(2)]), path)
        manifest["tasks"][task] = {"queue_path": str(path), "queue_sha256": sha256_file(path)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    args = SimpleNamespace(root=tmp_path, workers=workers, client_pools=pools,
                           base_url="unused", attempts=1, limit=0)
    assert asyncio.run(r.run(args, manifest))["ok"] == 4
    assert sum(c.capacity for c in clients) == workers
    assert all(c.closed for c in clients)
    assert calls == ["dili0", "carcinogens0", "dili1", "carcinogens1"]
    assert asyncio.run(r.run(args, manifest))["scheduled"] == 0
    assert len(calls) == 4
    assert json.loads((tmp_path / "completion.json").read_text())["status"] == "review_complete"
