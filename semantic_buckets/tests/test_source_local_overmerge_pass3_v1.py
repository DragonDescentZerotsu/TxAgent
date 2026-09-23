import json
from types import SimpleNamespace

from semantic_buckets import source_local_overmerge_pass3_v1 as run
from semantic_buckets import source_local_overmerge_reconcile_v1 as previous


def _group(index: int, *, parent: str = "p") -> dict:
    return {"task": "dili", "parent_id": parent, "level": "L2", "source_id": "source",
            "group_id": f"g{index}", "label": f"assay {index}",
            "values": [f"assay value {index}"], "rationale": "prior decision",
            "pass1_group_ids": [f"p{index}"]}


def test_pass3_rebatches_one_hundred_and_splits_models_evenly() -> None:
    schedule = run._schedule([_group(index) for index in range(200)])
    assert len(schedule) == 2
    assert sorted(len(row["items"]) for row in schedule) == [100, 100]
    assert {row["scope"]["provider_family"] for row in schedule} == {"luna", "deepseek"}
    assert all(row["scope"]["parent_id"] == "p" for row in schedule)


def test_reroute_preserves_completed_requests_and_changes_only_pending(tmp_path) -> None:
    schedule = run._schedule([_group(index) for index in range(200)])
    connection = previous._database(tmp_path / "requests.sqlite3")
    previous._insert(connection, schedule)
    finished = schedule[1]
    connection.execute("UPDATE requests SET status='complete' WHERE request_id=?",
                       (finished["request_id"],))
    connection.commit()
    connection.close()
    updated = run._rerouted_schedule(tmp_path, schedule)
    assert updated[1] == finished
    assert updated[0] == schedule[0]
    connection = previous._database(tmp_path / "requests.sqlite3")
    connection.execute("UPDATE requests SET status='pending' WHERE request_id=?",
                       (finished["request_id"],))
    connection.commit()
    connection.close()
    updated = run._rerouted_schedule(tmp_path, schedule)
    assert updated[1]["scope"]["provider_family"] == "luna"
    assert updated[1]["request_id"] != finished["request_id"]


def test_pass3_decisions_preserve_unmerged_and_recover_prior_rationale(tmp_path, monkeypatch) -> None:
    first, second = _group(0), _group(1)
    first.update(values=["same A", "same B"], rationale="Unmerged singleton")
    source = tmp_path / "predecessor"
    source.mkdir()
    (source / "pass1_groups.json").write_text(json.dumps([
        {"group_id": "p0", "values": first["values"], "rationale": "Pass-1 synonym review"}]))
    monkeypatch.setattr(run, "SOURCE", source)
    retained = run._successor_groups([first, second], [], {})
    assert retained[0]["rationale"] == "Pass-1 synonym review"
    assert retained[1] == second
    connection = previous._database(tmp_path / "ledger.sqlite3")
    schedule = [{"request_id": "request", "task": "dili", "pass_number": 3,
                 "scope": {"parent_id": "p", "provider_family": "luna"},
                 "items": ["g0", "g1"], "prompt": "review"}]
    previous._insert(connection, schedule)
    connection.execute("UPDATE requests SET status='complete', response_json=?", (
        json.dumps({"merge_sets": [{"member_ids": [0, 1], "label": "shared",
                                  "rationale": "same question"}]}),))
    connection.commit()
    merges, replacements = run._decisions(tmp_path, [first, second], connection, schedule)
    assert len(merges) == 1
    assert set(replacements) == {"g0", "g1"}
    successor = run._successor_groups([first, second], merges, replacements)
    assert len(successor) == 1
    assert successor[0]["values"] == ["assay value 1", "same A", "same B"]


def test_both_model_calls_enforce_high_reasoning() -> None:
    for family, model, limit_name in (("luna", run.LUNA_MODEL, "max_completion_tokens"),
                                      ("deepseek", run.DEEPSEEK_MODEL, "max_tokens")):
        sent = []
        response = SimpleNamespace(model=model, id="generation", usage=None,
                                   choices=[SimpleNamespace(message=SimpleNamespace(
                                       content='{"merge_sets":[]}', reasoning_content=None,
                                       reasoning=None))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (sent.append(kwargs), response)[1])))
        row = {"scope_json": json.dumps({"provider_family": family}),
               "items_json": '["a","b"]', "prompt": "review", "attempts": 0,
               "receipts_json": "[]"}
        result = run._call(client, row)
        assert result["status"] == "complete"
        assert sent[0]["reasoning_effort"] == "high"
        assert sent[0][limit_name] == run.MAX_TOKENS
