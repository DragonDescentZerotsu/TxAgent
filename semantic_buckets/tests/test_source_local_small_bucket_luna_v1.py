import json

import pandas as pd
import pytest

from semantic_buckets import source_local_small_bucket_luna_v1 as run


def test_small_bucket_schedule_stays_endpoint_local_and_uses_aggressive_prompt(monkeypatch):
    atoms = pd.DataFrame([
        {"atom_id": "a", "level": "L2", "source_id": "s", "record_count": 2,
         "values_json": json.dumps({"canonical_assay_context": "ROS assay",
                                    "canonical_endpoint_concept": "oxidative stress"}),
         "canonical_endpoint": "oxidative stress"},
        {"atom_id": "b", "level": "L2", "source_id": "s", "record_count": 3,
         "values_json": json.dumps({"canonical_assay_context": "cellular ROS",
                                    "canonical_endpoint_concept": "oxidative stress"}),
         "canonical_endpoint": "oxidative stress"},
        {"atom_id": "c", "level": "L2", "source_id": "s", "record_count": 11,
         "values_json": json.dumps({"canonical_assay_context": "injury",
                                    "canonical_endpoint_concept": "injury"}),
         "canonical_endpoint": "injury"},
    ])
    mapping = pd.DataFrame([
        {"atom_id": atom, "level": "L2", "source_id": "s",
         "source_semantic_bucket_id": f"bucket_{atom}"} for atom in "abc"
    ])
    samples = {atom: {"canonical_assay_context": label,
                      "canonical_endpoint_concept": endpoint}
               for atom, label, endpoint in (("a", "ROS assay", "oxidative stress"),
                                             ("b", "cellular ROS", "oxidative stress"),
                                             ("c", "injury", "injury"))}
    monkeypatch.setattr(run, "_inputs", lambda task: (atoms, mapping, samples))
    schedule, counts = run._task_schedule("dili")
    assert len(schedule) == 1
    assert set(schedule[0]["items"]) == {"bucket_a", "bucket_b"}
    assert schedule[0]["scope"]["canonical_endpoint"] == "oxidative stress"
    assert schedule[0]["scope"]["provider_family"] == "luna"
    assert "Actively look across all supplied candidates" in schedule[0]["prompt"]
    assert counts["eligible_buckets"] == 2


def test_prepared_schedule_detects_changed_request_bytes(tmp_path, monkeypatch):
    pinned = tmp_path / "input.json"
    pinned.write_text("{}")
    monkeypatch.setattr(run, "_paths", lambda: {"input": pinned})
    monkeypatch.setattr(run, "_task_schedule", lambda task: ([{
        "request_id": f"request_{task}", "task": task, "pass_number": 5,
        "scope": {"level": "L2", "source_id": "source", "canonical_endpoint": "endpoint",
                  "provider_family": "luna"},
        "items": ["one", "two"], "prompt": "review"
    }], {"eligible_buckets": 2, "scheduled_buckets": 2, "isolated_buckets": 0,
          "source_buckets": 2}))
    output = tmp_path / "candidate"
    run.prepare(output)
    connection = run.ledger._database(output / "requests.sqlite3")
    connection.execute("UPDATE requests SET prompt='changed' WHERE request_id='request_dili'")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="frozen schedule changed"):
        run._verify(output)
