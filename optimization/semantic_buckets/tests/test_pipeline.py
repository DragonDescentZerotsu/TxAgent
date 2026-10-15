from optimization.semantic_buckets import storage as semantic_storage
import json
import gzip
from types import SimpleNamespace
import pandas as pd
import pytest
from optimization.semantic_buckets import (
    build as construction,
    bucket_construction as grouping,
    inputs,
    bucket_refinement as refinement,
)
from optimization.semantic_buckets import execution, bucket_cards, publication
from predict.utils.json import sha256_file

"Offline construction/refinement checks using complete prepared request flows."


def test_construction_policies_are_isolated_and_resume_without_repeating_requests(
    tmp_path, monkeypatch
):
    source = "source"
    columns = (
        "canonical_endpoint_concept",
        "canonical_unit_text",
        "canonical_assay_context",
    )
    root = tmp_path / "build"
    config = construction.Workflow(
        "ames",
        tmp_path / "release",
        root,
        "fixture-model",
        ({"base_url": "http://fixture/v1", "max_inflight": 2},),
        None,
        {source: columns},
        {source: (columns[0], columns[2])},
        {source: columns[0]},
        tmp_path / "index.json",
        context_limit=1000000,
    )
    config.input.parent.mkdir(parents=True)
    config.canonical_records.parent.mkdir(parents=True)
    rows = [
        dict(
            canonical_record_id=f"r{i}",
            source_row_uid=f"u{i}",
            source_id=source,
            pair_bucket_key=json.dumps([source, "endpoint", "%", assay]),
            level="L2",
            node_key='{"canonical_endpoint_concept":"endpoint"}',
            relevance_bucket='{"canonical_endpoint_concept":"endpoint"}',
        )
        for i, assay in enumerate(("assay-a", "assay-b"))
    ]
    pd.DataFrame(rows).to_parquet(config.input, index=False)
    pd.DataFrame(
        [
            dict(
                canonical_record_id=f"r{i}",
                source_row_uid=f"u{i}",
                canonical_endpoint_concept="endpoint",
                canonical_assay_context=assay,
            )
            for i, assay in enumerate(("assay-a", "assay-b"))
        ]
    ).to_parquet(config.canonical_records)
    (config.input.parent / "manifest.json").write_text(
        json.dumps({"counts": {"records": 2, "atoms": 2}})
    )
    policy = construction.policy(config)
    other = construction.policy(config)
    other["TASK_NAME"] = "other"
    assert policy["TASK_NAME"] == "AMES"
    atoms, _ = inputs._load_atoms(policy=policy)
    assert len(atoms) == 2 and len(set(atoms.atom_id)) == 2
    counter = lambda text, *, policy: len(text) // 4
    monkeypatch.setattr(grouping, "_token_count", counter)
    monkeypatch.setattr(execution, "_token_count", counter)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        result = {
            "decision": "stop",
            "rationale": "Keep this intrinsic scientific family.",
        }
        return SimpleNamespace(
            model="fixture-model",
            usage=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(result), reasoning_content=""
                    )
                )
            ],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion)),
        snapshot=lambda: {},
    )
    monkeypatch.setattr(
        grouping, "_build_completion_pool", lambda *a, **k: (client, [], 2)
    )
    review = grouping.prepare_prompt_review(config.review, policy=policy)
    assert not calls
    approved = semantic_storage._file_sha256(config.review / "manifest.json")
    result = grouping.run_semantic(
        config.run,
        review_manifest_path=config.review / "manifest.json",
        approved_review_sha256=approved,
        parallelism=2,
        policy=policy,
    )
    assert result["status"] == "awaiting_agentic_review"
    saved = len(calls)
    monkeypatch.setattr(
        grouping,
        "_build_completion_pool",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("completed run must not construct a client")
        ),
    )
    again = grouping.run_semantic(
        config.run,
        review_manifest_path=config.review / "manifest.json",
        approved_review_sha256=approved,
        parallelism=2,
        policy=policy,
    )
    assert again["status"] == result["status"] and len(calls) == saved


def test_refinement_prepares_fresh_audits_and_detects_tampering(tmp_path, monkeypatch):
    dummy = tmp_path / "pinned.txt"
    dummy.write_text("frozen")
    families = pd.DataFrame(
        [
            dict(level="L2", source_id="s", semantic_bucket_id=b, record_count=count)
            for b, count in (("a", 2), ("b", 3), ("large", 100))
        ]
    )
    release = tmp_path / "release.json"
    release.write_text(json.dumps(dict(task="dili", evidence_library_version="frozen-v1")))
    mapping = families[["level", "source_id", "semantic_bucket_id"]].assign(
        atom_id=["atom-a", "atom-b", "atom-large"]
    )
    sources = {"fixture": dummy, "release_manifest": release}
    for name, table in (
        ("semantic_map", mapping),
        ("record_map", mapping.assign(source_row_uid=["u-a", "u-b", "u-large"])),
        ("weights", families[["level", "semantic_bucket_id"]].assign(weight=0.5)),
    ):
        sources[name] = tmp_path / f"{name}.parquet"
        table.to_parquet(sources[name], index=False)
    monkeypatch.setattr(refinement, "_sources", lambda task: sources.copy())
    groups = {
        row.semantic_bucket_id: {
            "level": "L2",
            "source_id": "s",
            "record_count": row.record_count,
            "endpoints": ["endpoint"],
            "values": ["activity"],
        }
        for row in families.itertuples()
    }
    monkeypatch.setattr(
        refinement, "_families", lambda sources: (pd.DataFrame(), families)
    )
    monkeypatch.setattr(refinement, "_samples", lambda sources: {})
    monkeypatch.setattr(refinement, "_cards", lambda *args: ({}, groups))

    def fit(task, kind, batch, *args):
        return [
            {
                "request_id": kind + ":" + ":".join(batch),
                "task": task,
                "pass_number": 1 if kind == "merge" else 2,
                "scope": {},
                "items": batch,
                "prompt": kind,
            }
        ]

    monkeypatch.setattr(refinement, "_fit_batches", fit)
    run = tmp_path / "run"
    manifest = refinement.prepare("dili", run)
    assert manifest["audit_requests"] == 1
    header, db = refinement._load(run)
    assert (
        db.execute("SELECT count(*) FROM requests WHERE pass_number=0").fetchone()[0]
        == 1
    )
    for row in db.execute("SELECT * FROM requests").fetchall():
        response = (
            {"merge_sets": []}
            if row["pass_number"] == 1
            else {
                "reviews": [
                    {
                        "id": 0,
                        "risk": "low",
                        "rationale": "One coherent activity.",
                        "suspected_activities": [],
                    }
                ]
            }
        )
        db.execute(
            "UPDATE requests SET status='complete',response_json=? WHERE request_id=?",
            (json.dumps(response), row["request_id"]),
        )
    db.commit()
    db.close()
    header["state"] = "complete_pass1"
    (run / "snapshot.json").write_text(json.dumps(header))
    assert refinement.run(run, 1)["state"] == "complete_pass1"
    monkeypatch.setattr(refinement, "_fit2", lambda *args: [])
    refinement.prepare_pass2(run)
    assert refinement.run(run, 2)["state"] == "complete_pass2"
    assert refinement.collect(run)["state"] == "complete"
    snapshot = json.loads((run / "snapshot.json").read_text())
    assert "release" not in snapshot
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps(dict(
        task="dili", reviewer="fixture-reviewer", merges=[], splits=[],
        review_manifest_sha256=sha256_file(run / "review_manifest.json"),
    )))
    candidate = tmp_path / "candidate"
    result = publication.materialize(run, decisions, candidate)
    assert result["evidence_release"] == "frozen-v1"
    assert set(pd.read_parquet(candidate / "record_semantic_bucket_map.parquet").source_row_uid) == {
        "u-a", "u-b", "u-large"
    }
    release.write_text(json.dumps(dict(task="dili")))
    blocked = tmp_path / "blocked-candidate"
    with pytest.raises(ValueError, match="snapshot input changed"):
        publication.materialize(run, decisions, blocked)
    assert not blocked.exists()
    snapshot["inputs"]["release_manifest"]["sha256"] = sha256_file(release)
    (run / "snapshot.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="matching evidence version"):
        publication.materialize(run, decisions, blocked)
    assert not blocked.exists()
    release.write_text(json.dumps(dict(task="dili", evidence_library_version="frozen-v1")))
    snapshot["inputs"]["release_manifest"]["sha256"] = sha256_file(release)
    (run / "snapshot.json").write_text(json.dumps(snapshot))
    audits = pd.read_csv(run / "large_family_reviews.tsv", sep="\t")
    assert list(audits.semantic_bucket_id) == ["large"]
    db = semantic_storage.review_database(run / "requests.sqlite3")
    db.execute("UPDATE requests SET prompt='changed'")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="request bytes changed"):
        refinement._load(run)


def test_scoring_inputs_preserve_deterministic_atom_cards(tmp_path):
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    atoms = pd.DataFrame(
        [
            dict(
                atom_id=f"a{i}",
                level="L2",
                source_id="fixed_mutation",
                values_json=json.dumps({"endpoint": "mutation", "assay": str(i)}),
            )
            for i in range(8)
        ]
    )
    atoms.to_parquet(semantic / "input_atoms.parquet")
    mapping = atoms[["atom_id", "level"]].assign(semantic_bucket_id="bucket")
    mapping.to_parquet(semantic / "semantic_bucket_map.parquet")
    with gzip.open(semantic / "sample_cards.json.gz", "wt") as handle:
        json.dump(
            {
                f"a{i}": {"canonical_record_id": f"r{i}", "endpoint": f"sample-{i}"}
                for i in range(8)
            },
            handle,
        )
    world = tmp_path / "world.parquet"
    mapping[["level", "semantic_bucket_id"]].drop_duplicates().to_parquet(world)
    output = tmp_path / "inputs"
    result = bucket_cards.prepare_scoring_inputs("ames", world, semantic, output)
    assert result["tasks"]["ames"]["payloads"] == "bucket_prompt_payloads.json.gz"
    with gzip.open(output / "bucket_prompt_payloads.json.gz", "rt") as handle:
        payload = json.load(handle)["bucket"]
    expected = bucket_cards._sample_atoms("bucket", atoms)
    assert [sample["endpoint"] for sample in payload["sample_records"]] == [
        "sample-" + atom[1:] for atom in expected
    ]
    assert len(expected) == 5
    assert all(
        ("canonical_record_id" not in sample for sample in payload["sample_records"])
    )
    mapping.loc[0, "semantic_bucket_id"] = "other"
    mapping.to_parquet(semantic / "semantic_bucket_map.parquet")
    world_frame = pd.read_parquet(world)
    world_frame.loc[0, "semantic_bucket_id"] = "missing"
    world_frame.to_parquet(world)
    with pytest.raises(ValueError, match="does not cover"):
        bucket_cards.prepare_scoring_inputs("ames", world, semantic, tmp_path / "bad")
