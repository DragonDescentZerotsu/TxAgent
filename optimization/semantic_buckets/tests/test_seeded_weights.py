import gzip
import json
import re
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from optimization.semantic_buckets import bucket_scoring as scoring, inputs


@pytest.fixture
def scoring_inputs(tmp_path):
    ids = [f"bucket-{i}" for i in range(5)]
    world = tmp_path / "world.parquet"
    pd.DataFrame(dict(level="L2", semantic_bucket_id=ids)).to_parquet(world)
    cards = tmp_path / "cards.json.gz"
    payload = dict(identity=dict(source_components=[dict(source_id="source", canonical_dimensions={"endpoint": ["oral exposure"]})]),
                   sample_records=[dict(source_id="source", endpoint="oral exposure")])
    with gzip.open(cards, "wt") as handle:
        json.dump(dict.fromkeys(ids, payload), handle)
    seed = tmp_path / "seed.parquet"
    pd.DataFrame(dict(task="bioavailability_ma", level="L2", semantic_bucket_id=ids,
                      weight=[.1, .9, .3, .8, .2])).to_parquet(seed)
    provider = inputs.REPOSITORY_ROOT / "predict/api_client/providers/gpt6_luna_openrouter_standard_medium_64.json"
    settings = dict(passes=2, chain_mode="auto", chains=1, chain_counts={}, max_chains=1,
                    overlap=1, advance_after=1, pass1_batch_size=2, pass2_batch_size=2,
                    pass1_fanout=1, pass1_average=1, pass2_fanout=3, pass2_average=3,
                    pass1_provider_config=str(provider), pass2_provider_config=str(provider),
                    seed_weights=str(seed), reasoning_effort="medium")
    return {"bioavailability_ma": dict(target="oral bioavailability", world=str(world), payloads=str(cards))}, settings, seed


def test_seeded_pass_scores_once_uses_new_anchors_and_resumes(tmp_path, monkeypatch, scoring_inputs):
    from predict.api_client import pool
    from optimization.semantic_buckets import publication
    from optimization.selection import pool as selection_pool
    from predict.utils.json import sha256_file

    descriptor, settings, seed = scoring_inputs
    release_root = tmp_path / "releases/bioavailability_ma/fixture"
    release_root.mkdir(parents=True)
    evidence = tmp_path / "repository/data/evidence_libraries/bioavailability_ma"
    (evidence / "fixture").mkdir(parents=True)
    (evidence / "CURRENT").write_text("fixture\n")
    monkeypatch.setattr(inputs, "RELEASE_ROOT", tmp_path / "releases")
    monkeypatch.setattr(inputs, "REPOSITORY_ROOT", tmp_path / "repository")
    monkeypatch.setattr(publication, "RELEASE_ROOT", tmp_path / "releases")
    monkeypatch.setattr(publication, "REPOSITORY_ROOT", tmp_path / "repository")
    buckets = pd.read_parquet(seed)
    mapping = buckets[["level", "semantic_bucket_id"]].assign(atom_id=[f"atom-{i}" for i in range(5)])
    mapping.to_parquet(release_root / "map.parquet", index=False)
    records = mapping.assign(source_row_uid=[f"uid-{i}" for i in range(5)])
    records.to_parquet(release_root / "records.parquet", index=False)
    ledger = [dict(source_row_uid=r.source_row_uid, level=r.level, assignment_status="assigned",
                   own_level_semantic_bucket_id=r.semantic_bucket_id,
                   semantic_memberships=[dict(level=r.level, semantic_bucket_id=r.semantic_bucket_id)])
              for r in records.itertuples(index=False)]
    ledger.append(dict(source_row_uid="direct", level="L1", assignment_status="unassigned",
                       own_level_semantic_bucket_id=None, semantic_memberships=None))
    pd.DataFrame(ledger).to_parquet(release_root / "ledger.parquet", index=False)
    buckets.to_parquet(release_root / "weights.parquet", index=False)
    (release_root / "map.json").write_text("{}")
    selected = dict(semantic_map="map.parquet", semantic_map_manifest="map.json",
                    record_semantic_bucket_map="records.parquet", central_ledger="ledger.parquet",
                    semantic_bucket_weights="weights.parquet")
    base = dict(schema_version=publication.SCHEMA_VERSION, task="bioavailability_ma", status="complete_reviewed",
                evidence_library_version="fixture", selected=selected,
                files={role: dict(path=p, sha256=sha256_file(release_root / p)) for role, p in selected.items()})
    (release_root / "manifest.json").write_text(json.dumps(base))
    card_manifest = tmp_path / "inputs.json"
    card_manifest.write_text(json.dumps(dict(source_bindings=dict(
        release_manifest=dict(path=str(release_root / "manifest.json"), sha256=sha256_file(release_root / "manifest.json")),
        world=dict(path=str(seed), sha256=sha256_file(seed))))))
    descriptor["bioavailability_ma"]["input_manifest"] = str(card_manifest)
    root = tmp_path / "run"
    manifest = scoring.prepare(descriptor, root, settings)
    assert manifest["completed_passes"] == 1 and manifest["completion_requests_made"] == 0
    schedule = pd.read_parquet(root / "pass2_schedule.parquet")
    assert json.loads(schedule.iloc[0].candidate_bucket_ids_json) == ["bucket-1", "bucket-3"]
    preview = (root / "prompt_review/bioavailability_ma_L2_initial.txt").read_text()
    assert "Candidate 1\n  Previous-pass weight (revisable): 0.90" in preview
    assert "Candidate 2\n  Previous-pass weight (revisable): 0.80" in preview
    calls = []
    fail_later_windows = True

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            assert kwargs["reasoning_effort"] == "medium"
            prompt = kwargs["messages"][-1]["content"]
            calls.append(prompt)
            if fail_later_windows and "Anchor 1" in prompt:
                raise ConnectionError("interrupted after the first complete window")
            candidates = re.findall(r"^Candidate \d+$", prompt, re.M)
            response = dict(scores=[dict(candidate=c, weight=.6, rationale="New Luna assessment") for c in candidates])
            return SimpleNamespace(model="openai/gpt-6-luna", usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response), reasoning_content=""))])

        async def close(self):
            pass

    monkeypatch.setattr(pool, "preflight_provider_models", lambda *a: {})
    monkeypatch.setattr(pool, "async_completion_pool", lambda *a, **k: Client())
    with pytest.raises(RuntimeError):
        scoring.run(root)
    fail_later_windows = False
    result = scoring.run(root)
    assert result["completed_passes"] == 2
    expected_calls = 3 * len(schedule) + 9
    assert len(calls) == expected_calls
    assert sum("Anchor 1" not in p for p in calls) == 3
    assert any("locked weight 0.6" in p and "New Luna assessment" in p for p in calls)
    assert not any("locked weight 0.9" in p for p in calls)
    scoring.collect(root)
    final = pd.read_parquet(root / "semantic_bucket_weights.parquet")
    assert len(final) == 5 and final.weight.eq(.6).all()
    scoring.run(root)
    assert len(calls) == expected_calls
    with sqlite3.connect(root / "requests.sqlite3") as db:
        assert not db.execute("SELECT 1 FROM requests WHERE phase LIKE 'pass1/%'").fetchone()
        previous = pd.read_parquet(seed).set_index("semantic_bucket_id").weight.to_dict()
        for prompt, validation in db.execute("SELECT prompt,validation_json FROM requests"):
            for index, bucket in enumerate(json.loads(validation)["candidate_bucket_ids"], 1):
                assert f"Candidate {index}\n  Previous-pass weight (revisable): {previous[bucket]:.2f}" in prompt
    old_map_mtime = (release_root / "map.parquet").stat().st_mtime_ns
    stage = tmp_path / "publication"
    published = publication.refresh_weights("bioavailability_ma", root, stage, "refreshed", activate=True)
    assert published["semantic_weight_resolution"] == inputs.MEMBERSHIP_WEIGHT_RESOLUTION
    assert "record_relevance_rankings" not in published["selected"]
    assert (release_root / "map.parquet").stat().st_mtime_ns == old_map_mtime
    assert (evidence / "CURRENT").read_text() == "fixture\n"
    rows, pins = selection_pool._semantic_ledger("bioavailability_ma")
    assert len(rows) == 5 and all(r["semantic_weight"] == .6 for r in rows.values())
    assert all("projection" not in p for p in pins)
    from predict.retrieval.assay_reranking.build_context_l2_semantic_cache import _semantic_index
    index, _ = _semantic_index("bioavailability_ma")
    assert len(index) == 5 and {rank for _, rank in index.values()} == {1, 2, 3, 4, 5}
    with pytest.raises(ValueError, match="do not accept projection extensions"):
        selection_pool._semantic_ledger("bioavailability_ma", extension=tmp_path / "obsolete.json")
    seed.write_bytes(seed.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source changed"):
        scoring.run(root)
    active_bytes = (release_root / "manifest.json").read_bytes()
    (release_root / "map.parquet").write_bytes(b"conflicting payload")
    with pytest.raises(FileExistsError, match="overwrite"):
        publication.select(stage, sha256_file(stage / "manifest.json"))
    assert (release_root / "manifest.json").read_bytes() == active_bytes
    inputs._cached_record_weights.cache_clear()
    with pytest.raises(ValueError, match="membership input changed"):
        inputs.load_reviewed_record_weights("bioavailability_ma")


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign", "nan", "infinite", "range"])
def test_invalid_seed_fails_before_preparation(tmp_path, scoring_inputs, change):
    descriptor, settings, seed = scoring_inputs
    frame = pd.read_parquet(seed)
    if change == "missing": frame = frame.iloc[:-1]
    elif change == "duplicate": frame = pd.concat([frame, frame.iloc[:1]])
    elif change == "foreign": frame.loc[0, "semantic_bucket_id"] = "foreign"
    else: frame.loc[0, "weight"] = {"nan": float("nan"), "infinite": float("inf"), "range": 2}[change]
    frame.to_parquet(seed)
    with pytest.raises(ValueError, match="seed weights"):
        scoring.prepare(descriptor, tmp_path / "run", settings)
    assert not (tmp_path / "run").exists()


def test_unseeded_preparation_still_starts_with_independent_batches(tmp_path, scoring_inputs):
    descriptor, settings, _ = scoring_inputs
    settings.pop("seed_weights")
    result = scoring.prepare(descriptor, tmp_path / "run", settings)
    assert result["completed_passes"] == 0 and "baseline_source" not in result
    assert not (tmp_path / "run/pass2_schedule.parquet").exists()


def test_full_membership_lookup_excludes_l1_and_rejects_missing_weights(tmp_path):
    ledger, weights = tmp_path / "ledger.parquet", tmp_path / "weights.parquet"
    memberships = [dict(level="L2", semantic_bucket_id="a"), dict(level="L3", semantic_bucket_id="b")]
    pd.DataFrame([
        dict(source_row_uid="uid-a", level="L2", assignment_status="assigned", own_level_semantic_bucket_id="a", semantic_memberships=memberships + memberships),
        dict(source_row_uid="uid-b", level="L3", assignment_status="assigned", own_level_semantic_bucket_id="b", semantic_memberships=memberships),
        dict(source_row_uid="direct", level="L1", assignment_status="unassigned", own_level_semantic_bucket_id=None, semantic_memberships=None),
    ]).to_parquet(ledger)
    table = pd.DataFrame([dict(level="L2", semantic_bucket_id="a", weight=.2), dict(level="L3", semantic_bucket_id="b", weight=.8)])
    table.to_parquet(weights)
    frame = inputs.reviewed_membership_weights(ledger, weights)
    assert set(frame.source_row_uid) == {"uid-a", "uid-b"}
    assert frame.weight.eq(.5).all() and frame.component_count.eq(2).all()
    table.iloc[:1].to_parquet(weights)
    with pytest.raises(ValueError, match="memberships lack weights"):
        inputs.reviewed_membership_weights(ledger, weights)
