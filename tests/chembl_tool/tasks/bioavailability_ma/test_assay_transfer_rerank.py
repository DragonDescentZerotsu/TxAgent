import hashlib
import json
import math
import sys
import time
from types import SimpleNamespace

import pytest

from tools.chembl_tool.tasks.bioavailability_ma.reranking.assay_transfer_rerank import (
    ASSAY_TRANSFER_MODEL_REVISION,
    CATALOG_SCHEMA_VERSION,
    SCORING_CONTRACT_VERSION,
    V6_5_NO_QUERY_EXTRA_DETAILS_QUERY_CONTEXT_POLICY,
    V6_5_NO_QUERY_EXTRA_DETAILS_TEMPLATE_PROFILE,
    V6_5_QUERY_CONTEXT_POLICY,
    V6_5_TEMPLATE_PROFILE,
    AssayTransferCacheMiss,
    AssayTransferCachedReranker,
    AssayTransferPromptRenderer,
    AssayTransferScoreCache,
    PromptScore,
    PromptTask,
    build_prompt_task,
    probability_from_log_likelihoods,
    template_bundle_hash,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.precompute_assay_transfer_rerank import (
    _check_free_vram,
    resolve_model_snapshot,
    run_spawned_workers,
)
from tools.chembl_tool.tasks.bioavailability_ma.reranking.build_assay_transfer_rerank_catalog import (
    _index_example_record,
)


def _record(record_id, smiles="CCN", concept="Fa", value="10"):
    return {
        "record_type": "assay_record",
        "record_id": record_id,
        "source_id": "test",
        "original_smiles": smiles,
        "canonical_smiles": smiles,
        "assay_concept": concept,
        "canonical_endpoint_key": "q2.intestinal_absorption.fraction_absorbed.percent",
        "endpoint_family": "intestinal_absorption",
        "endpoint_subtype": "fraction_absorbed",
        "unit_basis": "percent",
        "metric_type": "bounded_percentage",
        "threshold_display": "within 10 percentage points / at least 30 percentage points apart",
        "value": float(value),
        "value_display": value,
        "measurement_label": "fraction absorbed",
        "template_id": "Fa_intern_mcqa_v3",
        "template_context": {"study_or_assay_system": "human oral study"},
        "source_provenance": {},
    }


def _write_catalog(path, records, *, profile="legacy_v3"):
    metadata = {
        "record_type": "catalog_metadata",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": "test.catalog.v1",
        "template_hash": template_bundle_hash(profile=profile),
        "template_profile": profile,
        "n_records": len(records),
    }
    path.write_text(
        "\n".join(json.dumps(row) for row in [metadata, *records]) + "\n",
        encoding="utf-8",
    )


def _candidate(molecule_id, smiles, similarity):
    return {
        "rank": 1,
        "structural_rank": 1,
        "molecule_chembl_id": molecule_id,
        "canonical_smiles": smiles,
        "similarity": similarity,
        "evidence_rows": [{"value": molecule_id}],
    }


def test_vendored_fa_template_renders_training_contract_exactly():
    prompt = AssayTransferPromptRenderer().render(_record("r1"), "CCO")

    assert prompt.startswith("You are comparing two molecules for an intestinal absorption (Fa) assay.\n")
    assert "- <SMILES>CCN</SMILES>\n- known value: 10 percent" in prompt
    assert "- assay system: human oral study" in prompt
    assert prompt.endswith("(B) not transfer\n\nAnswer:\n")


def test_v6_5_template_is_exact_and_copies_context_while_hiding_query_value():
    record = _record("r1", smiles="CCN", value="10")
    record["template_context"].update(
        {
            "species_or_population": "rat",
            "dose": "5 mg/kg",
            "measured_process": "fraction absorbed",
            "extra_details": "fed state",
        }
    )
    renderer = AssayTransferPromptRenderer(profile=V6_5_TEMPLATE_PROFILE)
    prompt = renderer.render(record, "C(C)O")
    retrieval, query = prompt.split("Target query record (value hidden)", 1)

    assert renderer.template_hash == "e30f995988db7214cae4b170a2c36f3a5fd61b6bee6188b39ce54377fa67f5bd"
    assert renderer.query_context_policy == V6_5_QUERY_CONTEXT_POLICY
    assert "- <SMILES>CCN</SMILES>" in retrieval
    assert "- known value: 10 percent" in retrieval
    assert "- <SMILES>CCO</SMILES>" in query
    assert "- endpoint: q2.intestinal_absorption.fraction_absorbed.percent" in retrieval
    assert "- endpoint: q2.intestinal_absorption.fraction_absorbed.percent" in query
    for line in (
        "- species or population: rat",
        "- dose: 5 mg/kg",
        "- measured process: fraction absorbed",
        "- extra details: fed state",
    ):
        assert line in retrieval
        assert line in query
    assert "known value" not in query
    assert "scalar_value" not in query
    assert "value_display" not in query


def test_v6_5_no_query_extra_details_preserves_retrieval_context_only():
    record = _record("r1", smiles="CCN", value="10")
    record["template_context"].update(
        {
            "species_or_population": "rat",
            "dose": "5 mg/kg",
            "extra_details": "fed state",
        }
    )
    renderer = AssayTransferPromptRenderer(
        profile=V6_5_NO_QUERY_EXTRA_DETAILS_TEMPLATE_PROFILE
    )
    prompt = renderer.render(record, "C(C)O")
    retrieval, query = prompt.split("Target query record (value hidden)", 1)

    assert renderer.template_hash == template_bundle_hash(
        profile=V6_5_TEMPLATE_PROFILE
    )
    assert (
        renderer.query_context_policy
        == V6_5_NO_QUERY_EXTRA_DETAILS_QUERY_CONTEXT_POLICY
    )
    assert "- endpoint: q2.intestinal_absorption.fraction_absorbed.percent" in retrieval
    assert "- endpoint: q2.intestinal_absorption.fraction_absorbed.percent" in query
    assert "- species or population: rat" in retrieval
    assert "- species or population: rat" in query
    assert "- dose: 5 mg/kg" in retrieval
    assert "- dose: 5 mg/kg" in query
    assert "- extra details: fed state" in retrieval
    assert "- extra details: not specified" in query
    assert "known value" not in query


def test_corrected_profile_has_distinct_prompt_cache_and_provenance():
    common = {
        "record": _record("r1"),
        "query_smiles": "CCO",
        "group_id": "Fa.absorption_solubility_permeability",
        "molecule_id": "A",
        "model": "model",
        "model_revision": "a" * 40,
        "catalog_version": "catalog",
    }
    common["record"]["template_context"]["extra_details"] = "fed state"
    historical_renderer = AssayTransferPromptRenderer(
        profile=V6_5_TEMPLATE_PROFILE
    )
    corrected_renderer = AssayTransferPromptRenderer(
        profile=V6_5_NO_QUERY_EXTRA_DETAILS_TEMPLATE_PROFILE
    )
    historical = build_prompt_task(renderer=historical_renderer, **common)
    corrected = build_prompt_task(renderer=corrected_renderer, **common)

    assert historical_renderer.template_hash == corrected_renderer.template_hash
    assert (
        historical_renderer.query_context_policy
        != corrected_renderer.query_context_policy
    )
    assert historical.prompt_hash != corrected.prompt_hash
    assert historical.cache_key != corrected.cache_key


def test_v6_5_profile_changes_template_hash_prompt_hash_and_cache_key():
    common = {
        "record": _record("r1"),
        "query_smiles": "CCO",
        "group_id": "Fa.absorption_solubility_permeability",
        "molecule_id": "A",
        "model": "model",
        "model_revision": "a" * 40,
        "catalog_version": "catalog",
    }
    legacy = build_prompt_task(renderer=AssayTransferPromptRenderer(), **common)
    v6_5 = build_prompt_task(
        renderer=AssayTransferPromptRenderer(profile=V6_5_TEMPLATE_PROFILE), **common
    )

    assert legacy.template_hash != v6_5.template_hash
    assert legacy.prompt_hash != v6_5.prompt_hash
    assert legacy.cache_key != v6_5.cache_key


def test_catalog_hash_is_validated_against_selected_template_profile(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    cache = tmp_path / "scores.sqlite3"
    _write_catalog(catalog, [_record("r1")], profile=V6_5_TEMPLATE_PROFILE)
    AssayTransferScoreCache(cache, mode="read_write").close()

    with pytest.raises(ValueError, match="selected template profile"):
        AssayTransferCachedReranker(
            catalog_path=catalog,
            cache_path=cache,
            template_profile="legacy_v3",
        )
    reranker = AssayTransferCachedReranker(
        catalog_path=catalog,
        cache_path=cache,
        template_profile=V6_5_TEMPLATE_PROFILE,
        allow_missing=True,
    )
    reranker.cache.close()


def test_best_record_aggregation_and_deterministic_molecule_tie_breaking(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    cache_path = tmp_path / "scores.sqlite3"
    _write_catalog(catalog, [_record("a1", "CCN"), _record("a2", "CCN", value="20"), _record("b1", "CCC")])
    candidates = [_candidate("A", "CCN", 0.7), _candidate("B", "CCC", 0.9)]

    writer = AssayTransferCachedReranker(
        catalog_path=catalog,
        cache_path=cache_path,
        cache_mode="read_write",
        allow_missing=True,
    )
    tasks = writer.tasks_for_candidates(
        query_smiles="CCO", group_id="Fa.absorption_solubility_permeability", candidates=candidates
    )
    scores = {"a1": 0.2, "a2": 0.8, "b1": 0.8}
    flat_tasks = [task for values in tasks.values() for task in values]
    writer.cache.write_batch(
        flat_tasks,
        [
            PromptScore(task.cache_key, math.log(scores[task.record_id]), math.log(1 - scores[task.record_id]), scores[task.record_id])
            for task in flat_tasks
        ],
    )
    writer.cache.close()

    reranker = AssayTransferCachedReranker(
        catalog_path=catalog, cache_path=cache_path, cache_mode="read_only"
    )
    reranked = reranker.rerank(
        query_smiles="CCO", group_id="Fa.absorption_solubility_permeability", candidates=candidates
    )
    reranker.cache.close()

    assert [row["molecule_chembl_id"] for row in reranked] == ["B", "A"]
    row_a = next(row for row in reranked if row["molecule_chembl_id"] == "A")
    assert row_a["transfer_winning_record_id"] == "a2"
    assert all(row["transfer_scored_record_count"] >= 1 for row in reranked)
    # The winning-record payload is attached and matches the winning catalog record.
    winning = row_a["transfer_winning_record"]
    assert winning["record_id"] == "a2"
    assert winning["value_display"] == "20"  # a2 was written with value="20"
    assert winning["original_smiles"] == "CCN"
    assert winning["context"] == {"study_or_assay_system": "human oral study"}
    # canonical_smiles is carried in the payload (policy decides visibility, not rerank).
    assert winning["canonical_smiles"] == "CCN"


def test_rerank_records_selects_top_records_across_molecules(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    cache_path = tmp_path / "scores.sqlite3"
    # molecule A has two records; B has one.
    _write_catalog(catalog, [_record("a1", "CCN"), _record("a2", "CCN", value="20"), _record("b1", "CCC")])
    candidates = [_candidate("A", "CCN", 0.7), _candidate("B", "CCC", 0.9)]

    writer = AssayTransferCachedReranker(
        catalog_path=catalog, cache_path=cache_path, cache_mode="read_write", allow_missing=True
    )
    tasks = writer.tasks_for_candidates(
        query_smiles="CCO", group_id="Fa.absorption_solubility_permeability", candidates=candidates
    )
    # Both of molecule A's records outscore B's single record.
    scores = {"a1": 0.9, "a2": 0.8, "b1": 0.5}
    flat = [t for v in tasks.values() for t in v]
    writer.cache.write_batch(
        flat,
        [PromptScore(t.cache_key, math.log(scores[t.record_id]), math.log(1 - scores[t.record_id]), scores[t.record_id]) for t in flat],
    )
    writer.cache.close()

    reranker = AssayTransferCachedReranker(catalog_path=catalog, cache_path=cache_path, cache_mode="read_only")
    records = reranker.rerank_records(
        query_smiles="CCO", group_id="Fa.absorption_solubility_permeability", candidates=candidates
    )
    reranker.cache.close()

    # one item per scored record, ranked by transfer score (records, not molecules)
    assert [r["transfer_winning_record_id"] for r in records] == ["a1", "a2", "b1"]
    # the top-2 records are BOTH from molecule A
    top2 = records[:2]
    assert [r["molecule_chembl_id"] for r in top2] == ["A", "A"]
    assert {r["transfer_winning_record_id"] for r in top2} == {"a1", "a2"}
    # each carries its own record as the winning record, count 1
    assert all(r["transfer_scored_record_count"] == 1 for r in records)
    assert records[0]["transfer_winning_record"]["record_id"] == "a1"
    assert records[1]["transfer_winning_record"]["record_id"] == "a2"


def test_missing_cache_fails_without_partial_tanimoto_fallback(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    cache_path = tmp_path / "scores.sqlite3"
    _write_catalog(catalog, [_record("a1")])
    AssayTransferScoreCache(cache_path, mode="read_write").close()
    reranker = AssayTransferCachedReranker(
        catalog_path=catalog, cache_path=cache_path, cache_mode="read_only"
    )

    with pytest.raises(AssayTransferCacheMiss, match="run precompute"):
        reranker.rerank(
            query_smiles="CCO",
            group_id="Fa.absorption_solubility_permeability",
            candidates=[_candidate("A", "CCN", 0.9)],
        )
    reranker.cache.close()


def test_cache_key_changes_with_model_revision_and_probability_uses_both_full_likelihoods(tmp_path):
    assert probability_from_log_likelihoods(-2.0, -3.0) == pytest.approx(0.7310585786)
    assert probability_from_log_likelihoods(-20.0, -1.0) < 1e-8
    renderer = AssayTransferPromptRenderer()
    common = {
        "renderer": renderer,
        "record": _record("r1"),
        "query_smiles": "CCO",
        "group_id": "Fa.absorption_solubility_permeability",
        "molecule_id": "A",
        "model": "model",
        "catalog_version": "catalog",
    }
    first = build_prompt_task(model_revision="a" * 40, **common)
    second = build_prompt_task(model_revision="b" * 40, **common)
    different_catalog = build_prompt_task(
        model_revision="a" * 40, **{**common, "catalog_version": "other-catalog"}
    )
    assert first.prompt_hash == second.prompt_hash
    assert first.cache_key != second.cache_key
    assert first.cache_key == different_catalog.cache_key
    with pytest.raises(ValueError, match="immutable"):
        AssayTransferCachedReranker(
            catalog_path=tmp_path / "missing", cache_path=tmp_path / "missing", model_revision="main"
        )


def test_direct_bioavailability_generic_starling_schema_is_normalized():
    record = _index_example_record(
        "oral_bioavailability",
        {"molecule_chembl_id": "M1", "canonical_smiles": "CCO"},
        {"group_id": "Observed.direct_oral_bioavailability"},
        {
            "source_id": "observed_direct_bioavailability",
            "source_index": 7,
            "source_record_id": "ext_7",
            "endpoint_type": "bioavailability",
            "reported_value": "42.5",
            "reported_units": "%",
            "context": {
                "species": "rat",
                "statistic_type": "arithmetic_mean",
                "study_context": "single oral dose",
            },
        },
        0,
    )

    assert record is not None
    assert record["value"] == 42.5
    assert record["unit_basis"] == "percent"
    assert record["assay_concept"] == "oral_bioavailability"
    assert record["template_context"]["species_or_population"] == "rat"
    assert record["template_context"]["study_context"] == "single oral dose"


def test_flat_catalog_uses_exact_manifest_record_join(tmp_path):
    catalog = tmp_path / "catalog.jsonl"
    cache_path = tmp_path / "scores.sqlite3"
    manifest = tmp_path / "condition.jsonl"
    records = [_record("four-source", "CCN"), _record("fifth-source", "CCN")]
    catalog_version = "flat.test.catalog"
    metadata = {
        "record_type": "catalog_metadata",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": catalog_version,
        "source_mode": "flat_five_source_condition_union",
        "template_hash": template_bundle_hash(),
        "n_records": 2,
    }
    catalog.write_text(
        "\n".join(json.dumps(row) for row in [metadata, *records]) + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "record_type": "manifest_metadata",
                    "condition_id": "validation__four_source__operational",
                    "catalog_version": catalog_version,
                },
                {
                    "record_type": "candidate_group",
                    "query_smiles": "CCO",
                    "group_id": "Fa.absorption_solubility_permeability",
                    "candidates": [
                        {"molecule_id": "M1", "record_ids": ["four-source"]}
                    ],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    AssayTransferScoreCache(cache_path, mode="read_write").close()

    with pytest.raises(ValueError, match="candidate-manifest"):
        AssayTransferCachedReranker(
            catalog_path=catalog,
            cache_path=cache_path,
            cache_mode="read_only",
        )
    reranker = AssayTransferCachedReranker(
        catalog_path=catalog,
        cache_path=cache_path,
        cache_mode="read_only",
        candidate_manifest_path=manifest,
        allow_missing=True,
    )
    tasks = reranker.tasks_for_candidates(
        query_smiles="CCO",
        group_id="Fa.absorption_solubility_permeability",
        candidates=[_candidate("M1", "CCN", 0.8)],
    )
    reranker.cache.close()

    assert [task.record_id for task in tasks["M1"]] == ["four-source"]


def test_model_snapshot_resolution_records_sha_and_supports_offline_cache(monkeypatch):
    calls = []

    class FakeApi:
        def model_info(self, model, revision):
            calls.append(("info", model, revision))
            return SimpleNamespace(sha="a" * 40)

    def fake_snapshot_download(**kwargs):
        calls.append(("snapshot", kwargs))
        return "/local/snapshot"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_snapshot_download),
    )
    path, revision = resolve_model_snapshot(
        "model", "main", force_download=True, local_files_only=False
    )
    assert (path, revision) == ("/local/snapshot", "a" * 40)
    assert calls[-1][1]["revision"] == "a" * 40
    assert calls[-1][1]["force_download"] is True

    calls.clear()
    path, revision = resolve_model_snapshot(
        "model", "b" * 40, force_download=False, local_files_only=True
    )
    assert (path, revision) == ("/local/snapshot", "b" * 40)
    assert all(call[0] != "info" for call in calls)
    assert calls[-1][1]["local_files_only"] is True


def test_vram_preflight_uses_cuda_local_devices_and_supports_mig(monkeypatch):
    cuda = SimpleNamespace(
        device_count=lambda: 4,
        mem_get_info=lambda device: ((90 - device) * 1024**3, 90 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    _check_free_vram([0, 1, 2, 3], 64.0)
    with pytest.raises(RuntimeError, match="not visible"):
        _check_free_vram([4], 64.0)
    with pytest.raises(RuntimeError, match="at least 90.00 GiB"):
        _check_free_vram([1], 90.0)


def _tasks(count):
    output = []
    for index in range(count):
        cache_key = hashlib.sha256(f"task-{index}".encode()).hexdigest()
        output.append(
            PromptTask(
                cache_key=cache_key,
                prompt_hash=hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
                prompt=f"prompt {index}",
                query_smiles="CCO",
                group_id="Fa.absorption_solubility_permeability",
                molecule_id=f"m{index}",
                record_id=f"r{index}",
                model="model",
                model_revision=ASSAY_TRANSFER_MODEL_REVISION,
                scoring_contract_version=SCORING_CONTRACT_VERSION,
                template_hash="templates",
                catalog_version="catalog",
            )
        )
    return output


def _fake_worker(device, snapshot_path, dtype, work_queue, result_queue):
    result_queue.put({"type": "ready", "device": device})
    while True:
        item = work_queue.get()
        if item is None:
            return
        time.sleep(0.01 if device == 0 else 0.02)
        scores = []
        for task in item["tasks"]:
            probability = (int(task["cache_key"][-2:], 16) + 1) / 257.0
            scores.append(
                {
                    "cache_key": task["cache_key"],
                    "logp_transfer": math.log(probability),
                    "logp_not_transfer": math.log(1 - probability),
                    "transfer_probability": probability,
                }
            )
        result_queue.put(
            {
                "type": "result",
                "device": device,
                "batch_id": item["batch_id"],
                "scores": scores,
                "peak_vram_gib": 1.0 + device,
            }
        )


def _failing_worker(device, snapshot_path, dtype, work_queue, result_queue):
    result_queue.put({"type": "ready", "device": device})
    completed = 0
    while True:
        item = work_queue.get()
        if item is None:
            return
        if completed:
            result_queue.put(
                {
                    "type": "error",
                    "device": device,
                    "error_type": "OutOfMemoryError",
                    "message": "CUDA out of memory",
                    "cuda_oom": True,
                    "traceback": "fake traceback",
                }
            )
            return
        task = item["tasks"][0]
        result_queue.put(
            {
                "type": "result",
                "device": device,
                "batch_id": item["batch_id"],
                "scores": [{"cache_key": task["cache_key"], "logp_transfer": -1.0, "logp_not_transfer": -2.0, "transfer_probability": 0.7}],
                "peak_vram_gib": 1.0,
            }
        )
        completed += 1


def test_single_and_dual_worker_scheduling_produce_identical_cache_rows(tmp_path):
    tasks = _tasks(7)
    observed = []
    for devices in ([0], [0, 1]):
        cache = AssayTransferScoreCache(tmp_path / f"cache-{len(devices)}.sqlite3", mode="read_write")
        summary = run_spawned_workers(
            tasks,
            cache=cache,
            snapshot_path="unused",
            devices=list(devices),
            batch_size=2,
            dtype="bfloat16",
            worker_target=_fake_worker,
        )
        observed.append(cache.lookup(tasks))
        cache.close()
        assert summary["n_scored"] == len(tasks)
        assert summary["n_cache_rows_committed"] == len(tasks)
    assert observed[0] == observed[1]


def test_worker_failure_preserves_incrementally_committed_cache_for_resume(tmp_path):
    tasks = _tasks(3)
    cache = AssayTransferScoreCache(tmp_path / "cache.sqlite3", mode="read_write")
    with pytest.raises(RuntimeError, match="CUDA OOM"):
        run_spawned_workers(
            tasks,
            cache=cache,
            snapshot_path="unused",
            devices=[0],
            batch_size=1,
            dtype="bfloat16",
            worker_target=_failing_worker,
        )
    assert len(cache.lookup(tasks)) == 1
    cache.close()
