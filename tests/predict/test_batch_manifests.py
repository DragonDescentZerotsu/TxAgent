"""Shared records preserve scientific identity through migration and relocation."""
import json
from pathlib import Path
import shutil

import pytest

from predict.harnesses.branches.manifests import (
    read_record, read_run, require_query_ids, require_artifact_versions, write_batch,
)
from predict.harnesses.branches.migrate_manifests import convert_batch


def test_migration_preserves_payloads_state_overrides_and_relocation(tmp_path):
    source = tmp_path / 'valid.jsonl'
    rows = [{'benchmark_row_id': 'q0'}, {'benchmark_row_id': 'q1'}]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    batch = tmp_path / 'batch'
    batch.mkdir()
    (batch / 'manifest.json').write_text(json.dumps({
        'batch_id': 'batch', 'input_jsonl': str(source), 'model': 'model-v1',
        'input_sha256': 'obsolete', 'records_per_level': 10,
    }))
    for i in range(2):
        run = batch / 'runs' / f'batch_idx{i:05d}'
        run.mkdir(parents=True)
        (run / 'manifest.json').write_text(json.dumps({
            'query_index': i, 'model': 'model-v1', 'temperature': i,
            'expected_group_ids': ['a'], 'n_groups_with_neighbors': 1,
            'paths': {'retrieval': str(run / 'retrieval.json')},
            'stage_pool': {'events': [{'status': 'done'}]},
        }))
        (run / 'retrieval.json').write_bytes(b'FROZEN EVIDENCE\n')
    before = {p.relative_to(batch): p.read_bytes() for p in batch.rglob('retrieval.json')}
    assert convert_batch(batch)['status'] == 'planned'
    assert len(list(batch.glob('runs/*/manifest.json'))) == 2
    assert convert_batch(batch, apply=True)['deleted_queries'] == 2
    assert not list(batch.glob('runs/*/manifest.json'))
    manifest = json.loads((batch / 'manifest.json').read_text())
    assert manifest['queries'] == {'0': 'q0', '1': 'q1'}
    assert manifest['settings']['records_per_level'] == 10
    assert 'input_sha256' not in manifest['settings']
    assert manifest['query_overrides'] == {'0': {'temperature': 0}, '1': {'temperature': 1}}
    assert {p.relative_to(batch): p.read_bytes() for p in batch.rglob('retrieval.json')} == before
    moved = tmp_path / 'moved'
    shutil.move(batch, moved)
    record = read_record(moved / 'manifest.json')
    require_query_ids(record, rows, [0, 1])
    query = read_run(moved / 'runs/batch_idx00001')
    assert query['temperature'] == 1
    assert query['expected_group_ids'] == ['a']
    assert query['stage_pool']['events'][0]['status'] == 'done'
    assert convert_batch(moved, apply=True)['status'] == 'already_converted'
    with pytest.raises(ValueError, match='identity'):
        require_query_ids(record, [{'benchmark_row_id': 'changed'}, rows[1]], [0])
    with pytest.raises(ValueError, match='name'):
        require_artifact_versions(record, {'input_jsonl': tmp_path / 'other.jsonl'})


def test_unknown_uids_and_interrupted_query_removal(tmp_path):
    batch = tmp_path / 'unknown'
    run = batch / 'runs/unknown_idx00000'
    run.mkdir(parents=True)
    old = {'query_index': 0, 'prompt_version': 'legacy-v1'}
    (run / 'manifest.json').write_text(json.dumps(old))
    result = convert_batch(batch, apply=True)
    assert result['missing_query_uids'] == 1
    shared = read_record(batch / 'manifest.json')
    assert shared['queries'] == {'0': ''}
    (run / 'manifest.json').write_text(json.dumps(old))
    convert_batch(batch, apply=True)
    assert not (run / 'manifest.json').exists()
    assert read_record(batch / 'manifest.json')['run_uid'] == shared['run_uid']


def test_new_writer_uses_one_manifest(tmp_path):
    path = tmp_path / 'manifest.json'
    write_batch(path, {'batch_id': 'x', 'model': 'm', 'max_tokens': 17, 'paths': {},
                       'selection_audit': {'selection_policy': 'manual', 'inputs': {'/old/path': 'hash'}}},
                records=[(0, {'benchmark_row_id': 'q'})])
    read = read_record(path)
    assert read['model'] == 'm'
    assert read['max_tokens'] == 17
    assert read['selection_audit'] == {'selection_policy': 'manual'}
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize("query_version", ["v1", "v2"])
def test_query_override_keeps_local_snapshot_only_for_same_identity(tmp_path, query_version):
    from copy import deepcopy
    from predict.harnesses.branches.manifests import resolve_artifact

    batch = tmp_path / "relocated"
    local_input = batch / "inputs/input_jsonl.jsonl"
    local_input.parent.mkdir(parents=True)
    local_input.write_text('{"benchmark_row_id": "q0"}\n')
    identity = {"name": "test", "kind": "benchmark", "task": "Ames",
                "version": "v1", "partition": "scaffold"}
    query_identity = dict(identity, version=query_version)
    record = {"queries": {"0": "q0"},
              "artifacts": {"input_jsonl": dict(identity, local_format=".jsonl")},
              "query_overrides": {"0": {"artifacts": {"input_jsonl": query_identity}}}}
    before = deepcopy(record)
    query = read_run(batch / "runs/00000", batch=record)
    expected = local_input if query_version == "v1" else resolve_artifact(
        {"artifacts": {"input_jsonl": query_identity}}, "input_jsonl", batch)
    assert Path(query["input_jsonl"]) == expected
    assert resolve_artifact(query, "input_jsonl", batch) == expected
    assert ("local_format" in query["artifacts"]["input_jsonl"]) == (query_version == "v1")
    assert record == before


def test_resume_rejects_changed_scientific_settings_and_query_membership(tmp_path):
    path = tmp_path / "manifest.json"
    records = [(0, {"benchmark_row_id": "q"})]
    write_batch(path, {"model": "m1", "max_tokens": 10}, records=records)
    with pytest.raises(ValueError, match="scientific settings"):
        write_batch(path, {"model": "m2", "max_tokens": 10}, records=records)
    with pytest.raises(ValueError, match="query membership"):
        write_batch(path, {"model": "m1"}, records=[(0, {"benchmark_row_id": "other"})])
    assert read_record(path)["model"] == "m1"


def test_final_arm_reads_copied_source_settings(tmp_path):
    batch = tmp_path / "final/arm"
    source = batch / "source_manifests/batch_0.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"batch_id": "source", "max_tokens": 123,
                                  "model": "m", "paths": {"batch_dir": str(tmp_path / "original")}}))
    run = batch / "runs/00000"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"query_index": 0, "benchmark_row_id": "q"}))
    (batch / "manifest.json").write_text(json.dumps({
        "schema_version": "evidence_final_arm.v1", "task": "dili",
        "source_batches": [{"copied_manifest": "source_manifests/batch_0.json"}]}))
    convert_batch(batch, apply=True, source_uids={str(tmp_path / "original"): "source-uid"})
    record = read_record(batch / "manifest.json")
    assert record["max_tokens"] == 123
    assert record["model"] == "m"
    assert record["source_run_uids"] == ["source-uid"]


@pytest.mark.parametrize('field', ['benchmark', 'role', 'method', 'variant', 'level', 'operation'])
@pytest.mark.parametrize('benchmark', ['gold_v1', 'tdc'])
def test_artifact_checks_include_all_scientific_identity_fields(tmp_path, field, benchmark):
    from predict.harnesses.branches.manifests import artifact_name
    path = tmp_path / f'data/caches/assay_reranking/active/flat_v5/{benchmark}/ames/l2plus/assay_transfer/v27/parent100/scaffold/valid/L7/VERSION.json'
    identity = artifact_name(path)
    assert identity['level'] == 'L7'
    identity[field] = 'incompatible'
    with pytest.raises(ValueError, match=field):
        require_artifact_versions({'artifacts': {'retrieval': identity}}, {'retrieval': path})


def test_canonical_tdc_cache_identity_and_relocated_resume(tmp_path):
    from predict.harnesses.branches.manifests import artifact_name
    relative = Path('data/caches/assay_reranking/active/flat_v5/tdc/ames/l1_cards/RELEASE_INDEX.json')
    old = tmp_path / 'old' / relative
    moved = tmp_path / 'moved' / relative
    identity = artifact_name(old)
    assert identity == dict(name='RELEASE_INDEX', kind='retrieval', benchmark='tdc', task='ames', role='l1_cards')
    require_artifact_versions({'artifacts': {'retrieval': identity}}, {'retrieval': moved})
    batch = tmp_path / 'batch/manifest.json'
    records = [(0, {'benchmark_row_id': 'q'})]
    write_batch(batch, {'assay_transfer_cache': old}, records=records)
    original = batch.read_text()
    write_batch(batch, {'assay_transfer_cache': moved}, records=records)
    assert batch.read_text() == original
    changed = tmp_path / 'moved' / relative.parent.parent / 'l1/RELEASE_INDEX.json'
    with pytest.raises(ValueError, match='role'):
        write_batch(batch, {'assay_transfer_cache': changed}, records=records)
    assert batch.read_text() == original


def test_named_cache_snapshot_preserves_identity_and_resolves(tmp_path):
    from predict.retrieval.assay_reranking.artifact_bundle import DEFAULT_CACHE_SET
    from predict.harnesses.branches.manifests import artifact_name, resolve_artifact
    source = tmp_path / "cache.json"
    source.write_bytes(DEFAULT_CACHE_SET.read_bytes())
    target = tmp_path / "batch/manifest.json"
    result = write_batch(target, {"assay_transfer_cache": str(source), "task": "dili"})
    identity = result["artifacts"]["assay_transfer_cache"]
    assert identity["name"] == "scaffold_disjoint_parent100"
    assert identity["version"] == json.loads(source.read_text())["release"]
    pinned = resolve_artifact(result, "assay_transfer_cache", target.parent)
    assert pinned.read_bytes() == source.read_bytes()
    source.unlink()
    assert artifact_name(pinned)["version"] == identity["version"]


def test_prediction_resume_restores_pinned_composition(tmp_path, monkeypatch):
    from predict.harnesses.branches import flat
    from predict.retrieval.assay_reranking.artifact_bundle import DEFAULT_CACHE_SET, cache_task
    original = json.loads(DEFAULT_CACHE_SET.read_text())
    original.update(name="pinned_fixture", release="v9")
    index = tmp_path / "RELEASE_INDEX.json"
    evidence = tmp_path / "bbb_martins/v1/VERSION.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}")
    index.write_text(json.dumps({"evidence": {"manifest": str(evidence)},
        "splits": {"valid": {"levels": {"L1": {}, "L2": {}}}}}))
    original["benchmarks"]["gold_v1"]["bbb_martins"]["retrieval"] = {
        "L1": str(index), "later": str(index)}
    queries = tmp_path / "valid_molecule_condition_labels.jsonl"
    queries.write_text(json.dumps({"benchmark_row_id": "fixture", "drug": "CCO"}) + "\n")
    monkeypatch.setattr(flat, "split_path", lambda task, subset: queries)
    cache = tmp_path / "pinned.json"
    cache.write_text(json.dumps(original))
    observed = []
    def materialize(args):
        observed.append((cache_task(args.assay_transfer_cache, args.task, args.benchmark)["cache_set"], args.records_per_level))
        root = args.batch_root / args.batch_id / "cache_matched_retrieval"
        if not (root / "manifest.json").exists():
            row = json.loads(args.input_jsonl.read_text().splitlines()[0])
            write_batch(root / "manifest.json", {
                "organization": "flat", "task": args.task, "benchmark": args.benchmark,
                "prompt_version": args.prompt_version, "assay_transfer_cache": str(args.assay_transfer_cache),
                "records_per_level": args.records_per_level, "record_limits_by_level": args.record_limits_by_level,
                "max_level": args.max_level, "query_prior": "none",
            }, records=[(0, row)])
        return root, {"indices": [0]}
    monkeypatch.setattr(flat, "_materialize_cache_matched_retrievals", materialize)
    args = ["--task", "bbb_martins", "--batch-root", str(tmp_path), "--batch-id", "resume", "--query-prior", "none", "--prepare-only"]
    assert flat.run([*args, "--retrieval-cache", str(cache), "--records-per-level", "2"]) == 0
    cache.unlink()
    assert flat.run(args) == 0
    assert observed == [({"name": "pinned_fixture", "release": "v9"}, 2)] * 2


def _prepared_catalog_migration(tmp_path):
    from predict.utils.json import sha256_file
    source = tmp_path / 'queries.jsonl'
    query = dict(benchmark_row_id='q', drug='CCO')
    source.write_text(json.dumps(query) + '\n')
    old_manifest = tmp_path / 'old_level.json'
    new_manifest = tmp_path / 'new_level.json'
    header = dict(content_id='frozen-payload', database_sha256='frozen-database',
                  inputs=dict(query_sha256=sha256_file(source)))
    for path in (old_manifest, new_manifest):
        path.write_text(json.dumps(header))
    indexes = []
    for name, manifest in [('old', old_manifest), ('new', new_manifest)]:
        index = tmp_path / (name + '_index.json')
        index.write_text(json.dumps(dict(task_id='bbb_martins', status='complete', splits=dict(test=dict(levels={
            'L1': dict(manifest=manifest.name, manifest_sha256=sha256_file(manifest))})))))
        indexes.append(index)
    catalogs = []
    for name, scope, index in [('old', 'tdc_v2', indexes[0]), ('new', 'tdc', indexes[1])]:
        catalog = tmp_path / (name + '_catalog.json')
        catalog.write_text(json.dumps(dict(schema_version='retrieval_cache_set.v1', name='fixture', release='v1',
            benchmarks={scope: {'bbb_martins': dict(retrieval=dict(L1=str(index), later=str(index)))}})))
        catalogs.append(catalog)
    batch = tmp_path / 'batch'
    write_batch(batch / 'manifest.json', dict(task='bbb_martins', benchmark='tdc', benchmark_release='tdc_v2',
        evaluation_subset='test', input_jsonl=str(source), assay_transfer_cache=str(catalogs[0]),
        records_per_level=7), records=[(0, query)])
    return batch, catalogs[1], new_manifest


def test_prepared_catalog_migration_preserves_run_and_query_identity(tmp_path):
    from predict.harnesses.branches.migrate_manifests import migrate_cache_catalog
    batch, catalog, _ = _prepared_catalog_migration(tmp_path)
    original = (batch / 'manifest.json').read_bytes()
    snapshot = batch / 'inputs/assay_transfer_cache.json'
    old_snapshot = snapshot.read_bytes()
    assert migrate_cache_catalog(batch, catalog)['status'] == 'planned'
    assert (batch / 'manifest.json').read_bytes() == original
    assert migrate_cache_catalog(batch, catalog, apply=True)['status'] == 'converted'
    migrated = read_record(batch / 'manifest.json')
    previous = json.loads(original)
    assert migrated['run_uid'] == previous['run_uid']
    assert migrated['queries'] == previous['queries']
    assert migrated['benchmark_release'] == 'tdc' and migrated['records_per_level'] == 7
    assert snapshot.read_bytes() == catalog.read_bytes()
    assert (batch / 'inputs/historical_tdc/catalog.json').read_bytes() == old_snapshot
    assert migrate_cache_catalog(batch, catalog, apply=True)['status'] == 'already_converted'


@pytest.mark.parametrize('change', ['query', 'payload', 'unknown', 'completed'])
def test_prepared_catalog_migration_rejects_changed_or_unknown_bindings(tmp_path, change):
    from predict.harnesses.branches.migrate_manifests import migrate_cache_catalog
    from predict.utils.json import sha256_file
    batch, catalog, manifest = _prepared_catalog_migration(tmp_path)
    path = batch / 'manifest.json'
    document = json.loads(path.read_text())
    if change == 'unknown':
        document['queries']['0'] = ''
    elif change == 'query':
        document['queries']['0'] = 'different'
    elif change == 'payload':
        header = json.loads(manifest.read_text())
        header['database_sha256'] = 'different'
        manifest.write_text(json.dumps(header))
        index = tmp_path / 'new_index.json'
        entry = json.loads(index.read_text())
        entry['splits']['test']['levels']['L1']['manifest_sha256'] = sha256_file(manifest)
        index.write_text(json.dumps(entry))
    else:
        document['status'] = 'complete'
    path.write_text(json.dumps(document))
    before = path.read_bytes()
    if change == 'completed':
        assert migrate_cache_catalog(batch, catalog, apply=True)['status'] == 'completed_preserved'
    else:
        with pytest.raises(ValueError, match='identity|identities'):
            migrate_cache_catalog(batch, catalog, apply=True)
    assert path.read_bytes() == before


def test_catalog_migration_preserves_l1_membership_without_inventing_query_hash(tmp_path):
    from predict.harnesses.branches.migrate_manifests import migrate_cache_catalog
    from predict.utils.json import sha256_file
    batch, catalog, _ = _prepared_catalog_migration(tmp_path)
    for name in ('old', 'new'):
        manifest = tmp_path / (name + '_level.json')
        header = json.loads(manifest.read_text())
        header['inputs'].pop('query_sha256')
        manifest.write_text(json.dumps(header))
        index = tmp_path / (name + '_index.json')
        document = json.loads(index.read_text())
        document['splits']['test']['levels']['L1']['manifest_sha256'] = sha256_file(manifest)
        index.write_text(json.dumps(document))
    assert migrate_cache_catalog(batch, catalog, apply=True)['status'] == 'converted'
    assert json.loads((tmp_path / 'new_level.json').read_text())['inputs'] == {}


def test_catalog_migration_finishes_an_interrupted_parent_header(tmp_path):
    from predict.harnesses.branches.migrate_manifests import migrate_cache_catalog
    batch, catalog, _ = _prepared_catalog_migration(tmp_path)
    migrate_cache_catalog(batch, catalog, apply=True)
    child = json.loads((batch / 'manifest.json').read_text())
    parent = dict(schema_version='prediction_batch.v2', run_uid='parent-identity', queries=child['queries'],
                  settings=dict(cache_set=dict(name='fixture', release='v1'), benchmark_release='tdc_v2'))
    path = batch.parent / 'manifest.json'
    path.write_text(json.dumps(parent))
    assert migrate_cache_catalog(batch, catalog, apply=True)['status'] == 'converted'
    assert json.loads(path.read_text())['settings']['benchmark_release'] == 'tdc'
    assert json.loads(path.read_text())['run_uid'] == 'parent-identity'
    assert migrate_cache_catalog(batch, catalog, apply=True)['status'] == 'already_converted'
