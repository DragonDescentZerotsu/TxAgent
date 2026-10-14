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
        "schema_version": "joseph_final_arm.v1", "task": "dili",
        "source_batches": [{"copied_manifest": "source_manifests/batch_0.json"}]}))
    convert_batch(batch, apply=True, source_uids={str(tmp_path / "original"): "source-uid"})
    record = read_record(batch / "manifest.json")
    assert record["max_tokens"] == 123
    assert record["model"] == "m"
    assert record["source_run_uids"] == ["source-uid"]
