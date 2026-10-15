"""Named output paths and relocation preserve frozen prediction checkpoints."""

import fcntl
import json
from types import SimpleNamespace

import pytest

from data.processing import paths
from predict.harnesses.branches import manifests, migrate_manifests as migration


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "LEGACY_ROOT", tmp_path / "data/legacy")
    monkeypatch.setattr(manifests, "ROOT", tmp_path)
    monkeypatch.setattr(migration, "ROOT", tmp_path)
    monkeypatch.setattr(migration.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
    old = tmp_path / "outputs/old"
    batch = old / "semantic-top50/ames/batch"
    query = batch / "runs/batch_idx00000"
    query.mkdir(parents=True)
    manifests.write_batch(batch / "manifest.json", {
        "batch_id": "batch", "organization": "flat", "prompt_version": "final_full_flat",
        "benchmark": "gold", "task": "ames", "evaluation_subset": "test",
        "model": "fixed-model", "experiment_mode": "full_flat",
    }, records=[(0, {"benchmark_row_id": "query-1"})])
    for name in ("retrieval.json", "request.json", "final_reasoning_output.json", "stage_resume.json"):
        (query / name).write_text(json.dumps({"frozen": name}))
    output = tmp_path / "outputs/predictions"
    return old, batch, output


def test_named_output_and_prior_paths(tmp_path):
    root = manifests.prediction_run_root(tmp_path, "flat", "final_full_flat", "semantic-top50")
    batch = manifests.prediction_batch_directory(root, "gold", "ames", "test")
    assert batch == tmp_path / "flat/final_full_flat/semantic-top50/gold_v1/ames/test/default"
    for bad in ("", "../escape", "a/b", "two words"):
        with pytest.raises(ValueError):
            manifests.prediction_run_root(tmp_path, "flat", "final_full_flat", bad)
    prior = manifests.prediction_batch_directory(root, "tdc", "ames", "valid", "query-prior")
    manifests.write_batch(prior / "manifest.json", {"experiment_mode": "none"})
    assert manifests.query_prior_batch(root, "ames", "valid", "tdc") == prior


def test_relocation_preserves_bytes_identity_old_locators_and_is_idempotent(saved_run, tmp_path):
    old, batch, output = saved_run
    before = {str(path.relative_to(batch)): path.read_bytes() for path in batch.rglob("*") if path.is_file()}
    uid = manifests.read_record(batch / "manifest.json")["run_uid"]
    plan = migration.plan_output_relocations([old], output, {})
    assert len(plan["moves"]) == 1 and not plan["manual_classification"]
    target = output / "flat/final_full_flat/semantic-top50/gold_v1/ames/test/batch"
    assert plan["moves"][0]["destination"] == str(target)
    report, registry = tmp_path / "report.json", output / "relocations.json"
    migration.apply_output_relocations(plan, report, registry)
    assert not batch.exists()
    assert {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()} == before
    assert manifests.read_record(batch / "manifest.json")["run_uid"] == uid
    assert manifests.run_directory(batch, 0) == target / "runs/batch_idx00000"
    assert manifests.canonical_location(batch / "runs/batch_idx00000/request.json").read_bytes() == before["runs/batch_idx00000/request.json"]
    migration.apply_output_relocations(plan, report, registry)
    assert plan["moves"][0]["status"] == "moved"


def test_unknown_identity_is_left_for_manual_classification(saved_run):
    old, batch, output = saved_run
    header = batch / "manifest.json"
    document = json.loads(header.read_text())
    document["settings"].pop("prompt_version")
    header.write_text(json.dumps(document))
    plan = migration.plan_output_relocations([old], output, {})
    assert not plan["moves"]
    assert plan["manual_classification"][0]["missing"] == ["prompt"]
    assert batch.exists() and not output.exists()
    plan = migration.plan_output_relocations([old], output, {str(batch): {"prompt": "reviewed-prompt"}})
    assert len(plan["moves"]) == 1


def test_active_launcher_is_deferred_without_changes(saved_run, tmp_path):
    old, batch, output = saved_run
    lock = batch.parent / "launcher.lock"
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = migration.plan_output_relocations([old], output, {})
        migration.apply_output_relocations(plan, tmp_path / "report.json", output / "relocations.json")
    assert plan["moves"][0]["status"] == "active_deferred"
    assert batch.exists() and not output.exists()


def test_changed_header_blocks_a_stale_relocation_plan(saved_run, tmp_path):
    old, batch, output = saved_run
    plan = migration.plan_output_relocations([old], output, {})
    header = batch / "manifest.json"
    header.write_text(header.read_text() + "\n")
    with pytest.raises(ValueError, match="metadata changed"):
        migration.apply_output_relocations(plan, tmp_path / "report.json", output / "relocations.json")
    assert batch.exists() and not output.exists()


def test_nested_recovery_moves_before_its_parent(saved_run, tmp_path):
    old, parent, output = saved_run
    recovery = parent / 'recovery'
    (recovery / 'runs/recovered_idx00000').mkdir(parents=True)
    manifests.write_batch(recovery / 'manifest.json', {
        'batch_id': 'recovered', 'organization': 'flat', 'prompt_version': 'final_full_flat',
        'benchmark': 'gold', 'task': 'ames', 'evaluation_subset': 'test',
    }, records=[(0, {'benchmark_row_id': 'query-1'})])
    plan = migration.plan_output_relocations([old], output, {})
    assert [move['source'] for move in plan['moves']] == [str(recovery), str(parent)]
    migration.apply_output_relocations(plan, tmp_path/'report.json', output/'relocations.json')
    assert manifests.read_record(recovery/'manifest.json')['batch_id'] == 'recovered'
    assert manifests.read_record(parent/'manifest.json')['batch_id'] == 'batch'


def test_unknown_nested_run_prevents_parent_relocation(saved_run):
    old, parent, output = saved_run
    child = parent/'unknown/runs/query'
    child.mkdir(parents=True)
    (child.parent.parent/'manifest.json').write_text('{}')
    plan = migration.plan_output_relocations([old], output, {})
    assert not plan['moves']
    assert any('unclassified nested' in item.get('reason', '') for item in plan['manual_classification'])
    assert parent.exists()


def test_named_matrix_keeps_equal_profiles_distinct_across_tasks(tmp_path):
    from predict.harnesses.branches import matrix
    names = []
    for task in ('ames', 'dili'):
        args = matrix._selection_args(task, tmp_path, limit=0)
        args.batch_id, args.run_name, args.query_prior = 'default', 'shared-study', 'none'
        command = matrix._batch_command(args, tmp_path/'retrieval', base_url='http://localhost:1', api_key_env='EMPTY')
        names.append(command.experiment_name)
        assert '--run-name' in command.command
    assert names == ['ames__valid__default', 'dili__valid__default']


def test_existing_run_directory_names_disambiguate_old_repetitions(saved_run):
    old, batch, output = saved_run
    document = json.loads((batch/'manifest.json').read_text())
    for name in ('first-run', 'repeat-run'):
        repeated = old/'study/with_query_prior'/name/'ames/batch'
        (repeated/'runs').mkdir(parents=True)
        (repeated/'manifest.json').write_text(json.dumps(document))
    plan = migration.plan_output_relocations([old], output, {})
    runs = {move['identity']['run_name'] for move in plan['moves'] if move['kind']=='batch'}
    assert runs == {'semantic-top50', 'study__first-run', 'study__repeat-run'}
    assert not plan['manual_classification']


def test_named_study_catalog_reads_only_its_profile_metrics(tmp_path):
    import csv
    from predict.traces.catalog import _refresh_results_catalog
    campaign = tmp_path / 'flat/final_full_flat/study'
    for profile, accuracy in [('top10', 0.7), ('top50', 0.8)]:
        index = campaign / 'profiles' / profile / 'run.json'
        index.parent.mkdir(parents=True)
        index.write_text(json.dumps({
            'schema_version': 'organized_study_run.v1', 'tasks': ['ames'],
            'method': profile, 'metric_source_dir': str(campaign), 'selection_profile': profile,
        }))
        metrics = campaign / 'tdc/ames/valid' / profile / 'metrics.json'
        metrics.parent.mkdir(parents=True)
        metrics.write_text(json.dumps({'accuracy': accuracy, 'n_successful': 2}))
    _refresh_results_catalog(tmp_path)
    with (tmp_path / 'results_ledger.tsv').open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    assert {(row['method'], row['accuracy']) for row in rows} == {('top10', '0.7'), ('top50', '0.8')}
    assert len(rows) == 2


def test_query_prior_resume_rejects_changed_settings_before_preflight(tmp_path, monkeypatch):
    from predict.harnesses.branches import six_task_query_priors as prior
    root = manifests.prediction_run_root(tmp_path, 'query-prior', prior.FINAL_FULL_FLAT_PROMPT_VERSION, 'priors')
    root.mkdir(parents=True)
    execution = root / 'execution.json'
    execution.write_text(json.dumps({
        'model': 'frozen-model', 'prompt_version': prior.FINAL_FULL_FLAT_PROMPT_VERSION,
        'subset': 'valid', 'benchmark': 'gold', 'max_tokens': 20480, 'native_task_priors': False,
    }))
    before = execution.read_bytes()
    monkeypatch.setattr(prior, 'load_provider_pool_config', lambda _: SimpleNamespace(providers=[SimpleNamespace(model='changed-model')]))
    monkeypatch.setattr(prior, 'select_healthy_providers', lambda *a: pytest.fail('must reject before preflight'))
    with pytest.raises(ValueError, match='incompatible settings'):
        prior.main(['--output-root', str(tmp_path), '--run-name', 'priors', '--subset', 'valid', '--resume'])
    assert execution.read_bytes() == before
