from concurrent.futures import ThreadPoolExecutor
from copy import copy
import fcntl
import json
import threading
from types import SimpleNamespace

import pytest

from tools.chembl_tool.paper_experiments import budgeted_suite as suite


class Client:
    def snapshot(self):
        return {'providers': []}


class Process:
    def __init__(self):
        self.finished = threading.Event()

    def poll(self):
        return 0 if self.finished.is_set() else None

    def terminate(self):
        self.finished.set()

    def wait(self):
        assert self.finished.wait(5)
        return 0


def grouped_start(start):
    def start_all(cells):
        for c in cells:
            start(c)
        processes = [c.process for c in cells]
        class Group:
            finished = processes[0].finished
            def poll(self):
                codes = [p.poll() for p in processes]
                return None if None in codes else max(codes)
            def terminate(self):
                for p in processes: p.terminate()
            def wait(self):
                for p in processes: p.wait()
                return self.poll()
        group = Group()
        for c in cells: c.process = group
    return start_all


def args(tmp_path, **overrides):
    return SimpleNamespace(output_root=str(tmp_path), parallelism=2, preparation_processes=2,
                           max_stage_requeues=1, retry_delay_s=.1, **overrides)


def cell(tmp_path, task, n):
    root = tmp_path/'runs'/'none'/task
    queries = [suite.budgeted.runtime.PreparedQuery(task, i, root/'runs'/task/f'query_{i:05d}')
               for i in range(n)]
    return suite.Cell('none', task, root, SimpleNamespace(), [], queries)


def write_ready(query):
    query.query_dir.mkdir(parents=True, exist_ok=True)
    (query.query_dir/'prepared.json').write_text('{}')


def setup_manifest(c):
    c.root.mkdir(parents=True, exist_ok=True)
    suite.write_json_atomic(c.root/'experiment_manifest.json', {'code':{}})
    suite.write_json_atomic(c.root/'preparation_ready.json', {'manifest_hash':suite.content_hash({'code':{}})})


def test_preparation_does_not_inherit_inference_race_width(tmp_path, monkeypatch):
    c = cell(tmp_path, 'a', 0)
    c.args = SimpleNamespace(retry_race_width=6)
    observed = []
    def register(options, **kwargs):
        observed.append((options.prepare_only, options.parallelism, options.retry_race_width))
    monkeypatch.setattr(suite.budgeted, 'run', register)
    monkeypatch.setattr(suite, 'ProcessPoolExecutor', lambda **kwargs: ThreadPoolExecutor(1))
    options = args(tmp_path, preparation_cache_root='')
    assert suite._prepare_cells(options, [c]) == 0
    assert observed == [(True, 1, 1)]
    assert c.args.retry_race_width == 6


def test_discovery_only_checks_created_query_directories(tmp_path, monkeypatch):
    from pathlib import Path
    c = cell(tmp_path, 'a', 1000)
    c.manifest = {'code': {}}
    write_ready(c.queries[42])
    checked, exists = [], Path.exists
    def track(path):
        if path.name == 'prepared.json':
            checked.append(path)
        return exists(path)
    monkeypatch.setattr(Path, 'exists', track)
    assert suite._discover(c) == [c.queries[42]]
    assert checked == [c.queries[42].query_dir/'prepared.json']
    assert suite._discover(c) == []
    assert len(checked) == 1


def test_streams_queries_before_preparation_finishes_across_tasks(tmp_path, monkeypatch):
    a, b = cell(tmp_path, 'a', 2), cell(tmp_path, 'b', 1)
    started, other_finished = threading.Event(), threading.Event()
    threads, order = [], []
    running = peak = 0
    guard = threading.Lock()
    monkeypatch.setattr(suite.budgeted, '_code_hashes', lambda: {})
    monkeypatch.setattr(suite.budgeted, 'summarize', lambda *a: None, raising=False)

    def start(c):
        c.process = Process()
        setup_manifest(c)
        process = c.process
        def prepare():
            if c is a:
                write_ready(c.queries[0])
                assert other_finished.wait(5)
                write_ready(c.queries[1])
            else:
                assert started.wait(5)
                write_ready(c.queries[0])
            process.finished.set()
        thread = threading.Thread(target=prepare)
        threads.append(thread)
        thread.start()

    def execute(c, query, client, attempt):
        nonlocal running, peak
        with guard:
            running += 1
            peak = max(peak, running)
        try:
            if c is a and query.index == 0:
                assert not c.process.finished.is_set()
                started.set()
                assert other_finished.wait(5)
            if c is b:
                order.append('b')
                other_finished.set()
            else:
                order.append('a')
            return {'status':'ok'}
        finally:
            with guard:
                running -= 1

    monkeypatch.setattr(suite, '_start_preparation', grouped_start(start))
    monkeypatch.setattr(suite, '_execute', execute)
    assert suite._schedule(args(tmp_path), [a,b], Client()) == 0
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert peak == 2
    assert order[0] == 'b'
    status = json.loads((tmp_path/'suite_status.json').read_text())
    assert status['status'] == 'complete' and status['n_succeeded_queries'] == 3


def test_retry_wait_releases_slot_and_only_failed_query_is_repeated(tmp_path, monkeypatch):
    c = cell(tmp_path, 'a', 2)
    monkeypatch.setattr(suite.budgeted, '_code_hashes', lambda: {})
    monkeypatch.setattr(suite.budgeted, 'summarize', lambda *a: None, raising=False)
    def start(c):
        c.process = Process()
        setup_manifest(c)
        for q in c.queries:
            write_ready(q)
        c.process.finished.set()
    calls = []
    def execute(c, q, client, attempt):
        calls.append((q.index, attempt))
        return {'status':'error' if q.index == 0 and attempt == 0 else 'ok'}
    monkeypatch.setattr(suite, '_start_preparation', grouped_start(start))
    monkeypatch.setattr(suite, '_execute', execute)
    options = args(tmp_path)
    options.parallelism = 1
    assert suite._schedule(options, [c], Client()) == 0
    assert calls == [(0,0), (1,0), (0,1)]


def test_stale_manifest_does_not_unlock_checkpoint_after_failed_preparation(tmp_path, monkeypatch):
    c = cell(tmp_path, 'a', 1)
    setup_manifest(c)
    write_ready(c.queries[0])
    (c.root/'preparation_ready.json').unlink()
    monkeypatch.setattr(suite.budgeted, '_code_hashes', lambda: {})
    assert suite._discover(c) == []
    suite.write_json_atomic(c.root/'experiment_manifest.json', {'code':{'changed':'hash'}})
    suite.write_json_atomic(c.root/'preparation_ready.json', {
        'manifest_hash':suite.content_hash({'code':{'changed':'hash'}})})
    with pytest.raises(ValueError, match='implementation changed'):
        suite._discover(c)
    setup_manifest(c)
    assert suite._discover(c) == c.queries
    assert suite._discover(c) == []


def test_failed_preparation_does_not_drop_other_cells(tmp_path, monkeypatch):
    a, b = cell(tmp_path, 'a', 1), cell(tmp_path, 'b', 1)
    monkeypatch.setattr(suite.budgeted, '_code_hashes', lambda: {})
    monkeypatch.setattr(suite.budgeted, 'summarize', lambda *a: None, raising=False)
    def start(c):
        c.process = Process()
        if c is b:
            setup_manifest(c)
            write_ready(c.queries[0])
        else:
            c.process.poll = lambda: 1
        c.process.finished.set()
    monkeypatch.setattr(suite, '_start_preparation', grouped_start(start))
    monkeypatch.setattr(suite, '_stop_preparation', lambda p: p.terminate())
    monkeypatch.setattr(suite, '_execute', lambda *a: {'status':'ok'})
    assert suite._schedule(args(tmp_path), [a,b], Client()) == 1
    assert a.preparation_error and b.results[0]['status'] == 'ok'
    assert json.loads((tmp_path/'suite_status.json').read_text())['status'] == 'needs_attention'


def test_query_retry_state_is_not_shared_between_cells(tmp_path, monkeypatch):
    c = cell(tmp_path, 'ames', 1)
    original = copy(vars(c.args))
    observed = []
    monkeypatch.setattr(suite.budgeted, '_run_query', lambda a,q,client: observed.append(a.retry_round))
    suite._execute(c, c.queries[0], Client(), 3)
    assert observed == [3]
    assert vars(c.args) == original


@pytest.mark.parametrize('saved_status,expected_round', [(None, 0), ('ok', 0), ('error', 1)])
def test_resumed_failure_enters_retry_race_immediately(tmp_path, monkeypatch, saved_status, expected_round):
    c = cell(tmp_path, 'ames', 1)
    query = c.queries[0]
    query.query_dir.mkdir(parents=True)
    if saved_status is not None:
        (query.query_dir / 'output.json').write_text(json.dumps({'status': saved_status}))
    observed = []
    monkeypatch.setattr(suite.budgeted, '_run_query', lambda a,q,client: observed.append(a.retry_round))
    suite._execute(c, query, Client(), 0)
    assert observed == [expected_round]
    assert not hasattr(c.args, 'retry_round')


def test_next_preparation_starts_while_previous_cell_inference_is_still_running(tmp_path, monkeypatch):
    a, b = cell(tmp_path, 'a', 1), cell(tmp_path, 'b', 1)
    other_finished = threading.Event()
    monkeypatch.setattr(suite.budgeted, '_code_hashes', lambda: {})
    monkeypatch.setattr(suite.budgeted, 'summarize', lambda *a: None)
    def start(c):
        c.process = Process()
        setup_manifest(c)
        write_ready(c.queries[0])
        c.process.finished.set()
    def execute(c, q, client, attempt):
        if c is a:
            assert other_finished.wait(5)
        else:
            other_finished.set()
        return {'status':'ok'}
    monkeypatch.setattr(suite, '_start_preparation', grouped_start(start))
    monkeypatch.setattr(suite, '_execute', execute)
    options = args(tmp_path)
    options.preparation_processes = 1
    assert suite._schedule(options, [a,b], Client()) == 0


def test_suite_defaults_visible_but_explicit_blind_still_supported(tmp_path, monkeypatch):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as cli
    modes = []
    monkeypatch.setattr(suite, 'run', lambda options, argv: modes.append(options.visibility_mode) or 0)
    assert cli.main(['--baseline-suite','--output-root',str(tmp_path)]) == 0
    assert cli.main(['--baseline-suite','--output-root',str(tmp_path),
                     '--visibility-mode=identity_blind']) == 0
    assert modes == ['deployment_visible_prefetched', 'identity_blind']


def test_active_suite_lock_prevents_a_second_dispatcher(tmp_path, monkeypatch):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as cli
    def forbidden(*a, **kw):
        raise AssertionError('must not prepare or create a client under an active suite lock')
    monkeypatch.setattr(suite, '_cells', forbidden)
    monkeypatch.setattr(suite.budgeted.runtime, '_make_client', forbidden)
    with (tmp_path/'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            cli.main(['--baseline-suite','--output-root',str(tmp_path)])


@pytest.mark.parametrize('fail_first', [False, True])
@pytest.mark.parametrize('setting', ['none', 'direct_indirect'])
def test_shared_cpu_queue_serves_other_tasks_and_preserves_failures(tmp_path, monkeypatch, fail_first, setting):
    a, b = cell(tmp_path, 'a', 2), cell(tmp_path, 'b', 1)
    source = tmp_path/'input.jsonl'
    source.write_text('{"drug":"CC"}\n{"drug":"CO"}\n')
    for c in (a,b):
        c.args = SimpleNamespace(tasks=[c.task], output_root=str(c.root), evidence_setting=setting)
    def register(options, *, preparation_submit):
        assert options.prepare_only and options.parallelism == 1
        options.manifest_hash = 'verified'
        preparation_submit(options, {options.tasks[0]: SimpleNamespace(input_jsonl=source)}, {'code':{}}, None)
    monkeypatch.setattr(suite.budgeted, 'run', register)
    monkeypatch.setattr(suite.budgeted, '_code_hashes', lambda: {})
    monkeypatch.setattr(suite, 'ProcessPoolExecutor', lambda *, mp_context, **kw: ThreadPoolExecutor(**kw))
    finished = threading.Event()
    calls = []
    def prepare(job):
        task, index = job
        calls.append(job)
        if task == 'a' and index == 0:
            assert finished.wait(5)  # The other worker must move from a to b.
        if task == 'b': finished.set()
        return {str(a.root if task == 'a' else b.root): 'bad context' if fail_first and job == ('a',0) else ''}
    monkeypatch.setattr(suite, '_prepare_query_group', prepare)
    assert suite._prepare_cells(args(tmp_path, preparation_cache_root=''), [a,b]) == int(fail_first)
    assert set(calls) == {('a',0), ('a',1), ('b',0)}
    for c in (a,b):
        receipt = json.loads((c.root/'preparation_done.json').read_text())
        assert receipt['manifest_hash'] == 'verified'
        assert receipt['error'] == ('bad context' if fail_first and c is a else '')


def test_query_group_loads_candidates_once_for_all_settings(tmp_path, monkeypatch):
    entries=[]
    for setting in ('none','direct','direct_indirect'):
        q = suite.budgeted.runtime.PreparedQuery('ames', 0, tmp_path/setting)
        options = SimpleNamespace(evidence_setting=setting, split_scheme='scaffold', candidate_molecules=128, min_similarity=0)
        entries.append(dict(args=options, queries={0:q}, rows=[{'drug':'CC'}], manifest={}, config=None, root=q.query_dir, unresolved=[]))
    calls=[]; pool={'direct':[1], 'indirect':[2]}
    cache=SimpleNamespace(candidates=lambda *a,**kw: calls.append(kw) or pool)
    monkeypatch.setattr(suite, '_PREPARATION_CONTEXTS', {'ames':entries})
    monkeypatch.setattr(suite, '_PREPARATION_CACHE', cache)
    monkeypatch.setattr(suite, '_unresolved_parents', lambda *a: [])
    received=[]
    monkeypatch.setattr(suite.budgeted, '_prepare_simple_query', lambda *a: received.append(a[-1]))
    monkeypatch.setattr(suite.budgeted, '_run_query', lambda *a: {'status':'ok'})
    assert not any(suite._prepare_query_group(('ames',0)).values())
    assert len(calls)==1 and received == [{'direct':[], 'indirect':[]}, pool, pool]


def test_six_cells_launch_one_preparation_coordinator_at_suite_root(tmp_path, monkeypatch):
    source = tmp_path/'input.jsonl'
    source.write_text('{"drug":"CC","Y":0}\n')
    monkeypatch.setattr(suite.budgeted.runtime, '_progressive_task_specs',
                        lambda *a: {'ames': SimpleNamespace(input_jsonl=source)})
    monkeypatch.setattr(suite.budgeted.runtime, '_selected_indices', lambda *a: [0])
    options = args(tmp_path, tasks=['ames'], split_scheme='scaffold', evaluation_subset='valid',
                   python_executable='python', visibility_mode='deployment_visible_prefetched',
                   direct_budget=50, indirect_budget=50)
    argv = ['--baseline-suite', '--output-root', str(tmp_path), '--min-similarity', '0']
    cells = suite._cells(options, argv)
    launched = []
    def launch(command, **kw):
        assert kw['start_new_session'] is True
        launched.append(command)
        return Process()
    monkeypatch.setattr(suite.subprocess, 'Popen', launch)
    suite._start_preparation(cells)
    assert len(cells) == 6 and len(launched) == 1
    command = launched[0]
    assert command.count('--baseline-suite') == 1 and '--prepare-only' in command
    assert command[command.index('--output-root') + 1] == str(tmp_path)
    assert command[command.index('--min-similarity') + 1] == '0'
    assert all(c.process is cells[0].process for c in cells)


def test_scaling_uses_separate_direct_winners_and_all_three_indirect_controls(tmp_path):
    direct = suite._settings(SimpleNamespace(baseline_phase='direct'))
    assert len(direct) == 7 and direct[0][0] == 'none'
    assert {r[4] for r in direct[1:]} == {25,50,100}
    path = tmp_path/'selection.json'
    path.write_text(json.dumps({'status':'complete_verified','evaluation_subset':'valid',
                               'chosen':{'direct_fill':25,'direct_cap':100}}))
    indirect = suite._settings(SimpleNamespace(baseline_phase='indirect', direct_budget_selection=str(path)))
    assert len(indirect) == 9
    assert {r[4] for r in indirect if r[0].startswith('both_fill')} == {25}
    assert {r[4] for r in indirect if not r[0].startswith('both_fill')} == {100}
    assert {r[5] for r in indirect} == {25,50,100}


def test_test_scaling_freezes_validation_winners_without_selecting_on_test(tmp_path, monkeypatch):
    validation = tmp_path/'valid'/'direct_selection.json'
    receipt = {'status':'complete_verified', 'evaluation_subset':'valid', 'tasks':['a'],
               'chosen':{'direct_fill':50, 'direct_cap':100}}
    suite.write_json_atomic(validation, receipt)
    selected, stages = [], []
    def select(root):
        selected.append(root)
        assert root == validation.parent/'direct'
        return receipt
    def run(options, argv):
        stages.append(options.baseline_phase)
        if options.baseline_phase == 'indirect':
            assert suite.budgeted._read(options.direct_budget_selection) == receipt
        return 0
    monkeypatch.setattr(suite, 'select_direct_budgets', select)
    monkeypatch.setattr(suite, '_run_suite', run)
    options = SimpleNamespace(evaluation_subset='test', limit=0, indices=None,
        direct_budget_selection=str(validation), output_root=str(tmp_path/'test'),
        tasks=['a'], candidate_molecules=128, records_per_molecule=5,
        budgeted_prompt='full_flat_v5', preparation_cache_root='')
    assert suite._run_scaling(options, []) == 0
    assert stages == ['direct', 'indirect'] and len(selected) == 1
    assert suite.budgeted._read(tmp_path/'test'/'scaling_plan.json')['test_budget_selection'] is False
    receipt['chosen']['direct_fill'] = 25
    with pytest.raises(ValueError, match='receipt'):
        suite._run_scaling(options, [])


def test_test_scaling_rejects_missing_validation_receipt_before_execution(tmp_path):
    options = SimpleNamespace(evaluation_subset='test', limit=0, indices=None, direct_budget_selection='')
    with pytest.raises(ValueError, match='completed validation'):
        suite._run_scaling(options, [])


def test_winner_uses_equal_task_weights_with_small_budget_tie_break(tmp_path):
    from tools.chembl_tool.common.json_utils import sha256_file
    suite.write_json_atomic(tmp_path/'suite_status.json', {'status':'complete','jobs':[{'task':'a'},{'task':'b'}]})
    for task, n in [('a',4), ('b',8)]:
        gold = tmp_path/f'{task}.jsonl'
        labels = [0,1]*(n//2)
        gold.write_text(''.join(json.dumps({'Y':y})+'\n' for y in labels))
        for method in ('direct_fill','direct_cap'):
            for budget in suite.BUDGETS:
                correct = (budget==50 if method=='direct_cap' else (task=='a' if budget in (25,50) else task=='b'))
                cell = tmp_path/'runs'/f'{method}_b{budget}'/task
                suite.write_json_atomic(cell/'experiment_manifest.json', {'evaluation_subset':'valid','direct_budget':budget,
                    'inputs':{task:{'input':{'path':str(gold),'sha256':sha256_file(gold)}}}})
                p = cell/'summary'/task/'predictions.jsonl';p.parent.mkdir(parents=True)
                p.write_text(''.join(json.dumps({'index':i,'label':y,'pred_label':y if correct else 1-y})+'\n'
                                     for i,y in enumerate(labels)))
    selected = suite.select_direct_budgets(tmp_path)
    assert selected['chosen'] == {'direct_fill':25,'direct_cap':50}
    assert selected['scores']['direct_fill']['100']['mean_macro_f1'] == .5
    p = tmp_path/'runs/direct_fill_b25/a/summary/a/predictions.jsonl'
    p.write_text(p.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError,match='all validation'):
        suite.select_direct_budgets(tmp_path)


@pytest.mark.parametrize('already_exited', [False, True])
def test_stop_preparation_reaps_coordinator_and_signals_worker_group(monkeypatch, already_exited):
    calls = []
    def killpg(pid, sig):
        calls.append((pid, sig))
        if already_exited:
            raise ProcessLookupError()
    monkeypatch.setattr(suite.os, 'killpg', killpg)
    process = SimpleNamespace(pid=123, wait=lambda: calls.append('reaped'))
    suite._stop_preparation(process)
    assert calls == [(123, suite.signal.SIGTERM), 'reaped']


def test_preparation_launch_failure_is_reported_without_inference(tmp_path, monkeypatch):
    c = cell(tmp_path, 'ames', 1)
    def fail(cells):
        raise OSError('cannot spawn coordinator')
    monkeypatch.setattr(suite, '_start_preparation', fail)
    monkeypatch.setattr(suite, '_execute', lambda *a: pytest.fail('no context was prepared'))
    assert suite._schedule(args(tmp_path), [c], Client()) == 1
    assert c.finalized and 'cannot spawn' in c.preparation_error


def test_matched_knn_suite_does_not_duplicate_direct_or_rerun_none():
    options = SimpleNamespace(direct_knn_root='ml', direct_budget=3, indirect_budget=50,
                              baseline_phase='standard', baseline_record_scaling=False)
    settings = suite._settings(options)
    assert [s[0] for s in settings] == ['direct_knn', 'both_fill', 'both_cap', 'both_group']
    assert all(s[4:] == (3, 50) for s in settings)
    options.direct_budget = 50
    with pytest.raises(ValueError, match='direct budget 3'):
        suite._settings(options)


def test_explicit_direct_indirect_sweep_does_not_select_on_test(tmp_path, monkeypatch):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as cli
    captured = []
    monkeypatch.setattr(suite, 'run', lambda options, argv: captured.append(suite._settings(options)) or 0)
    assert cli.main(['--baseline-suite', '--baseline-phase', 'indirect_fixed',
        '--evaluation-subset', 'test', '--direct-budget', '100',
        '--direct-selector', 'neighbor_fill', '--indirect-selector', 'neighbor_fill',
        '--output-root', str(tmp_path)]) == 0
    assert captured == [[(f'both_fill_b{b}', 'direct_indirect', 'neighbor_fill', 'neighbor_fill', 100, b)
                         for b in (25, 50, 100)]]
    with pytest.raises(ValueError, match='explicit Direct settings'):
        cli.main(['--baseline-suite', '--baseline-phase', 'indirect_fixed',
                  '--baseline-record-scaling', '--output-root', str(tmp_path)])


def test_identity_audit_cache_is_bound_and_reuses_index_work(tmp_path):
    calls = []
    cache = SimpleNamespace(root=tmp_path, view=lambda *a: calls.append(a) or
                            {'budgeted_unresolved_parent_ids': ['unknown-parent']})
    manifest = {'inputs': {'ames': {'index': {'sha256': 'one'}}}, 'code': {'retrieval': 'v1'}}
    assert suite._unresolved_parents(cache, 'ames', manifest) == ['unknown-parent']
    assert suite._unresolved_parents(cache, 'ames', manifest) == ['unknown-parent']
    assert len(calls) == 1
    manifest['inputs']['ames']['index']['sha256'] = 'two'
    suite._unresolved_parents(cache, 'ames', manifest)
    assert len(calls) == 2
    binding = {'inputs': manifest['inputs']['ames'], 'code': manifest['code']}
    path = tmp_path / 'identity_audit' / f'{suite.content_hash(binding)}.json'
    saved = json.loads(path.read_text()); saved['parents'] = []
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='content mismatch'):
        suite._unresolved_parents(cache, 'ames', manifest)


def test_fixed_sweep_rejects_full_scaling_before_dispatch():
    options = SimpleNamespace(baseline_phase='indirect_fixed', baseline_record_scaling=True)
    with pytest.raises(ValueError, match='explicit Direct settings'):
        suite.run(options, [])


def test_standard_subset_cli_preserves_fill_budgets(tmp_path, monkeypatch):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as cli
    captured = []
    monkeypatch.setattr(suite, 'run', lambda options, argv: captured.append(suite._settings(options)) or 0)
    assert cli.main(['--baseline-suite', '--baseline-settings', 'none', 'direct_fill', 'both_fill',
                     '--direct-budget', '50', '--indirect-budget', '50',
                     '--output-root', str(tmp_path)]) == 0
    assert [s[0] for s in captured[0]] == ['none', 'direct_fill', 'both_fill']
    assert all(s[4:] == (50, 50) for s in captured[0])
    assert all(s[2:4] == ('neighbor_fill', 'neighbor_fill') for s in captured[0][1:])


@pytest.mark.parametrize('extra', [dict(baseline_phase='direct'), dict(baseline_record_scaling=True),
                                  dict(direct_knn_root='ml')])
def test_subset_rejects_nonstandard_modes(extra):
    options = SimpleNamespace(baseline_settings=['none'], **extra)
    with pytest.raises(ValueError, match='standard, non-KNN'):
        suite.run(options, [])


def test_subset_requires_suite_cli():
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as cli
    with pytest.raises(SystemExit):
        cli.main(['--baseline-settings', 'none'])


def test_batch_prefetch_exceeds_consumer_limit_and_skips_saved_output(tmp_path, monkeypatch):
    c = cell(tmp_path, 'a', 5)
    monkeypatch.setattr(suite.budgeted, '_code_hashes', lambda: {})
    monkeypatch.setattr(suite.budgeted, 'summarize', lambda *a: None)
    monkeypatch.setattr(suite.budgeted, 'task_contract', lambda *a: None)
    monkeypatch.setattr(suite.budgeted, 'validation_kwargs', lambda *a: {})
    monkeypatch.setattr(suite.budgeted, 'response_validation_errors', lambda *a, **k: [])
    class BatchClient(Client):
        supports_batch_prefetch = True
        def __init__(self): self.submitted = []
        def prefetch_batch_json(self, messages): self.submitted.append(messages)
    client = BatchClient()
    def start(c):
        c.process = Process()
        setup_manifest(c)
        for q in c.queries:
            messages = [{'role': 'user', 'content': str(q.index)}]
            prepared = {'messages': messages, 'request_hash': suite.content_hash(messages),
                        'manifest_hash': suite.content_hash({'code': {}}),
                        'context_ready': True, 'card_alias_map': {}}
            suite.write_json_atomic(q.query_dir/'prepared.json', prepared)
            if q.index == 0:
                suite.write_json_atomic(q.query_dir/'output.json',
                    {'status': 'ok', 'request_hash': prepared['request_hash'], 'llm': {}})
        c.process.finished.set()
    def execute(c, q, client, attempt):
        assert len(client.submitted) == 4  # All pending requests precede the first result.
        return {'status': 'ok'}
    monkeypatch.setattr(suite, '_start_preparation', grouped_start(start))
    monkeypatch.setattr(suite, '_execute', execute)
    options = args(tmp_path); options.parallelism = 1
    assert suite._schedule(options, [c], client) == 0
    assert [m[0]['content'] for m in client.submitted] == ['1', '2', '3', '4']
    prepared = suite.budgeted._read(c.queries[1].query_dir/'prepared.json')
    prepared['request_hash'] = 'tampered'
    suite.write_json_atomic(c.queries[1].query_dir/'prepared.json', prepared)
    with pytest.raises(ValueError, match='valid prepared context'):
        suite._prefetch_batch_query(c, c.queries[1], client)
    assert len(client.submitted) == 4
