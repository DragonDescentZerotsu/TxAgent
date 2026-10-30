"""Stream the six full-flat baselines into one provider pool across all tasks.

Preparation uses a shared CPU process pool through the family CLI. Atomic
prepared.json files are the handoff; prediction, validation and summaries reuse
budgeted_evidence. No scientific prompt or selection logic lives here.
"""
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from copy import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import heapq
import multiprocessing
import os
import signal
from pathlib import Path
from statistics import mean
import subprocess
import time

from tools.chembl_tool.common.budgeted_evidence import content_hash
from tools.chembl_tool.common.json_utils import read_jsonl, sha256_file, write_json_atomic
from tools.chembl_tool.paper_experiments import budgeted_evidence as budgeted

VERSION = 'budgeted_global_ready_pool.v1'
SETTINGS = (
    ('none', 'none', 'molecule_cap', 'neighbor_fill'),
    ('direct_fill', 'direct', 'neighbor_fill', 'neighbor_fill'),
    ('both_fill', 'direct_indirect', 'neighbor_fill', 'neighbor_fill'),
    ('direct_cap', 'direct', 'molecule_cap', 'molecule_cap'),
    ('both_cap', 'direct_indirect', 'molecule_cap', 'molecule_cap'),
    ('both_group', 'direct_indirect', 'molecule_cap', 'group_balanced'),
)
BUDGETS = (25, 50, 100)


def _settings(args):
    phase = getattr(args, 'baseline_phase', 'standard')
    selected = getattr(args, 'baseline_settings', None)
    if selected and (phase != 'standard' or getattr(args, 'baseline_record_scaling', False)
                     or getattr(args, 'direct_knn_root', '')):
        raise ValueError('--baseline-settings requires the standard, non-KNN baseline suite')
    if selected and (len(set(selected)) != len(selected) or set(selected) - {s[0] for s in SETTINGS}):
        raise ValueError('--baseline-settings must contain unique supported settings')
    if getattr(args, 'direct_knn_root', ''):
        if phase != 'standard' or getattr(args, 'baseline_record_scaling', False) or args.direct_budget != 3:
            raise ValueError('matched KNN suite requires standard phase and direct budget 3')
        return [(name, evidence, 'neighbor_fill', indirect, 3, args.indirect_budget)
                for name, evidence, indirect in (
                    ('direct_knn', 'direct', 'neighbor_fill'),
                    ('both_fill', 'direct_indirect', 'neighbor_fill'),
                    ('both_cap', 'direct_indirect', 'molecule_cap'),
                    ('both_group', 'direct_indirect', 'group_balanced'))]
    if phase == 'indirect_fixed':
        if getattr(args, 'baseline_record_scaling', False) or getattr(args, 'direct_budget_selection', ''):
            raise ValueError('indirect_fixed uses explicit Direct settings, not validation selection')
        if args.direct_budget <= 0 or args.indirect_selector not in ('neighbor_fill', 'molecule_cap', 'group_balanced'):
            raise ValueError('indirect_fixed requires a positive Direct budget and a simple Indirect selector')
        name = {'neighbor_fill': 'fill', 'molecule_cap': 'cap', 'group_balanced': 'group'}[args.indirect_selector]
        return [(f'both_{name}_b{b}', 'direct_indirect', args.direct_selector,
                 args.indirect_selector, args.direct_budget, b) for b in BUDGETS]
    if phase == 'standard':
        return [(*setting, args.direct_budget, args.indirect_budget) for setting in SETTINGS
                if not selected or setting[0] in selected]
    if phase == 'direct':
        return [('none', 'none', 'molecule_cap', 'neighbor_fill', 0, 0)] + [
            (f'{name}_b{b}', 'direct', policy, 'neighbor_fill', b, 0)
            for name, policy in (('direct_fill','neighbor_fill'), ('direct_cap','molecule_cap')) for b in BUDGETS]
    selection = budgeted._read(args.direct_budget_selection)
    if selection.get('status') != 'complete_verified' or selection.get('evaluation_subset') != 'valid':
        raise ValueError('indirect scaling requires verified validation direct-budget selection')
    return [(f'{name}_b{b}', 'direct_indirect', direct, indirect, selection['chosen'][parent], b)
        for name, direct, indirect, parent in (
            ('both_fill','neighbor_fill','neighbor_fill','direct_fill'),
            ('both_cap','molecule_cap','molecule_cap','direct_cap'),
            ('both_group','molecule_cap','group_balanced','direct_cap')) for b in BUDGETS]


@dataclass
class Cell:
    setting: str
    task: str
    root: Path
    args: object
    command: list[str]
    queries: list
    process: object = None
    manifest: dict | None = None
    preparation_done: bool = False
    preparation_error: str = ''
    discovered: set = field(default_factory=set)
    results: dict = field(default_factory=dict)
    finalized: bool = False


def _cells(args, argv):
    specs = budgeted.task_specs(args)
    tasks = args.tasks or list(specs)
    indices = {task: budgeted.runtime._selected_indices(args, len(read_jsonl(specs[task].input_jsonl)))
               for task in tasks}
    base = [value for value in argv if value != '--baseline-suite']
    cells = []
    for setting, evidence, direct, indirect, direct_budget, indirect_budget in _settings(args):
        for task in tasks:
            root = Path(args.output_root) / 'runs' / setting / task
            cell_args = copy(args)
            cell_args.tasks = [task]
            cell_args.output_root = str(root)
            cell_args.evidence_setting = evidence
            cell_args.direct_selector = direct
            cell_args.indirect_selector = indirect
            cell_args.direct_budget = direct_budget
            cell_args.indirect_budget = indirect_budget
            queries = [budgeted.runtime.PreparedQuery(task, i, root/'runs'/task/f'query_{i:05d}')
                       for i in indices[task]]
            command = [args.python_executable, '-m',
                'tools.chembl_tool.paper_experiments.run_conditioned_assay_family_curve',
                *base, '--baseline-suite', '--prepare-only', '--tasks', *tasks,
                '--visibility-mode', args.visibility_mode]
            cells.append(Cell(setting, task, root, cell_args, command, queries))
    return cells


def _start_preparation(cells):
    for cell in cells:
        cell.root.mkdir(parents=True, exist_ok=True)
        for name in ('preparation_ready.json', 'preparation_done.json'):
            (cell.root/name).unlink(missing_ok=True)
    first = cells[0]
    env = {**os.environ, 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
    with (first.root/'preparation.log').open('a') as log:
        process = subprocess.Popen(first.command, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    for cell in cells:
        cell.process = process


def _stop_preparation(process):
    # The coordinator owns spawned CPU workers; terminate the whole process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    process.wait()


_PREPARATION_CONTEXTS = None
_PREPARATION_CACHE = None


def _init_preparation_worker(contexts, cache_root):
    from tools.chembl_tool.common.budgeted_preparation import PreparationCache
    global _PREPARATION_CONTEXTS, _PREPARATION_CACHE
    _PREPARATION_CONTEXTS = contexts
    _PREPARATION_CACHE = PreparationCache(cache_root)


def _unresolved_parents(cache, task, manifest):
    """Load small identity-audit metadata once per source/code binding, on demand."""
    binding = {'inputs': manifest['inputs'][task], 'code': manifest['code']}
    path = cache.root / 'identity_audit' / f'{content_hash(binding)}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = budgeted._read(path)
            if saved['binding'] != binding or content_hash(saved['parents']) != saved['parents_hash']:
                raise ValueError('identity audit cache binding or content mismatch')
            return saved['parents']
        parents = cache.view(task, manifest).get('budgeted_unresolved_parent_ids', [])
        write_json_atomic(path, {'binding': binding, 'parents': parents, 'parents_hash': content_hash(parents)})
        return parents


def _prepare_query_group(job):
    """One candidate load serves every missing setting of this query."""
    task, index = job
    entries = _PREPARATION_CONTEXTS[task]
    pool, retrieval_error, unresolved = None, '', []
    evidence = [e for e in entries if e['args'].evidence_setting != 'none'
                and not (e['queries'][index].query_dir/'prepared.json').exists()]
    if evidence:
        first = evidence[0]
        options, manifest = first['args'], first['manifest']
        try:
            unresolved = _unresolved_parents(_PREPARATION_CACHE, task, manifest)
            pool = _PREPARATION_CACHE.candidates(first['rows'][index]['drug'], None,
                task=task, manifest=manifest, split_scheme=options.split_scheme,
                candidate_molecules=options.candidate_molecules, min_similarity=options.min_similarity,
                indirect_identity_policy=getattr(options, 'indirect_identity_policy', 'parent_disjoint'))
        except Exception as exc:
            retrieval_error = f'{type(exc).__name__}: {exc}'
    results = {}
    for entry in entries:
        options, query = entry['args'], entry['queries'][index]
        error = ''
        try:
            if not (query.query_dir/'prepared.json').exists():
                if options.evidence_setting != 'none' and retrieval_error:
                    raise RuntimeError(retrieval_error)
                # Cached pools omit unresolved identities; the view only supplies audit metadata.
                view = {'budgeted_unresolved_parent_ids': unresolved if options.evidence_setting != 'none' else []}
                budgeted._prepare_simple_query(options, task, entry['rows'][index], query,
                    view, entry['manifest'], entry['config'],
                    {'direct': [], 'indirect': []} if options.evidence_setting == 'none' else pool)
            budgeted._run_query(options, query, None)  # Validate caches without any model call.
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        results[str(entry['root'])] = error
    return results


def _prepare_cells(args, cells):
    """Bound CPU work globally; idle workers can serve any task or setting."""
    contexts, errors, remaining, rows_by_task = {}, {}, {}, {}
    def register(options, specs, manifest, config):
        task = options.tasks[0]
        c = next(c for c in cells if str(c.root) == options.output_root)
        if task not in rows_by_task:
            rows_by_task[task] = read_jsonl(specs[task].input_jsonl)
        contexts.setdefault(task, []).append({'args': options, 'root': c.root, 'manifest': manifest,
            'config': config, 'rows': rows_by_task[task],
            'queries': {q.index: q for q in c.queries}})
        remaining[str(c.root)] = len(c.queries)
    cache_root = args.preparation_cache_root or str(Path(args.output_root)/'candidate_cache')
    for c in cells:
        options = copy(c.args)
        options.prepare_only, options.parallelism = True, 1
        options.retry_race_width = 1  # CPU preparation never makes model requests.
        options.preparation_ready_file = str(c.root/'preparation_ready.json')
        errors[str(c.root)] = ''
        try:
            budgeted.run(options, preparation_submit=register)
        except Exception as exc:
            errors[str(c.root)] = f'{type(exc).__name__}: {exc}'
            write_json_atomic(c.root/'preparation_done.json', {
                'error': errors[str(c.root)], 'manifest_hash': getattr(options, 'manifest_hash', None)})
    # Workers load identity metadata and candidate pools on demand; registration loads no index.
    jobs = [(task, index) for task, entries in contexts.items()
            for index in entries[0]['queries']]
    entries_by_root = {str(e['root']): e for entries in contexts.values() for e in entries}
    with ProcessPoolExecutor(max_workers=args.preparation_processes,
            mp_context=multiprocessing.get_context('spawn'), initializer=_init_preparation_worker,
            initargs=(contexts, cache_root)) as pool:
        futures = {pool.submit(_prepare_query_group, job): job for job in jobs}
        for future in as_completed(futures):
            task, _ = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {str(e['root']): f'{type(exc).__name__}: {exc}' for e in contexts[task]}
            for root, error in result.items():
                errors[root] = errors[root] or error
                remaining[root] -= 1
                if remaining[root] == 0:
                    entry = entries_by_root[root]
                    if budgeted._code_hashes() != entry['manifest']['code']:
                        errors[root] = errors[root] or 'implementation changed during preparation'
                    write_json_atomic(Path(root)/'preparation_done.json', {
                        'error': errors[root], 'manifest_hash': entry['args'].manifest_hash,
                        'query_groups': len(entry['queries']), 'preparation_processes': args.preparation_processes})
    return int(any(errors.values()))


def _discover(cell):
    manifest_path = cell.root/'experiment_manifest.json'
    if cell.manifest is None:
        # A fresh receipt follows _pin's validation of ALL scientific inputs.
        # A stale manifest from a failed resume must not unlock old cards.
        receipt = cell.root/'preparation_ready.json'
        if not receipt.exists():
            return []
        manifest = budgeted._read(manifest_path)
        if budgeted._read(receipt)['manifest_hash'] != content_hash(manifest):
            raise ValueError('preparation readiness does not match the manifest')
        if manifest['code'] != budgeted._code_hashes():
            raise ValueError('implementation changed during preparation')
        cell.manifest = manifest
        cell.args.manifest_hash = content_hash(manifest)
    ready = []
    pending = {q.query_dir.name: q for q in cell.queries if q.index not in cell.discovered}
    if not pending:
        return ready
    # Enumerate published query directories instead of stat-ing every expected
    # query on each poll. The latter stalls the request queue on shared storage.
    try:
        with os.scandir(cell.queries[0].query_dir.parent) as entries:
            for entry in entries:
                query = pending.get(entry.name)
                if query is not None and (query.query_dir/'prepared.json').exists():
                    cell.discovered.add(query.index)
                    ready.append(query)
    except FileNotFoundError:
        pass  # The preparation worker has not created this task directory yet.
    return sorted(ready, key=lambda q: q.index)


def _execute(cell, query, client, attempt):
    args = copy(cell.args)  # Retry state belongs to the query, never the whole cell.
    args.retry_round = attempt
    output = query.query_dir / 'output.json'
    if not attempt and output.exists() and budgeted._read(output).get('status') == 'error':
        args.retry_round = 1  # A resumed failure should enter the configured race immediately.
    return budgeted._run_query(args, query, client)


def _prefetch_batch_query(cell, query, client):
    """Validate the prepared handoff and skip already valid saved predictions."""
    prepared = budgeted._read(query.query_dir/'prepared.json')
    if (prepared['request_hash'] != content_hash(prepared['messages'])
            or prepared['manifest_hash'] != cell.args.manifest_hash
            or not prepared.get('context_ready', prepared.get('tool_prefetch_complete'))):
        raise ValueError('Batch prefetch requires a valid prepared context')
    output = query.query_dir/'output.json'
    if output.exists():
        saved = budgeted._read(output)
        if saved.get('request_hash') != prepared['request_hash']:
            raise ValueError('prediction request hash mismatch')
        validation = budgeted.validation_kwargs(budgeted.task_contract(cell.args, query.task),
                                                prepared['card_alias_map'])
        if saved['status'] == 'ok' and not budgeted.response_validation_errors(saved['llm'], **validation):
            return
    client.prefetch_batch_json(prepared['messages'])


def _publish(root, cells, client, active, ready, delayed, phase='running'):
    jobs = []
    for cell in cells:
        succeeded = sum(r['status'] == 'ok' for r in cell.results.values())
        status = ('needs_attention' if cell.preparation_error or succeeded != len(cell.queries) else 'complete') \
            if cell.finalized else ('running' if cell.process is not None else 'queued')
        jobs.append({'setting':cell.setting, 'task':cell.task, 'status':status,
                     'expected_queries':len(cell.queries), 'n_prepared':len(cell.discovered),
                     'n_succeeded_queries':succeeded,
                     'preparation_done':cell.preparation_done, 'preparation_error':cell.preparation_error,
                     'failed_queries':[{'index':i, **r} for i,r in cell.results.items() if r['status'] != 'ok']})
    data = {'scheduler':VERSION, 'status':phase, 'pid':os.getpid(),
            'updated_at':datetime.now(timezone.utc).isoformat(), 'jobs':jobs,
            'expected_queries':sum(j['expected_queries'] for j in jobs),
            'n_succeeded_queries':sum(j['n_succeeded_queries'] for j in jobs),
            'active_queries':active, 'ready_queries':ready, 'delayed_retries':delayed,
            'preparing_tasks': sorted({c.task for c in cells if c.process is not None
                                      and not c.preparation_done}),
            'provider_pool':client.snapshot()}
    write_json_atomic(root/'suite_status.json', data)
    lines = ['# LLM baseline progress', '', f"Updated: {data['updated_at']} (15 s refresh)", '',
             f"{phase}: {data['n_succeeded_queries']:,} / {data['expected_queries']:,}", '',
             '| Setting | Prepared | Successful | Total |', '|---|---:|---:|---:|']
    for setting in dict.fromkeys(c.setting for c in cells):
        subset = [j for j in jobs if j['setting'] == setting]
        lines.append(f"| {setting} | {sum(j['n_prepared'] for j in subset)} | "
                     f"{sum(j['n_succeeded_queries'] for j in subset)} | "
                     f"{sum(j['expected_queries'] for j in subset)} |")
    lines += ['', f'Active queries: {active}; ready: {ready}; delayed retries: {delayed}.',
              'Preparing tasks: ' + ', '.join(data['preparing_tasks']) + '.',
              'Active queries include broker admission waits and JSON-repair calls.',
              'With a single Batch provider, prepared requests are submitted ahead of these result consumers; '
              'active/ready counts are not remote submission counts. See batch_jobs journals.', '']
    temporary = root/'PROGRESS.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(root/'PROGRESS.md')
    if (root.parent/'scaling_plan.json').exists():
        overall = root.parent/'PROGRESS.md'
        text = f'# Record-budget scaling\n\nCurrent stage: **{root.name}**\n\n'
        selection = root.parent/'direct_selection.json'
        if selection.exists():
            text += f"Selected direct budgets: {budgeted._read(selection)['chosen']}\n\n"
        temp = overall.with_suffix('.tmp')
        temp.write_text(text + '\n'.join(lines[2:]))
        temp.replace(overall)


def _schedule(args, cells, client):
    root = Path(args.output_root)
    pending, preparing = bool(cells), False
    ready, delayed, active = deque(), [], {}
    serial = 0
    last_scan = last_publish = 0.0
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        try:
            while pending or preparing or ready or delayed or active:
                now = time.monotonic()
                if pending:
                    pending = False
                    try:
                        _start_preparation(cells)
                        preparing = True
                    except OSError as exc:
                        for cell in cells:
                            cell.preparation_done = True
                            cell.preparation_error = str(exc)
                            cell.results = {q.index:{'status':'error', 'error':str(exc)} for q in cell.queries}
                if preparing and now - last_scan >= .5:
                    code = cells[0].process.poll()
                    for cell in cells:
                        if cell.preparation_done:
                            continue
                        try:
                            discovered = _discover(cell)
                            ready.extend((cell, q, 0) for q in discovered)
                            if getattr(client, 'supports_batch_prefetch', False):
                                for query in discovered:
                                    _prefetch_batch_query(cell, query, client)
                        except Exception as exc:
                            cell.preparation_error = str(exc)
                            _stop_preparation(cell.process)
                        done_path = cell.root/'preparation_done.json'
                        if code is None and not done_path.exists():
                            continue
                        cell.preparation_done = True
                        if done_path.exists():
                            receipt = budgeted._read(done_path)
                            cell.preparation_error = cell.preparation_error or receipt['error']
                            if cell.manifest and receipt['manifest_hash'] != content_hash(cell.manifest):
                                cell.preparation_error = 'preparation completion manifest mismatch'
                        elif code:
                            cell.preparation_error = cell.preparation_error or f'preparation exited {code}'
                        if len(cell.discovered) != len(cell.queries):
                            cell.preparation_error = cell.preparation_error or 'preparation did not publish all contexts'
                        for query in cell.queries:
                            if query.index not in cell.discovered:
                                cell.results[query.index] = {'status':'error', 'error':'context was not prepared'}
                    if code is not None:
                        if code:
                            _stop_preparation(cells[0].process)
                        preparing = False
                    last_scan = now
                while delayed and delayed[0][0] <= now:
                    _, _, job = heapq.heappop(delayed)
                    ready.append(job)
                while ready and len(active) < args.parallelism:
                    cell, query, attempt = ready.popleft()
                    active[executor.submit(_execute, cell, query, client, attempt)] = (cell, query, attempt)
                done = wait(active, timeout=.05, return_when=FIRST_COMPLETED).done if active else ()
                if not active:
                    time.sleep(.05)
                for future in done:
                    cell, query, attempt = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {'status':'error', 'error':str(exc)}
                        if getattr(exc, 'attempts', None):
                            result['provider_attempts'] = exc.attempts
                    if result['status'] != 'ok' and attempt < args.max_stage_requeues:
                        delay = min(args.retry_delay_s * 2 ** min(attempt, 20), 900)
                        serial += 1
                        heapq.heappush(delayed, (time.monotonic()+delay, serial, (cell, query, attempt+1)))
                    else:
                        cell.results[query.index] = result
                for cell in cells:
                    if cell.preparation_done and not cell.finalized and len(cell.results) == len(cell.queries):
                        if cell.manifest is not None and not cell.preparation_error:
                            budgeted.summarize(cell.args, cell.queries, cell.root)
                        if (root/'execution_provider.json').exists():
                            write_json_atomic(cell.root/'execution_provider.json',
                                              budgeted._read(root/'execution_provider.json'))
                        cell.finalized = True
                        write_json_atomic(cell.root/'execution_status.json', {
                            'scheduler':VERSION, 'phase':'complete' if not cell.preparation_error and all(
                                r['status']=='ok' for r in cell.results.values()) else 'needs_attention',
                            'n_total_queries':len(cell.queries),
                            'n_succeeded_queries':sum(r['status']=='ok' for r in cell.results.values()),
                            'failed_queries':[i for i,r in cell.results.items() if r['status']!='ok'],
                            'preparation_error':cell.preparation_error})
                if now - last_publish >= 15:
                    _publish(root, cells, client, len(active), len(ready), len(delayed))
                    last_publish = now
        finally:
            if preparing:
                _stop_preparation(cells[0].process)
    failed = any(c.preparation_error or any(r['status'] != 'ok' for r in c.results.values()) for c in cells)
    _publish(root, cells, client, 0, 0, 0, 'needs_attention' if failed else 'complete')
    return int(failed)


def select_direct_budgets(root):
    """Select one global budget per Direct strategy using equal task weights."""
    from sklearn.metrics import f1_score
    from tools.chembl_tool.common.json_utils import sha256_file
    status = budgeted._read(root/'suite_status.json')
    if status['status'] != 'complete':
        raise ValueError('cannot select budgets from an incomplete stage')
    tasks = sorted({j['task'] for j in status['jobs']})
    scores, sources = {}, []
    for method in ('direct_fill', 'direct_cap'):
        scores[method] = {}
        for b in BUDGETS:
            values = {}
            for task in tasks:
                cell = root/'runs'/f'{method}_b{b}'/task
                manifest = budgeted._read(cell/'experiment_manifest.json')
                if manifest['evaluation_subset'] != 'valid' or manifest['direct_budget'] != b:
                    raise ValueError('budget selection must use matching validation cells')
                source = manifest['inputs'][task]['input']
                if sha256_file(Path(source['path'])) != source['sha256']:
                    raise ValueError('budget selection input changed')
                gold = read_jsonl(Path(source['path']))
                p = cell/'summary'/task/'predictions.jsonl'
                predictions = read_jsonl(p)
                if (len(predictions) != len(gold) or [r['index'] for r in predictions] != list(range(len(gold)))
                        or [r['label'] for r in predictions] != [int(r['Y']) for r in gold]
                        or any(r['pred_label'] not in (0,1) for r in predictions)):
                    raise ValueError('budget selection requires all validation predictions')
                values[task] = float(f1_score([r['label'] for r in predictions],
                    [r['pred_label'] for r in predictions], average='macro'))
                sources.append({'path':str(p),'sha256':sha256_file(p)})
            scores[method][str(b)] = {'per_task': values, 'mean_macro_f1': mean(values.values())}
    chosen = {method: max(BUDGETS, key=lambda b: (round(values[str(b)]['mean_macro_f1'],12),-b))
              for method, values in scores.items()}
    return {'status':'complete_verified','evaluation_subset':'valid','metric':'unweighted task mean Macro-F1',
            'tie_break':'smaller budget (12 decimal places)', 'tasks':tasks,'chosen':chosen,'scores':scores,'sources':sources}


def _run_scaling(args, argv):
    if args.evaluation_subset not in ('valid', 'test') or args.limit or args.indices:
        raise ValueError('record scaling requires a complete validation or test set')
    selection_source = getattr(args, 'direct_budget_selection', '')
    if args.evaluation_subset == 'test':
        if not selection_source:
            raise ValueError('test scaling requires --direct-budget-selection from completed validation')
        selection_path = Path(selection_source).resolve()
        selection = budgeted._read(selection_path)
        # Recompute from validation predictions and current input hashes before
        # starting any test work. Never choose a budget from test metrics.
        verified = select_direct_budgets(selection_path.parent/'direct')
        if selection != verified or set(selection['tasks']) != set(args.tasks or budgeted.task_specs(args)):
            raise ValueError('validation budget receipt or task coverage does not match')
    elif selection_source:
        raise ValueError('validation scaling selects its own Direct budgets; omit --direct-budget-selection')
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'scaling.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {'budgets':list(BUDGETS),'tasks':args.tasks,
            'selection':'equal-task mean validation Macro-F1; separate Direct winners; smaller-budget tie-break',
            'group_balanced_direct':'Molecule-cap winner','candidate_molecules':args.candidate_molecules,
            'records_per_molecule':args.records_per_molecule,'prompt':args.budgeted_prompt}
        if args.evaluation_subset == 'test':
            plan.update(evaluation_subset='test', test_budget_selection=False,
                validation_selection={'path':str(selection_path),'sha256':sha256_file(selection_path)})
            budgeted._pin(root/'direct_selection.json', selection)
        budgeted._pin(root/'scaling_plan.json', plan)
        for phase in ('direct', 'indirect'):
            options = copy(args)
            options.output_root = str(root/phase)
            options.baseline_phase = phase
            options.direct_budget_selection = str(root/'direct_selection.json') if phase == 'indirect' else ''
            options.preparation_cache_root = args.preparation_cache_root or str(root/'candidate_cache')
            phase_argv = [*argv, '--baseline-phase', phase, '--output-root', options.output_root,
                          '--preparation-cache-root', options.preparation_cache_root]
            if phase == 'indirect':
                phase_argv.extend(['--direct-budget-selection', options.direct_budget_selection])
            if _run_suite(options, phase_argv):
                return 1
            if phase == 'direct' and args.evaluation_subset == 'valid':
                budgeted._pin(root/'direct_selection.json', select_direct_budgets(root/'direct'))
        write_json_atomic(root/'scaling_status.json', {'status':'complete', 'direct_selection':str(root/'direct_selection.json')})
        return 0


def run(args, argv):
    if getattr(args, 'baseline_settings', None):
        _settings(args)  # Reject incompatible filters before dispatching scaling.
    if getattr(args, 'baseline_phase', 'standard') == 'indirect_fixed':
        _settings(args)  # Reject conflicting selection modes before phase dispatch.
    if getattr(args, 'baseline_record_scaling', False) and not args.prepare_only:
        return _run_scaling(args, argv)
    return _run_suite(args, argv)


def _run_suite(args, argv):
    if any((args.evidence_setting, args.feature_inputs_only, args.annotation_db,
            args.transfer_scores, args.uniform_record_quality, args.skip_tool_prefetch,
            args.matched_progressive_root, args.replicate_ids, args.refresh_query_priors,
            args.level_reuse_source_root, args.prompt_adaptation, args.parallelism_per_task)):
        raise ValueError('--baseline-suite requires independent simple full-flat settings and one global pool')
    if not 1 <= args.parallelism <= args.endpoint_concurrency_budget or args.preparation_processes < 1:
        raise ValueError('invalid global request or preparation budget')
    if args.max_stage_requeues < 0 or args.retry_delay_s < 0:
        raise ValueError('retry limits must be nonnegative')
    args.retry_race_width = args.retry_race_width or 1
    if not 1 <= args.retry_race_width <= args.parallelism:
        raise ValueError('retry race width must fit the global request budget')
    if args.retry_race_width > 1 and args.transport_max_retries:
        raise ValueError('request races require transport-max-retries 0')
    budgeted.runtime.load_env_file(Path(args.env_file))
    provider = budgeted.runtime._resolve_provider_pool_config(args)
    budgeted.runtime._generation_settings(provider)
    if len({content_hash(p.request_extra_body) for p in provider.providers}) != 1:
        raise ValueError('all providers must use identical generation settings')
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        cells = _cells(args, argv)
        with (root/'preparation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _prepare_cells(args, cells)
    with (root/'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cells = _cells(args, argv)
        budgeted._pin(root/'suite_manifest.json', {'scheduler':VERSION,
            'cells':[{'setting':c.setting, 'task':c.task, 'indices':[q.index for q in c.queries]} for c in cells]})
        write_json_atomic(root/'execution_provider.json', asdict(provider))
        client = budgeted.runtime._make_client(args, provider)
        try:
            return _schedule(args, cells, client)
        finally:
            client.close()
