"""Reprepare frozen paper experiments using shared retrieval and model transport."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import argparse
import fcntl
import json
import math
import time
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path

from predict.prompts import paper_baseline as prompt
from predict.utils.json import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from predict.utils.response import call_with_json_validation, response_validation_errors, structured_response_is_valid

REVISION = 'paper_baseline_retrieval.v1'


def prepare(source_root, source_batch, target, indices=None, *, input_map=None):
    """Use frozen query/prior/config inputs; rebuild all eligible evidence before budgets."""
    from optimization.selection.record_budget import SIMPLE_POLICIES, select_simple_records
    from predict.retrieval.features import load_retrieval_index
    from predict.retrieval.paper_baseline import build_evidence_view, merge_train_labels, retrieve_candidates

    if (source_root is None) == (input_map is None):
        raise ValueError('provide exactly one source root or input map')
    relocated = json.loads(Path(input_map).read_text()) if input_map is not None else None
    if input_map is not None and (not isinstance(relocated, dict)
            or any(not isinstance(v, str) or not v for v in relocated.values())):
        raise ValueError('input map must map frozen path strings to file paths')
    source_root = Path(input_map).resolve().parent if input_map is not None else Path(source_root).resolve()
    source_batch, target = Path(source_batch).resolve(), Path(target).resolve()
    if target == source_batch or source_batch in target.parents:
        raise ValueError('preparation requires a separate output root')
    original = json.loads((source_batch / 'experiment_manifest.json').read_text())
    if (original.get('version') != 'direct_indirect_budgeted.v3'
            or original.get('prompt_version') != 'joseph_v5'
            or original.get('visibility_mode') != 'deployment_visible_prefetched'
            or original.get('assessment_required') or original.get('feature_inputs_only')
            or original.get('skip_tool_prefetch') or original.get('migration')):
        raise ValueError('requires a completed visible paper baseline source batch')
    if (original['direct_selector'] not in SIMPLE_POLICIES
            or original['selection']['policy'] not in SIMPLE_POLICIES
            or original['setting'] not in ('none', 'direct', 'direct_indirect')
            or any('direct_knn' in spec for spec in original['inputs'].values())):
        raise ValueError('unsupported paper baseline selection; matched KNN is a separate protocol')
    expected_direct = 'scaffold_disjoint' if original['split_scheme'] == 'scaffold' else 'parent_disjoint'
    if original['direct_identity_policy'] != expected_direct:
        raise ValueError('source direct identity policy differs from split contract')
    manifest = deepcopy(original)
    manifest['migration'] = {'version': REVISION, 'source_manifest_sha256': sha256_file(source_batch/'experiment_manifest.json'),
        'source_batch': str(source_batch), 'source_root': str(source_root), 'query_indices': {},
        'scope': 'frozen query/prior/generation; approved retrieval identity refinement; no inherited predictions'}
    from predict.retrieval import policies, task_identity, retrieve, features
    from predict.utils import evidence, response
    from optimization.selection import record_budget
    from predict.retrieval import paper_baseline
    manifest['code'] = {str(Path(module.__file__).relative_to(Path(__file__).parents[3])): sha256_file(Path(module.__file__))
                        for module in (prompt, paper_baseline, policies, task_identity, retrieve, features, evidence, response, record_budget)}
    manifest['code']['predict/harnesses/branches/paper_baseline.py'] = sha256_file(Path(__file__))
    inputs, cohorts = {}, {}
    for task, spec in original['inputs'].items():
        inputs[task] = {}
        required = ['input'] + ([] if original['setting'] == 'none' else ['index'])
        if original['benchmark_profile'] and original['setting'] != 'none':
            required += ['train', 'heldout']
        for key in required:
            locator = spec[key]['path']
            if relocated is not None and locator not in relocated:
                raise ValueError(f'{task}: missing {key} in input map')
            path = (source_root / (relocated[locator] if relocated is not None else locator)).resolve()
            if sha256_file(path) != spec[key]['sha256']:
                raise ValueError(f'{task}: frozen {key} input changed')
            inputs[task][key] = path
        cohorts[task] = read_jsonl(inputs[task]['input'])
        chosen = list(range(len(cohorts[task]))) if indices is None else list(indices)
        if not chosen or len(chosen) != len(set(chosen)) or any(i < 0 or i >= len(cohorts[task]) for i in chosen):
            raise ValueError('invalid or repeated query indices')
        manifest['migration']['query_indices'][task] = chosen
    if relocated is not None:
        required_paths = {original['inputs'][task][key]['path'] for task, paths in inputs.items() for key in paths}
        if set(relocated) != required_paths:
            raise ValueError('input map contains unused paths')
        manifest['migration']['retrieval_inputs'] = {task: {key: str(path) for key, path in paths.items()}
                                                    for task, paths in inputs.items()}
        manifest['migration']['evaluation_inputs'] = {task: str(paths['input']) for task, paths in inputs.items()}
    if (target/'experiment_manifest.json').exists():
        if json.loads((target/'experiment_manifest.json').read_text()) != manifest:
            raise ValueError('preparation contract changed; use a new output root')
    elif target.exists() and any(p.name != '.execution.lock' for p in target.iterdir()):
        raise ValueError('output root contains unbound artifacts')
    write_json_atomic(target/'experiment_manifest.json', manifest)
    reports = []
    for task, paths in inputs.items():
        view = None
        if original['setting'] != 'none':
            if paths['index'].suffix != '.pkl':
                raise ValueError('paper Morgan baseline requires its frozen pickle index')
            index = load_retrieval_index(paths['index'])
            if original['benchmark_profile']:
                index = merge_train_labels(index, task, read_jsonl(paths['train']), read_jsonl(paths['heldout']))
            view = build_evidence_view(index)
        for i in manifest['migration']['query_indices'][task]:
            source = source_batch/'runs'/task/f'query_{i:05d}'
            frozen = json.loads((source/'prepared.json').read_text())
            prompt.verify_prepared(frozen, original, json.loads((source/'output.json').read_text()))
            row = cohorts[task][i]
            query = {'benchmark_profile': original['benchmark_profile'], 'canonical_smiles': row['drug'],
                     'condition_group': row.get('condition_group', '')}
            evaluation = json.loads((source/'evaluation.json').read_text())
            if (frozen['query_hash'] != prompt.content_hash(query) or evaluation['label'] != int(row['Y'])
                    or frozen['prompt_payload']['task_definition']['task'] != task
                    or frozen['prompt_payload']['query']['canonical_smiles'] != row['drug']):
                raise ValueError('frozen query or label binding differs')
            pool = {'direct': [], 'indirect': []} if view is None else retrieve_candidates(row['drug'], view,
                split_scheme=original['split_scheme'], candidate_molecules=original['candidate_molecules'],
                min_similarity=original['min_similarity'], indirect_identity_policy=original['indirect_identity_policy'])
            direct, direct_audit = select_simple_records(pool['direct'], budget=original['direct_budget'],
                policy=original['direct_selector'], per_molecule=original['direct_records_per_molecule'],
                seed=original['selection']['seed'])
            candidates = pool['indirect'] if original['setting'] == 'direct_indirect' else []
            selected, audit = select_simple_records(candidates,
                **{k: original['selection'][k] for k in ('budget', 'policy', 'per_molecule', 'seed')})
            payload, aliases = prompt.project_records(frozen['prompt_payload'], direct + selected)
            messages = prompt.render(payload)
            prepared = {**frozen, 'manifest_hash': prompt.content_hash(manifest), 'messages': messages,
                'request_hash': prompt.content_hash(messages), 'prompt_payload': payload, 'card_alias_map': aliases,
                'direct_ids': [r['id'] for r in direct], 'indirect_ids': [r['id'] for r in selected],
                'candidate_pool_hash': prompt.content_hash(sorted(r['id'] for r in pool['indirect'])),
                'n_direct': len(direct), 'n_indirect': len(selected)}
            prompt.verify_request(prepared, manifest)
            directory = target/'runs'/task/f'query_{i:05d}'
            if (directory/'prepared.json').exists() and json.loads((directory/'prepared.json').read_text()) != prepared:
                raise ValueError('prepared evidence or frozen prior changed; use a new output root')
            comparison = {'task': task, 'index': i, 'messages_equal': messages == frozen['messages'],
                'direct_ids_equal': prepared['direct_ids'] == frozen['direct_ids'],
                'indirect_ids_equal': prepared['indirect_ids'] == frozen['indirect_ids'],
                'n_direct': len(direct), 'n_indirect': len(selected)}
            write_json_atomic(directory/'selection.json', {'query': query, 'direct': direct, 'selected': selected,
                'audit': audit, 'direct_audit': direct_audit, 'comparison': comparison})
            write_json_atomic(directory/'evaluation.json', evaluation)
            write_json_atomic(directory/'prepared.json', prepared)
            reports.append(comparison)
    write_json_atomic(target/'preparation_comparison.json', reports)
    return reports


def execute_query(directory, manifest, call):
    """Run or resume one bound request using the shared four-attempt validator."""
    directory = Path(directory)
    prepared = json.loads((directory/'prepared.json').read_text())
    prompt.verify_request(prepared, manifest)
    if not prepared.get('context_ready'):
        raise ValueError('prepared context is incomplete')
    output = directory/'output.json'
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get('request_hash') != prepared['request_hash']:
            raise ValueError('saved output belongs to a different request')
        if saved.get('status') == 'ok':
            prompt.verify_prepared(prepared, manifest, saved)
            return saved
    task = prepared['prompt_payload']['task_definition']['task']
    result = call_with_json_validation(call, prepared['messages'],
        **prompt.validation_kwargs(prepared['prompt_payload'], prepared['card_alias_map']),
        branch_name=f'{task} budgeted decision', max_attempts=4)
    saved = {'status': 'ok' if structured_response_is_valid(result) else 'error', 'model_called': True,
        'request_hash': prepared['request_hash'], 'llm': result, 'card_alias_map': prepared['card_alias_map']}
    if saved['status'] == 'ok':
        prompt.verify_prepared(prepared, manifest, saved)
    write_json_atomic(output, saved)
    return saved


def run(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-paper-baseline', type=Path, metavar='SOURCE_BATCH')
    parser.add_argument('--run-paper-baseline', type=Path, metavar='PREPARED_BATCH')
    parser.add_argument('--bundle-paper-baseline', type=Path, metavar='PREPARED_BATCH')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--input-map', type=Path,
                        help='Preparation-only JSON mapping frozen input paths to relocated files; relative to this JSON')
    parser.add_argument('--run-name')
    parser.add_argument('--query-indices', type=int, nargs='+')
    parser.add_argument('--provider-config', type=Path)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--parallelism', type=int, default=1)
    parser.add_argument('--retry-race-width', type=int, default=1)
    parser.add_argument('--max-stage-requeues', type=int, default=0)
    parser.add_argument('--retry-delay-s', type=float, default=0)
    args = parser.parse_args(argv)
    if sum(bool(value) for value in (args.prepare_paper_baseline, args.run_paper_baseline, args.bundle_paper_baseline)) != 1:
        parser.error('choose preparation, bundling or execution')
    if not args.run_paper_baseline and (args.retry_race_width != 1 or args.max_stage_requeues or args.retry_delay_s):
        parser.error('retry controls require --run-paper-baseline')
    if args.bundle_paper_baseline:
        if (not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in ('.', '..')
                or args.source_root or args.input_map or args.query_indices is not None or args.provider_config or args.env_file):
            parser.error('bundling requires a new --run-name and no preparation or provider options')
        manifest = json.loads((args.bundle_paper_baseline/'experiment_manifest.json').read_text())
        if manifest.get('migration', {}).get('version') != REVISION:
            raise ValueError('bundle only a newly prepared paper-baseline batch')
        with (args.bundle_paper_baseline/'.execution.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = Path('outputs/predictions/branches')/prompt.BUNDLE/args.run_name
            bundle(args.bundle_paper_baseline, target)
        print(target)
        return 0
    if args.prepare_paper_baseline:
        if (bool(args.source_root) == bool(args.input_map) or not args.run_name
                or Path(args.run_name).name != args.run_name or args.run_name in ('.', '..')):
            parser.error('preparation requires one of --source-root/--input-map and a single-component --run-name')
        if args.provider_config or args.env_file:
            parser.error('preparation never constructs a model client')
        target = Path('outputs/predictions/branches')/prompt.BUNDLE/args.run_name
    else:
        if not args.provider_config or args.source_root or args.input_map or args.run_name or args.query_indices is not None:
            parser.error('execution requires --provider-config and an already prepared batch')
        target = args.run_paper_baseline
        manifest = json.loads((target/'experiment_manifest.json').read_text())
        if manifest.get('migration', {}).get('version') != REVISION:
            raise ValueError('execute only a newly prepared paper-baseline batch')
    target.mkdir(parents=True, exist_ok=True)
    with (target/'.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_paper_baseline:
            print(json.dumps(prepare(args.source_root, args.prepare_paper_baseline, target, args.query_indices,
                                     input_map=args.input_map), indent=2))
            return 0
        return execute_batch(target, args.provider_config, args.env_file, args.parallelism,
            retry_race_width=args.retry_race_width, max_stage_requeues=args.max_stage_requeues, retry_delay_s=args.retry_delay_s)


def validate_batch(root):
    """Validate frozen inputs and checkpoints without constructing a model client."""
    manifest = json.loads((root/'experiment_manifest.json').read_text())
    if manifest.get('migration', {}).get('version') != REVISION:
        raise ValueError('execute only a newly prepared paper-baseline batch')
    source = root / manifest['migration']['source_batch']
    if sha256_file(source/'experiment_manifest.json') != manifest['migration']['source_manifest_sha256']:
        raise ValueError('frozen source manifest changed')
    original = json.loads((source/'experiment_manifest.json').read_text())
    for key in original:
        if key != 'code' and manifest.get(key) != original[key]:
            raise ValueError(f'frozen paper contract changed: {key}')
    if set(manifest['migration']['query_indices']) != set(manifest['inputs']):
        raise ValueError('prepared task set differs from the frozen inputs')
    queries = [(task, i, root/'runs'/task/f'query_{i:05d}')
               for task, indices in manifest['migration']['query_indices'].items() for i in indices]
    cohorts = {}
    for task, spec in manifest['inputs'].items():
        path = (root / manifest['migration']['evaluation_inputs'][task] if 'evaluation_inputs' in manifest['migration']
                else Path(manifest['migration']['source_root']) / spec['input']['path'])
        if sha256_file(path) != spec['input']['sha256']:
            raise ValueError('frozen evaluation cohort changed')
        cohorts[task] = read_jsonl(path)
        indices = manifest['migration']['query_indices'][task]
        if (not indices or len(indices) != len(set(indices))
                or any(type(i) is not int or i < 0 or i >= len(cohorts[task]) for i in indices)):
            raise ValueError('invalid or repeated query indices')
    # Validate every request before constructing a network client or dispatching any call.
    for task, i, directory in queries:
        prepared = json.loads((directory/'prepared.json').read_text())
        prompt.verify_request(prepared, manifest)
        if prepared['prompt_payload']['task_definition']['task'] != task or not prepared.get('context_ready'):
            raise ValueError('prepared task or readiness mismatch')
        row = cohorts[task][i]
        query = {'benchmark_profile': manifest['benchmark_profile'], 'canonical_smiles': row['drug'],
                 'condition_group': row.get('condition_group', '')}
        frozen = json.loads((source/'runs'/task/f'query_{i:05d}'/'prepared.json').read_text())
        prompt.verify_request(frozen, original)
        if (prepared['query_hash'] != prompt.content_hash(query)
                or json.loads((directory/'evaluation.json').read_text())['label'] != int(row['Y'])
                or {k: v for k, v in prepared['prompt_payload'].items() if k != 'neighbors'}
                != {k: v for k, v in frozen['prompt_payload'].items() if k != 'neighbors'}):
            raise ValueError('frozen query, prior or evaluation binding changed')
        if (directory/'output.json').exists():
            saved = json.loads((directory/'output.json').read_text())
            if saved.get('request_hash') != prepared['request_hash']:
                raise ValueError('saved output belongs to a different request')
            if saved.get('status') == 'ok':
                prompt.verify_prepared(prepared, manifest, saved)
    return manifest, queries


def bundle(root, target):
    """Export a fresh execution batch with its small, hash-bound source inputs."""
    root, target = Path(root).resolve(), Path(target).resolve()
    if target.exists() or root in target.parents:
        raise ValueError('bundle requires a new, separate output directory')
    manifest, queries = validate_batch(root)
    migration = manifest['migration']
    source = root / migration['source_batch']
    portable = deepcopy(manifest)
    portable['migration'].update(source_batch='source', evaluation_inputs={
        task: f'source/inputs/{task}.jsonl' for task in manifest['inputs']})
    portable['migration'].pop('source_root', None)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix='.paper-bundle-') as staging:
        stage = Path(staging)/'batch'
        (stage/'source').mkdir(parents=True)
        shutil.copyfile(source/'experiment_manifest.json', stage/'source/experiment_manifest.json')
        for task, spec in manifest['inputs'].items():
            path = (root / migration['evaluation_inputs'][task] if 'evaluation_inputs' in migration
                    else Path(migration['source_root']) / spec['input']['path'])
            destination = stage / portable['migration']['evaluation_inputs'][task]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        write_json_atomic(stage/'experiment_manifest.json', portable)
        for task, i, directory in queries:
            relative = Path('runs')/task/f'query_{i:05d}'
            (stage/relative).mkdir(parents=True)
            (stage/'source'/relative).mkdir(parents=True)
            shutil.copyfile(source/relative/'prepared.json', stage/'source'/relative/'prepared.json')
            for name in ('evaluation.json', 'selection.json'):
                if (directory/name).exists():
                    shutil.copyfile(directory/name, stage/relative/name)
            prepared = json.loads((directory/'prepared.json').read_text())
            prepared['manifest_hash'] = prompt.content_hash(portable)
            write_json_atomic(stage/relative/'prepared.json', prepared)
        if (root/'preparation_comparison.json').exists():
            shutil.copyfile(root/'preparation_comparison.json', stage/'preparation_comparison.json')
        validate_batch(stage)
        stage.rename(target)


def execute_batch(root, provider_config, env_file, parallelism, *, retry_race_width=1, max_stage_requeues=0, retry_delay_s=0):
    """Execute frozen generation settings through the selected transport and bounded recovery."""
    from predict.api_client.pool import load_provider_pool_config, build_provider_pool, validate_parallelism, ProviderPoolExhausted
    from sklearn.metrics import accuracy_score, f1_score

    manifest, queries = validate_batch(root)
    config = load_provider_pool_config(provider_config)
    validate_parallelism(config, parallelism)
    if not 1 <= retry_race_width <= parallelism or max_stage_requeues < 0 or not math.isfinite(retry_delay_s) or retry_delay_s < 0:
        raise ValueError('invalid retry width, requeue count or delay')
    is_batch = config.providers[0].transport == 'openrouter_batch'
    if is_batch and retry_race_width != 1:
        raise ValueError('Batch requires retry-race-width 1; paid remote jobs cannot be raced')
    if retry_race_width > 1 and config.spend_budget is not None:
        raise ValueError('retry races cannot use the synchronous spend ledger')
    if any(p.model != manifest['model'] or dict(p.request_extra_body or {}) != manifest['request_extra_body'] for p in config.providers):
        raise ValueError('provider model or extra generation parameters differ from the frozen experiment')
    generation = manifest['generation']
    if generation['thinking'] != 'provider_default':
        raise ValueError('unsupported thinking contract')
    client = build_provider_pool(config, env_file=env_file, timeout_s=240, max_tokens=manifest['max_tokens'],
        temperature=generation['temperature'], reasoning_effort='' if generation['reasoning_effort'] == 'omitted' else generation['reasoning_effort'],
        enable_thinking=False, tool_service_url='', enable_group_tools=False, max_tool_rounds=0, transport_max_retries=0,
        token_limit_parameter='max_tokens', batch_dir=root/'batch_transport')
    race = None
    try:
        if retry_race_width > 1:
            from predict.api_client.retry import ParallelRetryClient
            race = ParallelRetryClient(client, parallelism=parallelism,
                task_limits={task: parallelism for task in manifest['migration']['query_indices']})
        write_json_atomic(root/'execution_provider.json', config.public_dict())
        write_json_atomic(root/'execution_options.json', dict(retry_race_width=retry_race_width,
            max_stage_requeues=max_stage_requeues, retry_delay_s=retry_delay_s, parallelism=parallelism))
        if is_batch:
            for _, _, directory in queries:
                saved = json.loads((directory/'output.json').read_text()) if (directory/'output.json').exists() else {}
                if saved.get('status') != 'ok':
                    client.prefetch_json(json.loads((directory/'prepared.json').read_text())['messages'])
        def run_query(query):
            task, _, directory = query
            prepared = json.loads((directory/'prepared.json').read_text())
            validation = prompt.validation_kwargs(prepared['prompt_payload'], prepared['card_alias_map'])
            calls = 0
            retry_pending = (directory/'output.json').exists()

            def call(messages):
                nonlocal calls
                calls += 1
                if race is None:
                    return client.chat_json(messages)
                return race.chat_validated(messages, task=task,
                    width=retry_race_width if calls > 1 or retry_pending else 1,
                    validate=lambda response: response_validation_errors(response, **validation),
                    receipt_path=directory/'retry_races'/f'{time.time_ns()}.json')

            for attempt in range(max_stage_requeues + 1):
                try:
                    result = execute_query(directory, manifest, call)
                except ProviderPoolExhausted as exc:
                    result = dict(status='error', model_called=True, request_hash=prepared['request_hash'],
                        llm={'execution_provider_attempts': exc.attempts}, card_alias_map=prepared['card_alias_map'])
                    write_json_atomic(directory/'output.json', result)
                    if not max_stage_requeues:
                        raise
                if result['status'] == 'ok' or attempt == max_stage_requeues:
                    return result
                write_json_atomic(directory/'stage_requeues'/f'{time.time_ns()}.json',
                    dict(attempt=attempt+1, result=result, delay_seconds=retry_delay_s))
                time.sleep(retry_delay_s)

        with ThreadPoolExecutor(max_workers=parallelism) as workers:
            try:
                results = list(workers.map(run_query, queries))
            except BaseException:
                if config.providers[0].transport == 'openrouter_batch':
                    client.close()  # Release local waiters; paid remote jobs remain in the journal.
                raise
    finally:
        race.close() if race is not None else client.close()
    failures = 0
    for task in manifest['migration']['query_indices']:
        rows = []
        for (query_task, i, directory), result in zip(queries, results, strict=True):
            if query_task != task:
                continue
            prepared = json.loads((directory/'prepared.json').read_text())
            label = json.loads((directory/'evaluation.json').read_text())['label']
            prediction = prompt.verify_prepared(prepared, manifest, result) if result['status'] == 'ok' else None
            rows.append({'index': i, 'label': label, 'pred_label': prediction, 'status': result['status']})
        successful = [r for r in rows if r['status'] == 'ok']
        metrics = {'n_total': len(rows), 'n_successful': len(successful), 'n_failed_runs': len(rows)-len(successful)}
        if successful:
            labels, predictions = ([r[k] for r in successful] for k in ('label', 'pred_label'))
            metrics.update(accuracy=accuracy_score(labels, predictions),
                           macro_f1=f1_score(labels, predictions, labels=[0, 1], average='macro', zero_division=0))
        failures += metrics['n_failed_runs']
        write_jsonl_atomic(root/'summary'/task/'predictions.jsonl', rows)
        write_json_atomic(root/'summary'/task/'metrics.json', metrics)
    return int(bool(failures))
