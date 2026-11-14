"""MolRAG: frozen direct references, Sim-CoT, and shared model transport."""

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import threading
import time

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from predict.api_client.pool import (DEEPSEEK_V4_ALIASES, ProviderPoolExhausted,
                                    build_provider_pool, canonical_provider_model, load_provider_pool_config,
                                    validate_parallelism)
from predict.api_client.transport import DEFAULT_ENV_FILE
from predict.baselines.conditioned_knn import NO_REPORTED_CONDITION
from predict.harnesses.branches.manifests import (prediction_batch_directory,
    prediction_run_root, read_record, write_batch)
from predict.prompts import bundle_directory, template_environment
from predict.prompts.paper_baseline import profiles
from predict.retrieval.policies import (IDENTITY_NORMALIZER_VERSION, RELATION_POLICY_VERSION,
    decide_candidate, normalize_molecule_identity)
from predict.utils.json import (canonical_json_bytes, read_json, read_jsonl, sha256_file,
                                write_json_atomic, write_jsonl_atomic)

ROOT = Path(__file__).resolve().parents[2]
PROMPT = 'molrag_evidence_v1'
MODEL = 'deepseek/deepseek-v4-flash-0731'
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def model_matches(served, requested):
    return served == requested or (requested in DEEPSEEK_V4_ALIASES and served in DEEPSEEK_V4_ALIASES)


def condition(row):
    return row.get('condition_group') or NO_REPORTED_CONDITION


def generation_settings(model):
    model = MODEL if model in DEEPSEEK_V4_ALIASES else model
    limits = {MODEL: 16384, 'openai/gpt-6-luna': 8192, 'google/gemini-3.8-flash': 16384}
    if model not in limits:
        raise ValueError('Unsupported MolRAG backbone')
    return dict(temperature='provider_default' if model == 'openai/gpt-6-luna' else 0,
                max_tokens=limits[model], reasoning_effort='low', response_format='text')


def select_examples(query, reference, fingerprints):
    identity = normalize_molecule_identity(query['drug'])
    if identity.status != 'ok':
        raise ValueError(f"Unresolved query: {query['benchmark_row_id']}")
    fp = GENERATOR.GetFingerprint(Chem.MolFromSmiles(query['drug']))
    similarities = DataStructs.BulkDiceSimilarity(fp, fingerprints)
    selected, seen = [], set()
    for i in sorted(range(len(reference)), key=lambda i: (-similarities[i], i)):
        row = reference[i]
        if row['canonical_smiles'] in seen or decide_candidate(identity, row, 'scaffold_disjoint').excluded:
            continue
        selected.append({k: row[k] for k in ('reference_id', 'smiles', 'card')})
        selected[-1]['dice'] = float(similarities[i])
        seen.add(row['canonical_smiles'])
        if len(selected) == 4:
            return selected
    raise ValueError('MolRAG requires four eligible distinct reference molecules')


def render(query, examples, task):
    env = template_environment()
    return [dict(role='system', content=env.get_template(f'{PROMPT}/system.jinja').render().strip()),
            dict(role='user', content=env.get_template(f'{PROMPT}/user.jinja').render(
                task=task, smiles=query['drug'], condition=condition(query), examples=examples).strip())]


def parse_answer(text):
    marker = r'Question\s+Answer["\']?\s*:\s*["\'<*\s]*(YES|NO)\b'
    # Some providers wrap the final answer in JSON, occasionally malformed JSON.
    suffix = r'["\'>*\s]*\.?\s*'
    if text.lstrip().startswith('{'):
        suffix += r'(?:[}\]]\s*)*'
    match = re.search(marker + suffix + r'$', text, re.IGNORECASE)
    answers = {answer.upper() for answer in re.findall(marker, text, re.IGNORECASE)}
    if not match or len(answers) != 1:
        raise ValueError('Missing terminal Question Answer: YES/NO or conflicting answers')
    return int(match[1].upper() == 'YES')


def prepare(args):
    library = read_json(args.reference_library)
    root = prediction_run_root(args.output_root, 'molrag', PROMPT, args.run_name)
    if root.exists():
        raise ValueError('Preparation requires a new run name; use run/verify to resume')
    source = read_json(args.sources)
    provider = read_json(args.provider_config)
    config = load_provider_pool_config(args.provider_config)
    model = canonical_provider_model(config)
    generation = generation_settings(model)
    if config.max_failovers or any(p.base_url.rstrip('/') != 'https://openrouter.ai/api/v1'
            or p.transport != 'chat_completions'
            or (p.request_extra_body or {}).get('reasoning_effort_override', 'low') != 'low'
            for p in config.providers):
        raise ValueError('MolRAG requires low-effort OpenRouter chat completions without automatic failovers')
    datasets = [d for d in source['datasets'] if (not args.task or d['task'] == args.task)
                and (not args.benchmark or d['benchmark'] == args.benchmark)]
    if not datasets:
        raise ValueError('Choose a nonempty dataset suite')
    loaded = []
    for dataset in datasets:
        splits = {}
        for name, pin in dataset['splits'].items():
            path = args.source_root / pin['path']
            if sha256_file(path) != pin['sha256']:
                raise ValueError(f'Frozen source changed: {path}')
            rows = read_jsonl(path)
            ids = [r['benchmark_row_id'] for r in rows]
            if (not rows or not all(isinstance(uid, str) and uid for uid in ids) or len(ids) != len(set(ids))
                    or any(type(r['Y']) is not int or r['Y'] not in (0, 1) for r in rows)):
                raise ValueError(f'Invalid frozen rows: {path}')
            splits[name] = rows
        loaded.append((dataset, splits))
    root.mkdir(parents=True)
    shutil.copyfile(args.provider_config, root / 'provider.json')
    shutil.copyfile(args.sources, root / 'sources.json')
    prompt_hashes = {name: sha256_file(bundle_directory(PROMPT) / name)
                     for name in ('system.jinja', 'user.jinja')}
    paths = []
    for dataset, splits in loaded:
        benchmark, task = dataset['benchmark'], dataset['task']
        batch = prediction_batch_directory(root, benchmark, task, 'test', 'sim_cot')
        batch.mkdir(parents=True)
        queries = splits['test']
        fields = profiles()['conditioned'][task]['task']
        if benchmark == 'tdc':
            fields = profiles()['tdc_overrides'].get(task, {'task': fields})['task']
        fields = {k: fields[k] for k in ('endpoint_name', 'positive_prediction', 'negative_prediction', 'label_scope')}
        reference_receipt = next(d for d in library['datasets'] if d['benchmark'] == benchmark and d['task'] == task)
        reference_path = args.reference_library.parent / reference_receipt['path']
        if sha256_file(reference_path) != reference_receipt['sha256']:
            raise ValueError('Frozen evidence reference pool changed')
        original_path = args.source_root / reference_receipt['source']['batch'] / 'experiment_manifest.json'
        source_inputs = read_json(original_path)['inputs'][task]
        if (sha256_file(original_path) != reference_receipt['source_manifest_sha256']
                or source_inputs['input']['sha256'] != dataset['splits']['test']['sha256']
                or (benchmark == 'tdc' and source_inputs['train']['sha256'] != dataset['splits']['train']['sha256'])):
            raise ValueError('Evidence source belongs to different query or training cohorts')
        reference = read_jsonl(reference_path)
        (batch / 'inputs').mkdir()
        shutil.copyfile(reference_path, batch / 'inputs/references.jsonl')
        fps = [GENERATOR.GetFingerprint(Chem.MolFromSmiles(r['smiles'])) for r in reference]
        requests = []
        for index, query in enumerate(queries):
            examples = select_examples(query, reference, fps)
            messages = render(query, examples, fields)
            requests.append(dict(index=index, uid=query['benchmark_row_id'], examples=examples,
                                 messages=messages, request_hash=digest(messages)))
        settings = dict(protocol='molrag_direct_evidence.v1', benchmark=benchmark, task=task,
            subset='test', source_release=source['release'], prompt=PROMPT, model=model,
            method='Sim-CoT', direct_budget=4, indirect_budget=0, query_prior='none',
            condition_policy='dice_only_source_conditions_visible', identity_policy='scaffold_disjoint',
            normalizer=IDENTITY_NORMALIZER_VERSION, relation_policy=RELATION_POLICY_VERSION,
            representation='full_molecule', molecular_fingerprint=dict(type='Morgan bit', radius=2, bits=2048, chirality=False),
            similarity='Dice', tie_break='canonical_smiles', selection='top4',
            reference_release=library['release'], duplicate_policy=library['record_policy'], records_per_molecule=1,
            max_attempts=2, retry_token_exhaustion=False, task_definition=fields, generation=generation,
            provider_route=[dict(model=p.model, transport=p.transport, request_extra_body=p.request_extra_body)
                            for p in config.providers])
        artifacts = dict(references=dict(kind='evidence', name=f'{benchmark}_{task}_direct_references',
                                        version=library['release'], local_format='.jsonl'))
        for name in ('train', 'test'):
            shutil.copyfile(args.source_root / dataset['splits'][name]['path'], batch / 'inputs' / f'{name}.jsonl')
            artifacts[name] = dict(kind='benchmark', name=f'{benchmark}_{task}_{name}',
                                   version=source['release'], local_format='.jsonl')
        write_batch(batch / 'manifest.json', {**settings, 'run_name': args.run_name, 'artifacts': artifacts},
                    records=list(enumerate(queries)))
        write_jsonl_atomic(batch / 'requests.jsonl', requests)
        write_json_atomic(batch / 'receipt.json', dict(source=dataset, provider_sha256=digest(provider),
            prompt_sha256=prompt_hashes, requests_sha256=sha256_file(batch / 'requests.jsonl'),
            manifest_sha256=sha256_file(batch / 'manifest.json'), reference_library=reference_receipt,
            rdkit=rdBase.rdkitVersion, shots=dict(Counter(len(r['examples']) for r in requests))))
        paths.append(str(batch.relative_to(root)))
        print(json.dumps(dict(prepared=f'{benchmark}/{task}', queries=len(queries))), flush=True)
    write_json_atomic(root / 'suite.json', dict(batches=paths))
    print(root, flush=True)
    return root


def validate(root):
    """Validate every frozen input, ordered selection and request before any model call."""
    entries = []
    provider = read_json(root / 'provider.json')
    config = load_provider_pool_config(root / 'provider.json')
    model = canonical_provider_model(config)
    for path in read_json(root / 'suite.json')['batches']:
        batch = root / path
        manifest = read_record(batch / 'manifest.json')
        receipt = read_json(batch / 'receipt.json')
        if sha256_file(batch / 'manifest.json') != receipt['manifest_sha256']:
            raise ValueError('Frozen batch contract changed')
        prompt = manifest['prompt']
        if (prompt != PROMPT or manifest['protocol'] != 'molrag_direct_evidence.v1'
                or manifest['subset'] != 'test' or not model_matches(manifest['model'], model)
                or manifest['generation'] != generation_settings(model)):
            raise ValueError('Incompatible MolRAG protocol/model')
        if receipt['provider_sha256'] != digest(provider):
            raise ValueError('Provider configuration changed')
        generation = manifest['generation']
        if any((p.request_extra_body or {}).get('reasoning_effort_override', generation['reasoning_effort'])
               != generation['reasoning_effort'] for p in config.providers):
            raise ValueError('Provider effort differs from the frozen generation settings')
        for name, expected in receipt['prompt_sha256'].items():
            if sha256_file(bundle_directory(prompt) / name) != expected:
                raise ValueError('Prompt changed; use a new run identity')
        for name in ('train', manifest['subset']):
            if sha256_file(batch / 'inputs' / f'{name}.jsonl') != receipt['source']['splits'][name]['sha256']:
                raise ValueError('Frozen split changed')
        if sha256_file(batch / 'requests.jsonl') != receipt['requests_sha256']:
            raise ValueError('Frozen requests changed')
        queries = read_jsonl(batch / 'inputs' / f"{manifest['subset']}.jsonl")
        requests = read_jsonl(batch / 'requests.jsonl')
        if len(requests) != len(queries) or list(manifest['queries'].values()) != [q['benchmark_row_id'] for q in queries]:
            raise ValueError('Query membership changed')
        if sha256_file(batch / 'inputs/references.jsonl') != receipt['reference_library']['sha256']:
            raise ValueError('Frozen evidence reference pool changed')
        reference = read_jsonl(batch / 'inputs/references.jsonl')
        canonical = [r['canonical_smiles'] for r in reference]
        if canonical != sorted(set(canonical)) or any(
                r['card']['evidence_family_level'] != 1 or
                r['smiles'] != r['molecule_identity']['canonical_smiles'] or
                digest(normalize_molecule_identity(r['molecule_identity']['input_smiles']).to_dict()) != digest(r['molecule_identity'])
                for r in reference):
            raise ValueError('Invalid direct evidence identity, order or membership')
        fps = [GENERATOR.GetFingerprint(Chem.MolFromSmiles(r['smiles'])) for r in reference]
        for index, (query, request) in enumerate(zip(queries, requests, strict=True)):
            selected = select_examples(query, reference, fps)
            messages = render(query, selected, manifest['task_definition'])
            if (request['index'] != index or request['uid'] != query['benchmark_row_id']
                    or request['examples'] != selected or request['messages'] != messages
                    or request['request_hash'] != digest(messages)):
                raise ValueError('Selection, Dice score, label or rendered request changed')
            directory = batch / 'runs' / f'query_{index:05d}'
            artifacts = list(directory.glob('attempt_*.json'))
            if (directory / 'output.json').exists():
                artifacts.append(directory / 'output.json')
            for artifact in artifacts:
                saved = read_json(artifact)
                if saved['request_hash'] != request['request_hash'] or saved['uid'] != request['uid']:
                    raise ValueError('Saved attempt belongs to another request')
                if saved['status'] == 'ok' and (saved['prediction'] != parse_answer(saved['llm']['raw_content'])
                        or not model_matches(saved['llm'].get('model'), manifest['model'])):
                    raise ValueError('Saved prediction is incompatible with the frozen request')
            if (directory / 'stage_resume.json').exists() and read_json(directory / 'stage_resume.json')['request_hash'] != request['request_hash']:
                raise ValueError('Resume state belongs to another request')
            entries.append((batch, manifest, query, request, directory))
        print(f'validated {path}: {len(queries)}', flush=True)
    return entries


def repeat(source, target, provider_config):
    """Copy validated frozen inputs into a new run; never inherit attempts."""
    if target.exists():
        raise ValueError('Repetition requires a new output directory')
    entries = validate(source)
    provider = read_json(provider_config)
    config = load_provider_pool_config(provider_config)
    if (not model_matches(canonical_provider_model(config), entries[0][1]['model']) or config.max_failovers
            or any(p.transport != 'chat_completions'
                   or (p.request_extra_body or {}).get('reasoning_effort_override', 'low') != 'low'
                   or (p.request_extra_body or {}).get('chat_template_kwargs', {}).get('reasoning_effort', 'low') != 'low'
                   for p in config.providers)):
        raise ValueError('Repetition requires the same model and low-effort chat completions without failovers')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix='.molrag-repeat-') as staging:
        stage = Path(staging)/'batch'
        stage.mkdir()
        shutil.copyfile(provider_config, stage/'provider.json')
        for name in ('sources.json', 'suite.json'):
            shutil.copyfile(source/name, stage/name)
        for relative in read_json(source/'suite.json')['batches']:
            old, batch = source/relative, stage/relative
            document = read_json(old/'manifest.json')
            shutil.copytree(old/'inputs', batch/'inputs')
            shutil.copyfile(old/'requests.jsonl', batch/'requests.jsonl')
            settings = dict(document['settings'], run_name=target.name, artifacts=document['artifacts'],
                source_run_uids=[document['run_uid']],
                provider_route=[dict(model=p.model, transport=p.transport, request_extra_body=p.request_extra_body)
                                for p in config.providers])
            write_batch(batch/'manifest.json', settings,
                        records=list(enumerate(read_jsonl(batch/'inputs/test.jsonl'))))
            receipt = dict(read_json(old/'receipt.json'), provider_sha256=digest(provider),
                manifest_sha256=sha256_file(batch/'manifest.json'),
                repetition_source=dict(run_uid=document['run_uid'], manifest_sha256=sha256_file(old/'manifest.json')))
            write_json_atomic(batch/'receipt.json', receipt)
        validate(stage)
        stage.rename(target)


def execute_query(entry, client, stop, *, attempt_limit=None):
    batch, manifest, query, request, directory = entry
    output = directory / 'output.json'
    if output.exists():
        return 'reused'
    # Recover a completed response if interruption happened before output publication.
    for path in sorted(directory.glob('attempt_*.json')):
        saved = read_json(path)
        if saved['uid'] != request['uid'] or saved['request_hash'] != request['request_hash']:
            raise ValueError('Saved attempt belongs to a different request')
        llm = saved.get('llm', {})
        if not model_matches(llm.get('model'), manifest['model']):
            if saved['status'] == 'ok':
                raise ValueError('Saved successful attempt has the wrong model')
            continue
        try:
            prediction = parse_answer(llm['raw_content'])
        except ValueError:
            continue
        if saved['status'] == 'ok' and saved['prediction'] != prediction:
            raise ValueError('Saved successful attempt is invalid')
        recovered = dict(saved, status='ok', prediction=prediction, recovered_from=path.name)
        recovered.pop('error', None)
        write_json_atomic(output, recovered)
        write_json_atomic(directory / 'stage_resume.json', dict(status='ok',
            recovered_from=path.name, request_hash=request['request_hash']))
        return 'recovered'
    if len(request['examples']) != 4:
        return 'insufficient_neighbors'
    directory.mkdir(parents=True, exist_ok=True)
    previous = len(list(directory.glob('attempt_*.json')))
    if previous and read_json(directory / f'attempt_{previous:04d}.json').get('retryable') is False:
        return 'error'
    limit = attempt_limit if attempt_limit is not None else manifest['max_attempts']
    remaining = max(0, limit - previous)
    for attempt in range(remaining):
        if stop.is_set():
            return 'cancelled_after_fatal_error'
        result = dict(uid=request['uid'], request_hash=request['request_hash'], started_at=time.time())
        fatal = False
        try:
            response = client.chat_json(request['messages'])
            result['llm'] = response
            if not model_matches(response.get('model'), manifest['model']):
                fatal = True
                raise ValueError('Unexpected served model; stopped before accepting predictions')
            result['prediction'] = parse_answer(response['raw_content'])
            result['status'] = 'ok'
        except ProviderPoolExhausted as error:
            result.update(status='error', execution_provider_attempts=error.attempts)
            fatal = any(a.get('status_code') in (400, 401, 402, 403, 404) for a in error.attempts)
        except ValueError as error:
            result.update(status='error', error=str(error))
        except RuntimeError as error:
            if 'spend budget' not in str(error) and 'reserved maximum' not in str(error):
                raise
            stop.set()
            return 'budget_stopped'
        if (result['status'] == 'error' and manifest.get('retry_token_exhaustion') is False
                and result.get('llm', {}).get('usage', {}).get('completion_tokens', 0)
                >= manifest['generation']['max_tokens'] - 32):
            result['retryable'] = False
        result['finished_at'] = time.time()
        write_json_atomic(directory / f'attempt_{previous + attempt + 1:04d}.json', result)
        write_json_atomic(directory / 'stage_resume.json', dict(status=result['status'],
            attempt=previous + attempt + 1, request_hash=request['request_hash']))
        if result['status'] == 'ok':
            write_json_atomic(output, result)
            return 'ok'
        if fatal:
            stop.set()
            return 'fatal_error'
        if result.get('retryable') is False:
            return 'error'
        if attempt + 1 < remaining:
            rate_limited = any(a.get('status_code') == 429 for a in result.get('execution_provider_attempts', []))
            time.sleep(60 if rate_limited else 2 ** attempt)
    return 'error'


def summarize(root, entries, *, require_complete=False, write=True):
    cells, usage = [], Counter()
    batches = list(dict.fromkeys(entry[0] for entry in entries))
    for batch in batches:
        rows = []
        for _, manifest, query, request, directory in (e for e in entries if e[0] == batch):
            row = dict(uid=request['uid'], label=query['Y'], condition=condition(query),
                       shots=len(request['examples']), prediction=None, status='pending')
            if len(request['examples']) != 4:
                row['status'] = 'insufficient_neighbors'
            if (directory / 'output.json').exists():
                saved = read_json(directory / 'output.json')
                row.update(prediction=parse_answer(saved['llm']['raw_content']), status='ok')
            elif (directory / 'stage_resume.json').exists():
                row['status'] = 'error'
            rows.append(row)
            for path in directory.glob('attempt_*.json'):
                attempt = read_json(path)
                usage['attempts'] += 1
                llm = attempt.get('llm', {})
                tokens = llm.get('usage', {})
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    usage[key] += tokens.get(key) or 0
                if isinstance(tokens.get('cost'), (int, float)):
                    usage['reported_cost_usd'] += tokens['cost']
                    usage['priced_attempts'] += 1
                else:
                    usage['unpriced_attempts'] += 1
        successful = [r for r in rows if r['status'] == 'ok']
        metrics = dict(benchmark=manifest['benchmark'], task=manifest['task'], n_total=len(rows),
                       n_successful=len(successful), coverage=len(successful) / len(rows),
                       shots=dict(Counter(r['shots'] for r in rows)),
                       statuses=dict(Counter(r['status'] for r in rows)), auroc=None,
                       auroc_reason='Paper YES/NO output has no continuous score')
        if successful:
            labels, predictions = ([r[key] for r in successful] for key in ('label', 'prediction'))
            metrics.update(macro_f1=float(f1_score(labels, predictions, labels=[0, 1], average='macro', zero_division=0)),
                           accuracy=float(accuracy_score(labels, predictions)),
                           confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist())
        metrics['conditions'] = []
        for group in sorted({r['condition'] for r in rows}):
            cohort = [r for r in rows if r['condition'] == group]
            valid = [r for r in cohort if r['status'] == 'ok']
            metrics['conditions'].append(dict(condition=group, n_total=len(cohort), n_successful=len(valid),
                macro_f1=float(f1_score([r['label'] for r in valid], [r['prediction'] for r in valid],
                                       labels=[0, 1], average='macro', zero_division=0)) if valid else None))
        if write:
            write_jsonl_atomic(batch / 'predictions.jsonl', rows)
            write_json_atomic(batch / 'metrics.json', metrics)
        cells.append(metrics)
    summary = dict(cells=cells, n_total=sum(c['n_total'] for c in cells),
                   n_successful=sum(c['n_successful'] for c in cells), usage=dict(usage))
    summary['complete'] = summary['n_total'] == summary['n_successful']
    summary['means'] = {benchmark: sum(c['macro_f1'] for c in cells if c['benchmark'] == benchmark) /
                        sum(c['benchmark'] == benchmark for c in cells)
                        for benchmark in {c['benchmark'] for c in cells}
                        if all(c['n_successful'] == c['n_total'] for c in cells if c['benchmark'] == benchmark)}
    if write:
        write_json_atomic(root / 'results.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cells'}), flush=True)
    if require_complete and not summary['complete']:
        raise ValueError('Run is incomplete; coverage is not a full-cohort result')
    return summary


def score_saved_runs(run_roots, output_dir):
    """Score frozen predictions offline; keep failure penalties separate from model answers."""
    cells, scored, sources = [], [], []
    for root in run_roots:
        for relative in read_json(root / 'suite.json')['batches']:
            batch = root / relative
            manifest = read_record(batch / 'manifest.json')
            if (manifest['prompt'] != PROMPT or manifest['protocol'] != 'molrag_direct_evidence.v1'
                    or manifest['subset'] != 'test'
                    or manifest['generation'] != generation_settings(manifest['model'])):
                raise ValueError('Only the selected MolRAG evidence/low-effort setting is supported')
            gold_path = batch / 'inputs' / f"{manifest['subset']}.jsonl"
            receipt = read_json(batch / 'receipt.json')
            if (sha256_file(batch / 'manifest.json') != receipt['manifest_sha256']
                    or sha256_file(batch / 'requests.jsonl') != receipt['requests_sha256']):
                raise ValueError('Frozen reparse manifest or requests changed')
            if sha256_file(gold_path) != receipt['source']['splits'][manifest['subset']]['sha256']:
                raise ValueError('Frozen scoring labels changed')
            rows, gold = read_jsonl(batch / 'predictions.jsonl'), read_jsonl(gold_path)
            requests = read_jsonl(batch / 'requests.jsonl')
            if not rows or len({r['uid'] for r in rows}) != len(rows):
                raise ValueError('Empty or duplicate scoring queries')
            labels, predictions, failed, recovered = [], [], 0, 0
            identity = dict(run_name=root.name, model=manifest['model'], benchmark=manifest['benchmark'],
                            task=manifest['task'], subset=manifest['subset'])
            for index, (row, query, request) in enumerate(zip(rows, gold, requests, strict=True)):
                if row['uid'] != query['benchmark_row_id'] or row['label'] != query['Y'] or type(row['label']) is not int or row['label'] not in (0, 1):
                    raise ValueError('Scoring query or gold label mismatch')
                penalty = row['status'] != 'ok'
                recovery = {}
                if row['status'] == 'error' and row['prediction'] is None:
                    if request['uid'] != row['uid']:
                        raise ValueError('Reparse request identity mismatch')
                    for path in sorted((batch / 'runs' / f'query_{index:05d}').glob('attempt_*.json')):
                        attempt = read_json(path)
                        if attempt['uid'] != row['uid'] or attempt['request_hash'] != request['request_hash']:
                            raise ValueError('Reparse attempt belongs to a different request')
                        llm = attempt.get('llm', {})
                        if not model_matches(llm.get('model'), manifest['model']):
                            continue
                        if llm.get('messages', [])[:len(request['messages'])] != request['messages']:
                            raise ValueError('Saved response messages differ from the frozen request')
                        try:
                            prediction = parse_answer(llm.get('raw_content', ''))
                        except ValueError:
                            continue
                        recovery = dict(reparsed_prediction=prediction,
                            reparsed_attempt=str(path.relative_to(root)), reparsed_attempt_sha256=sha256_file(path))
                        penalty = False
                        recovered += 1
                        break
                if penalty:
                    if row['status'] != 'error' or row['prediction'] is not None:
                        raise ValueError('Only explicit failed predictions may be counted wrong')
                    prediction = 1 - row['label']
                    failed += 1
                elif not recovery:
                    prediction = row['prediction']
                    if type(prediction) is not int or prediction not in (0, 1):
                        raise ValueError('Invalid saved prediction')
                labels.append(row['label'])
                predictions.append(prediction)
                scored.append({**identity, **row, **recovery, 'scoring_prediction': prediction,
                               'failure_counted_wrong': penalty})
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            cells.append(dict(**identity, n_total=len(rows), n_model_predictions=len(rows) - failed,
                n_failures_counted_wrong=failed,
                macro_f1=float(f1_score(labels, predictions, labels=[0, 1], average='macro', zero_division=0)),
                accuracy=float(accuracy_score(labels, predictions)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
            cells[-1]['n_reparsed_answers'] = recovered
            source_files = [batch / 'manifest.json', batch / 'predictions.jsonl', gold_path, batch / 'requests.jsonl']
            sources.append(dict(run_root=str(root.resolve()), batch=relative, run_uid=manifest['run_uid'],
                sha256={str(p.relative_to(root)): sha256_file(p) for p in source_files}))
    means = []
    for run, benchmark in dict.fromkeys((c['run_name'], c['benchmark']) for c in cells):
        group = [c for c in cells if c['run_name'] == run and c['benchmark'] == benchmark]
        if len(group) != 6 or len({c['task'] for c in group}) != 6:
            raise ValueError('Six-task means require six distinct task cells')
        means.append(dict(run_name=run, model=group[0]['model'], benchmark=benchmark,
            n_total=sum(c['n_total'] for c in group),
            n_model_predictions=sum(c['n_model_predictions'] for c in group),
            n_failures_counted_wrong=sum(c['n_failures_counted_wrong'] for c in group),
            macro_f1=sum(c['macro_f1'] for c in group) / 6,
            accuracy=sum(c['accuracy'] for c in group) / 6))
    output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version='molrag_performance.v1',
        failure_policy='opposite_gold_for_scoring_only',
        mean_weighting='equal_task', cells=cells, means=means, sources=sources)
    result['reparsing'] = dict(policy='first_parseable_saved_attempt_for_failed_queries',
                              implementation_sha256=sha256_file(Path(__file__)))
    write_json_atomic(output_dir / 'results.json', result)
    write_jsonl_atomic(output_dir / 'scored_predictions.jsonl', scored)
    for name, rows in (('scores.csv', cells), ('means.csv', means)):
        with (output_dir / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    scoring = sub.add_parser('score', help='Offline final metrics from frozen saved predictions')
    scoring.add_argument('--run-root', type=Path, nargs='+', required=True)
    scoring.add_argument('--output-dir', type=Path, required=True)
    repetition = sub.add_parser('repeat', help='New inference run from exact frozen inputs')
    repetition.add_argument('--run-root', type=Path, required=True)
    repetition.add_argument('--run-name', required=True)
    repetition.add_argument('--output-root', type=Path, default=ROOT/'outputs/predictions')
    repetition.add_argument('--provider-config', type=Path, required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source-root', type=Path, required=True)
    prep.add_argument('--sources', type=Path, default=ROOT / 'experiments/baselines/tinker_sft_paper_sources.json')
    prep.add_argument('--provider-config', type=Path, required=True)
    prep.add_argument('--output-root', type=Path, default=ROOT / 'outputs/predictions')
    prep.add_argument('--run-name', required=True)
    prep.add_argument('--task')
    prep.add_argument('--benchmark', choices=('tdc', 'conditioned'))
    prep.add_argument('--reference-library', type=Path, required=True, help='Frozen naive direct pool manifest')
    for command in ('run', 'verify'):
        p = sub.add_parser(command)
        p.add_argument('--run-root', type=Path, required=True)
        if command == 'run':
            p.add_argument('--env-file', type=Path, default=DEFAULT_ENV_FILE)
            p.add_argument('--provider-config', type=Path, help='Reject resume if this requested provider configuration differs')
            p.add_argument('--credential-env', help='Explicit credential alias; saved for subsequent resumes')
            p.add_argument('--parallelism', type=int, default=32)
            p.add_argument('--provider-max-inflight', type=int,
                           help='Override runtime capacity per provider without changing frozen requests or routing')
            p.add_argument('--min-request-interval-ms', type=int, default=0,
                           help='Increase per-provider dispatch spacing without changing model inputs')
            p.add_argument('--attempt-cap', type=int,
                           help='Lower the saved-attempt limit on resume; never increases the frozen allowance')
        else:
            p.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.command == 'repeat':
        target = prediction_run_root(args.output_root, 'molrag', PROMPT, args.run_name)
        repeat(args.run_root, target, args.provider_config)
        print(target, flush=True)
        return
    if args.command == 'score':
        result = score_saved_runs(args.run_root, args.output_dir)
        print(json.dumps(result['means']), flush=True)
        return
    if args.command == 'prepare':
        prepare(args)
        return
    root = args.run_root.resolve()
    if args.command == 'run' and args.provider_config and read_json(args.provider_config) != read_json(root / 'provider.json'):
        raise ValueError('Requested provider configuration differs; use a new run identity')
    with (root / '.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entries = validate(root)
        if args.command == 'verify':
            summarize(root, entries, require_complete=args.require_complete, write=False)
            return
        jobs = [(e, e[1]['max_attempts']) for e in entries]
        if args.attempt_cap is not None:
            if args.attempt_cap < 1:
                raise ValueError('Attempt cap must be positive')
            jobs = [(e, min(limit, args.attempt_cap)) for e, limit in jobs]
        config = load_provider_pool_config(root / 'provider.json')
        if args.provider_max_inflight is not None:
            if args.provider_max_inflight < 1:
                raise ValueError('Provider capacity must be positive')
            config = replace(config, providers=tuple(replace(p, max_inflight=args.provider_max_inflight)
                                                     for p in config.providers))
        if args.min_request_interval_ms < 0:
            raise ValueError('Request interval must be nonnegative')
        if args.min_request_interval_ms:
            config = replace(config, providers=tuple(replace(p, request_extra_body={
                **(p.request_extra_body or {}), 'min_request_interval_ms': max(args.min_request_interval_ms,
                    (p.request_extra_body or {}).get('min_request_interval_ms', 0))}) for p in config.providers))
        sessions = read_json(root / 'execution_sessions.json') if (root / 'execution_sessions.json').exists() else []
        alias = args.credential_env or (sessions[-1].get('credential_env') if sessions else None)
        if alias:
            config = replace(config, providers=tuple(replace(p, api_key_env=alias) for p in config.providers))
        validate_parallelism(config, args.parallelism)
        generation = entries[0][1]['generation']
        client = build_provider_pool(config, env_file=args.env_file, timeout_s=7200,
            max_tokens=generation['max_tokens'],
            temperature=None if generation['temperature'] == 'provider_default' else generation['temperature'],
            reasoning_effort=generation['reasoning_effort'], response_format={'type': 'text'},
            token_limit_parameter='max_tokens', enable_thinking=False, tool_service_url='',
            enable_group_tools=False, max_tool_rounds=0, transport_max_retries=0)
        sessions.append(dict(started_at=time.time(), credential_env=alias,
                             implementation_sha256=sha256_file(Path(__file__)), parallelism=args.parallelism,
                             provider_max_inflight=args.provider_max_inflight,
                             min_request_interval_ms=args.min_request_interval_ms,
                             attempt_cap=args.attempt_cap,
                             n_selected_queries=len(jobs)))
        write_json_atomic(root / 'execution_sessions.json', sessions)
        stop, counts = threading.Event(), Counter()
        try:
            with ThreadPoolExecutor(max_workers=args.parallelism) as workers:
                futures = [workers.submit(execute_query, entry, client, stop, attempt_limit=limit) for entry, limit in jobs]
                for done, future in enumerate(as_completed(futures), 1):
                    try:
                        counts[future.result()] += 1
                    except Exception:
                        stop.set()
                        raise
                    if done % 25 == 0 or done == len(jobs):
                        progress = dict(done=done, total=len(jobs), counts=dict(counts), updated_at=time.time())
                        write_json_atomic(root / 'execution_status.json', progress)
                        print(json.dumps(progress), flush=True)
        finally:
            client.close()
        summarize(root, entries, require_complete=True)


if __name__ == '__main__':
    main()
