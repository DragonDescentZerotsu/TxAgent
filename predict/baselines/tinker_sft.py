"""Train-only label-token SFT with validation selection and held-out evaluation."""

import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import log_loss

from predict.baselines.minimol.run_bioavailability_ma import choose_threshold, evaluate_metrics
from predict.prompts import template_environment
from predict.prompts.paper_baseline import profiles
from predict.retrieval.policies import normalize_molecule_identity
from predict.utils.json import (canonical_json_bytes, read_json, read_jsonl, sha256_file,
                                write_json_atomic, write_jsonl_atomic)

PROMPT = Path(__file__).parents[1] / 'prompts/tinker_sft_v1/user.jinja'
MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144'
RENDERER = 'deepseekv4_1_disable_thinking'
TTL = 30 * 24 * 3600
TEMPLATE = template_environment().get_template('tinker_sft_v1/user.jinja')


def fingerprint(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def parent_key(row):
    identity = normalize_molecule_identity(row['drug'])
    if identity.status != 'ok':
        raise ValueError(f"Cannot normalize {row['benchmark_row_id']}")
    return identity.parent_connectivity_key or identity.parent_smiles


def render_prompt(benchmark, task, row, examples):
    specs = profiles()
    spec = specs['conditioned'][task]
    if benchmark == 'tdc':
        spec = specs['tdc_overrides'].get(task, spec)
    fields = spec['task']
    return TEMPLATE.render(
        endpoint=fields['endpoint_name'], negative=fields['negative_prediction'],
        positive=fields['positive_prediction'], scope=fields.get('label_scope', ''),
        conditioned=benchmark == 'conditioned', query=row, examples=examples).strip()


def select_examples(train, keys, query_key, rng=None, count=10, similarities=None):
    """Train examples only; exclude every row of the query's normalized parent."""
    eligible = [i for i, key in enumerate(keys) if key != query_key]
    if rng is not None:
        indices = rng.sample(eligible, min(count, len(eligible)))
    else:
        indices = sorted(eligible, key=lambda i: (-similarities[i], i))[:count]
    return [train[i] for i in indices]


def prepare(args):
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    source = read_json(args.sources)
    config = dict(protocol='txgemma_style_label_sft.v1', source_release=source['release'],
                  model=args.model, renderer=RENDERER, seed=args.seed, rank=args.rank,
                  epochs=args.epochs, learning_rates=args.learning_rates, batch_size=args.batch_size,
                  fewshot_fraction=0.3, training_shots=[1, 10], evaluation_shots=10,
                  train_unembed=False, checkpoint_selection='valid_binary_cross_entropy',
                  threshold_selection='valid_macro_f1', prompt_sha256=sha256_file(PROMPT),
                  profiles_sha256=fingerprint(profiles()), sources_sha256=sha256_file(args.sources))
    if (args.output / 'plan.json').exists():
        old = read_json(args.output / 'plan.json')
        if {k: v for k, v in old['config'].items() if k != 'label_tokens'} != config:
            raise ValueError('Run already prepared with different settings; choose a new output.')
        verify_prepared(args.output, old)
        print(json.dumps(old['cost_estimate_usd'], indent=2), flush=True)
        return
    tokenizer = get_tokenizer(args.model)
    renderer = get_renderer(RENDERER, tokenizer)
    labels = [tokenizer.encode(x, add_special_tokens=False) for x in ('A', 'B')]
    if any(len(x) != 1 for x in labels) or labels[0] == labels[1]:
        raise ValueError('A/B must be distinct single tokens for this protocol.')
    config['label_tokens'] = [x[0] for x in labels]
    # Check the rendered assistant prefix, not just the requested renderer name.
    probe = renderer.build_generation_prompt([{'role': 'user', 'content': 'A or B?'}]).to_ints()
    if not tokenizer.decode(probe).endswith('</think>'):
        raise ValueError('Expected an explicit non-thinking assistant prefix.')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    RDLogger.DisableLog('rdApp.warning')
    datasets, train_tokens, valid_tokens, test_tokens, max_length = [], 0, 0, 0, 0
    for dataset in source['datasets']:
        benchmark, task = dataset['benchmark'], dataset['task']
        root = args.output / benchmark / task
        splits, audit, hashes = {}, {}, {}
        for split, locator in dataset['splits'].items():
            path = args.source_root / locator['path']
            if sha256_file(path) != locator['sha256']:
                raise ValueError(f'Frozen source changed: {path}')
            rows = read_jsonl(path)
            uids = [row['benchmark_row_id'] for row in rows]
            if len(uids) != len(set(uids)) or not rows:
                raise ValueError(f'Empty or duplicate IDs: {path}')
            if any(type(row['Y']) is not int or row['Y'] not in (0, 1) for row in rows):
                raise ValueError(f'Nonbinary labels: {path}')
            if set(row['Y'] for row in rows) != {0, 1}:
                raise ValueError(f'Both classes required for AUROC: {path}')
            splits[split] = rows
            audit[split] = dict(rows=len(rows), labels=dict(Counter(row['Y'] for row in rows)))
        keys = {split: [parent_key(row) for row in rows] for split, rows in splits.items()}
        audit['parent_connectivity_overlap'] = {
            f'{a}/{b}': len(set(keys[a]) & set(keys[b]))
            for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]}
        if benchmark == 'conditioned' and any(audit['parent_connectivity_overlap'].values()):
            raise ValueError(f'Unexpected conditioned parent overlap: {task}')
        train = splits['train']
        train_fp = [generator.GetFingerprint(Chem.MolFromSmiles(row['drug'])) for row in train]
        for split, rows in splits.items():
            epochs = range(args.epochs) if split == 'train' else [None]
            for epoch in epochs:
                rng = random.Random(f'{args.seed}:{benchmark}:{task}:{epoch}')
                fewshot = set(rng.sample(range(len(rows)), round(0.3 * len(rows)))) if split == 'train' else set()
                records = []
                for i, row in enumerate(rows):
                    if split == 'train':
                        examples = select_examples(train, keys['train'], keys[split][i],
                            rng=rng, count=rng.randint(1, 10)) if i in fewshot else []
                    else:
                        fp = generator.GetFingerprint(Chem.MolFromSmiles(row['drug']))
                        sim = DataStructs.BulkTanimotoSimilarity(fp, train_fp)
                        examples = select_examples(train, keys['train'], keys[split][i], similarities=sim)
                    prompt = render_prompt(benchmark, task, row, examples)
                    tokens = renderer.build_generation_prompt([{'role': 'user', 'content': prompt}]).to_ints()
                    max_length = max(max_length, len(tokens) + 1)
                    records.append(dict(uid=row['benchmark_row_id'], label=row['Y'],
                                        condition=row.get('condition_group'), tokens=tokens,
                                        example_uids=[x['benchmark_row_id'] for x in examples]))
                if split == 'train':
                    rng.shuffle(records)
                    name = f'train_epoch_{epoch + 1}.jsonl'
                    train_tokens += sum(len(x['tokens']) for x in records) * len(args.learning_rates)
                else:
                    name = f'{split}.jsonl'
                    count = sum(2 * (len(x['tokens']) + 1) for x in records)
                    if split == 'valid':
                        valid_tokens += count * args.epochs * len(args.learning_rates)
                    else:
                        test_tokens += count
                write_jsonl_atomic(root / name, records)
                hashes[name] = sha256_file(root / name)
        entry = dict(benchmark=benchmark, task=task, audit=audit, prepared_sha256=hashes,
                     sources=dataset['splits'])
        datasets.append(entry)
        print(f'prepared {benchmark}/{task}: {audit}', flush=True)
    if max_length > 262144:
        raise ValueError('Context overflow; inputs are never silently truncated.')
    # Logprob SDK requests generate one discarded token per candidate; include it.
    eval_requests = sum(2 * (d['audit']['valid']['rows'] * args.epochs * len(args.learning_rates)
                            + d['audit']['test']['rows']) for d in datasets)
    cost = dict(training=train_tokens * 4.98 / 1e6,
                evaluation_prefill=(valid_tokens + test_tokens) * 0.60 / 1e6,
                discarded_sampling=eval_requests * 3.60 / 1e6)
    cost['total_without_retries_or_storage'] = sum(cost.values())
    plan = dict(config=config, datasets=datasets, max_sequence_length=max_length,
                token_counts=dict(training=train_tokens, validation_prefill=valid_tokens,
                                  test_prefill=test_tokens, discarded_sampling=eval_requests),
                cost_estimate_usd=cost, pricing_source='https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/',
                pricing_date='2026-10-31', packages={name: importlib.metadata.version(name)
                    for name in ('tinker', 'tinker-cookbook', 'transformers', 'rdkit')})
    write_json_atomic(args.output / 'plan.json', plan)
    print(json.dumps(cost, indent=2), flush=True)


def verify_prepared(root, plan):
    if sha256_file(PROMPT) != plan['config']['prompt_sha256']:
        raise ValueError('Prompt changed since preparation.')
    if fingerprint(profiles()) != plan['config']['profiles_sha256']:
        raise ValueError('Task profiles changed since preparation.')
    for dataset in plan['datasets']:
        for name, expected in dataset['prepared_sha256'].items():
            if sha256_file(root / dataset['benchmark'] / dataset['task'] / name) != expected:
                raise ValueError(f'Prepared data changed: {dataset["task"]}/{name}')


def datum(record, label_tokens):
    import tinker
    tokens = record['tokens']
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(tokens), loss_fn_inputs={
        'target_tokens': tokens[1:] + [label_tokens[record['label']]],
        'weights': [0.0] * (len(tokens) - 1) + [1.0]})


def probabilities(logprobs):
    a, b = logprobs
    if not all(math.isfinite(x) for x in (a, b)):
        raise ValueError('Nonfinite candidate logprob')
    return math.exp(b - np.logaddexp(a, b))


def read_predictions(records, output, *, complete=False):
    """Reject incompatible cached results before scoring or reporting them."""
    expected = {r['uid']: r for r in records}
    existing = {}
    for row in read_jsonl(output) if output.exists() else []:
        uid = row['uid']
        if uid not in expected or uid in existing:
            raise ValueError(f'Unexpected or duplicate prediction ID: {output}: {uid}')
        if any(row[k] != expected[uid][k] for k in ('label', 'condition')):
            raise ValueError(f'Prediction label/condition mismatch: {output}: {uid}')
        probability = probabilities([row['logprob_A'], row['logprob_B']])
        if not math.isclose(row['probability'], probability, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'Prediction probability mismatch: {output}: {uid}')
        existing[uid] = row
    if complete and len(existing) != len(records):
        raise ValueError(f'Incomplete predictions: {output}')
    return existing


async def score(sampler, records, output, label_tokens, semaphore):
    import tinker
    existing = read_predictions(records, output)
    output.parent.mkdir(parents=True, exist_ok=True)

    async def one(record):
        if record['uid'] in existing:
            return
        async with semaphore:
            lps = await asyncio.gather(*(sampler.compute_logprobs_async(
                tinker.ModelInput.from_ints(record['tokens'] + [label]), prompt_logprobs_last_n=1)
                for label in label_tokens))
            pair = [values[-1] for values in lps]
            result = dict(uid=record['uid'], label=record['label'], condition=record['condition'],
                          logprob_A=pair[0], logprob_B=pair[1], probability=probabilities(pair))
            # No await while appending: each completed query becomes immediately resumable.
            with output.open('a') as handle:
                handle.write(json.dumps(result) + '\n')
            existing[record['uid']] = result

    await asyncio.gather(*(one(record) for record in records))
    ordered = [existing[r['uid']] for r in records]
    return np.asarray([x['probability'] for x in ordered])


def verify_results(root, plan):
    """Recompute saved validation selection and test metrics without provider calls."""
    results = []
    for dataset in plan['datasets']:
        task_root = root / dataset['benchmark'] / dataset['task']
        trials = []
        valid = read_jsonl(task_root / 'valid.jsonl')
        labels = [r['label'] for r in valid]
        for lr in plan['config']['learning_rates']:
            for epoch in range(1, plan['config']['epochs'] + 1):
                out = task_root / f'lr_{lr:g}'
                trial = read_json(out / f'epoch_{epoch}.json')
                rows = read_predictions(valid, out / f'epoch_{epoch}_valid.jsonl', complete=True)
                scores = np.asarray([rows[r['uid']]['probability'] for r in valid])
                threshold, _ = choose_threshold(labels, scores, 'valid_macro_f1')
                expected = dict(trial, learning_rate=lr, epoch=epoch,
                    valid_bce=float(log_loss(labels, scores, labels=[0, 1])),
                    valid_metrics=evaluate_metrics(labels, scores, threshold))
                if trial != expected:
                    raise ValueError(f'Validation metrics mismatch: {out}/epoch_{epoch}.json')
                trials.append(trial)
        selected = min(trials, key=lambda x: (x['valid_bce'], x['learning_rate'], x['epoch']))
        if read_json(task_root / 'selected.json') != selected:
            raise ValueError(f'Validation selection mismatch: {task_root}')
        test = read_jsonl(task_root / 'test.jsonl')
        rows = read_predictions(test, task_root / 'test_predictions.jsonl', complete=True)
        scores = np.asarray([rows[r['uid']]['probability'] for r in test])
        labels = [r['label'] for r in test]
        result = dict(benchmark=dataset['benchmark'], task=dataset['task'], rows=len(test),
                      selected=selected, test=evaluate_metrics(labels, scores, selected['valid_metrics']['threshold']),
                      test_fixed_0_5=evaluate_metrics(labels, scores, 0.5))
        if read_json(task_root / 'test_metrics.json') != result:
            raise ValueError(f'Test metrics mismatch: {task_root}')
        results.append(result)
    if read_json(root / 'results.json') != results:
        raise ValueError('Aggregate results mismatch')
    return dict(tasks=len(results), test_predictions=sum(r['rows'] for r in results))


def event(path, **fields):
    with path.open('a') as handle:
        handle.write(json.dumps(dict(time=time.time(), **fields), default=str) + '\n')
    print(json.dumps(fields, default=str), flush=True)


async def candidate(service, config, root, lr, semaphore):
    import tinker
    out = root / f'lr_{lr:g}'
    out.mkdir(parents=True, exist_ok=True)
    progress = read_json(out / 'progress.json') if (out / 'progress.json').exists() else None
    completed = [read_json(out / f'epoch_{e}.json') for e in range(1, config['epochs'] + 1)
                 if (out / f'epoch_{e}.json').exists()]
    if len(completed) == config['epochs']:
        return completed
    if progress:
        client = await service.create_training_client_from_state_with_optimizer_async(progress['state_path'])
        event(out / 'events.jsonl', kind='resume', progress=progress)
    else:
        client = await service.create_lora_training_client_async(
            base_model=config['model'], rank=config['rank'], seed=config['seed'],
            train_unembed=config['train_unembed'],
            user_metadata={'experiment': str(root), 'learning_rate': str(lr)})
    model_info = (await client.get_info_async()).model_dump(mode='json')
    write_json_atomic(out / 'model_info.json', model_info)
    event(out / 'events.jsonl', kind='training_client', model_info=model_info)
    valid = read_jsonl(root / 'valid.jsonl')
    labels = [r['label'] for r in valid]
    for epoch in range(1, config['epochs'] + 1):
        if (out / f'epoch_{epoch}.json').exists():
            continue
        records = read_jsonl(root / f'train_epoch_{epoch}.jsonl')
        start = progress['next_row'] if progress and progress['epoch'] == epoch else 0
        for block in range(start, len(records), 25 * config['batch_size']):
            pending = []
            end = min(block + 25 * config['batch_size'], len(records))
            # Tinker preserves operation sequence IDs. Queue complete FB/optimizer
            # pairs in order, then drain before saving a checkpoint at this cursor.
            for offset in range(block, end, config['batch_size']):
                batch = records[offset:offset + config['batch_size']]
                fb = await client.forward_backward_async([datum(r, config['label_tokens']) for r in batch], 'cross_entropy')
                opt = await client.optim_step_async(tinker.AdamParams(learning_rate=lr))
                pending.append((offset + len(batch), batch, fb, opt))
            for next_row, batch, fb, opt in pending:
                loss, _ = await asyncio.gather(fb.result_async(), opt.result_async())
                event(out / 'events.jsonl', kind='step', epoch=epoch, next_row=next_row,
                      input_tokens=sum(len(r['tokens']) for r in batch), metrics=loss.metrics)
            saved = await (await client.save_state_async(f'e{epoch}-r{end}', ttl_seconds=TTL)).result_async()
            progress = dict(epoch=epoch, next_row=end, state_path=saved.path)
            write_json_atomic(out / 'progress.json', progress)
        sampler_record = out / f'epoch_{epoch}_sampler.json'
        if sampler_record.exists():
            sampler_path = read_json(sampler_record)['path']
        else:
            saved = await (await client.save_weights_for_sampler_async(f'epoch-{epoch}', ttl_seconds=TTL)).result_async()
            sampler_path = saved.path
            write_json_atomic(sampler_record, {'path': sampler_path})
        sampler = await service.create_sampling_client_async(model_path=sampler_path)
        scores = await score(sampler, valid, out / f'epoch_{epoch}_valid.jsonl', config['label_tokens'], semaphore)
        threshold, _ = choose_threshold(labels, scores, 'valid_macro_f1')
        result = dict(learning_rate=lr, epoch=epoch, sampler_path=sampler_path,
                      state_path=progress['state_path'], valid_bce=float(log_loss(labels, scores, labels=[0, 1])),
                      valid_metrics=evaluate_metrics(labels, scores, threshold))
        write_json_atomic(out / f'epoch_{epoch}.json', result)
        completed.append(result)
        event(out / 'events.jsonl', kind='validation', **result)
    return completed


async def run(args, plan):
    from dotenv import dotenv_values
    import tinker

    key = dotenv_values(args.env_file).get('TINKER_API_KEY') if args.env_file else None
    if args.env_file and not key:
        raise ValueError('Explicit --env-file must contain TINKER_API_KEY')
    auth = {'api_key': key} if key else {}
    service = tinker.ServiceClient(**auth, display_name=f'{args.output.name}:test',
                                  user_metadata={'run_name': args.output.name, 'purpose': 'test'})
    rest = tinker.RestClient(**auth)
    config = plan['config']
    capabilities = (await service.get_server_capabilities_async()).model_dump()
    if not any(x['model_name'] == config['model'] and x.get('trainable', True)
               for x in capabilities['supported_models']):
        raise ValueError('Exact pinned model is not trainable; no automatic substitution.')
    write_json_atomic(args.output / 'capabilities.json', capabilities)
    train_slots, inference_slots = asyncio.Semaphore(args.parallel_tasks), asyncio.Semaphore(args.inference_concurrency)

    async def task(dataset):
        root = args.output / dataset['benchmark'] / dataset['task']
        if (root / 'test_metrics.json').exists():
            return
        async with train_slots:
            selected_path = root / 'selected.json'
            if selected_path.exists():
                selected = read_json(selected_path)
            else:
                trials = []
                for lr in config['learning_rates']:
                    trial_service = tinker.ServiceClient(**auth,
                        display_name=f'{args.output.name}:{dataset["benchmark"]}:{dataset["task"]}:lr{lr:g}',
                        user_metadata={'run_name': args.output.name, 'benchmark': dataset['benchmark'],
                                       'task': dataset['task'], 'learning_rate': str(lr)})
                    try:
                        trials.extend(await candidate(trial_service, config, root, lr, inference_slots))
                    except BaseException:
                        await trial_service.close('interrupted')
                        raise
                    else:
                        await trial_service.close('success')
                selected = min(trials, key=lambda x: (x['valid_bce'], x['learning_rate'], x['epoch']))
                write_json_atomic(selected_path, selected)
            await rest.set_checkpoint_ttl_from_tinker_path_async(selected['sampler_path'], None)
            # Test is first sent to the API only after all candidates and selection are frozen.
            sampler = await service.create_sampling_client_async(model_path=selected['sampler_path'])
            test = read_jsonl(root / 'test.jsonl')
            scores = await score(sampler, test, root / 'test_predictions.jsonl', config['label_tokens'], inference_slots)
            labels = [x['label'] for x in test]
            metrics = dict(benchmark=dataset['benchmark'], task=dataset['task'], rows=len(test),
                           selected=selected, test=evaluate_metrics(labels, scores, selected['valid_metrics']['threshold']),
                           test_fixed_0_5=evaluate_metrics(labels, scores, 0.5))
            write_json_atomic(root / 'test_metrics.json', metrics)
            event(args.output / 'events.jsonl', kind='task_complete', **metrics)
    await asyncio.gather(*(task(dataset) for dataset in plan['datasets']))
    results = [read_json(args.output / d['benchmark'] / d['task'] / 'test_metrics.json') for d in plan['datasets']]
    write_json_atomic(args.output / 'results.json', results)
    await service.close('success')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--model', choices=[MODEL], default=MODEL)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[1e-4, 5e-4])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--parallel-tasks', type=int, default=2)
    parser.add_argument('--inference-concurrency', type=int, default=32)
    parser.add_argument('--max-estimated-usd', type=float, default=1000)
    args = parser.parse_args()
    if min(args.epochs, args.rank, args.batch_size, args.parallel_tasks, args.inference_concurrency) < 1:
        parser.error('epochs, rank, batch size and concurrency must be positive')
    if any(lr <= 0 or not math.isfinite(lr) for lr in args.learning_rates):
        parser.error('learning rates must be finite and positive')
    if len({f'{lr:g}' for lr in args.learning_rates}) != len(args.learning_rates):
        parser.error('learning rates must have distinct output directory names')
    if not math.isfinite(args.max_estimated_usd) or args.max_estimated_usd <= 0:
        parser.error('max estimated USD must be finite and positive')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            if not args.sources or not args.source_root:
                parser.error('prepare requires --sources and --source-root')
            prepare(args)
        else:
            plan = read_json(args.output / 'plan.json')
            verify_prepared(args.output, plan)
            if args.command == 'verify' or (args.output / 'results.json').exists():
                print(json.dumps(verify_results(args.output, plan)), flush=True)
                return
            if plan['cost_estimate_usd']['total_without_retries_or_storage'] > args.max_estimated_usd:
                raise ValueError('Prepared token cost exceeds --max-estimated-usd')
            write_json_atomic(args.output / 'status.json', {'status': 'running', 'updated': time.time()})
            try:
                asyncio.run(run(args, plan))
            except BaseException as error:
                # Keep provider exception details in the local process log, not public metadata.
                write_json_atomic(args.output / 'status.json',
                                  {'status': 'interrupted', 'error_type': type(error).__name__, 'updated': time.time()})
                raise
            write_json_atomic(args.output / 'status.json', {'status': 'complete', 'updated': time.time()})


if __name__ == '__main__':
    main()
