"""Prepare retrieval features, fit one RF candidate, or finalize a validation sweep."""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from predict.baselines.minimol.run_bioavailability_ma import choose_threshold, evaluate_metrics
from predict.baselines.retrieval_features import load_features, prepare
from predict.utils.json import sha256_file, write_json_atomic


def fit(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'training.json').exists():
        raise ValueError('Completed candidate exists')
    feature_set = getattr(args, 'feature_set', 'direct_indirect')
    manifest, features, rows = load_features(args.prepared, args.policy, feature_set)
    started = time.monotonic()
    max_features = args.max_features if args.max_features == 'sqrt' else float(args.max_features)
    scores, members = [], []
    for seed in args.seeds:
        model = RandomForestClassifier(n_estimators=args.trees, max_features=max_features,
                                       min_samples_leaf=args.min_leaf, random_state=seed,
                                       n_jobs=args.workers, bootstrap=True, class_weight=None)
        model.fit(features['train'], [r['Y'] for r in rows['train']])
        scores.append(model.predict_proba(features['valid'])[:, 1])
        model.n_jobs = 1
        path = args.output / f'seed_{seed}.joblib'
        joblib.dump(model, path, compress=0)
        members.append(dict(seed=seed, path=str(path), sha256=sha256_file(path)))
        print(f'Completed seed {seed}', flush=True)
    scores = np.asarray(scores)
    np.save(args.output / 'valid_scores.npy', scores)
    write_json_atomic(args.output / 'training.json', dict(
        status='complete', prepared_sha256=sha256_file(args.prepared / 'manifest.json'),
        policy=args.policy, members=members, n_estimators=args.trees,
        max_features=max_features, min_samples_leaf=args.min_leaf,
        max_depth=None, criterion='gini', class_weight=None, bootstrap=True,
        valid_bce=float(log_loss([r['Y'] for r in rows['valid']], scores.mean(axis=0), labels=[0, 1])),
        valid_scores_sha256=sha256_file(args.output / 'valid_scores.npy'),
        seconds=time.monotonic() - started, train_rows=len(rows['train']),
        feature_count=manifest['features'], feature_set=feature_set, test_used=False,
    ))


def finalize(args):
    feature_set = getattr(args, 'feature_set', 'direct_indirect')
    manifest, features, rows = load_features(args.prepared, args.policy, feature_set)
    candidates = []
    for path in sorted((args.output / 'candidates').glob('*/training.json')):
        receipt = json.loads(path.read_text())
        if (receipt['prepared_sha256'] != sha256_file(args.prepared / 'manifest.json')
                or receipt['policy'] != args.policy or receipt['test_used']
                or receipt.get('feature_set', 'direct_indirect') != feature_set
                or receipt['feature_count'] != manifest['features']):
            raise ValueError(f'Candidate input mismatch: {path}')
        score_path = path.parent / 'valid_scores.npy'
        if sha256_file(score_path) != receipt['valid_scores_sha256']:
            raise ValueError(f'Candidate validation scores changed: {path}')
        for member in receipt['members']:
            if sha256_file(Path(member['path'])) != member['sha256']:
                raise ValueError(f'Candidate model changed: {path}')
        scores = np.load(score_path).mean(axis=0)
        loss = float(log_loss([r['Y'] for r in rows['valid']], scores, labels=[0, 1]))
        if not np.isclose(loss, receipt['valid_bce']):
            raise ValueError(f'Candidate validation loss changed: {path}')
        candidates.append((loss, str(path), receipt, scores))
    if len(candidates) != args.expected_candidates:
        raise ValueError(f'Expected {args.expected_candidates} candidates, found {len(candidates)}')
    loss, path, winner, valid = min(candidates, key=lambda c: (c[0], c[1]))
    threshold, _ = choose_threshold([r['Y'] for r in rows['valid']], valid, 'valid_macro_f1')
    selection = dict(candidate=path, candidate_sha256=sha256_file(Path(path)),
                     validation_bce=loss, threshold=threshold, test_used=False,
                     candidate_validation_bce={p: v for v, p, *_ in candidates})
    if (args.output / 'metrics.json').exists():
        previous = json.loads((args.output / 'selection.json').read_text())
        metrics = json.loads((args.output / 'metrics.json').read_text())
        if (metrics.get('feature_set', 'direct_indirect') == feature_set
                and previous == selection and all(sha256_file(args.output / f'{s}_predictions.jsonl') == h
                                         for s, h in metrics['prediction_sha256'].items())):
            return
        raise ValueError('Completed evaluation differs; use a new output directory')
    write_json_atomic(args.output / 'selection.json', selection)
    test_members = []
    for member in winner['members']:
        model = joblib.load(member['path'])
        model.n_jobs = args.workers
        test_members.append(model.predict_proba(features['test'])[:, 1])
    test = np.mean(test_members, axis=0)
    metrics = {}
    for split, scores in [('valid', valid), ('test', test)]:
        metrics[split] = evaluate_metrics([r['Y'] for r in rows[split]], scores, threshold)
        with (args.output / f'{split}_predictions.jsonl').open('w') as stream:
            for i, (row, score) in enumerate(zip(rows[split], scores)):
                stream.write(json.dumps(dict(row, query_index=i, score=float(score),
                                              prediction=int(score >= threshold))) + '\n')
    metrics.update(selected_parameters={k: winner[k] for k in ('n_estimators', 'max_features', 'min_samples_leaf')},
                   threshold=threshold, prepared_sha256=sha256_file(args.prepared / 'manifest.json'),
                   policy=manifest['policies'][args.policy],
                   feature_set=feature_set, feature_count=manifest['features'],
                   prediction_sha256={s: sha256_file(args.output / f'{s}_predictions.jsonl') for s in ('valid','test')})
    write_json_atomic(args.output / 'metrics.json', metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'fit', 'finalize'])
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--source-prepared', type=Path)
    p.add_argument('--feature-set', choices=['direct', 'direct_indirect'], default='direct_indirect',
                   help='Model inputs: direct uses only the three label/similarity pairs from the audited cache')
    p.add_argument('--output', type=Path)
    p.add_argument('--condition-policy', default='same_condition_then_null',
                   choices=['same_condition_then_null', 'same_condition_then_null_then_all'])
    p.add_argument('--policy', type=int, choices=[0, 1], default=0)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--trees', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--max-features', default='sqrt')
    p.add_argument('--min-leaf', type=int, default=1)
    p.add_argument('--expected-candidates', type=int, default=6)
    args = p.parse_args()
    if args.phase == 'prepare':
        if not args.data_dir or not args.source_prepared:
            p.error('prepare requires --data-dir and --source-prepared')
        prepare(args.data_dir, args.source_prepared, args.prepared, args.condition_policy, args.workers)
    elif args.output is None:
        p.error('fit/finalize requires --output')
    else:
        (fit if args.phase == 'fit' else finalize)(args)


if __name__ == '__main__':
    main()
