"""Repeated inference statistics require complete cohorts and run-level means."""

import json
import os
from pathlib import Path
import statistics
import subprocess

import pytest

from analysis.prediction.llm_repeats import METHODS, TASKS, summarize


@pytest.mark.parametrize('methods', [METHODS, ('deepseek_none',)])
def test_complete_repetitions_and_run_level_task_means(methods):
    expected, rows = {}, []
    for benchmark in ('conditioned', 'tdc'):
        for method in methods:
            for i, task in enumerate(TASKS):
                expected[benchmark, task, method] = i + 1
                for repetition in range(1, 6):
                    value = .5 + i*.01 + (-1)**i * (repetition-3)*.02
                    rows.append(dict(benchmark=benchmark, method=method, task=task, repetition=repetition,
                                     n=i+1, macro_f1=value, accuracy=value, failures_counted_wrong=0, run_root='fixture'))
    members, summary = summarize(rows, expected)
    assert len(members) == 70*len(methods) and len(summary) == 28*len(methods) and len(rows) == 60*len(methods)
    overall = [r for r in summary if r['task'] == 'six_task_mean']
    assert all(r['macro_f1_mean'] == pytest.approx(.525) and r['macro_f1_sd'] == pytest.approx(0)
               for r in overall)
    bbb = next(r for r in summary if r['task'] == TASKS[0] and r['scope'] == 'historical_plus_four')
    assert bbb['macro_f1_sd'] == pytest.approx(statistics.stdev([.46, .48, .50, .52, .54]))
    for bad in (rows[:-1], rows + [rows[0]], [{**rows[0], 'n': 99}, *rows[1:]],
                [{**rows[0], 'macro_f1': float('nan')}, *rows[1:]]):
        with pytest.raises(ValueError):
            summarize(bad, expected)


def test_recipe_uses_json_plan_and_keeps_failure_receipts(tmp_path):
    study = tmp_path/'study'
    study.mkdir()
    jobs = []
    for kind in ('naive', 'molrag'):
        root = tmp_path/kind
        root.mkdir()
        jobs.append(dict(kind=kind, root=str(root), provider='provider.json', source='unused'))
    (study/'jobs.json').write_text(json.dumps(jobs))
    stub = tmp_path/'python-stub'
    stub.write_text('''#!/usr/bin/env python3
import os, sys
if sys.argv[1] == '-':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
assert '--env-file' not in sys.argv
if 'predict.baselines.molrag' in sys.argv:
    assert sys.argv[sys.argv.index('--parallelism')+1] == '256'
    assert sys.argv[sys.argv.index('--provider-max-inflight')+1] == '256'
    sys.exit(1)
''')
    stub.chmod(0o755)
    recipe = Path(__file__).resolve().parents[3]/'experiments/baselines/llm_repeats.sh'
    result = subprocess.run(['bash', str(recipe), str(study)], env={**os.environ,
        'PYTHON_BIN': str(stub), 'TXAGENT_PARCC_REPEAT_KEY': 'test-only', 'MOLRAG_PARALLELISM': '256'})
    assert result.returncode == 123
    assert (study/'logs/naive.done').exists() and not (study/'logs/molrag.done').exists()
    assert (study/'logs/molrag.exit').read_text().strip() == '1'
    before = {p: p.read_bytes() for p in (study/'logs').iterdir()}
    (study/'aggregation_status.json').write_text('{"status":"complete"}')
    env = {**os.environ, 'PYTHON_BIN': str(stub)}
    env.pop('TXAGENT_PARCC_REPEAT_KEY', None)
    subprocess.run(['bash', str(recipe), str(study)], env=env, check=True)
    assert all(p.read_bytes() == value for p, value in before.items())
