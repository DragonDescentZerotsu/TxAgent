"""Scientific boundaries of the label-only SFT runner; no network requests."""
import asyncio
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from predict.baselines import tinker_sft as sft


def test_query_label_is_not_rendered_and_conditions_are_preserved():
    query = dict(drug='CCO', Y=0, condition_group='species=human', benchmark_row_id='q')
    a = sft.render_prompt('conditioned', 'carcinogens', query, [])
    b = sft.render_prompt('conditioned', 'carcinogens', dict(query, Y=1), [])
    assert a == b
    assert 'species=human' in a
    assert 'species=human' not in sft.render_prompt('tdc', 'carcinogens', query, [])


def test_demonstrations_exclude_all_query_parent_rows():
    train = [{'uid': 'same_a'}, {'uid': 'same_b'}, {'uid': 'other_a'}, {'uid': 'other_b'}]
    keys = ['same', 'same', 'other_a', 'other_b']
    for kwargs in ({'rng': random.Random(42)}, {'similarities': [1.0, 0.99, 0.2, 0.8]}):
        examples = sft.select_examples(train, keys, 'same', count=10, **kwargs)
        assert {x['uid'] for x in examples} == {'other_a', 'other_b'}
    examples = sft.select_examples(train, keys, 'same', count=1, similarities=[1, 1, .2, .8])
    assert examples == [train[3]]


def test_loss_supervises_only_next_label_token():
    pytest.importorskip('tinker')
    data = sft.datum({'tokens': [7, 8, 9], 'label': 1}, [35, 36])
    assert data.model_input.to_ints() == [7, 8, 9]
    assert data.loss_fn_inputs['target_tokens'].to_numpy().tolist() == [8, 9, 36]
    assert data.loss_fn_inputs['weights'].to_numpy().tolist() == [0, 0, 1]
    assert sft.probabilities([-1001, -1000]) == pytest.approx(0.7310585786)


def test_validation_selection_precedes_test_and_resume_skips_training(tmp_path, monkeypatch):
    tinker = pytest.importorskip('tinker')
    dataset = dict(benchmark='tdc', task='test_task')
    root = tmp_path / 'tdc/test_task'
    root.mkdir(parents=True)
    sft.write_jsonl_atomic(root / 'test.jsonl', [dict(uid='n', label=0), dict(uid='p', label=1)])
    calls = []

    class Service:
        async def close(self, status):
            pass

        async def get_server_capabilities_async(self):
            return SimpleNamespace(model_dump=lambda: {'supported_models': [dict(model_name='model', trainable=True)]})

        async def create_sampling_client_async(self, model_path):
            assert (root / 'selected.json').exists()
            assert model_path == 'better_valid'
            calls.append('test_sampler')
            return object()

    async def candidate(service, config, root, lr, semaphore):
        assert not (root / 'selected.json').exists()
        calls.append(lr)
        return [dict(valid_bce=lr, learning_rate=lr, epoch=1,
                     sampler_path='better_valid' if lr == .1 else 'worse_valid',
                     valid_metrics={'threshold': .6})]

    async def score(sampler, records, output, labels, semaphore):
        assert calls[:2] == [.2, .1]
        assert sft.read_json(root / 'selected.json')['learning_rate'] == .1
        calls.append('test_score')
        return np.array([.3, .7])

    monkeypatch.setattr(tinker, 'ServiceClient', lambda **kwargs: Service())
    class Rest:
        async def set_checkpoint_ttl_from_tinker_path_async(self, path, ttl):
            assert path == 'better_valid' and ttl is None
    monkeypatch.setattr(tinker, 'RestClient', lambda **kwargs: Rest())
    monkeypatch.setattr(sft, 'candidate', candidate)
    monkeypatch.setattr(sft, 'score', score)
    args = SimpleNamespace(env_file=None, output=tmp_path, parallel_tasks=1, inference_concurrency=2)
    plan = dict(config=dict(model='model', learning_rates=[.2, .1], label_tokens=[35, 36]), datasets=[dataset])
    asyncio.run(sft.run(args, plan))
    assert calls == [.2, .1, 'test_sampler', 'test_score']
    assert sft.read_json(root / 'test_metrics.json')['test']['threshold'] == .6
    asyncio.run(sft.run(args, plan))
    assert calls == [.2, .1, 'test_sampler', 'test_score']


def test_resume_after_training_before_validation_does_not_repeat_updates(tmp_path, monkeypatch):
    pytest.importorskip('tinker')
    rows = [dict(uid=str(i), tokens=[1, 2], label=i % 2, condition=None) for i in range(4)]
    sft.write_jsonl_atomic(tmp_path / 'train_epoch_1.jsonl', rows)
    sft.write_jsonl_atomic(tmp_path / 'valid.jsonl', rows)
    updates = []

    class Future:
        def __init__(self, **fields):
            self.value = SimpleNamespace(**fields)

        async def result_async(self):
            return self.value

    class Client:
        async def get_info_async(self):
            return SimpleNamespace(model_dump=lambda **kwargs: {'base_model': 'model'})

        async def forward_backward_async(self, batch, loss):
            return Future(metrics={})

        async def optim_step_async(self, params):
            updates.append(1)
            return Future()

        async def save_state_async(self, name, **kwargs):
            return Future(path='state')

        async def save_weights_for_sampler_async(self, name, **kwargs):
            return Future(path='sampler')

    class Service:
        async def create_lora_training_client_async(self, **kwargs):
            return Client()

        async def create_training_client_from_state_with_optimizer_async(self, path):
            assert path == 'state'
            return Client()

        async def create_sampling_client_async(self, **kwargs):
            return object()

    async def failing_score(*args):
        raise ConnectionError('interrupted after epoch checkpoint')

    async def working_score(*args):
        return np.array([.1, .9, .2, .8])

    config = dict(model='model', rank=32, seed=42, train_unembed=False, epochs=1,
                  batch_size=2, label_tokens=[35, 36])
    monkeypatch.setattr(sft, 'score', failing_score)
    with pytest.raises(ConnectionError):
        asyncio.run(sft.candidate(Service(), config, tmp_path, .1, asyncio.Semaphore(1)))
    assert len(updates) == 2
    monkeypatch.setattr(sft, 'score', working_score)
    result = asyncio.run(sft.candidate(Service(), config, tmp_path, .1, asyncio.Semaphore(1)))
    assert len(updates) == 2
    assert len(result) == 1


@pytest.mark.parametrize('damage', ['duplicate', 'unknown', 'label', 'condition', 'probability', 'nonfinite'])
def test_invalid_cached_predictions_fail_before_inference(tmp_path, damage):
    pytest.importorskip('tinker')
    records = [dict(uid=str(i), label=i, condition=None, tokens=[1, 2]) for i in range(2)]
    row = dict(records[0], logprob_A=-1., logprob_B=-2., probability=sft.probabilities([-1., -2.]))
    if damage == 'unknown':
        row['uid'] = 'unexpected'
    elif damage in ('label', 'condition'):
        row[damage] = 'changed'
    elif damage == 'probability':
        row['probability'] = .9
    elif damage == 'nonfinite':
        row['logprob_A'] = float('nan')
    output = tmp_path / 'predictions.jsonl'
    sft.write_jsonl_atomic(output, [row, row] if damage == 'duplicate' else [row])

    class Sampler:
        async def compute_logprobs_async(self, *args, **kwargs):
            pytest.fail('Invalid cache must be rejected before scoring missing rows')

    with pytest.raises(ValueError):
        asyncio.run(sft.score(Sampler(), records, output, [35, 36], asyncio.Semaphore(1)))


def test_completed_run_verifies_offline_without_rewriting_artifacts(tmp_path, monkeypatch):
    tinker = pytest.importorskip('tinker')
    monkeypatch.setattr(tinker, 'ServiceClient', lambda **kwargs: pytest.fail('Unexpected provider call'))
    task = tmp_path / 'tdc/test_task'
    records = [dict(uid=str(i), label=i, condition=None) for i in range(2)]
    rows = [dict(r, logprob_A=a, logprob_B=b, probability=sft.probabilities([a, b]))
            for r, (a, b) in zip(records, [(-.1, -2.), (-2., -.1)])]
    scores = np.array([r['probability'] for r in rows])
    threshold, _ = sft.choose_threshold([0, 1], scores, 'valid_macro_f1')
    metrics = sft.evaluate_metrics([0, 1], scores, threshold)
    trial = dict(learning_rate=.1, epoch=1, sampler_path='saved', state_path='saved',
                 valid_bce=float(sft.log_loss([0, 1], scores, labels=[0, 1])), valid_metrics=metrics)
    for split in ['valid', 'test']:
        sft.write_jsonl_atomic(task / f'{split}.jsonl', records)
    sft.write_jsonl_atomic(task / 'lr_0.1/epoch_1_valid.jsonl', rows)
    sft.write_jsonl_atomic(task / 'test_predictions.jsonl', rows)
    sft.write_json_atomic(task / 'lr_0.1/epoch_1.json', trial)
    sft.write_json_atomic(task / 'selected.json', trial)
    result = dict(benchmark='tdc', task='test_task', rows=2, selected=trial, test=metrics,
                  test_fixed_0_5=sft.evaluate_metrics([0, 1], scores, .5))
    sft.write_json_atomic(task / 'test_metrics.json', result)
    sft.write_json_atomic(tmp_path / 'results.json', [result])
    plan = dict(config=dict(epochs=1, learning_rates=[.1], prompt_sha256=sft.sha256_file(sft.PROMPT),
                            profiles_sha256=sft.fingerprint(sft.profiles())),
                datasets=[dict(benchmark='tdc', task='test_task', prepared_sha256={
                    f'{s}.jsonl': sft.sha256_file(task / f'{s}.jsonl') for s in ['valid', 'test']})])
    sft.write_json_atomic(tmp_path / 'plan.json', plan)
    sft.write_json_atomic(tmp_path / 'status.json', dict(status='complete', updated=1))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*.json*')}
    for command in ['verify', 'run']:
        monkeypatch.setattr(sys, 'argv', ['tinker_sft', command, '--output', str(tmp_path)])
        sft.main()
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    result['test'] = dict(metrics, macro_f1=0.)
    sft.write_json_atomic(task / 'test_metrics.json', result)
    with pytest.raises(ValueError, match='Test metrics mismatch'):
        sft.verify_results(tmp_path, plan)
