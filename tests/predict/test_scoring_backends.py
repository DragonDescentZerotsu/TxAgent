"""Legacy scorers keep their tokenizer; FA3 is an explicit frozen-token workflow."""
import sys
from types import SimpleNamespace

import pytest

from predict.retrieval.assay_reranking import runtime, build_ranked_uid_retrieval as builder


@pytest.mark.parametrize('gpu_major', [9, 10])
@pytest.mark.parametrize('backend', [None, 'sgl_fa3'])
def test_gpu_generation_cannot_opt_legacy_scorers_into_fa3(monkeypatch, gpu_major, backend):
    calls = []
    tokenizer = SimpleNamespace(pad_token=None, eos_token='eos', padding_side='right')
    model = SimpleNamespace(config=SimpleNamespace())
    model.to = lambda device: model
    model.eval = lambda: None
    def load_model(*args, **kwargs):
        calls.append(kwargs)
        return model
    def load_tokenizer(*args, **kwargs):
        calls.append('tokenizer')
        return tokenizer
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(get_device_capability=lambda _: (gpu_major, 0))))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        __version__='5.12.1' if backend else runtime.REQUIRED_TRANSFORMERS_VERSION,
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
    ))
    monkeypatch.setattr(runtime, '_register_sgl_fa3', lambda: calls.append('fa3'))
    _, loaded_tokenizer = runtime.load_model('frozen-model', attention_backend=backend)
    if backend:
        assert loaded_tokenizer is None and 'tokenizer' not in calls and 'fa3' in calls
        assert model.config._attn_implementation == 'sgl_fa3'
        with pytest.raises(ValueError, match='frozen tokenized'):
            runtime.score_prompt_batch(model, loaded_tokenizer, [])
    else:
        assert loaded_tokenizer is tokenizer
        assert tokenizer.padding_side == 'left' and tokenizer.pad_token == 'eos'
        assert calls[-1]['attn_implementation'] == 'flash_attention_2'
    before = list(calls)
    sys.modules['transformers'].__version__ = 'unsupported'
    with pytest.raises(RuntimeError, match='requires transformers'):
        runtime.load_model('frozen-model', attention_backend=backend)
    assert calls == before


def test_fa3_rejects_missing_frozen_tokens_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match='tokenized-root'):
        builder.score_level('ames', 'valid', 'L2', tmp_path, 0, 1, 0, 4, attention_backend='sgl_fa3')


@pytest.mark.parametrize('benchmark,expected', [('gold', 'v1'), ('tdc', {'benchmark': 'tdc'})])
def test_builder_cli_records_correct_benchmark_identity(tmp_path, monkeypatch, benchmark, expected):
    for name in ('EVIDENCE_RELEASE', 'PROFILE', 'SCORE_REUSE_ROOTS', 'TRUST_PREDECESSOR_ROWS',
                 'QUERY_BENCHMARK', 'LABEL_RELEASE', 'LATER_IDENTITY_POLICY',
                 'QUERY_IDENTITY_ROOT', 'RESCORE_CONFLICTING_REUSE'):
        monkeypatch.setattr(builder, name, getattr(builder, name))
    seen = []
    def inventory(*args):
        seen.append(builder.LABEL_RELEASE)
        return {'status': 'complete'}
    monkeypatch.setattr(builder, 'write_index', inventory)
    monkeypatch.setattr(sys, 'argv', ['builder', 'index', '--task', 'ames', '--benchmark', benchmark,
                                    '--evidence-manifest', str(tmp_path / 'VERSION.json'), '--output-root', str(tmp_path)])
    builder.main()
    assert seen == [expected]


def test_scoring_cli_forwards_explicit_fa3_with_tokens(tmp_path,monkeypatch):
    for name in ('EVIDENCE_RELEASE', 'PROFILE', 'SCORE_REUSE_ROOTS', 'TRUST_PREDECESSOR_ROWS',
                 'QUERY_BENCHMARK', 'LABEL_RELEASE', 'LATER_IDENTITY_POLICY',
                 'QUERY_IDENTITY_ROOT', 'RESCORE_CONFLICTING_REUSE'):
        monkeypatch.setattr(builder,name,getattr(builder,name))
    seen=[]
    monkeypatch.setattr(builder,'score_level',lambda *a,**kw:seen.append((a,kw)) or {'status':'complete'})
    monkeypatch.setattr(sys,'argv',['builder','score','--task','ames','--subset','valid','--level','L2',
                                  '--benchmark','tdc','--evidence-manifest',str(tmp_path/'VERSION.json'),
                                  '--output-root',str(tmp_path),'--attention-backend','sgl_fa3',
                                  '--tokenized-root',str(tmp_path/'tokens')])
    builder.main()
    assert seen[0][0][-1]==tmp_path/'tokens'
    assert seen[0][1]=={'attention_backend':'sgl_fa3'}


def test_backend_mismatch_rejects_even_a_completed_shard_before_loading(tmp_path, monkeypatch):
    import json
    target = tmp_path / 'ames/scaffold/valid/L2'
    target.mkdir(parents=True)
    (target / 'VERSION.json').write_text(json.dumps({
        'status': 'prepared', 'scoring_contract_version': runtime.SCORING_CONTRACT_VERSION,
    }))
    monkeypatch.setattr(runtime, 'load_model', lambda *a, **kw: pytest.fail('model must not load'))
    with pytest.raises(ValueError, match='contract differs'):
        builder.score_level('ames', 'valid', 'L2', tmp_path, 0, 1, 0, 1,
                            tmp_path / 'tokens', attention_backend='sgl_fa3')
    assert not (target / '.scores').exists()


def test_backends_have_distinct_prompt_keys_without_global_mutation():
    renderer = SimpleNamespace(render=lambda *a: 'same prompt', template_hash='template',
                               projection_hash='projection', task_id='ames')
    options = dict(query_smiles='CCO', group_id='L2', molecule_id='m', model='fixture', model_revision='a'*40)
    record = {'record_id': 'physical-uid'}
    fa2 = runtime.build_prompt_task(renderer, record, **options)
    fa3 = runtime.build_prompt_task(renderer, record, **options,
                                   scoring_contract_version=runtime.FA3_SCORING_CONTRACT_VERSION)
    assert fa2.prompt == fa3.prompt and fa2.cache_key != fa3.cache_key
    assert fa2.scoring_contract_version == runtime.SCORING_CONTRACT_VERSION


@pytest.mark.parametrize('command', ['prepare-level', 'score'])
def test_safety_fa3_alias_forwards_backend_and_keeps_global_contract(tmp_path, monkeypatch, command):
    from predict.retrieval.assay_reranking import build_safety_v27_ranked_retrieval as safety
    for name, value in vars(builder).copy().items():
        if name.isupper():
            monkeypatch.setattr(builder, name, value)
    original = runtime.SCORING_CONTRACT_VERSION
    seen = []
    function = 'prepare_level' if command == 'prepare-level' else 'score_level'
    monkeypatch.setattr(builder, function, lambda *a, **kw: seen.append(kw) or {'status': 'complete'})
    monkeypatch.setattr(sys, 'argv', [
        'safety', command, '--task', 'ames', '--benchmark', 'tdc', '--subset', 'valid', '--level', 'L2',
        '--fa3-score-generation', '--tokenized-root', str(tmp_path/'tokens'),
        '--evidence-manifest', str(tmp_path/'VERSION.json'),
    ])
    safety.main()
    assert seen == [{'attention_backend': 'sgl_fa3'}]
    assert runtime.SCORING_CONTRACT_VERSION == original
    assert builder.LABEL_RELEASE == {'benchmark': 'tdc'}
