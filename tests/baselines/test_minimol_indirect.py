import json

import numpy as np
import pytest
import torch

from baselines.minimol.indirect_data import aggregate_targets
from baselines.minimol.finetune_runtime import auxiliary_loss


def row(smiles, bucket, values, binary=False):
    return dict(parent_smiles=smiles,
                pair_bucket_key=json.dumps(["source", bucket, "binary_outcome_class" if binary else "unit"]),
                canonical_measurements=[str(v) for v in values])


def test_heldout_connectivity_removed_before_bucket_normalization():
    rows = [row("a", "x", [1, 3, 99]), row("b", "x", [5]), row("heldout", "x", [10000]),
            row("a", "y", [100]), row("a", "b", [1, 1, -1], True)]
    identities = {"a": ("a", "A"), "b": ("b", "B"), "heldout": ("different_stereo", "H")}
    targets, buckets, counts = aggregate_targets(rows, {"H"}, identities)
    assert counts["heldout_rows_removed"] == 1
    x = next(b for b in buckets if json.loads(b["key"])[1] == "x")
    assert x["mean"] == 4 and x["scale"] == 1
    assert sorted(r["target"] for r in targets if json.loads(buckets[r["bucket"]]["key"])[1] == "x") == [-1, 1]
    binary = next(r for r in targets if r["binary"])
    assert binary["target"] == pytest.approx(2 / 3)
    assert any(b["constant"] for b in buckets)  # Retained, not quality filtered.
    assert all(np.isfinite(r["target"]) for r in targets)


def test_equivalent_parent_aliases_aggregate_only_within_bucket():
    rows = [row("salt", "x", [2]), row("parent", "x", [8]), row("parent", "y", [100])]
    targets, buckets, _ = aggregate_targets(rows, set(), {"salt": ("parent", "P"), "parent": ("parent", "P")})
    assert len(targets) == 2
    assert sorted(r["raw_target"] for r in targets) == [5, 100]


def test_mixed_loss_has_only_observed_head_gradients():
    logits = torch.tensor([[0.0, 99.0], [99.0, 2.0]], requires_grad=True)
    selected = logits[torch.arange(2), torch.tensor([0, 1])]
    loss = auxiliary_loss(selected, torch.tensor([0.75, 1.0]), torch.tensor([True, False]))
    loss.backward()
    assert logits.grad[0, 1] == logits.grad[1, 0] == 0
    assert logits.grad[0, 0] == pytest.approx(-0.125)
    assert logits.grad[1, 1] == pytest.approx(1.0)


def test_invalid_binary_domain_fails_instead_of_relabeling():
    with pytest.raises(ValueError, match="binary domain"):
        aggregate_targets([row("a", "x", [0, 1], True)], set(), {"a": ("a", "A")})


def test_valid_polyvalent_ion_preserves_charge_and_heldout_identity():
    from baselines.minimol.indirect_data import parent_keys

    smiles = "F[P-](F)(F)(F)F"
    parent, key = parent_keys(smiles)
    assert parent == smiles and key == "PLRGVWCDUXXBBZ"
    targets, _, counts = aggregate_targets([row(smiles, "x", [1])], {key},
                                           {smiles: (parent, key)})
    assert not targets and counts["heldout_rows_removed"] == 1
    with pytest.raises(ValueError, match="Unresolved parent"):
        parent_keys("not a smiles")


def test_graph_cache_tampering_is_rejected(tmp_path):
    from baselines.minimol.run_multitask import load_prepared
    from tools.chembl_tool.common.json_utils import sha256_file

    torch.save({'smiles': ['CCO']}, tmp_path / 'data.pt')
    torch.save([1], tmp_path / 'graphs.pt')
    digest = sha256_file(tmp_path / 'data.pt')
    (tmp_path / 'manifest.json').write_text(json.dumps({'data_sha256': digest}))
    (tmp_path / 'graphs.json').write_text(json.dumps({
        'data_sha256': digest, 'graph_sha256': sha256_file(tmp_path / 'graphs.pt'), 'n_graphs': 1}))
    assert load_prepared(tmp_path, with_graphs=False)[1] is None
    torch.save([2], tmp_path / 'graphs.pt')
    with pytest.raises(ValueError, match='graph cache hash'):
        load_prepared(tmp_path, with_graphs=False)


@pytest.mark.parametrize('corruption', ['input', 'scores', 'weight'])
def test_candidate_binding_rejects_mixed_inputs_or_changed_scores(tmp_path, corruption):
    from baselines.minimol.run_multitask import ensemble_bce, validated_member
    from tools.chembl_tool.common.json_utils import sha256_file

    labels = np.array([0., 1.])
    scores = np.array([0.2, 0.8])
    loss = ensemble_bce(labels, scores)
    np.save(tmp_path / 'valid_scores.npy', scores)
    torch.save(dict(args=dict(seed=1, aux_weight=0.1, epochs=1), epoch=1,
                    prepared_sha256='expected', valid_bce=loss,
                    valid_scores_sha256=sha256_file(tmp_path / 'valid_scores.npy')),
               tmp_path / 'best.pt')
    (tmp_path / 'training.json').write_text(json.dumps(dict(
        checkpoint_sha256=sha256_file(tmp_path / 'best.pt'), test_used=False,
        best_epoch=1, history=[dict(epoch=1, valid_bce=loss)])))
    kwargs = dict(prepared_sha256='expected', seed=1, weight=0.1, labels=labels)
    validated_member(tmp_path, **kwargs)
    if corruption == 'input':
        kwargs['prepared_sha256'] = 'different-cohort'
    elif corruption == 'weight':
        kwargs['weight'] = 0.6
    else:
        np.save(tmp_path / 'valid_scores.npy', 1 - scores)
    with pytest.raises(ValueError):
        validated_member(tmp_path, **kwargs)


@pytest.mark.parametrize('indirect_only', [False, True])
def test_finalize_reuses_verified_predictions_without_loading_encoder(tmp_path, monkeypatch, indirect_only):
    from argparse import Namespace
    from baselines.minimol import run_multitask as runner
    from tools.chembl_tool.common.json_utils import sha256_file

    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    (prepared / 'manifest.json').write_text('{}')
    (prepared / 'graphs.json').write_text(json.dumps({'graph_sha256': 'graph-hash'}))
    labels = torch.tensor([0., 1.])
    data = {'splits': {'valid': {'labels': labels}}}
    monkeypatch.setattr(runner, 'load_prepared', lambda *a, **k: (data, None))
    config = dict(hidden_dim=8, depth=3, lr=1e-4, encoder_lr=1e-5, epochs=25, batch_size=32)
    def member(path, *, weight, **kwargs):
        assert not indirect_only or weight > 0
        score = {0: .3, .1: .25, .6: .1, 1: .2}[weight]
        return dict(path=path, config=config, scores=np.array([score, 1-score]),
                    checkpoint_sha256=f'checkpoint-{weight:g}', valid_scores_sha256=f'scores-{weight:g}')
    monkeypatch.setattr(runner, 'validated_member', member)
    def forbidden(*args, **kwargs):
        raise AssertionError('Reusable evaluations must not load an encoder or access test batches')
    monkeypatch.setattr(runner, 'MultiTaskModel', forbidden)
    for name, weight in ([] if indirect_only else [('finetuned', 0)]) + [('indirect', .6)]:
        output = tmp_path / name
        output.mkdir()
        m = member(tmp_path / f'aux_{weight:g}' / 'seed_1', weight=weight)
        threshold, _ = runner.choose_threshold(labels.tolist(), m['scores'], 'valid_macro_f1')
        for split in ['valid', 'test']:
            (output / f'{split}_predictions.jsonl').write_text('frozen prediction bytes\n')
        (output / 'metrics.json').write_text(json.dumps(dict(
            evaluation_inputs=runner.evaluation_inputs(prepared, [m], threshold),
            prediction_sha256={s: sha256_file(output / f'{s}_predictions.jsonl') for s in ['valid', 'test']})))
    runner.finalize(Namespace(prepared=prepared, output=tmp_path, threads=1, seeds=[1],
                              aux_weights=[.1, .6, 1], indirect_only=indirect_only))
    assert json.loads((tmp_path / 'selection.json').read_text())['selected_aux_weight'] == .6
    output = tmp_path / 'indirect'
    metrics = json.loads((output / 'metrics.json').read_text())
    (output / 'test_predictions.jsonl').write_text('changed')
    assert runner.reusable_evaluation(output, metrics['evaluation_inputs']) is None


@pytest.mark.parametrize('condition_field', [None, 'model_condition'])
def test_external_main_split_preserved_without_relaxing_auxiliary_exclusion(tmp_path, monkeypatch, condition_field):
    from baselines.minimol import indirect_data
    from concurrent.futures import ThreadPoolExecutor
    import pyarrow as pa
    import pyarrow.parquet as pq

    monkeypatch.setattr(indirect_data, 'ProcessPoolExecutor',
                        lambda max_workers, mp_context: ThreadPoolExecutor(max_workers=max_workers))
    main = tmp_path / 'main'
    main.mkdir()
    record = dict(drug='CCO', Y=1, condition_group='no_reported_external_condition',
                  model_condition='TDC:molecular_label')
    for split in ('train', 'valid', 'test'):
        (main / f'{split}.jsonl').write_text(json.dumps(record) + '\n')
    aux = tmp_path / 'aux.parquet'
    pq.write_table(pa.Table.from_pylist([row('CCO', 'x', [1000]), row('CCN', 'x', [2])]), aux)
    with pytest.raises(ValueError, match='train/heldout parent overlap'):
        indirect_data.prepare(main, aux, tmp_path / 'strict', workers=1, condition_field=condition_field)
    out = tmp_path / 'preserved'
    manifest = indirect_data.prepare(main, aux, out, workers=1, condition_field=condition_field,
                                     preserve_main_split=True)
    data = torch.load(out / 'data.pt', weights_only=False)
    assert manifest['main_heldout_identity_overlap'] > 0
    assert manifest['counts']['heldout_rows_removed'] == 1
    assert [data['smiles'][i] for i in data['auxiliary']['indices']] == ['CCN']
    assert data['buckets'][0]['mean'] == 2
    assert data['splits']['train']['labels'].tolist() == [1]
    assert data['splits']['test']['condition_groups'] == ['no_reported_external_condition']
    assert data['splits']['train']['conditions'].shape == (1, 0 if condition_field is None else 1)
    assert data['conditions'] == ([] if condition_field is None else ['TDC:molecular_label'])
