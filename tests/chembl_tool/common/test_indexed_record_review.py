import copy
import hashlib
import json

import pytest

from tools.chembl_tool.common.indexed_record_review import (
    apply_record_review, load_record_review, surface_sha256,
)
from tools.chembl_tool.common.progressive_assay_reasoning import (
    _card_id, _card_surface, stable_analog_id,
)


def fixture_index():
    molecule = {"molecule_chembl_id": "m", "canonical_smiles": "CC"}
    example = {"evidence_family": "near", "evidence_family_level": 2,
               "endpoint_type": "negative bacterial mutation", "support_text": "negative"}
    row = {"source_record_examples": [example], "assay_chembl_id": "a"}
    surface = _card_surface(example, row)
    decision = {"card_id": _card_id(stable_analog_id(molecule), surface),
                "canonical_smiles": "CC", "surface_sha256": surface_sha256(surface),
                "source_family": "near", "action": "move", "reason": "DNA endpoint",
                "target_family": "dna", "target_level": 4}
    return {"molecules": [molecule], "evidence_by_molecule_group": {
        "m": {"a": [row], "duplicate": [copy.deepcopy(row)]}}}, decision


def test_all_occurrences_move_without_losing_negative_or_mutating_base():
    index, decision = fixture_index()
    before = copy.deepcopy(index)
    result, receipt = apply_record_review(index, {"decisions": [decision]})
    assert index == before
    assert receipt["matched_occurrences_by_action"]["move"] == 2
    for rows in result["evidence_by_molecule_group"]["m"].values():
        example = rows[0]["source_record_examples"][0]
        assert example["evidence_family_level"] == 4
        assert example["support_text"] == "negative"
        assert "reason" not in example


def test_exclusion_removes_empty_groups_and_molecule():
    index, decision = fixture_index()
    decision["action"] = "exclude"
    result, _ = apply_record_review(index, {"decisions": [decision]})
    assert result["evidence_by_molecule_group"] == {}
    assert index["evidence_by_molecule_group"]


def test_source_scope_correction_changes_all_cards_without_rewriting_support():
    index, decision = fixture_index()
    before = copy.deepcopy(index)
    decision.pop("target_level")
    decision.pop("target_family")
    decision.update(action="correct", field_updates={
        "reported_value": "no detected modulation of challenge-induced damage",
    })
    result, receipt = apply_record_review(index, {"decisions": [decision]})
    assert index == before
    assert receipt["matched_occurrences_by_action"]["correct"] == 2
    for rows in result["evidence_by_molecule_group"]["m"].values():
        ex = rows[0]["source_record_examples"][0]
        assert ex["reported_value"].startswith("no detected modulation")
        assert ex["support_text"] == "negative"
        assert ex["evidence_family_level"] == 2
        assert _card_id(stable_analog_id(index["molecules"][0]), _card_surface(ex, rows[0])) != decision["card_id"]


@pytest.mark.parametrize("field", ["support_text", "evidence_family", "canonical_smiles"])
def test_correction_cannot_rewrite_identity_support_or_membership(tmp_path, field):
    _, decision = fixture_index()
    decision.update(action="correct", field_updates={field: "changed"})
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"version": "indexed_record_review.v1", "task": "example",
                               "base_index_sha256": "hash", "decisions": [decision]}))
    with pytest.raises(ValueError, match="immutable"):
        load_record_review(path, task="example", index_sha256="hash",
                           levels=[{"level": 1, "endpoint_group": "direct"}])


def test_review_selection_policy_is_explicit_and_does_not_mutate_base():
    index, decision = fixture_index()
    result, receipt = apply_record_review(index, {
        "decisions": [decision], "selection_policy": "endpoint_diverse_delta.v1",
    })
    assert result["record_review_selection_policy"] == "endpoint_diverse_delta.v1"
    assert receipt["selection_policy"] == "endpoint_diverse_delta.v1"
    assert "record_review_selection_policy" not in index


def test_original_document_support_correction_preserves_base_and_changes_all_ids(tmp_path):
    index, decision = fixture_index()
    before = copy.deepcopy(index)
    document = tmp_path / "source.txt"
    document.write_text("Original table: endpoint reported per mille.")
    decision.pop("target_level")
    decision.pop("target_family")
    decision.update(action="correct", field_updates={"support_text": "Measured values are per mille."},
                    support_text_provenance={"document_path": str(document),
                                             "document_sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
                                             "locator": "Table 6"})
    result, receipt = apply_record_review(index, {"decisions": [decision]})
    assert index == before
    assert receipt["matched_occurrences_by_action"]["correct"] == 2
    for rows in result["evidence_by_molecule_group"]["m"].values():
        ex = rows[0]["source_record_examples"][0]
        assert ex["support_text"] == "Measured values are per mille."
        assert ex["evidence_family"] == "near"
        assert _card_id(stable_analog_id(index["molecules"][0]), _card_surface(ex, rows[0])) != decision["card_id"]
    document.write_text("Changed document")
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_record_review(index, {"decisions": [decision]})


def test_stale_payload_and_missing_card_fail_closed():
    index, decision = fixture_index()
    decision["surface_sha256"] = "stale"
    with pytest.raises(ValueError, match="payload mismatch"):
        apply_record_review(index, {"decisions": [decision]})
    decision["card_id"] = "missing"
    with pytest.raises(ValueError, match="absent"):
        apply_record_review(index, {"decisions": [decision]})


def reassignment_fixture(tmp_path):
    index, decision = fixture_index()
    index['molecules'].append({'molecule_chembl_id': 'target', 'canonical_smiles': 'CCC',
                               'standard_inchi_key': 'target-key', 'molecule_identity': {'parent_smiles': 'CCC'}})
    document = tmp_path / 'paper.txt'
    document.write_text('Compound 2 is the tested target, not compound 1.')
    decision.update(action='reassign', target_canonical_smiles='CCC', identity_provenance={
        'document_path': str(document), 'document_sha256': hashlib.sha256(document.read_bytes()).hexdigest(),
        'locator': 'Compound 2 definition and toxicity prediction'})
    decision.pop('target_family')
    decision.pop('target_level')
    for rows in index['evidence_by_molecule_group']['m'].values():
        rows[0]['canonical_smiles'] = 'CC'
        rows[0]['molecule_chembl_id'] = 'm'
        rows[0]['minimal_evidence'] = {'molecule': {'id': 'm', 'canonical_smiles': 'CC'}}
    return index, decision


def test_identity_reassignment_moves_all_occurrences_without_changing_evidence(tmp_path):
    index, decision = reassignment_fixture(tmp_path)
    before = copy.deepcopy(index)
    result, receipt = apply_record_review(index, {'decisions': [decision]})
    assert index == before
    assert receipt['matched_occurrences_by_action']['reassign'] == 2
    assert 'm' not in result['evidence_by_molecule_group']
    for rows in result['evidence_by_molecule_group']['target'].values():
        row = rows[0]
        assert row['canonical_smiles'] == 'CCC' and row['standard_inchi_key'] == 'target-key'
        assert row['minimal_evidence']['molecule'] == {'id': 'target', 'canonical_smiles': 'CCC'}
        assert row['source_record_examples'] == before['evidence_by_molecule_group']['m']['a'][0]['source_record_examples']
        assert _card_id(stable_analog_id(result['molecules'][1]), _card_surface(row['source_record_examples'][0], row)) != decision['card_id']
    assert result['molecules'][1]['n_evidence_rows'] == 2
    assert result['molecules'][1]['molecule_identity'] == before['molecules'][1]['molecule_identity']


@pytest.mark.parametrize('problem', ['missing_target', 'duplicate_target', 'changed_document', 'partial_row'])
def test_identity_reassignment_fails_closed(tmp_path, problem):
    index, decision = reassignment_fixture(tmp_path)
    if problem == 'missing_target':
        index['molecules'].pop()
    elif problem == 'duplicate_target':
        index['molecules'].append(dict(index['molecules'][-1]))
    elif problem == 'changed_document':
        (tmp_path / 'paper.txt').write_text('different source')
    else:
        index['evidence_by_molecule_group']['m']['a'][0]['source_record_examples'].append({'support_text': 'another observation'})
    with pytest.raises(ValueError):
        apply_record_review(index, {'decisions': [decision]})


def test_identity_reassignment_cannot_edit_actual_voters(tmp_path):
    _, decision = reassignment_fixture(tmp_path)
    decision['source_family'] = 'direct'
    path = tmp_path / 'review.json'
    path.write_text(json.dumps({'version': 'indexed_record_review.v1', 'task': 'example',
                               'base_index_sha256': 'hash', 'decisions': [decision]}))
    with pytest.raises(ValueError, match='actual voter'):
        load_record_review(path, task='example', index_sha256='hash',
                           levels=[{'level': 1, 'endpoint_group': 'direct'}])


def test_opted_in_complete_aggregate_split_preserves_other_molecule_and_cleans_mirrors(tmp_path):
    index, decision = reassignment_fixture(tmp_path)
    decision.update(allow_complete_aggregate_split=True, resolved_compound_name='Correct target',
                    field_updates={'support_text': 'Corrected target negative result.'},
                    support_text_provenance=decision['identity_provenance'])
    sibling = {'support_text': 'Genuine source-molecule positive result.', 'reported_value': 'positive',
               'endpoint_type': 'tumour outcome', 'species_context': 'rat',
               'evidence_family': 'near', 'evidence_family_level': 2}
    for rows in index['evidence_by_molecule_group']['m'].values():
        row = rows[0]
        row.update(source_record_count=2, source_support_texts=['negative', sibling['support_text']],
                   standard_value='negative; positive', assay_description='mixed other-molecule summary',
                   source_molecule_names=['Wrong source name'], uncertainty=['mixed aggregate uncertainty'])
        row['source_record_examples'][0]['reported_value'] = 'negative'
        row['source_record_examples'].append(copy.deepcopy(sibling))
    row = index['evidence_by_molecule_group']['m']['a'][0]
    surface = _card_surface(row['source_record_examples'][0], row)
    decision.update(card_id=_card_id(stable_analog_id(index['molecules'][0]), surface),
                    surface_sha256=surface_sha256(surface))
    before = copy.deepcopy(index)
    sibling_surface = _card_surface(sibling, index['evidence_by_molecule_group']['m']['a'][0])
    result, receipt = apply_record_review(index, {'decisions': [decision]})
    assert index == before
    assert receipt['matched_occurrences_by_action']['reassign'] == 2
    for group in ['a', 'duplicate']:
        retained = result['evidence_by_molecule_group']['m'][group][0]
        moved = result['evidence_by_molecule_group']['target'][group][0]
        assert _card_surface(retained['source_record_examples'][0], retained) == sibling_surface
        assert retained['standard_value'] == 'positive' and retained['source_record_count'] == 1
        assert moved['source_record_count'] == 1
        assert moved['standard_value'] == 'negative'
        assert moved['minimal_evidence']['molecule']['names'] == ['Correct target']
        assert 'Genuine source-molecule' not in json.dumps(moved)
        assert 'mixed other-molecule summary' not in json.dumps(result)
        assert 'Wrong source name' not in json.dumps(result)
        assert moved['source_support_texts'] == ['Corrected target negative result.']
        assert moved['minimal_evidence']['examples'][0]['support_text'] == 'Corrected target negative result.'


@pytest.mark.parametrize('problem', ['incomplete', 'voter', 'missing_support'])
def test_opted_in_split_rejects_unrecoverable_aggregate(tmp_path, problem):
    index, decision = reassignment_fixture(tmp_path)
    decision['allow_complete_aggregate_split'] = True
    row = index['evidence_by_molecule_group']['m']['a'][0]
    row['source_record_count'] = 2
    sibling = {'support_text': 'other', 'evidence_family': 'near', 'evidence_family_level': 2}
    if problem == 'incomplete':row['source_record_count'] = 3
    elif problem == 'voter':sibling['evidence_family_level'] = 1
    else:sibling.pop('support_text')
    row['source_record_examples'].append(sibling)
    with pytest.raises(ValueError, match='complete nonvoter'):
        apply_record_review(index, {'decisions': [decision]})


def test_split_cannot_transfer_a_mixed_aggregate_value_without_source_correction(tmp_path):
    index, decision = reassignment_fixture(tmp_path)
    decision['allow_complete_aggregate_split'] = True
    for rows in index['evidence_by_molecule_group']['m'].values():
        row = rows[0]
        row.update(source_record_count=2, standard_value='mixed unrelated measurements')
        row['source_record_examples'].append({'support_text': 'other', 'evidence_family_level': 2})
    surface = _card_surface(row['source_record_examples'][0], row)
    decision.update(card_id=_card_id(stable_analog_id(index['molecules'][0]), surface), surface_sha256=surface_sha256(surface))
    with pytest.raises(ValueError, match='inherited card fields'):
        apply_record_review(index, {'decisions': [decision]})


def test_reassignment_can_correct_expanded_name_with_document_provenance(tmp_path):
    index, decision = reassignment_fixture(tmp_path)
    for rows in index['evidence_by_molecule_group']['m'].values():
        rows[0]['source_support_texts'] = ['negative']
        rows[0]['minimal_evidence']['text'] = {'evidence': 'negative'}
    before = copy.deepcopy(index)
    decision.update(field_updates={'support_text': 'Negative result for corrected test article.'},
                    support_text_provenance=decision['identity_provenance'])
    result, receipt = apply_record_review(index, {'decisions': [decision]})
    assert index == before
    assert receipt['matched_occurrences_by_action']['reassign'] == 2
    for rows in result['evidence_by_molecule_group']['target'].values():
        row = rows[0]
        assert row['canonical_smiles'] == 'CCC'
        ex = row['source_record_examples'][0]
        assert ex['support_text'] == decision['field_updates']['support_text']
        assert row['source_support_texts'] == [ex['support_text']]
        assert row['minimal_evidence']['text']['evidence'] == ex['support_text']
        assert ex['evidence_family'] == 'near'
        assert ex['endpoint_type'] == 'negative bacterial mutation'


def test_reassignment_can_also_fix_placement_without_losing_endpoint(tmp_path):
    index, decision = reassignment_fixture(tmp_path)
    before = copy.deepcopy(index)
    decision.update(target_level=4, target_family='dna')
    result, _ = apply_record_review(index, {'decisions': [decision]})
    assert index == before
    for rows in result['evidence_by_molecule_group']['target'].values():
        ex = rows[0]['source_record_examples'][0]
        assert ex['evidence_family'] == 'dna'
        assert ex['evidence_family_level'] == 4
        assert ex['support_text'] == 'negative'
    decision.update(target_level=1, target_family='direct')
    with pytest.raises(ValueError, match='voter membership'):
        apply_record_review(index, {'decisions': [decision]})


def test_reassignment_placement_requires_known_family(tmp_path):
    _, decision = reassignment_fixture(tmp_path)
    decision.update(target_level=4, target_family='unknown')
    path = tmp_path / 'review.json'
    path.write_text(json.dumps({'version': 'indexed_record_review.v1', 'task': 'example',
                               'base_index_sha256': 'hash', 'decisions': [decision]}))
    with pytest.raises(ValueError, match='unknown family'):
        load_record_review(path, task='example', index_sha256='hash',
                           levels=[{'level': 1, 'endpoint_group': 'direct'},
                                   {'level': 4, 'endpoint_group': 'dna'}])


@pytest.mark.parametrize('updates', [
    {'support_text': 'Unsupported rewrite'}, {'canonical_smiles': 'CCCC'},
    {'evidence_family': 'direct'}, {},
])
def test_reassignment_does_not_bypass_correction_validation(tmp_path, updates):
    index, decision = reassignment_fixture(tmp_path)
    decision['field_updates'] = updates
    with pytest.raises(ValueError):
        apply_record_review(index, {'decisions': [decision]})
    path = tmp_path / 'review.json'
    path.write_text(json.dumps({'version': 'indexed_record_review.v1', 'task': 'example',
                               'base_index_sha256': 'hash', 'decisions': [decision]}))
    with pytest.raises(ValueError):
        load_record_review(path, task='example', index_sha256='hash',
                           levels=[{'level': 1, 'endpoint_group': 'direct'}])


def test_review_cannot_grant_l1(tmp_path):
    _, decision = fixture_index()
    decision.update(target_level=1, target_family="direct")
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"version": "indexed_record_review.v1", "task": "example",
                               "base_index_sha256": "hash", "decisions": [decision]}))
    with pytest.raises(ValueError, match="voter membership"):
        load_record_review(path, task="example", index_sha256="hash",
                           levels=[{"level": 1, "endpoint_group": "direct"}])


def test_independent_reuse_can_resume_after_changed_level(tmp_path, monkeypatch):
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runner
    monkeypatch.setattr(runner, "_levels", lambda task: [{"level": i} for i in (1, 2, 3)])
    source, target = tmp_path / "old", tmp_path / "new"
    for level in (1, 2, 3):
        relative = f"example/queries/query_idx00000/levels/level_{level}"
        for root in (source, target):
            path = root / relative
            path.mkdir(parents=True)
            (path / "prepared.json").write_text(json.dumps({
                "level": level, "new_card_ids": ["changed" if level == 2 and root == target else "old"],
            }))
        (source / relative / "output.json").write_text(json.dumps({
            "status": "ok", "model_called": True, "state": {"prediction": "positive"},
        }))
    result = runner._reuse_unchanged_progressive_prefixes(
        prepared_queries=[runner.PreparedQuery("example", 0, target / "example/queries/query_idx00000")],
        output_root=target, source_root=source, independent_levels=True,
    )
    assert result["n_reused_levels"] == 2
    before = (target / "example/queries/query_idx00000/levels/level_1/output.json").read_bytes()
    runner._reuse_unchanged_progressive_prefixes(
        prepared_queries=[runner.PreparedQuery("example", 0, target / "example/queries/query_idx00000")],
        output_root=target, source_root=source, independent_levels=True,
    )
    assert (target / "example/queries/query_idx00000/levels/level_1/output.json").read_bytes() == before

    assert not (target / "example/queries/query_idx00000/levels/level_2/output.json").exists()
    assert (target / "example/queries/query_idx00000/levels/level_3/output.json").exists()


def test_matched_provider_switch_preserves_scientific_contract():
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_family_curve as family
    from tools.chembl_tool.paper_experiments import run_conditioned_assay_progressive_curve as runtime
    old = {'model': 'deepseek-ai/DeepSeek-V4-Flash-0731', 'base_url': 'local',
           'provider_pool': {'providers': ['local']}, 'inputs': {'hash': 'frozen'},
           'max_tokens': 20480, 'prepared_sha256': {'card': 'frozen'},
           'parallelism': 256, 'parallelism_per_task': 256, 'endpoint_concurrency_budget': 1024}
    current = {**old, 'model': 'deepseek/deepseek-v4-flash-0731', 'base_url': 'openrouter',
               'provider_pool': {'providers': ['openrouter']},
               'parallelism': 128, 'parallelism_per_task': 128, 'endpoint_concurrency_budget': 512}
    merged = family._merge_matched_resume_manifest(runtime, old, current)
    assert merged['provider_pool_history'] == [old['provider_pool'], current['provider_pool']]
    assert merged['inputs'] == old['inputs']
    assert merged['parallelism_per_task'] == 128 and merged['endpoint_concurrency_budget'] == 512
    for field, value in [('max_tokens', 8192), ('prepared_sha256', {'card': 'changed'}),
                         ('model', 'deepseek/deepseek-v4.1-flash')]:
        with pytest.raises(ValueError, match='cannot resume'):
            family._merge_matched_resume_manifest(runtime, old, {**current, field: value})


def test_corrected_endpoint_can_move_without_changing_original_support():
    index, decision = fixture_index()
    decision.update(action='correct', field_updates={'endpoint_type': 'cell viability'},
                    target_level=3, target_family='growth')
    result, _ = apply_record_review(index, {'decisions': [decision]})
    ex = result['evidence_by_molecule_group']['m']['a'][0]['source_record_examples'][0]
    assert ex['endpoint_type'] == 'cell viability'
    assert ex['evidence_family_level'] == 3
    assert ex['support_text'] == 'negative'
    assert index['evidence_by_molecule_group']['m']['a'][0]['source_record_examples'][0]['evidence_family_level'] == 2


def test_new_corrected_identity_rebuilds_fingerprint_and_disjoint_keys(tmp_path):
    from tools.chembl_tool.common.task_workflows.evidence_library import standardize_smiles_and_fp, fingerprint_metadata
    from tools.chembl_tool.common.starling.new_task_identity import VERSION
    from tools.chembl_tool.common.starling.new_task_retrieval_identity import query_identity
    from rdkit import DataStructs
    index, decision = fixture_index()
    index['fingerprint'] = fingerprint_metadata()
    index['fingerprints'] = [standardize_smiles_and_fp('CC')[2]]
    index['source'] = {'retrieval_identity_contract': VERSION, 'retrieval_identity_task': 'carcinogens'}
    doc = tmp_path / 'original.txt'; doc.write_text('Tested compound is ethanol.')
    decision.pop('target_level'); decision.pop('target_family')
    target, ik, fp = standardize_smiles_and_fp('CCO')
    decision.update(action='reassign', target_canonical_smiles=target, allow_new_target=True,
                    target_inchi_key=ik, identity_provenance={'document_path': str(doc),
                    'document_sha256': hashlib.sha256(doc.read_bytes()).hexdigest(), 'locator': 'identity'})
    edited, _ = apply_record_review(index, {'decisions': [decision]})
    assert len(index['molecules']) == 1 and len(edited['molecules']) == 2
    m = edited['molecules'][-1]
    assert m['canonical_smiles'] == 'CCO'
    assert DataStructs.TanimotoSimilarity(edited['fingerprints'][-1], fp) == 1
    assert m['retrieval_leakage_group'] == query_identity('carcinogens', target, VERSION)['leakage_group']
    assert edited['group_to_molecule_indices']['a'] == [1]
    assert m['molecule_chembl_id'] in edited['evidence_by_molecule_group']
    decision['target_inchi_key'] = 'wrong'
    with pytest.raises(ValueError, match='InChIKey mismatch'):
        apply_record_review(index, {'decisions': [decision]})
