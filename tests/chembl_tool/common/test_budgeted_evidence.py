"""Budgeted objective, rules and selector controls under a fixed candidate pool."""
import itertools
import math
import random
from copy import deepcopy

import pytest

from tools.chembl_tool.common.budgeted_evidence import (
    BUDGET_GRID, DEFAULT_BUDGET, SelectionConfig, normalized_features, objective_value,
    score_candidate, select_records,
)
from tools.chembl_tool.common.record_diversity import prepare_record
from tools.chembl_tool.common.record_selection_context import record_quality


def record(name, molecule='m', group='g', p=1., features=None):
    return dict(id=name, molecule_id=molecule, group=group, utility=p, quality=p, transfer=1.,
                similarity=.5, features=features if features is not None else {'x': 1.})


def test_defaults_and_invalid_contract():
    assert DEFAULT_BUDGET == 64 and BUDGET_GRID == (16, 64, 128)
    for kwargs in [dict(budget=-1), dict(weights=(1, 1, 1)), dict(weights=(math.nan, 0, 0)), dict(per_molecule=0)]:
        with pytest.raises(ValueError):
            SelectionConfig(**kwargs)
    with pytest.raises(ValueError, match='duplicate'):
        select_records([record('a'), record('a')])
    with pytest.raises(ValueError, match='mass'):
        select_records([record('a', features={'x': 1., 'y': 1.})])


def test_objective_numerical_example_and_group_counts_molecule_once():
    a, b, c = record('a'), record('b'), record('c', molecule='n')
    assert objective_value([a, b], (1, 0, 0)) == 2
    assert objective_value([a, b], (0, 1, 0)) == pytest.approx(math.sqrt(2))
    assert objective_value([a, b], (0, 0, 1)) == 1
    assert objective_value([a, b, c], (0, 0, 1)) == pytest.approx(math.sqrt(2))
    assert objective_value([a, record('weak', p=.1)], (0, 0, 1)) == 1
    assert objective_value([a, record('different', features={'y': 1.})], (0, 1, 0)) == 2


def test_submodular_diminishing_returns_all_small_subsets():
    pool = [record('a'), record('b', p=.5), record('c', molecule='n'),
            record('d', group='h'), record('e', features={'x': .25, 'y': .25})]
    for n in range(len(pool)):
        for positions in itertools.combinations(range(len(pool)), n):
            small = [pool[i] for i in positions]
            for i in set(range(len(pool))) - set(positions):
                for j in set(range(len(pool))) - set(positions) - {i}:
                    large = small + [pool[j]]
                    gain = objective_value(small+[pool[i]]) - objective_value(small)
                    later = objective_value(large+[pool[i]]) - objective_value(large)
                    assert gain >= later - 1e-12 and later >= -1e-12


def test_lazy_greedy_matches_naive_and_has_nested_budgets():
    rng = random.Random(15)
    pool = [record(str(i), str(i%4), str(i%3), rng.random(), {str(i%5): 1.}) for i in range(30)]
    naive = []
    for _ in range(12):
        remaining = [r for r in pool if r not in naive]
        remaining.sort(key=lambda r: (-r['similarity'], r['id']))
        naive.append(max(remaining, key=lambda r: objective_value(naive+[r])-objective_value(naive)))
    chosen, audit = select_records(pool, SelectionConfig(budget=12))
    assert [r['id'] for r in chosen] == [r['id'] for r in naive]
    assert chosen[:5] == select_records(list(reversed(pool)), SelectionConfig(budget=5))[0]
    assert audit['objective'] == pytest.approx(objective_value(chosen))
    assert all(0 <= t['gain'] <= t['utility']+1e-12 for t in audit['selection_trace'])


def test_quality_only_matches_joint_a1_and_zero_utility_does_not_fill():
    pool = [record('a', p=.1), record('b', p=.9), record('c', p=0)]
    assert select_records(pool, SelectionConfig(policy='quality'))[0] == select_records(pool, SelectionConfig(weights=(1,0,0)))[0]
    assert len(select_records(pool)[0]) == 2
    assert len(select_records(pool, SelectionConfig(policy='neighbor_fill'))[0]) == 3


def test_random_controls_are_deterministic_caps_and_groups_work():
    pool = [record(str(i), str(i//5), str(i%2)) for i in range(20)]
    for policy in ('neighbor_fill', 'molecule_cap', 'group_balanced'):
        config = SelectionConfig(budget=8, policy=policy, per_molecule=2)
        selected = select_records(pool, config)[0]
        assert selected == select_records(list(reversed(pool)), config)[0]
        if policy == 'molecule_cap':
            assert max(sum(r['molecule_id']==m for r in selected) for m in {r['molecule_id'] for r in selected}) <= 2
        if policy == 'group_balanced':
            assert {r['group'] for r in selected[:2]} == {'0', '1'}
    assert {r['molecule_id'] for r in select_records(pool, SelectionConfig(policy='neighbor_fill', budget=5))[0]} == {'0'}


def prepared(endpoint='bacterial reverse mutation', result='positive', **fields):
    row = dict(canonical_record_id='r', molecule_name='compound', canonical_smiles='CCO',
        canonical_endpoint_name=endpoint, canonical_measurement_text=result,
        canonical_assay_context='bacterial assay', support_text='The compound was tested.',
        tested_entity_role='test_compound', **fields)
    return prepare_record(row), {'endpoint': endpoint, 'reported_value': result}


def test_quality_direction_symmetric_unknown_nonzero_offtask_zero():
    pos, c = prepared()
    neg, n = prepared(result='negative')
    assert record_quality('ames', '', pos, c)['score'] == record_quality('ames', '', neg, n)['score']
    unknown, u = prepared(endpoint='unrecognized measurement')
    off, o = prepared(endpoint='antibacterial minimum inhibitory concentration')
    qu = record_quality('ames', '', unknown, u)
    assert qu['statuses']['relevance'] == 'unknown' and 0 < qu['score'] < record_quality('ames', '', pos, c)['score']
    assert record_quality('ames', '', off, o)['score'] == 0


def test_condition_and_hepatic_scope_change_quality_not_gold_label():
    rodent, c = prepared(endpoint='tumor incidence', canonical_species_context='rat')
    assert record_quality('carcinogens', 'species=rodent', rodent, c)['score'] > record_quality('carcinogens', 'species=human', rodent, c)['score']
    liver, c = prepared(endpoint='intracellular ROS', biological_system='human hepatocytes')
    generic, _ = prepared(endpoint='intracellular ROS', biological_system='HeLa cells')
    assert record_quality('dili', '', liver, c)['score'] > record_quality('dili', '', generic, c)['score']


def test_missing_system_is_not_a_separator_and_native_hepatic_model_is_read():
    source = prepare_record(dict(canonical_record_id='r', canonical_endpoint_name='brain uptake',
        canonical_measurement_text='decreased', support_text='Uptake decreased.'))
    card = {'endpoint': 'brain uptake', 'reported_value': 'decreased'}
    assert not record_quality('bbb_martins', '', source, card)['completeness_checks']['assay_or_system']
    liver, card = prepared(endpoint='autophagic activity', biological_model_context='LO2 hepatocytes')
    assert record_quality('dili', '', liver, card)['components']['relevance'] == 1


def test_unclear_semantic_attribution_is_not_explicitly_wrong_object():
    source, card = prepared()
    annotation = {'outcomes': [{'endpoint': 'mutation', 'kind': 'positive', 'focal_scope': 'unclear'}]}
    quality = record_quality('ames', '', source, card, annotation)
    assert quality['components']['object'] == .5
    assert quality['statuses']['object'] == 'unknown'


@pytest.mark.parametrize('task,endpoint', [
    ('ames', 'single-nucleotide variants'), ('ames', 'indels'), ('ames', 'CNV'),
    ('carcinogens', 'pheochromocytoma induction'), ('skin_reaction', 'allergenicity'),
    ('dili', 'formation of thiol conjugates'),
])
def test_semantic_endpoint_aliases_keep_known_task_relevance(task, endpoint):
    source, card = prepared(endpoint=endpoint)
    assert record_quality(task, '', source, card)['statuses']['relevance'] == 'related'


def test_explicit_patient_age_matches_query_without_reading_doses_as_age():
    young, card = prepared(endpoint='drug-induced liver injury', qualifying_conditions='31-year-old woman')
    old, _ = prepared(endpoint='drug-induced liver injury', qualifying_conditions='71-year-old woman')
    dose, _ = prepared(endpoint='drug-induced liver injury', qualifying_conditions='31 mg/kg daily')
    values = [record_quality('dili', 'age_group=older_than_65', p, card) for p in (young, old, dose)]
    assert [v['statuses']['condition'] for v in values] == ['mismatched', 'matched', 'unknown']


def test_content_features_ignore_field_aliases_repetition_and_species_spelling():
    p, _ = prepared()
    p['groups']['system'] = {'species':'rat'}
    alternate = deepcopy(p)
    alternate['groups']['endpoint'] = {'endpoint_name':p['endpoint'], 'another_native_alias':p['endpoint']}
    alternate['groups']['system'] = {'biological_system':'Rattus norvegicus'}
    assert normalized_features('ames',p) == normalized_features('ames',alternate)
    changed = deepcopy(alternate)
    changed['groups']['system'] = {'species':'mouse'}
    assert normalized_features('ames',p) != normalized_features('ames',changed)


def test_feature_numbers_ignore_duplicate_raw_encoding_and_keep_sign():
    p,_=prepared(endpoint='permeability',result='0.0000000191',
        canonical_unit_text='cm/s',measurement_text='1.91 × 10^-8',unit_text='cm/s',
        finite_scalar_value='1.91e-08',embedded_unit='False')
    alternate=deepcopy(p)
    alternate['groups']['outcome'].update(canonical_measurement_text='1.910e-8',
        measurement_text='0.191',unit_text='nm/s',embedded_unit='True',finite_scalar_value='0.0000000191')
    assert normalized_features('bbb_martins',p)==normalized_features('bbb_martins',alternate)
    negative=deepcopy(p)
    negative['groups']['outcome']['canonical_measurement_text']='-0.0000000191'
    assert normalized_features('bbb_martins',p)!=normalized_features('bbb_martins',negative)


def test_assessed_readout_features_do_not_inherit_native_direction_labels():
    p,_=prepared()
    a=assessed()
    alternative=deepcopy(p)
    alternative['groups']['endpoint']['endpoint_name']='wrong native label'
    alternative['groups']['outcome']['result_label']='opposite native direction'
    assert normalized_features('ames',p,a)==normalized_features('ames',alternative,a)


def assessed(kind='positive', basis='measured', limitations=(), scope='focal_effect'):
    return {'outcomes':[{'endpoint':'bacterial reverse mutation','kind':kind,'focal_scope':'focal',
        'evidence_basis':basis,'result_scope':scope,'limitations':[{'kind':l} for l in limitations],
        'conditions':[],'comparator':[]}], 'unresolved':['Additional background has no dose reported.']}


def test_assessment_distinguishes_experiment_inference_and_inadequate_studies():
    p,c = prepared()
    def score(a):return record_quality('ames','',p,c,a)['score']
    measured = score(assessed())
    assert measured == score(assessed(kind='negative')) == 1
    assert measured > score(assessed(basis='inferred')) > score(assessed(kind='not_reported',basis='methods_only'))
    assert score(assessed(kind='unclear',basis='classification',limitations=['insufficient_evaluation'])) <= .2
    assert score(assessed(kind='negative',limitations=['method_limitation'])) <= .5
    assert score(assessed(scope='perturbation_effect')) < measured


def test_weak_background_does_not_cap_the_usable_bound_observation():
    p,c=prepared()
    a=assessed(); a['outcomes'] += assessed(kind='not_reported',basis='methods_only')['outcomes']
    result=record_quality('ames','',p,c,a)
    assert result['score']==1 and result['observation_index']==0
    assert len(result['observation_scores'])==2


def test_native_narrative_codes_do_not_receive_authoritative_classification_strength():
    p,c = prepared()
    a = assessed(basis='classification')
    a['outcomes'][0]['endpoint'] = 'carcinogenicity'
    a['outcomes'][0]['basis_evidence'] = [
        {'field':'carcinogenic_role','quote':'general_or_unspecified_carcinogen'},
        {'field':'evidence_basis','quote':'review_list_or_narrative_assertion'}]
    result = record_quality('carcinogens','',p,c,a)
    assert result['components']['evidence_strength'] == .35
    assert 'classification_supported_only_by_source_codes' in result['reasons']
    # A reported evidence evaluation provides an independent narrative basis.
    a['outcomes'][0]['basis_evidence'].append(
        {'field':'support_text','quote':'IARC evaluated the agent as carcinogenic to humans.'})
    assert record_quality('carcinogens','',p,c,a)['components']['evidence_strength'] == .6


def test_normalization_and_transfer_affect_all_terms():
    p, card = prepared()
    features = normalized_features('ames', p)
    assert sum(math.sqrt(v) for v in features.values()) == pytest.approx(1.)
    candidate = {**record('r'), 'card': card}
    full = score_candidate('ames', '', candidate, p)
    half = score_candidate('ames', '', candidate, p, transfer=.5)
    for weights in [(1,0,0), (0,1,0), (0,0,1)]:
        assert objective_value([half], weights) == pytest.approx(objective_value([full], weights)/2)
    with pytest.raises(ValueError):
        score_candidate('ames', '', candidate, p, transfer=1.1)


def test_duplicate_source_ambiguity_uses_minimum_quality_and_common_features():
    from tools.chembl_tool.common.budgeted_evidence import score_bound_candidate
    good, c = prepared(canonical_species_context='rat')
    weak, _ = prepared(endpoint='unrecognized measurement', canonical_species_context='human')
    bound = {'prepared':good, 'alternatives':[good,weak], 'ambiguous_native_metadata':True,
             'source_record_ids':['a','b'], 'source_input_hashes':['x','y']}
    candidate = {**record('r'), 'card':c}
    merged = score_bound_candidate('ames','',candidate,bound)
    individual = [score_candidate('ames','',candidate,p) for p in [good,weak]]
    assert merged['quality'] == min(r['quality'] for r in individual)
    assert set(merged['features']) <= set(individual[0]['features']) & set(individual[1]['features'])
    assert merged['quality_audit']['ambiguous_native_metadata']


def test_bound_conditions_preserve_unchanged_species_and_override_strain():
    p,c=prepared(metabolic_activation='absent',test_system='TA102')
    a=assessed();a['outcomes'][0]['endpoint']='bacterial_reverse_mutation'
    a['outcomes'][0]['conditions']=[{'field':'canonical_assay_context','quote':'Salmonella typhimurium TA102'},
                                  {'field':'qualifying_conditions','quote':'tested without S9.'}]
    q=record_quality('ames','metabolic_activation=absent+strain_panel=TA98',p,c,a)
    assert q['components']['condition']==0
    p,c=prepared(endpoint='tumor incidence',canonical_species_context='human')
    a['outcomes'][0].update(endpoint='tumor incidence',conditions=[{'field':'qualifying_conditions','quote':'higher dose'}])
    assert record_quality('carcinogens','species=rodent',p,c,a)['components']['condition']==0


def test_source_replicate_arithmetic_conflict_is_not_a_new_label():
    p,c=prepared(endpoint='CD83',experimental_conditions='ratio >130 considered significant.',
        extra_details='Ratios: 107, 175, 133, 123, 69; significant in 1/5 experiments.')
    a=assessed();a['outcomes'][0]['endpoint']='CD83'
    q=record_quality('skin_reaction','',p,c,a)
    assert q['score']<=.35 and 'replicate_threshold_count_conflict' in q['reasons']
    assert a['outcomes'][0]['kind']=='positive'


@pytest.mark.parametrize('system', [
    'mouse RAW264.7 macrophages; no hepatic cell model.',
    'mouse RAW264.7 macrophages; non-hepatic cell model.',
    'mouse RAW264.7 macrophages without liver cells.',
])
def test_negated_hepatic_scope_does_not_rescue_dili_relevance(system):
    p,c=prepared(endpoint='cell viability',experimental_system=system)
    a=assessed();a['outcomes'][0]['endpoint']='cell viability'
    q=record_quality('dili','',p,c,a)
    assert q['components']['relevance']==.5
    assert 'mechanism_without_identified_hepatic_scope' in q['reasons']
    p,c=prepared(endpoint='liver injury',experimental_system=system)
    a['outcomes'][0].update(endpoint='liver injury',kind='negative')
    assert record_quality('dili','',p,c,a)['components']['relevance']==1


def test_plant_redox_is_distant_evidence_and_equivocal_is_not_negative():
    p,c=prepared(endpoint='H2O2 level',biological_system='wounded leaves of Pelargonium peltatum')
    a=assessed();a['outcomes'][0]['endpoint']='H2O2 level'
    q=record_quality('carcinogens','',p,c,a)
    assert q['components']['relevance']==.25
    p,c=prepared(endpoint='patch test')
    a=assessed(kind='unclear',limitations=['inconclusive_result']);a['outcomes'][0]['endpoint']='patch test'
    assert record_quality('skin_reaction','',p,c,a)['score']<=.5
    assert a['outcomes'][0]['kind']=='unclear'


def test_assessed_features_ignore_direction_codes_embedded_in_measurement_text():
    p,_ = prepared(result='effect_direction=increased | endpoint_result=4.2', canonical_unit_text='fold')
    other = deepcopy(p)
    other['groups']['outcome']['canonical_measurement_text'] = 'effect_direction=decreased | endpoint_result=4.2'
    a = assessed(kind='numeric_only')
    assert normalized_features('ames', p, a) == normalized_features('ames', other, a)
    changed = deepcopy(other)
    changed['groups']['outcome']['canonical_measurement_text'] = 'effect_direction=decreased | endpoint_result=1.2'
    assert normalized_features('ames', p, a) != normalized_features('ames', changed, a)
    # Native-only diagnostics retain the source's stated direction.
    assert normalized_features('ames', p) != normalized_features('ames', other)


def test_shared_visible_paragraph_contributes_overlap_across_assay_foci():
    p,_=prepared()
    p['input']['support_text']='Mutation was negative whereas micronucleus induction was positive.'
    a=assessed(kind='negative');b=assessed(kind='positive')
    b['outcomes'][0]['endpoint']='micronucleus induction'
    f=normalized_features('ames',p,a);g=normalized_features('ames',p,b)
    shared={k for k in f if k.startswith('source_content/')}
    assert shared and shared=={k for k in g if k.startswith('source_content/')}
    assert f!=g  # Different readouts remain represented.
    assert math.isclose(sum(math.sqrt(v) for v in f.values()),1)
    changed=deepcopy(p);changed['input']['support_text']='DNA repair increased after irradiation.'
    assert normalized_features('ames',changed,a)!=f


def test_bibliography_in_unstructured_notes_is_not_exposure_diversity():
    p,_=prepared(extra_details='Natsch et al. 2013; Urbisch et al. 2015.',
                 qualifying_conditions='extra_details=Natsch et al. 2013.',
                 exposure_duration='24 h')
    a=assessed();a['exposure']=['24 h']
    f=normalized_features('ames',p,a)
    changed=deepcopy(p)
    changed['groups']['exposure'].update(extra_details='Different authors 2020.',
                                        qualifying_conditions='extra_details=Different authors 2020.')
    assert normalized_features('ames',changed,a)==f
    assert not any(k.startswith('exposure/') and ('natsch' in k or 'urbisch' in k) for k in f)
    assert 'exposure/24' in f
    assert p['groups']['exposure']['extra_details'].startswith('Natsch')


def test_group_balance_skips_capped_records_within_each_turn():
    # Group a's exhausted nearest molecule must not delay eligible a records
    # while b gets extra turns. No quality/features are supplied.
    pool = [dict(id=f'{g}{m}{i}', molecule_id=m, group=g, similarity=sim)
            for g,m,n,sim in [('a','near',12,.9),('a','far',10,.5),('b','other',20,.8)]
            for i in range(n)]
    selected, audit = select_records(pool, SelectionConfig(policy='group_balanced', budget=8, per_molecule=3))
    assert [r['group'] for r in selected[:6]] == ['a','b']*3
    assert sum(r['molecule_id']=='near' for r in selected) == 3
    assert sum(r['molecule_id']=='other' for r in selected) == 3
    assert len(selected) == 8 and audit['assessment_required'] is False
