import pytest

from tools.chembl_tool.tasks.dili.build_reviewed_starling import validate, study_key, reused_metadata


def record():
    return {'pmid':'123456','support_text':'Our patient had no liver injury. Smith et al. (2001) reported another cohort.',
            'qualifying_conditions':'Children aged 8 to 12 years.', 'extra_details':None}


def verdict():
    return {'study_kind':'own','study_ref':'123456','study_quote':'Our patient',
            'conditions':[], 'reason':'A directly reported individual observation.'}


def test_does_not_reject_individual_negative_or_missing_conditions():
    assert validate(verdict(),record())['conditions']==[]
    assert study_key(verdict())=='pmid:123456'


def test_no_invented_cited_pmid_or_condition_quote():
    v=verdict();v.update(study_kind='cited_pmid',study_ref='987654')
    with pytest.raises(ValueError,match='explicitly'):validate(v,record())
    v=verdict();v['conditions']=[{'axis':'age_group','value':'pediatric','quote':'A young child'}]
    with pytest.raises(ValueError,match='verbatim'):validate(v,record())


def test_author_year_requires_literal_grounding_and_normalizes_reference():
    v=verdict();v.update(study_kind='author_year',study_ref='Smith et al. (2001)',study_quote='Smith et al. (2001)')
    validate(v,record())
    assert study_key(v)=='cited:smith:2001'
    v['study_ref']='Jones 2001'
    with pytest.raises(ValueError,match='grounded'):validate(v,record())


def test_unresolved_study_is_explicit_not_a_reporting_pmid_vote():
    v=verdict();v.update(study_kind='unresolved',study_ref='',study_quote='')
    validate(v,record())
    assert study_key(v) is None


def test_reuse_accepts_review_assertion_and_scoped_negative_without_model_gate():
    raw=record()
    raw.update(human_evidence_basis='review_or_reference_assertion',qualifying_conditions=None)
    value=reused_metadata(raw)
    assert value['study_kind']=='reporting_pmid'
    assert study_key(value)=='pmid:123456'
    assert value['conditions']==[]


def test_reuse_pools_formulation_and_indication_but_keeps_overdose():
    raw=record()
    raw.update(exposure_context='overdose_or_supratherapeutic',qualifying_conditions='Extended release; rheumatoid arthritis.')
    prior={'condition_atoms':['formulation=long_acting_injectable','population=human','population_context=rheumatoid_arthritis']}
    assert reused_metadata(raw,prior)['conditions']==[
        {'axis':'exposure','value':'overdose_or_supratherapeutic','quote':''}]


def test_reuse_does_not_turn_diagnostic_exclusion_or_post_injury_transplant_into_host():
    raw=record()
    raw.update(qualifying_conditions='HCV negative; no history of liver disease. Required liver transplantation after injury.')
    assert reused_metadata(raw)['conditions']==[]
    raw['qualifying_conditions']='Pre-existing chronic HCV infection.'
    assert reused_metadata(raw)['conditions']==[
        {'axis':'population_context','value':'hepatitis_c','quote':''}]


def test_reuse_coarse_age_and_unknown_qualifier_do_not_reject_record():
    raw=record()
    raw.update(support_text='A 72-year-old woman had no injury.',qualifying_conditions='A rare indication with unspecified dose restrictions.')
    assert reused_metadata(raw)['conditions']==[
        {'axis':'age_group','value':'older_than_65','quote':''}]


@pytest.mark.parametrize('qualifier',['Pediatric population.','Paediatric population (<18 years).'])
def test_both_spellings_map_to_same_pediatric_group(qualifier):
    raw=record();raw['qualifying_conditions']=qualifier
    assert reused_metadata(raw)['conditions']==[
        {'axis':'age_group','value':'pediatric','quote':''}]
