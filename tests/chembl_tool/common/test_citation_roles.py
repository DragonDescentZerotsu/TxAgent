import json

import pytest

from tools.chembl_tool.common.budgeted_evidence import content_hash
from tools.chembl_tool.paper_experiments.analyze_progressive_trace_adoption import (
    _budgeted_citation_row, citation_role_metrics,
)


def test_roles_deduplicate_keep_overlap_and_separate_evidence_pools():
    cards = {'C01': {'parent': 'p1', 'kind': 'direct'},
             'C02': {'parent': 'p1', 'kind': 'indirect'},
             'C03': {'parent': 'p2', 'kind': 'indirect'}}
    result = citation_role_metrics({'supportive_card_ids': ['C01', 'C01', 'C02', 'C03'],
                                    'contradictory_card_ids': ['C02']}, cards)
    support = result['roles']['supportive']
    assert support['all']['records'] == 3 and support['all']['parents'] == 2
    assert support['all']['effective_parents'] == pytest.approx(9/5)
    assert support['direct']['effective_parents'] == 1
    assert support['indirect']['effective_parents'] == 2
    assert result['roles']['combined']['all']['records'] == 3
    assert result['overlap_records'] == result['overlap_parents'] == 1
    assert result['roles']['contradictory']['indirect']['record_fraction'] == .5


def test_empty_citations_and_empty_context_have_distinct_denominators():
    cards = {'C01': {'parent': 'p1', 'kind': 'direct'}}
    result = citation_role_metrics({'claims': []}, cards)['roles']['supportive']
    assert result['all']['effective_parents'] is None
    assert result['all']['record_fraction'] == 0
    assert result['indirect']['record_fraction'] is None
    assert result['all']['parents'] == result['all']['records'] == 0


def test_claim_roles_support_upstream_schema_and_reject_invisible_citations():
    cards = {'C01': {'parent': 'p1', 'kind': 'direct'}}
    content = {'claims': [{'card_ids': ['C01'], 'evidence_role': 'contradictory'}]}
    assert citation_role_metrics(content, cards)['roles']['contradictory']['all']['records'] == 1
    with pytest.raises(ValueError, match='absent'):
        citation_role_metrics(content, {})


def test_trace_reader_checks_binding_and_skips_unselected_ledger(tmp_path):
    query = tmp_path/'runs/ames/query_00000'
    query.mkdir(parents=True)
    messages = [{}, {'content': json.dumps({'neighbors': [{'records': [
        {'card_id': 'C01', 'evidence_kind': 'direct'}]}]})}]
    prepared = {'manifest_hash': 'manifest', 'request_hash': content_hash(messages),
                'messages': messages, 'card_alias_map': {'C01': 'stable'},
                'direct_ids': ['stable'], 'indirect_ids': []}
    output = {'status': 'ok', 'request_hash': prepared['request_hash'],
              'card_alias_map': prepared['card_alias_map'],
              'llm': {'content': {'supportive_card_ids': ['C01'], 'contradictory_card_ids': []}}}
    (query/'prepared.json').write_text(json.dumps(prepared))
    (query/'output.json').write_text(json.dumps(output))
    (query/'selection.json').write_text(json.dumps({'direct': [
        {'id': 'stable', 'molecule_id': 'parent', 'kind': 'direct'}],
        'selected': [], 'candidates': [{'irrelevant': 'x'*100000}]}))
    job = ('ames', 'direct_cap', 0, tmp_path, 'manifest')
    assert _budgeted_citation_row(job)['roles']['supportive']['all']['parents'] == 1
    output['request_hash'] = 'changed'
    (query/'output.json').write_text(json.dumps(output))
    with pytest.raises(ValueError, match='binding'):
        _budgeted_citation_row(job)
