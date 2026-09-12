import json

from tools.chembl_tool.tasks.carcinogens.starling_levels import classify, direct_guard


def record(endpoint, text='', source='carcinogens_v4', extra=''):
    return {'source_id': source, 'canonical_endpoint_name': endpoint,
            'support_text': text, 'retrieval_eligible': True,
            'raw_record_json': json.dumps({'support_text': text, 'extra_details': extra})}


def test_only_actual_voters_receive_l1():
    row = record('tumor induction', 'The carcinogen induced liver tumors in rats.')
    assert classify(row)['level'] == 2
    assert classify(row, is_voter=True)['level'] == 1
    row['retrieval_eligible'] = False
    assert classify(row, is_voter=True)['level'] == 0


def test_embedded_extra_details_and_other_agent_assertion_contained():
    row = record('glutathione_redox', 'Melatonin reduced adduct formation.',
                 extra='Safrole is the administered carcinogen; melatonin protects.')
    assert direct_guard(row)
    assert classify(row)['level'] == 2


def test_site_negative_xenograft_and_prediction_never_mechanism():
    for text in ['No lung tumors were induced in the NTP study.',
                 'DPN increased tumor volume in MCF7 xenografts.',
                 'Structural alerts suggest no carcinogenic potential.']:
        assert classify(record('dna_methylation', text))['level'] == 2


def test_cell_line_cancer_name_is_not_hazard_assertion():
    row = record('cell proliferation', 'Curcumol reduced proliferation of breast cancer cells.')
    assert not direct_guard(row)
    assert classify(row)['level'] == 3


def test_cross_source_endpoint_assignments_and_uncertain_direction_retained():
    for endpoint, expected in [('mammalian_cell_gene_mutation', 4),
                               ('glutathione_depletion', 5),
                               ('gap_junction_communication', 6),
                               ('regenerative_or_compensatory_proliferation', 7)]:
        for source in ['carcinogens_v1', 'carcinogens_v5']:
            row = record(endpoint, 'Equivocal result; further context required.', source=source)
            assert classify(row)['level'] == expected


def test_domain_alone_cannot_promote_motor_behavior_to_regeneration():
    row = record('phenotype_domain=injury_driven_regeneration',
                 'Rotarod and inverted-screen motor strength were unchanged by ketoprofen.')
    assert classify(row)['level'] == 0
