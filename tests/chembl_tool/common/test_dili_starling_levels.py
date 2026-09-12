"""DILI family boundary checks: direct leakage, task scope and endpoint semantics."""
import json
from pathlib import Path
import pytest
from tools.chembl_tool.tasks.dili.starling_levels import classify, direct_guard
from tools.chembl_tool.common.starling.source_gold_review import payload_hash


def row(endpoint='', context='', support='', **extra):
    return {'source_id': 'arbitrary_new_source', 'retrieval_eligible': True,
            'canonical_endpoint_name': endpoint, 'canonical_assay_context': context,
            'support_text': support, **extra}


@pytest.mark.parametrize('text', [
    'No hepatotoxicity was detected in patients.',
    'ATP depletion was measured; the same drug causes clinical liver injury.',
    'The model predicts DILI risk.',
    '4-PBA reduced APAP-induced liver injury in mice.',
    'The biopsy revealed hepatic fibrosis under drug treatment.',
    'Serum ALT was increased after dosing.',
])
def test_direct_containment_is_not_gold_voting(text):
    r=row('mitochondrial membrane potential', 'hepatocytes', text)
    assert direct_guard(r)
    assert classify(r)['level']==2


def test_actual_voter_membership_required_and_identity_hold_precedes_it():
    r=row(support='The drug caused DILI in this patient.')
    assert classify(r)['level']==2
    assert classify(r,is_voter=True)['level']==1
    assert classify({**r,'identity_review_reason':'name_structure_mismatch'},is_voter=True)['level']==0


def test_baseline_viral_hepatitis_does_not_make_redox_a_dili_outcome():
    r=row('lipid_peroxidation serum hydroperoxide', 'human patients with chronic hepatitis C',
          'Ribavirin and interferon reduced serum hydroperoxides during therapy.')
    assert not direct_guard(r)
    assert classify(r)['level']==7


def test_measured_endpoint_precedes_platform_and_source_run():
    r=row('lysosomal_or_autophagic_function immunofluorescence co-staining of Lamp1',
          'mouse hepatoma cells','Lamp1/cathepsin colocalization decreased.')
    assert classify(r)['level']==7
    assert classify({**r,'source_id':'dili_v4'})==classify({**r,'source_id':'dili_v1'})
    assert classify(row('bile-acid concentration','tilapia liver','Targeted metabolomics showed lower bile acids.'))['level']==4


def test_explicit_nonhepatic_ros_excluded_but_normal_hepatic_comparator_retained():
    assert classify(row('reactive_species ROS','A549 lung cells','GSH reduced ROS.'))['level']==0
    assert classify(row('viability MTT','normal L-O2 liver cells','Anticancer compound IC50 in normal cells.'))['level']==3


def test_no_source_only_fallback():
    assert classify({'source_id':'dili_v2','retrieval_eligible':True})['level']==0


def test_review_payloads_are_hash_bound_and_direct_not_left_in_mechanisms():
    root=Path('data/starling_data/dili/level_review_v1')
    records={r['source_row_uid']:r for r in map(json.loads,(root/'sample.jsonl').read_text().splitlines())}
    decisions=list(map(json.loads,(root/'placement_decisions.jsonl').read_text().splitlines()))
    assert len(decisions)>=60
    assert len({r['source_row_uid'] for r in decisions})==len(decisions)
    for d in decisions:
        r=records[d['source_row_uid']]
        assert payload_hash(json.loads(r['raw_record_json']))==d['source_payload_sha256']
        if d['level']>=3:
            assert not direct_guard(r) or d.get('direct_guard_exemption'), d['source_row_uid']
