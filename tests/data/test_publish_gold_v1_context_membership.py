from __future__ import annotations

import json

import pytest

from data.processing.gold_labels import publish_gold_v1_context_membership as publisher
from predict.utils.json import sha256_file


def test_reviewed_voter_exclusion_updates_mean_without_changing_label(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = tmp_path / "pinned.txt"
    pinned.write_text("source authority\n", encoding="utf-8")
    correction = tmp_path / "correction.json"
    correction.write_text(
        json.dumps(
            {
                "version": "test.v1",
                "status": "reviewed",
                "task": "dili",
                "inputs": [{"path": "pinned.txt", "sha256": sha256_file(pinned)}],
                "decisions": [
                    {
                        "action": "exclude",
                        "source_row_uid": "remove",
                        "expected_split": "train",
                        "expected_benchmark_row_id": "card",
                        "expected_molecule_identity_key": "molecule",
                        "expected_condition_group": "condition",
                        "expected_vote_label": 1,
                        "expected_source_payload_sha256": "payload",
                        "expected_gold_label_before": 1,
                        "expected_gold_label_after": 1,
                        "expected_voter_mean_before": 2 / 3,
                        "expected_voter_mean_after": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(publisher.TASKS, "dili", {"corrections": "correction.json"})
    cards = [
        (
            "train",
            {
                "benchmark_row_id": "card",
                "molecule_identity_key": "molecule",
                "condition_group": "condition",
                "Y": 1,
                "source_record_ids": ["keep-positive", "remove", "keep-negative"],
                "source_votes": [
                    {
                        "source_record_id": "keep-positive",
                        "Y": 1,
                        "source_payload_sha256": "a",
                    },
                    {
                        "source_record_id": "remove",
                        "Y": 1,
                        "source_payload_sha256": "payload",
                    },
                    {
                        "source_record_id": "keep-negative",
                        "Y": 0,
                        "source_payload_sha256": "b",
                    },
                ],
            },
        )
    ]
    corrected, provenance = publisher._apply_reviewed_corrections("dili", cards)

    assert corrected[0][1]["source_record_ids"] == ["keep-positive", "keep-negative"]
    assert provenance is not None
    assert provenance["applied"] == [
        {
            "source_row_uid": "remove",
            "benchmark_row_id": "card",
            "old_voter_mean": 2 / 3,
            "new_voter_mean": 0.5,
            "old_vote_count": 3,
            "new_vote_count": 2,
            "gold_label": 1,
        }
    ]


@pytest.mark.parametrize('failure', [None, 'weight', 'nan_weight', 'missing_uid', 'gold_label', 'duplicate', 'wrong_condition'])
def test_membership_audit_checks_vote_weights_labels_and_canonical_rows(tmp_path, failure):
    import pyarrow as pa
    import pyarrow.parquet as pq
    gold, library = tmp_path / 'gold', tmp_path / 'library'
    gold.mkdir()
    (library / '03_pair_buckets').mkdir(parents=True)
    card = dict(benchmark_row_id='card', drug='CC', Y=1,
                molecule_identity_key='parent', condition_group='condition')
    for split in publisher.SPLITS:
        (gold / f'{split}.jsonl').write_text(json.dumps(card) + '\n' if split == 'train' else '')
    unit = dict(benchmark_row_id='card', split='train', vote_id='claim', vote_label=1,
                molecule_identity_key='parent', condition_group='condition', physical_member_count=2,
                physical_source_row_uids=['sr_a', 'sr_b'], physical_source_record_ids=['a', 'b'])
    members = [{**unit, 'source_row_uid': uid, 'physical_vote_weight': 0.5}
               for uid in ('sr_a', 'sr_b')]
    index = dict(benchmark_row_id='card', split='train', Y=1, vote_unit_count=1,
                 physical_member_count=2, voter_mean=1.0)
    if failure == 'weight':
        members[0]['physical_vote_weight'] = 1.0
    elif failure == 'nan_weight':
        members[0]['physical_vote_weight'] = float('nan')
    elif failure == 'wrong_condition':
        members[0]['condition_group'] = 'different'
    elif failure == 'gold_label':
        index['Y'] = 0
    elif failure == 'duplicate':
        members.append(dict(members[0]))
    for name, rows in [('voter_membership', members), ('vote_units', [unit]),
                       ('gold_label_record_index', [index])]:
        pq.write_table(pa.Table.from_pylist(rows), gold / f'{name}.parquet')
    pq.write_table(pa.table({'source_row_uid': ['sr_a'] if failure == 'missing_uid' else ['sr_a', 'sr_b']}),
                   library / '03_pair_buckets/records.parquet')
    if failure:
        with pytest.raises(ValueError):
            publisher.verify_membership(gold, library)
    else:
        result = publisher.verify_membership(gold, library)
        assert result['vote_units'] == 1 and result['physical_members'] == 2


def test_skin_projection_uses_canonical_source_rows_and_no_card_examples(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from predict.retrieval.assay_reranking import build_skin_gold_v1_l1_adapter as skin
    library = tmp_path / 'data/evidence_libraries/skin_reaction'
    library.mkdir(parents=True)
    (library / 'CURRENT').write_text('release')
    stage3 = library / 'release/03_pair_buckets'
    stage3.mkdir(parents=True)
    contract = library / 'release/02_canonicalized'
    contract.mkdir()
    (contract / 'source_contract.json').write_text(json.dumps({'sources': {'direct': {
        'source_visible_fields': ['support_text', 'outcome_label']}}}))
    uid = 'sr_' + 'a' * 32
    pq.write_table(pa.Table.from_pylist([dict(source_row_uid=uid, source_id='direct',
                                            source_record_id='ext_7', canonical_smiles='CC',
                                            support_text='Actual source quotation', outcome_label='negative')]),
                   stage3 / 'records.parquet')
    membership = tmp_path / 'gold'
    membership.mkdir()
    (membership / 'voter_membership.parquet').write_bytes(b'pinned membership')
    monkeypatch.setattr(skin, 'REPO', tmp_path)
    cards = {context: dict(source_row_uids=[uid], drug='CC',
                           molecule_identity_key=skin.normalize_molecule_identity('CC').parent_inchi_key,
                           raw_value_examples=['WRONG EXAMPLE'])
             for context in ('null', 'conditioned')}
    output = tmp_path / 'output'
    path = skin._projection(output, cards, set(cards), gold=membership)
    rows = pq.read_table(path.with_name('records.parquet')).to_pylist()
    assert len(rows) == 1 and rows[0]['source_row_uid'] == uid
    payload = json.loads(rows[0]['payload'])
    assert payload['source_fields'] == dict(support_text='Actual source quotation', outcome_label='negative')
    assert 'Y' not in payload and 'benchmark_row_id' not in payload
    cards['null']['source_row_uids'] = ['sr_' + 'b' * 32]
    with pytest.raises(ValueError, match='absent or nonunique'):
        skin._projection(tmp_path / 'invalid', cards, set(cards), gold=membership)
    assert not (tmp_path / 'invalid').exists()


def test_restore_skin_voter_uses_its_own_source_payload(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data.processing.evidence_library import import_main_source_universe as importer
    monkeypatch.setattr(importer, 'ROOT', tmp_path)
    monkeypatch.setattr(importer, 'UID_LEDGER', tmp_path / 'ledger')
    importer.UID_LEDGER.mkdir()
    first, missing = 'sr_' + 'a' * 32, 'sr_' + 'b' * 32
    pq.write_table(pa.Table.from_pylist([dict(task_id='skin_reaction', source_row_uid=missing,
        source_id='direct_skin_reaction', source_record_id='ext_2',
        acquisition_source_row_number=1, legacy_cleaned_record_id='cleaned-b')]),
        importer.UID_LEDGER / 'part.parquet')
    raw_root = tmp_path / 'data/raw/starling/skin_reaction/direct_skin_reaction'
    raw_root.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(source_row_uid=missing, extraction_id='ext_2',
        SMILES='CCC', reaction_type='sensitization', effect_metric='12%',
        support_text='Actual missing row', outcome_label='positive')]), raw_root / 'extractions.parquet')
    base = tmp_path / 'base'
    (base / 'skin_reaction').mkdir(parents=True)
    source = base / 'skin_reaction/records.parquet'
    pq.write_table(pa.Table.from_pylist([dict(source_row_uid=first, source_id='direct_skin_reaction',
        source_record_id='ext_1', source_row_number=0, cleaned_record_id='cleaned-a',
        canonical_smiles='CC', measurement_text='99%', support_text='Different source row')]), source)
    before = source.read_bytes()
    (base / 'manifest.json').write_text(json.dumps(dict(version='main_source_universe.v1', tasks={
        'skin_reaction': dict(path='skin_reaction/records.parquet', rows=1, sha256=sha256_file(source))})))
    membership = tmp_path / 'members.parquet'
    pq.write_table(pa.table({'source_row_uid': [first, missing, missing]}), membership)
    target = tmp_path / 'restored'
    result = importer.restore_voter_rows('skin_reaction', base, membership, target)
    rows = pq.read_table(target / 'skin_reaction/records.parquet').to_pylist()
    assert len(rows) == 2 and result['tasks']['skin_reaction']['gold_voter_rows_restored'] == 1
    assert rows[1]['source_row_uid'] == missing and rows[1]['source_row_number'] == 1
    assert rows[1]['support_text'] == 'Actual missing row'
    assert rows[1]['measurement_text'] == '12%' and rows[1]['canonical_smiles'] == 'CCC'
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        importer.restore_voter_rows('skin_reaction', base, membership, target)


def test_skin_level_mapping_preserves_legacy_l1_and_shared_voters(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data.processing.evidence_library.build_release_level_mapping import build_subset
    records = tmp_path / 'records.parquet'
    pq.write_table(pa.Table.from_pylist([dict(source_row_uid=uid, canonical_record_id=uid,
        pair_bucket_key='bucket', measurement_kind='categorical', assay_transfer_eligible=True,
        assay_transfer_ineligibility_reason='') for uid in ('old-l1', 'voter', 'other-l2')]), records)
    prior = tmp_path / 'prior.parquet'
    pq.write_table(pa.table({'source_row_uid': ['old-l1', 'voter', 'other-l2'], 'level': [1, 1, 2]}), prior)
    members = tmp_path / 'members.parquet'
    pq.write_table(pa.table({'source_row_uid': ['voter', 'voter']}), members)
    output = tmp_path / 'levels'
    build_subset('skin_reaction', records, prior, output, current_voter_membership=members)
    doc = json.loads((output / 'manifest.json').read_text())
    assert doc['validations']['l1_covers_current_physical_voters']
    assert not doc['current_l1_validation']['l1_equals_current_physical_voters']
    result = pq.read_table(output / 'records.parquet')
    assert dict(zip(result['source_row_uid'].to_pylist(), result['level'].to_pylist())) == {
        'old-l1': 1, 'voter': 1, 'other-l2': 2}


def test_skin_voter_correction_extends_protection_before_normalization(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data.processing.evidence_library import publish_gold_level_mappings as levels
    monkeypatch.setattr(levels, 'ROOT', tmp_path)
    prior = tmp_path / 'data/gold_labels/Skin_Reaction/level_mappings/v1'
    (prior / 'level_mapping').mkdir(parents=True)
    old = [dict(source_row_uid=uid, canonical_record_id=uid, source_group_id='group',
                family_key='family', level=level) for uid, level in [('legacy', 1), ('indirect', 2)]]
    pq.write_table(pa.Table.from_pylist(old), prior / 'level_mapping/part-00000.parquet')
    (prior / 'manifest.json').write_text(json.dumps(dict(version='gold_original_level_mapping.v1')))
    members = tmp_path / 'members.parquet'
    pq.write_table(pa.table({'source_row_uid': ['restored', 'restored']}), members)
    output = tmp_path / 'corrected'
    levels.publish_skin_voter_correction(members, output)
    assert set(pq.read_table(output / 'stage1_voter_protection.parquet')['source_row_uid'].to_pylist()) == {
        'legacy', 'restored'}
    canonical = tmp_path / 'canonical.parquet'
    pq.write_table(pa.table({'source_row_uid': ['restored'], 'canonical_record_id': ['physical-record']}), canonical)
    levels.publish_skin_voter_correction(members, output, canonical)
    result = pq.read_table(output / 'level_mapping').to_pylist()
    assert result[:2] == old
    assert result[2] == dict(source_row_uid='restored', canonical_record_id='physical-record',
                             source_group_id='Direct.skin_reaction', family_key='direct_skin_reaction', level=1)
