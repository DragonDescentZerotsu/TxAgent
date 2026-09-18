import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v10.tasks.bbb_martins import starling_measurement_resolution as config


class BBBMappingProvenanceTest(unittest.TestCase):
    def test_reviewed_unit_replay_preserves_other_fields_and_rejects_drift(self):
        decision = json.loads(config.SOURCE_UNIT_OVERRIDES.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, mapping, output = (root / name for name in ('source.parquet', 'paid.parquet', 'derived.parquet'))
            rows = [{'source_row_uid': r['source_row_uid'], 'cleaned_record_id': r['source_row_uid'],
                     'status': 'ok', 'measurements_json': json.dumps([{'measurement': r['measurement'], 'unit': decision['expected_model_unit']}]),
                     'raw_response_json': 'preserve original paid response'} for r in decision['rows']]
            rows.append({'source_row_uid': 'unrelated', 'cleaned_record_id': 'unrelated', 'status': 'relative',
                         'measurements_json': '[]', 'raw_response_json': 'also preserved'})
            pq.write_table(pa.Table.from_pylist([{'source_row_uid': r['source_row_uid'], 'cleaned_record_id': r['cleaned_record_id'], 'unit_text': decision['expected_source_unit']} for r in rows]), source)
            pq.write_table(pa.Table.from_pylist(rows), mapping)
            mapping.with_suffix('.manifest.json').write_text(json.dumps({'mapping_sha256': config.file_sha256(mapping), 'cleaned_records_path': str(source), 'cleaned_records_sha256': config.file_sha256(source)}))
            config.apply_reviewed_source_units(mapping, output)
            result = pq.read_table(output).to_pylist()
            self.assertEqual(result[-1], rows[-1])
            for before, after in zip(rows[:-1], result[:-1]):
                expected = dict(before)
                entries = json.loads(expected['measurements_json'])
                entries[0]['unit'] = decision['expected_source_unit']
                expected['measurements_json'] = json.dumps(entries, ensure_ascii=False)
                self.assertEqual(after, expected)
            self.assertEqual(pq.read_table(mapping).to_pylist(), rows)
            with self.assertRaisesRegex(ValueError, 'must not be overwritten'):
                config.apply_reviewed_source_units(mapping, mapping)
            source.write_bytes(b'drifted source')
            with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
                config.apply_reviewed_source_units(mapping, output)

    def test_mixed_reuse_and_pinned_delta_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base, mapping = (root / name for name in ('source', 'base', 'mapping.parquet'))
            source.write_bytes(b'source')
            base.write_bytes(b'previous scientific assignments')
            rows = [
                {'source_row_uid': 'old', 'inference_source': 'base_mapping', 'served_provider': 'OpenAI', 'inference_base_url': 'https://api.openai.com/v1'},
                {'source_row_uid': 'new', 'inference_source': 'delta_inference', 'served_provider': 'Baidu', 'inference_base_url': 'https://openrouter.ai/api/v1'},
            ]
            manifest = {
                'task_id': 'bbb_martins', 'mapping_version': config.MAPPING_VERSION,
                'cleaned_records_sha256': config.file_sha256(source),
                'prompt': {'prompt_version': config.PROMPT_VERSION},
                'base_mapping': {'path': str(base), 'sha256': config.file_sha256(base)},
                'delta_inference': {'models': ['deepseek/deepseek-v4-flash-0731']},
                'validations': dict.fromkeys(('one_row_per_candidate', 'unique_cleaned_record_ids', 'only_ok_carries_measurements', 'maximum_measurements_per_row'), True),
            }
            for provider, valid in [('Baidu', True), ('Other host', False)]:
                rows[1]['served_provider'] = provider
                pq.write_table(pa.Table.from_pylist(rows), mapping)
                manifest['mapping_sha256'] = config.file_sha256(mapping)
                mapping.with_suffix('.manifest.json').write_text(json.dumps(manifest))
                with patch.object(config, 'DEFAULT_CLEANED_RECORDS', source):
                    if valid:
                        config.validate_mapping_provenance(mapping)
                        archive, ledger = root / 'archived-source', root / 'compatibility.json'
                        archive.write_bytes(source.read_bytes())
                        ledger.write_text(json.dumps({'tasks': {'bbb_martins': {
                            'previous_stage1_path': str(archive),
                            'previous_stage1_sha256': config.file_sha256(archive)}}}))
                        source.write_bytes(b'rebuilt source')
                        with patch.object(config, 'COMPATIBILITY_LEDGER', ledger):
                            config.validate_mapping_provenance(mapping)
                            archive.write_bytes(b'tampered archive')
                            with self.assertRaisesRegex(ValueError, 'cleaned_records_sha256'):
                                config.validate_mapping_provenance(mapping)
                        source.write_bytes(b'source')
                    else:
                        with self.assertRaisesRegex(ValueError, 'delta_provider'):
                            config.validate_mapping_provenance(mapping)
            base.write_bytes(b'tampered')
            with patch.object(config, 'DEFAULT_CLEANED_RECORDS', source):
                with self.assertRaisesRegex(ValueError, 'base_mapping'):
                    config.validate_mapping_provenance(mapping)

    def test_main_universe_accepts_only_high_effort_openai_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base, mapping = (
                root / name for name in ('source', 'base', 'mapping.parquet')
            )
            source.write_bytes(b'main universe')
            base.write_bytes(b'base assignments')
            pq.write_table(pa.Table.from_pylist([{
                'source_row_uid': 'new',
                'inference_source': 'delta_inference',
                'served_provider': None,
                'inference_base_url': 'https://api.openai.com/v1',
            }]), mapping)
            manifest = {
                'task_id': 'bbb_martins',
                'mapping_version': config.MAPPING_VERSION,
                'mapping_sha256': config.file_sha256(mapping),
                'cleaned_records_path': str(source),
                'cleaned_records_sha256': config.file_sha256(source),
                'prompt': {'prompt_version': config.PROMPT_VERSION},
                'base_mapping': {'path': str(base), 'sha256': config.file_sha256(base)},
                'delta_inference': {'models': ['gpt-5.4-mini']},
                'inference': {'reasoning_mode': 'high'},
                'validations': dict.fromkeys(
                    ('one_row_per_candidate', 'unique_cleaned_record_ids',
                     'only_ok_carries_measurements', 'maximum_measurements_per_row'),
                    True,
                ),
            }
            manifest_path = mapping.with_suffix('.manifest.json')
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(config, 'MAIN_UNIVERSE_CLEANED_RECORDS', source):
                config.validate_mapping_provenance(mapping)
                manifest['inference']['reasoning_mode'] = 'low'
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'delta_inference'):
                    config.validate_mapping_provenance(mapping)


if __name__ == '__main__':
    unittest.main()


def test_context_redistribution_preserves_labels_and_reports_exclusions():
    from data.processing.evidence_library.versions.v10.tasks.bbb_martins.direct_record_mapping import redistribute_training_context_membership
    import pytest
    a = dict(benchmark_row_id='a', molecule_identity_key='old', condition_group='none', Y=1,
             label_counts={'1': 1}, source_record_count=1, source_record_ids=['row:1'])
    b = {**a, 'benchmark_row_id': 'b', 'molecule_identity_key': 'new', 'source_record_ids': ['row:2']}
    c = {**a, 'benchmark_row_id': 'c', 'molecule_identity_key': 'third', 'source_record_ids': ['row:3']}
    records = {f'row:{i}': dict(source_row_uid=f'u{i}', canonical_record_id=f'r{i}', parent_identity_key='new') for i in (1,2,3)}
    rows = redistribute_training_context_membership([a,b,c], records, {'u1':1,'u2':1}, {'row:1'})
    assert rows[0]['destination_context_id']=='b' and rows[0]['source_row_uid']=='u1'
    assert rows[2]['status']=='excluded_unmapped' and rows[2]['destination_context_id'] is None
    with pytest.raises(ValueError, match='changes labels'):
        redistribute_training_context_membership([a,{**b,'Y':0}], records, {'u1':1,'u2':1}, {'row:1'})
    assert redistribute_training_context_membership([a], records, {'u1':1}, set())[0]['status']=='unresolved_parent_change'
    assert redistribute_training_context_membership([a], records, {'u1':2}, set())[0]['status']=='mapped_other_level'


def test_mixed_context_moves_replay_votes_and_check_joint_source_gate():
    from data.processing.evidence_library.versions.v10.tasks.bbb_martins.direct_record_mapping import redistribute_training_context_membership
    import pytest
    a = dict(benchmark_row_id='a', molecule_identity_key='old', condition_group='none', Y=0,
             label_counts={'0':4, '1':1}, source_record_count=5, source_record_ids=list('abcde'))
    b = dict(benchmark_row_id='b', molecule_identity_key='new', condition_group='none', Y=0,
             label_counts={'0':1}, source_record_count=1, source_record_ids=['f'])
    records = {s: dict(source_row_uid=s, canonical_record_id=s, parent_identity_key='new' if s in 'abf' else 'old',
                      direct_vote_label=int(s == 'e')) for s in 'abcdef'}
    levels = {s:1 for s in records}
    rows = redistribute_training_context_membership([a,b], records, levels, {'a'})
    assert rows[0]['destination_context_id'] == 'b'
    with pytest.raises(ValueError, match='Joint context transfer'):
        redistribute_training_context_membership([a,b], records, levels, {'a','b'})
    records['e']['direct_vote_label'] = 0
    with pytest.raises(ValueError, match='do not replay'):
        redistribute_training_context_membership([a,b], records, levels, {'a'})
