"""Executable invariants for the source-reviewed Oral Bio extraction corpus."""

import json
import hashlib
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v9.tasks.bioavailability_ma.starling_measurement_resolution import MAX_MEASUREMENTS_PER_ROW, route_measurement, render_prompt, SOURCE_IDS
from data.processing.evidence_library.versions.v9.build_measurement_resolution_mapping import validate_row
from tests.chembl_tool.common.measurement_resolution_quality.evaluate_mapping import _score_case


class OralGoldTests(unittest.TestCase):
    def test_complete_current_source_bound_corpus(self):
        path = Path('tests/chembl_tool/common/measurement_resolution_quality/gold/bioavailability_ma.v9.2.jsonl')
        manifest, *cases = [json.loads(line) for line in path.read_text().splitlines()]
        previous = Path(manifest['previous_gold_path'])
        self.assertEqual(hashlib.sha256(previous.read_bytes()).hexdigest(), manifest['previous_gold_sha256'])
        old_cases = [json.loads(line) for line in previous.read_text().splitlines()[1:]]
        receipt = json.loads(path.with_suffix('.changes.json').read_text())
        self.assertEqual(receipt['revised_gold_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        actual_changes = {a['audit_case_id'] for a,b in zip(cases,old_cases) if a['expected'] != b['expected']}
        self.assertEqual(actual_changes, {c['audit_case_id'] for c in receipt['changes']})
        for case, old in zip(cases, old_cases):
            self.assertEqual(case['input'], old['input'])
            self.assertEqual(case['source_row_uid'], old['source_row_uid'])
            self.assertEqual(case['audit_case_id'], old['audit_case_id'])
        self.assertEqual(len(cases), 500)
        self.assertEqual(len({c['source_row_uid'] for c in cases}), 500)
        self.assertEqual(len({c['audit_case_id'] for c in cases}), 500)
        self.assertEqual(Counter(c['source_id'] for c in cases),
                         dict.fromkeys(('fa','fg','fh','hf_bioavailability','oral_exposure'),100))
        fields = ('endpoint_name','measurement_text','unit_text','support_text')
        records = pq.read_table(manifest['cleaned_records_path'],
                                columns=['cleaned_record_id','source_row_uid',*fields]).to_pylist()
        current = {r['cleaned_record_id']: r for r in records}
        for case in cases:
            with self.subTest(case=case['audit_case_id']):
                row = current[case['audit_case_id'].split(':',1)[1]]
                self.assertEqual(row['source_row_uid'],case['source_row_uid'])
                self.assertTrue(all(row[k] == case['input'].get(k) for k in fields))
                self.assertEqual(route_measurement(case['input']).bucket,'extract')
                self.assertTrue(case['label_note'])
                for answer in [case['expected'],*case['expected'].get('alternatives',[])]:
                    self.assertIn(answer['status'],('ok','unsure','relative','unavailable'))
                    self.assertEqual(bool(answer['measurements']),answer['status']=='ok')
                    self.assertLessEqual(len(answer['measurements']), MAX_MEASUREMENTS_PER_ROW)
                    for pair in answer['measurements']:
                        self.assertTrue(Decimal(pair['measurement']).is_finite())
                        self.assertTrue(pair['unit'].strip())

    def test_single_output_validation_and_rendering(self):
        source = {'id':'sample', 'source_id':'fa'}
        pair = {'measurement':'-0.5', 'unit':'cm/s'}
        for status, pairs, valid in [('ok',[pair],True), ('ok',[pair,pair],False),
                                     ('ok',[],False), ('relative',[],True),
                                     ('relative',[pair],False)]:
            result, error = validate_row({'id':'sample','status':status,'measurements':pairs},
                                         source, task='bioavailability_ma',
                                         max_measurements=MAX_MEASUREMENTS_PER_ROW)
            self.assertEqual(error is None, valid)
            if valid:
                self.assertEqual(json.loads(result['measurements_json']), pairs)
        for source_id in SOURCE_IDS:
            rendered = render_prompt(source_id)
            self.assertTrue(rendered.strip())
            self.assertEqual(rendered, render_prompt(source_id, endpoint_profiles=('unrelated batch statistics',)))

    def test_reviewed_unit_equivalence_is_case_specific(self):
        path = Path('tests/chembl_tool/common/measurement_resolution_quality/gold/bioavailability_ma.v9.2.jsonl')
        cases = [json.loads(line) for line in path.read_text().splitlines()[1:]]
        case = next(c for c in cases if c['input']['measurement_text'] == 'Papp 151 nm/s')
        prediction = {'cleaned_record_id':case['audit_case_id'].split(':',1)[1], 'status':'ok'}
        for value, unit, correct in [('151','nm/s',True), ('0.0000151','cm/s',True), ('151','cm/s',False)]:
            prediction['measurements_json'] = json.dumps([{'measurement':value,'unit':unit}])
            self.assertEqual(_score_case(case,prediction,task='bioavailability_ma')['record_match'],correct)
