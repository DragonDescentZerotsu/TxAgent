"""Delta routing and context delivery, without assertions about prompt prose."""
import tempfile
import unittest
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.versions.v10.percentage_delta import review_percentage, merge_pilot, digest
from data.processing.evidence_library.versions.v10.measurement_routing import RouteDecision
from data.processing.evidence_library.versions.v10.build_measurement_resolution_mapping import TaskConfig, candidate_rows, validate_row, gold_extract_ids


class PercentageDeltaTests(unittest.TestCase):
    def test_accept_only_overlay(self):
        accept=RouteDecision('accept','source_rule','17.2','%')
        reject=RouteDecision('reject','bound_rule')
        extract=RouteDecision('extract')
        for text,unit in [('17.2','%'),('17.2%',''),('17.2','%/h'),('17.2','percent'),('17.2','per cent'),('17.2','％')]:
            row={'measurement_text':text,'unit_text':unit}
            self.assertEqual(review_percentage(row,accept).bucket,'extract')
            self.assertIs(review_percentage(row,reject),reject)
            self.assertIs(review_percentage(row,extract),extract)
        self.assertIs(review_percentage({'measurement_text':'17.2','unit_text':'cm/s','support_text':'5% vehicle'},accept),accept)

    def test_context_reaches_candidate_and_reuse_stays_out(self):
        config=TaskConfig('bioavailability_ma')
        row={'cleaned_record_id':'id','source_row_uid':'uid','source_id':'hf_bioavailability',
             'canonical_endpoint_name':'oral_bioavailability','endpoint_name':'oral_bioavailability',
             'measurement_text':'17.2%','unit_text':None,'support_text':'relative to solution',
             'bioavailability_report_type':'relative_comparison','comparator':'PEG 400 solution',
             'measurement_resolution_route':'extract'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'records.parquet'
            pq.write_table(pa.Table.from_pylist([row]),path)
            payload=candidate_rows(path,config)[0]
            self.assertEqual(payload['bioavailability_report_type'],'relative_comparison')
            self.assertEqual(payload['comparator'],'PEG 400 solution')
            row['measurement_resolution_route']='reject'
            pq.write_table(pa.Table.from_pylist([row]),path)
            self.assertEqual(candidate_rows(path,config),[])

    def test_four_status_contract(self):
        row={'id':'a','source_id':'fa'}
        for status in ['ok','relative','unsure','unavailable']:
            pairs=[{'measurement':'17.2','unit':'%'}] if status=='ok' else []
            _,error=validate_row({'id':'a','status':status,'measurements':pairs},row,task='bioavailability_ma',max_measurements=1)
            self.assertIsNone(error)
        _,error=validate_row({'id':'a','status':'relative','measurements':[{'measurement':'17.2','unit':'%'}]},row,task='bioavailability_ma',max_measurements=1)
        self.assertIsNotNone(error)

    def test_gold_source_identity_need_not_be_duplicated_inside_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gold.jsonl'
            case={'audit_case_id':'bioavailability_ma:x','source_id':'fa',
                  'input':{'endpoint_name':'fraction_absorbed','measurement_text':'17.2','unit_text':'%'}}
            path.write_text(json.dumps({'task_id':'bioavailability_ma'})+'\n'+json.dumps(case)+'\n')
            self.assertEqual(gold_extract_ids(path),['x'])

    def test_merge_preserves_old_assignment_and_rejects_bad_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old={'cleaned_record_id':'old','source_row_uid':'u0','status':'relative',
                 'measurements_json':'[]','assignment_method':'model_single_pass','assignment_origin':'v9_reused'}
            new={'cleaned_record_id':'new','source_row_uid':'u1','status':'ok',
                 'measurements_json':'[{"measurement":"17.2","unit":"%"}]','assignment_method':'model_single_pass'}
            pq.write_table(pa.Table.from_pylist([old]),root/'reused_assignments.parquet')
            pq.write_table(pa.Table.from_pylist([{'cleaned_record_id':'new','source_row_uid':'u1'}]),root/'delta_records.parquet')
            (root/'manifest.json').write_text(json.dumps({'task_id':'test','delta_sha256':'delta','source_sha256':'source','prior_extraction_pending':5}))
            mapping=root/'pilot.parquet'
            pq.write_table(pa.Table.from_pylist([new]),mapping)
            m={'mapping_sha256':digest(mapping),'cleaned_records_sha256':'wrong'}
            mapping.with_suffix('.manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):merge_pilot(root,mapping)
            m['cleaned_records_sha256']='delta'
            mapping.with_suffix('.manifest.json').write_text(json.dumps(m))
            receipt=merge_pilot(root,mapping)
            self.assertTrue(receipt['partial'])
            self.assertEqual(receipt['prior_extraction_pending'],5)
            self.assertEqual(pq.read_table(root/'partial_measurement_resolution.parquet').to_pylist()[0],old)
            with self.assertRaisesRegex(ValueError,'already exists'):merge_pilot(root,mapping)
