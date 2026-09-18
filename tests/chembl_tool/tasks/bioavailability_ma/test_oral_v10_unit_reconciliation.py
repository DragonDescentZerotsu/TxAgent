"""Executable unit-only identity and frozen-input checks, without production assets."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.evidence_library.shared.v2.normalization.cleaning import file_sha256
from data.processing.evidence_library.shared.v2.normalization.measurement_resolution import load_exact_unit_mapping
from data.processing.evidence_library.versions.v10.tasks.bioavailability_ma.data_processing.build_unit_reconciliation import (
    TASK, _review_entries, build_mapping,
)


class OralV10UnitReconciliationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root / 'source.parquet'
        pq.write_table(pa.Table.from_pylist([
            dict(cleaned_record_id='a', source_row_uid='uid-a', unit_text='µg/mL',
                 measurement_resolution_route='accept', measurement_resolution_exact_unit='µg/mL'),
            dict(cleaned_record_id='b', source_row_uid='uid-b', unit_text='ratio',
                 measurement_resolution_route='extract', measurement_resolution_exact_unit=None),
        ]), source)
        extraction = self.root / 'extraction.parquet'
        pq.write_table(pa.Table.from_pylist([dict(cleaned_record_id='b', source_row_uid='uid-b',
            status='ok', measurements_json=json.dumps([dict(value=-2, unit='μg/ml')]))]), extraction)
        vocabulary = self.root / 'vocabulary.json'
        vocabulary.write_text(json.dumps(dict(units=[dict(unit=u, sources=[TASK+':source'])
            for u in ['%', 'fraction', '×10^-?? cm/s', 'cm/s', 'cm sec^-1',
                      'ng·h/mL', 'ng/mL/h', 'mL mg^-2/3 h^-1']])))
        self.units = {'µg/mL', 'μg/ml', 'ratio', 'fraction', '%', '×10^-?? cm/s',
                      'cm/s', 'cm sec^-1', 'ng·h/mL', 'ng/mL/h',
                      'mL mg^-2/3 h^-1'}
        aliases = {'μg/ml': 'µg/mL', 'cm sec^-1': 'cm/s'}
        self.reviews = [dict(input_unit=u, canonical_unit=aliases.get(u, u),
            decision='merge_alias' if u in aliases else 'keep_distinct',
            rationale='Reviewed spelling or notation alias' if u in aliases
            else 'Preserve literal identity', review_round=1) for u in sorted(self.units)]
        self.review = self.root / 'review.jsonl'
        self.review.write_text(''.join(json.dumps(r)+'\n' for r in self.reviews))
        selected = dict(cleaned_records=str(source), measurement_resolution=str(extraction),
                        unit_vocabulary=str(vocabulary))
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(task=TASK, selected_inputs=selected,
            inputs={v: dict(sha256=file_sha256(Path(v))) for v in selected.values()},
            review_sha256=file_sha256(self.review))))

    def test_source_and_llm_aliases_converge_without_conversion(self):
        payload = build_mapping(self.manifest, self.review)
        path = self.root / 'map.json'
        path.write_text(json.dumps(payload))
        rules = load_exact_unit_mapping(path)
        self.assertEqual(len(rules), len(self.units))
        self.assertEqual(rules[(TASK, '*', 'μg/ml')]['canonical_unit'], 'µg/mL')
        self.assertEqual(rules[(TASK, '*', 'cm sec^-1')]['canonical_unit'], 'cm/s')
        for unit in self.units:
            rule = rules[(TASK, '*', unit)]
            self.assertEqual((rule['action'], rule['scale'], rule['domain']), ('map', '1', 'any'))
        for unit in ['ratio', 'fraction', '%', '×10^-?? cm/s', 'ng·h/mL',
                     'ng/mL/h', 'mL mg^-2/3 h^-1']:
            self.assertEqual(rules[(TASK, '*', unit)]['canonical_unit'], unit)
        contract = payload['bioavailability_v10_contract']
        self.assertEqual(contract['per_round_counts'], {'1': len(self.units)})
        self.assertEqual(contract['review_sha256'], file_sha256(self.review))

    def test_input_and_review_drift_fail_closed(self):
        self.review.write_text(self.review.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'decision hash'):
            build_mapping(self.manifest, self.review)
        (self.root/'vocabulary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'input hash'):
            build_mapping(self.manifest, self.review)

    def test_invalid_review_contracts_fail(self):
        for patch in [dict(decision='exclude'), dict(scale='100'),
                      dict(endpoint_rules=[dict(canonical_endpoint='oral_bioavailability')]),
                      dict(canonical_unit='unobserved', decision='merge_alias')]:
            rows = copy.deepcopy(self.reviews)
            rows[0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                _review_entries(rows, self.units)
        with self.assertRaises(ValueError):
            _review_entries(self.reviews[:-1], self.units)
        with self.assertRaises(ValueError):
            _review_entries(self.reviews+self.reviews[:1], self.units)
        rows = copy.deepcopy(self.reviews)
        next(r for r in rows if r['input_unit']=='ratio').update(canonical_unit='%', decision='merge_alias')
        with self.assertRaisesRegex(ValueError, 'remain distinct'):
            _review_entries(rows, self.units)

    def test_extraction_uid_drift_fails_even_with_updated_hash(self):
        path = self.root/'extraction.parquet'
        rows = pq.read_table(path).to_pylist()
        rows[0]['source_row_uid'] = 'wrong-source'
        pq.write_table(pa.Table.from_pylist(rows), path)
        manifest = json.loads(self.manifest.read_text())
        manifest['inputs'][str(path)]['sha256'] = file_sha256(path)
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            build_mapping(self.manifest, self.review)


if __name__ == '__main__':
    unittest.main()
