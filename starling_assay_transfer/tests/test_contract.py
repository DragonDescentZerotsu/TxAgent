"""Check per-task overrides and reject scientifically ambiguous settings."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from assay_transfer.canonical import degree_cap, load_contract, task_contract


class CanonicalContractTest(unittest.TestCase):
    def test_task_initialization(self):
        self.assertEqual(degree_cap('BBB_Martins', 'L4'), 96)
        self.assertEqual(degree_cap('bioavailability_ma', 'L2'), 6)
        self.assertEqual(degree_cap('Bioavailability_Ma', 'L4'), 6)
        self.assertEqual(degree_cap('Bioavailability_Ma', component='context_indirect'), 2)
        for task in ('Ames', 'Carcinogens', 'DILI'):
            self.assertIsNone(task_contract(task)['ood_bucket_records'])
            self.assertEqual(task_contract(task)['evaluation_levels'], [])
        self.assertEqual(task_contract('Skin_Reaction')['ood_bucket_records'], [8, 75])
        for task in ('BBB_Martins','Bioavailability_Ma'):
            self.assertNotIn('L5',task_contract(task)['record_levels'])
            self.assertNotIn('L5',task_contract(task)['evaluation_levels'])
            with self.assertRaises(ValueError):
                degree_cap(task,'L5')
        with self.assertRaises(ValueError):
            degree_cap('Bioavailability_Ma', 'L1')

    def test_effective_hash_tracks_semantics_and_isolates_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'contract.json'
            original = load_contract()
            changed = copy.deepcopy(original)
            changed['tasks']['Bioavailability_Ma']['record_degree_caps_by_level']['L4'] = 12
            path.write_text(json.dumps(changed))
            self.assertEqual(degree_cap('Bioavailability_Ma', 'L4', path=path), 12)
            self.assertNotEqual(task_contract('Bioavailability_Ma', path)['contract_sha256'],
                                task_contract('Bioavailability_Ma')['contract_sha256'])
            self.assertEqual(task_contract('BBB_Martins', path)['contract_sha256'],
                             task_contract('BBB_Martins')['contract_sha256'])
            changed['common']['numeric_target']['temperature_sd'] = 0.2
            path.write_text(json.dumps(changed))
            self.assertNotEqual(task_contract('BBB_Martins', path)['contract_sha256'],
                                task_contract('BBB_Martins')['contract_sha256'])

    def test_reject_invalid_policies(self):
        mutations = [('record_degree_cap', 3), ('record_degree_cap', True),
                     ('record_degree_cap', 0), ('variance_ratio_maximum', -0.1),
                     ('variance_ratio_maximum', float('nan')), ('variance_ratio_maximum', True),
                     ('ood_bucket_records', [11, 9]), ('ood_bucket_records', [True, 11]),
                     ('ood_bucket_records', None), ('evaluation_levels', ['L99']),
                     ('record_degree_caps_by_level', {'L99': 24}), ('unknown_gate', 0.5)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'contract.json'
            for key, value in mutations:
                with self.subTest(key=key, value=value):
                    changed = load_contract()
                    changed['tasks']['BBB_Martins'][key] = value
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        load_contract(path)

    def test_shared_invariants_and_ood_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'contract.json'
            for key, value in [('gold_version', 'v2'), ('test_values_fit_targets', True),
                               ('query_measurement_visible', True), ('variance_ratio_undefined', 'allow'),
                               ('completion_requires', ['artifact_hashes'])]:
                with self.subTest(key=key):
                    changed = load_contract()
                    changed['common'][key] = value
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        load_contract(path)
            changed = load_contract()
            changed['tasks']['DILI'].update(ood_bucket_records=[8, 75], evaluation_levels=['L2', 'L3'],
                                           evaluation_policy='explicit_new_ood_evaluation_contract')
            path.write_text(json.dumps(changed))
            self.assertEqual(task_contract('DILI', path)['ood_bucket_records'], [8, 75])
            self.assertNotEqual(task_contract('DILI', path)['contract_sha256'],
                                task_contract('DILI')['contract_sha256'])


if __name__ == '__main__':
    unittest.main()
