import copy
import tempfile
import unittest
from pathlib import Path

from assay_transfer.storage import digest, stage, verify_manifest, write_json
from assay_transfer.build import supported
from assay_transfer.canonical import task_contract
from assay_transfer.targets import binary_targets,probability
from assay_transfer.filtering import domain,ratio_diagnostics
from assay_transfer.inputs import gold_value,identity
from assay_transfer.split import assign
from assay_transfer.filtering import fit_bucket
from assay_transfer.verify import constraints


def rows(split='train'):
    return [{'record_id':f'{i}:{j}', 'parent_id':str(i), 'parent_smiles':'C', 'scaffold':f's{i}',
             'geometry_value':float(i), 'measurement_kind':'continuous', 'category_id':'',
             'dataset_split':split,'pair_bucket_key':'["assay","L2"]','upstream_pair_bucket_key':'["assay"]',
             'level':'L2','canonical_unit_text':'%'} for i in range(6) for j in range(2)]


class ScientificContracts(unittest.TestCase):
    def test_gold_vote_formats(self):
        self.assertEqual(gold_value({'source_record_count':2,'label_counts':{'0':2}},{}),(2,0))
        row={'benchmark_row_id':'id','Y':0}
        voters={'id':{'vote_unit_count':3,'voter_mean':2/3}}
        self.assertEqual(gold_value(row,voters),(3,2/3))
        with self.assertRaises(KeyError):gold_value(row,{})

    def test_geometry_and_targets(self):
        data = rows()
        entry, reason = domain(data)
        self.assertIsNone(reason)
        self.assertEqual(ratio_diagnostics(data,entry)['within_to_parent_mean_variance_ratio'],0)
        target = {'center_sd':.4,'temperature_sd':.1}
        self.assertAlmostEqual(probability({'geometry_value':0},{'geometry_value':.4},
                                          {'calibration_sample_standard_deviation':1},target),.5)
        self.assertAlmostEqual(probability({'geometry_value':0},{'geometry_value':0},
                                          {'calibration_sample_standard_deviation':1},target),.9820137900379085)
        self.assertIsNone(ratio_diagnostics(data[::2],entry)['within_to_parent_mean_variance_ratio'])
        self.assertEqual(identity('CCO.[Na+]')[0],identity('CCO')[0])

    def test_missing_binary_target_has_no_fallback(self):
        record = {'record_id':'binary','measurement_kind':'binary','level':'L2'}
        pairs = [{'a_id':'binary','b_id':'binary'}]
        with self.assertRaises(ValueError):
            binary_targets(pairs,{'binary':record},{},{})
        result = binary_targets(pairs,{'binary':record},{},{},reject_missing=True)
        self.assertEqual(result['L2']['status'],'rejected')

    def test_holdout_and_test_fit(self):
        data, policy = rows(),task_contract('Ames')
        gold = {'train':[],'valid':[{'parent_id':'1','scaffold':'s1'}],
                'test':[{'parent_id':'2','scaffold':'s2'}]}
        assign(data,gold,policy)
        self.assertTrue(all(r['dataset_split']=='gate_only' for r in data if r['parent_id'] in ('1','2')))
        evidence = rows() + [{**r,'record_id':'gate:'+r['record_id'],'dataset_split':'gate_only'} for r in rows()]
        entry, reason = fit_bucket(evidence,policy,{'bucket_overrides':{},'identity_units':['%'],'default_positive_transform':'log10'})
        self.assertIsNone(reason)
        self.assertTrue(all(not key.startswith('gate:') for key in entry['calibration_fit_record_ids']))
        self.assertTrue(any(key.startswith('gate:') for key in entry['ratio_gate_record_ids']))

    def test_unknown_policy_and_caps_rejected(self):
        policy = task_contract('DILI')
        policy['record_sampling_policy'] = 'made_up_sampler'
        with self.assertRaises(ValueError):supported(policy)
        policy = task_contract('Bioavailability_Ma')
        data = rows()
        index = {r['record_id']:r for r in data}
        bad = {'pair_id':'x','a_id':'0:0','b_id':'0:1','split':'train','component':'indirect','probability':.9}
        with self.assertRaises(ValueError):constraints([bad],index,policy,'record')

    def test_stages_are_immutable_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'stage'
            def produce(path):
                write_json(path/'data.json',{'value':1})
                return {'scope':'smoke'}
            original = stage(root,{'source':'a'},produce)
            self.assertEqual(stage(root,{'source':'a'},produce,True),original)
            with self.assertRaises(FileExistsError):stage(root,{'source':'b'},produce,True)
            (root/'data.json').write_text('{}')
            with self.assertRaises(ValueError):verify_manifest(root)


if __name__ == '__main__':
    unittest.main()
