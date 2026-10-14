"""Regression checks for separate builders, augmentation, and task-wide holdouts."""

import copy
import json
import hashlib
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock

from assay_transfer.storage import digest, read_json, read_parquet, write_json
from assay_transfer.build import arguments, selected_inputs
from assay_transfer.canonical import DEFAULT_CONTRACT, TASKS, load_contract, task_contract
from assay_transfer.filtering import gold_calibration
from assay_transfer.augmentation import mixture
from assay_transfer.split import check_disjointness, filter_gold, filter_training
from assay_transfer.filtering import prepare_record_pool,calibrate
from assay_transfer.sampling import sampled


def build_gold(root,smoke=False):
    gold=root/'source/gold_labels/DILI/v1/scaffold';gold.mkdir(parents=True)
    for split,count in (('train',32),('valid',2),('test',2)):
        offset={'train':1,'valid':40,'test':45}[split]
        labels=[{'benchmark_row_id':f'{split}:{i}','drug':'C'*(offset+i),'Y':int(i%11>=5),
                 'source_record_count':10,'label_counts':{'0':10-i%11,'1':i%11}} for i in range(count)]
        (gold/f'{split}_molecule_condition_labels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in labels))
    document=load_contract();document['tasks']['DILI']['context_direct_degree_cap']=4
    contract=root/'contract.json';write_json(contract,document)
    inputs=root/'inputs.json'
    subprocess.run([sys.executable,'-m','assay_transfer.build','pin','--task','DILI','--gold-only',
                    '--contract',str(contract),'--source-root',str(root/'source'),'--input-manifest',str(inputs)],check=True,capture_output=True)
    command=[sys.executable,'-m','assay_transfer.build_context','all','--task','DILI','--no-augment',
             '--contract',str(contract),'--input-manifest',str(inputs),'--output-root',str(root/'outputs')]
    if smoke:command.append('--smoke')
    result=subprocess.run(command,capture_output=True)
    assert result.returncode==0,result.stderr.decode()
    return root/'outputs/context_molecule/dili',contract,command


class SeparateBuilders(unittest.TestCase):
    def test_cross_level_disjointness_and_modes(self):
        heldout = [{'record_id':'val', 'parent_id':'heldout', 'scaffold':'shared', 'level':'L4'}]
        training = [{'record_id':'same_parent', 'parent_id':'heldout', 'scaffold':'other', 'level':'L2'},
                    {'record_id':'same_scaffold', 'parent_id':'different', 'scaffold':'shared', 'level':'L1'},
                    {'record_id':'safe', 'parent_id':'safe', 'scaffold':'safe', 'level':'L3'}]
        for task in TASKS:
            parent = task_contract(task, split_disjointness='parent')
            scaffold = task_contract(task)
            self.assertEqual([r['record_id'] for r in filter_training(training, heldout, parent)[0]], ['same_scaffold','safe'])
            kept, excluded = filter_training(training, heldout, scaffold)
            self.assertEqual([r['record_id'] for r in kept], ['safe'])
            self.assertEqual([r['exclusion_reason'] for r in excluded], ['heldout_parent','heldout_scaffold'])
            with self.assertRaises(ValueError):
                check_disjointness(training, heldout, scaffold)

    def test_gold_filter_preserves_holdouts_and_refits(self):
        groups = {'train':[{'record_id':f't{i}', 'parent_id':f'p{i}', 'scaffold':f's{i}', 'geometry_value':value}
                           for i,value in enumerate((100, .1, .2, .4))],
                  'valid':[{'record_id':'val', 'parent_id':'val', 'scaffold':'s0', 'geometry_value':.8}], 'test':[]}
        original = copy.deepcopy(groups)
        retained, excluded = filter_gold(groups, task_contract('DILI'))
        self.assertEqual(len(excluded), 1)
        self.assertEqual(retained['valid'], original['valid'])
        self.assertEqual(groups, original)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'gold.json'
            write_json(source, retained)
            fit = gold_calibration(retained, source)
            self.assertAlmostEqual(fit['calibration_sample_standard_deviation'], statistics.stdev((.1,.2,.4,.8)))
            self.assertNotIn('t0', fit['fit_record_ids'])
        with self.assertRaises(ValueError):
            filter_gold({**groups, 'train':[groups['train'][0]]}, task_contract('DILI'))

    def test_bbb_assignment_uses_gold_only_across_levels(self):
        from assay_transfer.split import assign
        parents = ('GZCGUPFRVQAUEE-SLPGGIOYSA-N', 'GZCGUPFRVQAUEE-IFLZFJRASA-N',
                   'XOAAWQZATWQOTB-UHFFFAOYSA-N', 'DUYJMQONPNNFPI-UHFFFAOYSA-N')
        data = [{'parent_id':parent, 'level':level} for parent in parents for level in ('L1','L4')]
        gold = {'train':[], 'valid':[], 'test':[]}
        assign(data, gold, task_contract('BBB_Martins'))
        self.assertTrue(all(row['dataset_split']=='train' for row in data))
        gold['valid'] = [{'parent_id':parents[0]}]
        gold['test'] = [{'parent_id':parents[1]}]
        assign(data, gold, task_contract('BBB_Martins'))
        self.assertTrue(all(row['dataset_split']=='validation' for row in data if row['parent_id']==parents[0]))
        self.assertTrue(all(row['dataset_split']=='test_gate_only' for row in data if row['parent_id']==parents[1]))
        self.assertTrue(all(row['dataset_split']=='train' for row in data if row['parent_id'] in parents[2:]))

    def test_bbb_ood_panel_requires_gold_validation_membership(self):
        from assay_transfer.evaluation import select_panel
        base = {'level':'L4', 'measurement_kind':'continuous', 'pair_bucket_key':'bucket',
                'evaluation_regime':'ood', 'dataset_split':'ood', 'bucket_total_record_count':10}
        rows = [{**base, 'record_id':'extra', 'parent_id':'extra', 'validation_parent_eligible':False},
                {**base, 'record_id':'gold', 'parent_id':'gold', 'validation_parent_eligible':True}]
        select_panel(rows, {'L4':set()})
        self.assertFalse(rows[0]['query_validation'])
        self.assertTrue(rows[1]['query_validation'])

    def test_flag_hashes_and_invalid_ratios(self):
        base = task_contract('DILI')
        self.assertNotEqual(base['contract_sha256'], task_contract('DILI', split_disjointness='parent')['contract_sha256'])
        self.assertNotEqual(task_contract('DILI', augmentation_ratio=1)['contract_sha256'],
                            task_contract('DILI', augmentation_ratio=2)['contract_sha256'])
        for ratio in (-1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                task_contract('DILI', augmentation_ratio=ratio)

    def test_ratio_budgets_and_skin_proportions(self):
        records = {f'q{i}':{'measurement_kind':'continuous', 'level':level,'pair_bucket_key':level} for i,level in
                   enumerate(['L1']*24 + ['L2']*24 + ['L3']*3)}
        candidates = [{'a_id':f'a{i}', 'b_id':key, 'pair_id':str(i), 'near':True}
                      for i,key in enumerate(records)]
        for limit in (0, 5, 17):
            selected, budget = mixture(copy.deepcopy(candidates), records, task_contract('Skin_Reaction'), limit, False)
            self.assertEqual(len(selected), limit)
            self.assertEqual(budget['shortfall'], 0)
        with self.assertRaises(ValueError):
            mixture([], {}, task_contract('DILI'), 2, False)
        self.assertEqual(mixture([], {}, task_contract('DILI'), 2, True)[1]['shortfall'], 2)

    def test_oral_augmentation_uses_unscored_near_far_classes(self):
        records={f'q{i}':{'measurement_kind':'continuous','source_id':'oral_transfer','pair_bucket_key':'bucket'} for i in range(4)}
        candidates=[{'pair_id':str(i),'a_id':f'a{i}','b_id':f'q{i}','near':i%2==0,'probability':None,'similarity':.8} for i in range(4)]
        chosen,budget=mixture(candidates,records,task_contract('Bioavailability_Ma'),4,False)
        self.assertEqual(len(chosen),4)
        self.assertEqual(budget['shortfall'],0)
        self.assertTrue(all(row['probability'] is None for row in chosen))

    def test_selected_gold_inputs_ignore_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gold = root / 'gold.json'; write_json(gold, {})
            inputs = {'schema_version':'assay_transfer.inputs.v1', 'tasks':{'DILI':{'files':{
                'gold_train':{'path':str(gold), 'sha256':hashlib.sha256(gold.read_bytes()).hexdigest()},
                'records':{'path':str(root / 'missing.parquet'), 'sha256':'missing'}}}}}
            inputs['tasks']['DILI']['files']['gold_valid'] = inputs['tasks']['DILI']['files']['gold_train']
            inputs['tasks']['DILI']['files']['gold_test'] = inputs['tasks']['DILI']['files']['gold_train']
            inputs['inputs_sha256'] = digest(inputs)
            source = root / 'inputs.json'; write_json(source, inputs)
            args = SimpleNamespace(input_manifest=source, contract=DEFAULT_CONTRACT, task=['DILI'])
            selected = selected_inputs(args, gold_only=True)
            self.assertEqual(set(selected['tasks']['DILI']['files']), {'gold_train','gold_valid','gold_test'})
            with self.assertRaises(ValueError):
                selected_inputs(args)

    def test_record_filter_rejects_lost_support(self):
        from test_science import rows
        data = rows()
        heldout = {'record_id':'val', 'parent_id':'val', 'scaffold':'s0', 'geometry_value':.5}
        gold = {'train':[{'record_id':'gold', 'parent_id':'gold', 'scaffold':'gold', 'geometry_value':.2}],
                'valid':[heldout], 'test':[]}
        for mode, expected_count in (('scaffold',0), ('parent',12)):
            with tempfile.TemporaryDirectory() as temporary:
                policy = task_contract('Ames', split_disjointness=mode)
                with patch('assay_transfer.filtering.gold_groups', return_value=gold), \
                     patch('assay_transfer.filtering.project', return_value=(copy.deepcopy(data),{})), \
                     patch('assay_transfer.filtering.read_json', return_value={'bucket_overrides':{},'identity_units':['%'],'default_positive_transform':'log10'}):
                    if mode=='scaffold':
                        with self.assertRaisesRegex(ValueError, 'No eligible training'):
                            prepare_record_pool(Path(temporary), {'geometry_policy':{'path':'unused'}}, policy, True)
                    else:
                        prepare_record_pool(Path(temporary), {'geometry_policy':{'path':'unused'},'stage3_manifest':{'sha256':'source'}}, policy, True)
                        self.assertEqual(len(read_parquet(Path(temporary) / 'records.parquet')), expected_count)

    def test_separate_flags_and_default_augmentation(self):
        cli = ['builder','all','--input-manifest','inputs.json','--output-root','output']
        with patch.object(sys, 'argv', cli):
            args = arguments('context')
            self.assertTrue(args.augment)
            self.assertEqual(args.augmentation_ratio, 1.0)
            self.assertEqual(args.split_disjointness, 'scaffold')
        with patch.object(sys, 'argv', cli + ['--no-augment','--augmentation-ratio','2']):
            self.assertFalse(arguments('context').augment)
        with patch.object(sys, 'argv', cli):
            self.assertFalse(hasattr(arguments('record'), 'augment'))

    def test_safety_exhaustive_candidates_work_in_full_scope(self):
        from test_science import rows
        from assay_transfer.filtering import fit_bucket
        data, policy = rows(), task_contract('DILI')
        entry, reason = fit_bucket(data, policy, {'bucket_overrides':{},'identity_units':['%'],'default_positive_transform':'log10'})
        self.assertIsNone(reason)
        pairs, report = sampled(data, {data[0]['pair_bucket_key']:entry}, policy,
                                component='context_indirect', limit=None, smoke=False)
        self.assertTrue(pairs)
        self.assertIn('rejected_binary_levels',report)
        self.assertTrue(all(p['probability'] is None for p in pairs))

    def test_ood_selection_is_frozen_during_refit(self):
        from test_science import rows
        policy, data = task_contract('Skin_Reaction'), rows()
        reviewed = {'decisions':{}}
        with patch('assay_transfer.filtering.choose_skin_ood') as choose:
            retained, _, _, _, selection = calibrate(data, policy, reviewed,
                                                     {data[0]['pair_bucket_key']:12}, frozen_ood=[])
            choose.assert_not_called()
        self.assertEqual(selection['selected_buckets'], [])
        self.assertTrue(retained)
        self.assertTrue(all(row['evaluation_regime']=='id' for row in retained))

    def test_gold_only_cli_without_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            task,contract,command=build_gold(root)
            subprocess.run(command+['--resume'],check=True,capture_output=True)
            changed=subprocess.run(command+['--resume','--split-disjointness','parent'],capture_output=True)
            self.assertNotEqual(changed.returncode,0)
            self.assertIn(b'Immutable stage exists',changed.stderr)
            self.assertEqual(read_json(task/'verification/context/receipt.json')['status'],'verified')
            self.assertEqual(read_json(task/'augmentation/coverage.json')['indirect_budget']['requested'],0)
            self.assertFalse((task/'pool/indirect').exists())
            self.assertFalse((root/'outputs/record').exists())
            for stage in ('sampling','augmentation'):
                self.assertTrue(all(row['probability'] is None for row in read_parquet(task/stage/'train.parquet')))
            self.assertTrue(all(0<=row['probability']<=1 for row in read_parquet(task/'targets/train.parquet')))

    def test_upload_full_verified_release_and_reject_tampering(self):
        from assay_transfer.dataset import upload_verified
        with tempfile.TemporaryDirectory() as temporary:
            task,contract,_=build_gold(Path(temporary))
            policy=task_contract('DILI',contract,augmentation_ratio=0)
            api=Mock();api.create_commit.return_value=SimpleNamespace(oid='test-commit',commit_url='https://huggingface.co/test/commit')
            with patch('huggingface_hub.HfApi',return_value=api):
                uploaded=upload_verified(task,policy,'context','test/canonical',private=True)
            self.assertEqual(uploaded['commit_sha'],'test-commit')
            self.assertEqual(api.create_repo.call_args.kwargs['repo_type'],'dataset')
            self.assertTrue(api.create_repo.call_args.kwargs['private'])
            paths={op.path_in_repo for op in api.create_commit.call_args.kwargs['operations']}
            self.assertIn('provenance/verification.json',paths)
            self.assertIn('train/data.parquet',paths)
            self.assertFalse(any(path.startswith('pool/') for path in paths))
            (task/'targets/train.parquet').write_bytes(b'tampered')
            with patch('huggingface_hub.HfApi') as remote:
                with self.assertRaises(ValueError):upload_verified(task,policy,'context','test/canonical')
                remote.assert_not_called()

    def test_upload_rejects_smoke_before_remote_calls(self):
        from assay_transfer.dataset import upload_verified
        with tempfile.TemporaryDirectory() as temporary:
            task,contract,_=build_gold(Path(temporary),smoke=True)
            policy=task_contract('DILI',contract,augmentation_ratio=0)
            with patch('huggingface_hub.HfApi') as remote:
                with self.assertRaisesRegex(ValueError,'verified full release'):
                    upload_verified(task,policy,'context','test/canonical')
                remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
