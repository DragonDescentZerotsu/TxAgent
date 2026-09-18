"""Budget, host restriction and resume checks for the paid extraction schedule."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data.processing.evidence_library.versions.v9 import build_measurement_resolution_mapping as runner
from data.processing.evidence_library.versions.v9.build_reference_semantics_mapping import TokenLedger


class ExtractionHandoffTests(unittest.TestCase):
    def test_two_independent_caps_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('one.json', 'two.json')]
            ledgers = [TokenLedger(p, epoch='fresh', start_new_epoch=False, max_tokens=100) for p in paths]
            self.assertTrue(ledgers[0].reserve('a', 80))
            self.assertFalse(ledgers[0].reserve('b', 30))
            self.assertTrue(ledgers[1].reserve('b', 30))
            ledgers[0].complete('a', {'input_tokens': 30, 'output_tokens': 20})
            self.assertTrue(ledgers[0].reserve('interrupted', 40))
            resumed = TokenLedger(paths[0], epoch='fresh', start_new_epoch=False, max_tokens=100)
            self.assertEqual(resumed.spent(), 90)
            self.assertFalse(resumed.reserve('c', 11))

    def test_schedule_excludes_third_key_and_pins_fallback(self):
        argv = ['--task', 'bioavailability_ma', '--cache-dir', '/tmp/test-cache',
                '--budget-epoch', 'fresh', '--two-key-baidu-run']
        args = runner.parse_args(argv)
        with patch.object(runner, 'main', side_effect=[75, 75, 0]) as invoke:
            self.assertEqual(runner.run_two_key_baidu(argv, args), 0)
        calls = [runner.parse_args(call.args[0]) for call in invoke.call_args_list]
        self.assertEqual([c.api_key_env for c in calls],
                         ['OPENAI_API_KEY', 'OPENAI_API_KEY_TWO', 'OPEN_ROUTER_KEY'])
        self.assertEqual([c.budget_max_tokens for c in calls[:2]], [10_000_000] * 2)
        self.assertNotEqual(calls[0].token_ledger, calls[1].token_ledger)
        self.assertEqual(calls[2].provider_only, 'baidu/fp8')
        self.assertTrue(calls[2].retry_failed)
        self.assertTrue(calls[2].require_complete)

    def test_provider_lock_and_mismatch(self):
        requests = []
        response = SimpleNamespace(
            id='generation-1', model='deepseek/deepseek-v4-flash-0731', provider='Baidu',
            usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))],
        )
        def create(**kwargs):
            requests.append(kwargs)
            return response
        client = SimpleNamespace(base_url='https://openrouter.ai/api/v1',
                                 chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        call = runner.openai_compatible_llm(client, provider_only='baidu/fp8')
        result = call({'system': 'extract', 'user': '{}'}, model=response.model,
                      max_tokens=100, temperature=1)
        self.assertEqual(result['api_metadata']['served_provider'], 'Baidu')
        self.assertEqual(requests[0]['extra_body']['provider'],
                         {'only': ['baidu/fp8'], 'allow_fallbacks': False, 'require_parameters': True})
        response.provider = 'Another host'
        with self.assertRaises(ValueError):
            call({'system': 'extract', 'user': '{}'}, model=response.model,
                 max_tokens=100, temperature=1)

    def test_only_failures_are_retried(self):
        cache = SimpleNamespace(
            attempted={'valid', 'uncertain', 'failed', 'interrupted'},
            assignments={
                'valid': {'assignment_method': 'model_single_pass', 'status': 'ok'},
                'uncertain': {'assignment_method': 'model_single_pass', 'status': 'unsure'},
                'failed': {'assignment_method': 'api_failure', 'status': 'unavailable'},
            },
        )
        self.assertEqual(runner.retryable_assignment_ids(cache), {'failed', 'interrupted'})

    def test_quota_rejection_is_unbilled_and_available_for_fallback(self):
        error = RuntimeError('no credits')
        error.body = {'code': 'credit_balance_exhausted'}
        batch = runner.RequestBatch('batch', 'fa', ({'id': 'row', 'source_id': 'fa'},), 'extract', 100)
        rows, usage, status = runner.query_batch(
            batch, llm=Mock(side_effect=error), model='gpt-5.4-mini',
            task='bioavailability_ma',
        )
        self.assertEqual(status, 'api_failure_quota_exhausted')
        self.assertEqual(usage, {'input_tokens': 0, 'output_tokens': 0})
        self.assertEqual(rows[0]['assignment_method'], 'api_failure_quota_exhausted')
