"""Budget, host restriction and resume checks for the paid extraction schedule."""

import tempfile
import unittest
import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from data.processing.evidence_library.versions.v10 import build_measurement_resolution_mapping as runner
from data.processing.evidence_library.versions.v10.build_reference_semantics_mapping import TokenLedger
from data.processing.evidence_library.versions.v10.build_reference_semantics_mapping import CACHE_VERSION
from data.processing.evidence_library.versions.v10.run_measurement_resolution_full_matrix import (
    DURABLE_RUNS_ROOT,
    _interleaved_batches,
    _paid_usage,
    _plan_retries,
    _preflight_local,
    _proportional_quotas,
    _retry_epoch_start,
    _retry_ids,
)
from data.processing.evidence_library.versions.v10.tasks.ames.data_processing.consolidate_measurement_resolution import (
    build_mixed_mapping,
    validate_mixed_mapping,
)


class ExtractionHandoffTests(unittest.TestCase):
    @patch("data.processing.evidence_library.versions.v10.run_measurement_resolution_full_matrix._validate_provider_config")
    @patch("data.processing.evidence_library.versions.v10.run_measurement_resolution_full_matrix.socket.gethostname", return_value="worker")
    def test_local_preflight_requires_one_durable_run_root(self, _hostname, validate):
        run_id = "durable-test"
        durable = DURABLE_RUNS_ROOT / run_id
        args = SimpleNamespace(
            run_id=run_id,
            expected_host="worker",
            staging_root=durable,
            canonical_cache_root=durable,
            tasks=["dili", "carcinogens"],
            retry_max_completion_tokens=65_536,
            provider_config=Path("pool.json"),
            retry_only=True,
            dili_retry_batch_size=20,
            request_timeout_s=3_600,
        )
        _preflight_local(args)
        validate.assert_called_once()

        args.retry_max_completion_tokens = 524_288
        args.request_timeout_s = 7_200
        _preflight_local(args)

        args.request_timeout_s = 0
        with self.assertRaisesRegex(ValueError, "request timeout must be positive"):
            _preflight_local(args)
        args.request_timeout_s = 7_200

        args.staging_root = Path("/local/lost-on-node")
        with self.assertRaisesRegex(ValueError, "staging and canonical roots"):
            _preflight_local(args)

    def test_phase_cap_stops_without_closing_shared_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = TokenLedger(
                root / "ledger.json",
                epoch="fresh",
                start_new_epoch=False,
                max_tokens=100_000,
            )
            batch = runner.RequestBatch(
                "batch",
                "source",
                ({"id": "row", "source_id": "source"},),
                "prompt",
                100,
            )
            args = SimpleNamespace(workers=1, phase_budget_max_tokens=1)
            outcome = runner._execute_batches(
                [batch],
                args,
                SimpleNamespace(task_id="ames", module=SimpleNamespace()),
                SimpleNamespace(),
                ledger,
                Mock(),
                "gpt-5.4-mini",
                "https://api.openai.com/v1",
                "OPENAI_API_KEY_ONE",
            )
            self.assertEqual(outcome, (False, False, True))
            self.assertEqual(
                runner._execution_exit_code(*outcome, ledger),
                runner.PHASE_BUDGET_EXHAUSTED_EXIT_CODE,
            )
            self.assertEqual(ledger.state["status"], "active")

    def test_remaining_paid_budget_is_split_by_extraction_rows(self):
        counts = {"ames": 2, "dili": 3, "carcinogens": 5}
        self.assertEqual(
            _proportional_quotas(100, counts),
            {"ames": 20, "dili": 30, "carcinogens": 50},
        )

    def test_paid_usage_counts_only_terminal_gpt_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "requests.jsonl"
            events = [
                {"status": "submitted", "request_id": "gpt", "model": "gpt-5.4-mini"},
                {"status": "terminal", "request_id": "gpt", "usage": {"input_tokens": 7, "output_tokens": 3}},
                {"status": "submitted", "request_id": "local", "model": "deepseek-ai/DeepSeek-V4-Flash-0731"},
                {"status": "terminal", "request_id": "local", "usage": {"input_tokens": 100, "output_tokens": 100}},
            ]
            cache.write_text("".join(json.dumps(event) + "\n" for event in events))
            self.assertEqual(_paid_usage(cache), 10)

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

    def test_schedule_carries_base_mapping_and_original_budget_directory(self):
        argv = ['--task','bioavailability_ma','--cache-dir','/tmp/new-cache',
                '--budget-epoch','original','--two-key-baidu-run',
                '--base-mapping','/tmp/reused.parquet','--budget-ledger-dir','/tmp/original-ledgers']
        with patch.object(runner,'main',side_effect=[75,75,0]) as invoke:
            self.assertEqual(runner.run_two_key_baidu(argv,runner.parse_args(argv)),0)
        calls=[runner.parse_args(call.args[0]) for call in invoke.call_args_list]
        self.assertTrue(all(c.base_mapping==Path('/tmp/reused.parquet') for c in calls))
        self.assertTrue(all(c.token_ledger.parent==Path('/tmp/original-ledgers') for c in calls[:2]))
        self.assertTrue(all(not c.start_new_budget_epoch for c in calls))

    def test_mixed_base_provenance_survives_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.parquet'
            profile=root/'profile.json'
            source.write_bytes(b'fixture source')
            profile.write_text('{}')
            rows=[{'cleaned_record_id':key,'source_row_uid':'uid-'+key,
                   'status':'relative','measurements_json':'[]','quantity_count':0,
                   'assignment_method':'model_single_pass','rejected_response_json':None,
                   'inference_model':model,'inference_base_url':url,'inference_credential_env':credential}
                  for key,model,url,credential in [('a','gpt','openai','KEY_ONE'),('b','deepseek','openrouter','ROUTER')]]
            rows[1]['reuse_mapping_sha256']='original-receipt'
            base=root/'base.parquet'
            pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in set().union(*(r.keys() for r in rows))} for r in rows]),base)
            base.with_suffix('.manifest.json').write_text(json.dumps({'mapping_sha256':runner.file_sha256(base)}))
            assignments=runner.load_base_mapping(base)
            candidates=[{'id':r['cleaned_record_id'],'source_row_uid':r['source_row_uid']} for r in rows]
            runner.materialize(candidates,SimpleNamespace(events=[]),runner.TaskConfig('bioavailability_ma'),
                mapping_path=root/'result.parquet',records_path=source,profile_path=profile,
                model='unused',api_base_url='unused',max_completion_tokens=100,reasoning_mode='low',
                base_assignments=assignments,base_mapping_path=base)
            result=pq.read_table(root/'result.parquet').to_pylist()
            for original,actual in zip(rows,result):
                for key,value in original.items():self.assertEqual(actual[key],value)
            base.with_suffix('.manifest.json').write_text('{"mapping_sha256":"wrong"}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):runner.load_base_mapping(base)

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

    def test_provider_pool_receives_task_reasoning_effort(self):
        args = SimpleNamespace(provider_pool_config=Path("pool.json"))
        cache = SimpleNamespace()
        expected = (Mock(), "model", "provider_pool", "")
        with patch.object(runner, "_run_provider_pool", return_value=expected) as build:
            self.assertIs(runner._run_client(args, cache, None, "high"), expected)
        build.assert_called_once_with(args, cache, "high")

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

    def test_schema_retry_keeps_valid_rows_and_sends_feedback(self):
        calls = []

        def llm(prompt, **kwargs):
            payload = json.loads(prompt["user"])
            calls.append((payload, kwargs))
            if len(calls) == 1:
                rows = [
                    {"id": "r1", "status": "ok", "measurements": [
                        {"measurement": "1", "unit": "ratio A/B"}
                    ]},
                    {"id": "r2", "status": "ok", "measurements": [
                        {"measurement": "2", "unit": ""}
                    ]},
                ]
            else:
                rows = [{"id": "r2", "status": "unsure", "measurements": []}]
            return {
                "content": json.dumps({"rows": rows}),
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                "api_metadata": {},
            }

        batch = runner.RequestBatch(
            "batch",
            "source",
            (
                {"id": "one", "source_id": "source"},
                {"id": "two", "source_id": "source"},
            ),
            "prompt",
            32768,
        )
        rows, usage, _ = runner.query_batch(
            batch,
            llm=llm,
            model="model",
            task="ames",
            retry_validation_feedback=True,
            max_schema_attempts=2,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual([row["id"] for row in calls[1][0]["rows"]], ["r2"])
        self.assertEqual(
            calls[1][0]["validation_feedback"],
            [{"id": "r2", "error": "entry_without_a_unit"}],
        )
        self.assertEqual([row["status"] for row in rows], ["ok", "unsure"])
        self.assertEqual(usage, {"input_tokens": 6, "output_tokens": 8})

    def test_schema_attempt_cap_is_actual_call_cap(self):
        llm = Mock(
            return_value={
                "content": "",
                "usage": {"prompt_tokens": 1, "completion_tokens": 32768},
                "api_metadata": {},
            }
        )
        batch = runner.RequestBatch(
            "batch",
            "source",
            ({"id": "row", "source_id": "source"},),
            "prompt",
            8192,
        )
        rows, usage, _ = runner.query_batch(
            batch,
            llm=llm,
            model="model",
            task="ames",
            retry_validation_feedback=True,
            max_schema_attempts=3,
            initial_validation_feedback={"row": "entry_without_a_unit"},
            retry_max_completion_tokens=65536,
        )
        self.assertEqual(llm.call_count, 3)
        self.assertEqual(
            [call.kwargs["max_tokens"] for call in llm.call_args_list],
            [8192, 65536, 65536],
        )
        self.assertEqual(rows[0]["assignment_method"], "invalid_response_after_structural_retry")
        self.assertEqual(usage["output_tokens"], 3 * 32768)
        first_user = json.loads(llm.call_args_list[0].args[0]["user"])
        self.assertEqual(
            first_user["validation_feedback"],
            [{"id": "r1", "error": "entry_without_a_unit"}],
        )

    def test_combined_queue_interleaves_and_caps_total_submissions(self):
        first = {
            "task": "dili",
            "batches": [SimpleNamespace(request_id=value) for value in ("d1", "d2")],
        }
        second = {
            "task": "carcinogens",
            "batches": [SimpleNamespace(request_id=value) for value in ("c1", "c2", "c3")],
        }
        order = [batch.request_id for _, batch in _interleaved_batches([first, second])]
        self.assertEqual(order, ["d1", "c1", "d2", "c2", "c3"])

        cache = SimpleNamespace(
            attempted={"retry", "capped"},
            assignments={
                "retry": {"assignment_method": "api_failure"},
                "capped": {"assignment_method": "worker_failure"},
            },
            events=[
                {"status": "submitted", "row_ids": ["retry", "capped"]},
                {"status": "submitted", "row_ids": ["capped"]},
                {"status": "submitted", "row_ids": ["capped"]},
            ],
        )
        cache.events = [
            {"status": "submitted", "model": "old-model", "row_ids": ["retry", "capped"]},
            {"status": "submitted", "model": "new-model", "row_ids": ["capped"]},
            {"status": "submitted", "model": "new-model", "row_ids": ["capped"]},
            {"status": "submitted", "model": "new-model", "row_ids": ["capped"]},
        ]
        self.assertEqual(_retry_ids(cache, "new-model"), {"retry"})

    def test_retry_epoch_counts_only_successor_attempts(self):
        cache = SimpleNamespace(
            events=[
                {"status": "submitted", "model": "same-model", "row_ids": ["row"]}
            ],
            attempted={"row"},
            assignments={"row": {"assignment_method": "api_failure"}},
            _append=lambda event: cache.events.append(event),
        )
        start = _retry_epoch_start(cache, "successor")
        self.assertEqual(_retry_ids(cache, "same-model", start), {"row"})
        self.assertEqual(_retry_epoch_start(cache, "successor"), start)

    def test_dili_retry_batch_is_split_without_changing_prompt(self):
        rows = tuple({"id": f"row-{index}"} for index in range(20))
        payload = tuple({"id": f"r{index + 1}"} for index in range(20))
        planned = runner.RequestBatch("request", "source", rows, "prompt", 65_536, payload)
        cache = SimpleNamespace(
            events=[], attempted=set(), allow_retry=Mock()
        )
        context = {
            "task": "dili",
            "retry_batch_size": 10,
            "cache": cache,
            "args": SimpleNamespace(model="model"),
            "delta_candidates": list(rows),
            "config": SimpleNamespace(),
            "endpoint_profile": {},
            "profile_digest": "digest",
            "retry_epoch_start": 0,
        }
        with patch.object(runner, "retryable_assignment_ids", return_value={row["id"] for row in rows}), patch.object(
            runner, "plan_batches", return_value=[planned]
        ):
            _plan_retries(context)
        self.assertEqual([len(batch.rows) for batch in context["batches"]], [10, 10])
        self.assertEqual([batch.prompt for batch in context["batches"]], ["prompt", "prompt"])
        self.assertEqual(context["batches"][1].api_rows[0]["id"], "r11")

    def test_mixed_mapping_preserves_layer_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.parquet"
            profile = root / "profile.json"
            pq.write_table(
                pa.Table.from_pylist([
                    {
                        "cleaned_record_id": record_id,
                        "source_row_uid": "uid-" + record_id,
                        "source_id": "mutagenicity_mechanism",
                        "measurement_resolution_route": "extract",
                    }
                    for record_id in ("a", "b")
                ]),
                records,
            )
            profile.write_text("{}")
            layers = []
            for index, record_id in enumerate(("a", "b"), 1):
                layer_root = root / f"layer-{index}"
                cache_path = layer_root / "requests.jsonl"
                contract_path = layer_root / "input_contract.json"
                layer_root.mkdir()
                assignment = {
                    "cleaned_record_id": record_id,
                    "source_id": "mutagenicity_mechanism",
                    "status": "unsure",
                    "measurements_json": "[]",
                    "quantity_count": 0,
                    "assignment_method": "model_single_pass",
                    "rejected_response_json": None,
                    "raw_response_json": "{}",
                    "returned_model": "model",
                    "base_url": "endpoint",
                }
                events = [
                    {
                        "cache_version": CACHE_VERSION,
                        "status": "submitted",
                        "request_id": f"request-{index}",
                        "row_ids": [record_id],
                        "model": "model",
                        "base_url": "endpoint",
                    },
                    {
                        "cache_version": CACHE_VERSION,
                        "status": "terminal",
                        "request_id": f"request-{index}",
                        "row_ids": [record_id],
                        "assignments": [assignment],
                    },
                ]
                cache_path.write_text("".join(json.dumps(event) + "\n" for event in events))
                contract_path.write_text(json.dumps({
                    "task": "ames",
                    "records_sha256": runner.file_sha256(records),
                    "profile_sha256": runner.file_sha256(profile),
                    "reasoning_effort": "low" if index == 1 else "high",
                    "prompt": {
                        "prompt_version": f"prompt-{index}",
                        "template_sha256": f"template-{index}",
                    },
                }))
                layers.append({
                    "layer_id": f"layer-{index}",
                    "cache_path": str(cache_path),
                    "contract_path": str(contract_path),
                    "max_completion_tokens": 32768,
                })
            output = root / "mapping.parquet"
            manifest = build_mixed_mapping(
                records_path=records,
                profile_path=profile,
                layers=layers,
                output_path=output,
            )
            self.assertEqual(manifest["layer_counts"], {"layer-1": 1, "layer-2": 1})
            validate_mixed_mapping(output)
            with (root / "layer-1" / "requests.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "cache changed"):
                validate_mixed_mapping(output)
