"""Behavioral checks for UID joins and level-scoped relevance comparisons."""

import json
import tempfile
import unittest
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from analysis.evidence_library import relevance_bucket_levels as levels
from analysis.evidence_library import bioavailability_relevance_bucket_levels as oral_levels
from analysis.evidence_library import bioavailability_relevance_bucket_tournament as oral


class LevelRelevanceTests(unittest.TestCase):
    def test_oral_endpoint_groups_and_three_record_sampling_are_outcome_blind(self):
        identity_record = {
            "source_id": "hf_bioavailability",
            "canonical_endpoint_concept": "oral_bioavailability",
            "canonical_bioavailability_report_type": "absolute",
            "canonical_bioavailability_evidence_scope": "direct",
            "canonical_endpoint_name": "must_not_be_in_identity",
            "measurement_kind": "must_not_be_in_identity",
        }
        _, identity = oral.relevance_bucket(identity_record)
        self.assertEqual(identity, {
            "source_id": "hf_bioavailability",
            "canonical_endpoint_concept": "oral_bioavailability",
            "canonical_bioavailability_report_type": "absolute",
            "canonical_bioavailability_evidence_scope": "direct",
        })

        records = []
        for index, (pair_bucket, unit) in enumerate((
            ("pair-a", "%"), ("pair-a", "fraction"), ("pair-b", "ratio"),
        )):
            records.append({
                **identity_record,
                "canonical_record_id": f"record-{index}",
                "pair_bucket_key": pair_bucket,
                "canonical_unit_text": unit,
                "finite_scalar_value": 99,
                "direct_vote_label": "positive",
            })
        samples = oral.select_diverse_records(records)
        self.assertEqual(len(samples), 3)
        self.assertEqual({sample["canonical_unit_text"] for sample in samples}, {
            "%", "fraction", "ratio",
        })
        self.assertTrue(all("finite_scalar_value" not in sample for sample in samples))
        self.assertTrue(all("direct_vote_label" not in sample for sample in samples))

    def test_single_comparison_payload_and_exact_winner_contract(self):
        calls = levels.calls
        with tempfile.TemporaryDirectory() as temporary:
            connection = levels.bbb._connect(Path(temporary) / "requests.sqlite3")
            for key, source in (("a", "fa"), ("b", "fg")):
                connection.execute(
                    "INSERT INTO buckets VALUES (?, ?, ?, ?, 1, 1, NULL)",
                    (key, source, json.dumps({"source_id": source}), json.dumps([{"unit": "%"}])),
                )
            connection.execute(
                "INSERT INTO comparisons VALUES ('L2-000', 'L2', 'L2-000', 'a', 'b', NULL, NULL, NULL)"
            )
            prompt = calls.render_request(
                connection, "L2-000", template_path=oral_levels.TEMPLATE,
                compact=True, single_comparison=True,
            )
            payload = json.JSONDecoder().raw_decode(prompt[prompt.index('{"groups":'):])[0]
            self.assertEqual(set(payload), {"groups"})
            self.assertEqual(set(payload["groups"]), {
                calls.bucket_id("a"), calls.bucket_id("b"),
            })
            comparison = calls._comparisons(connection, "L2-000")
            schema = calls.response_format(
                comparison, allow_ties=False, single_comparison=True,
            )["json_schema"]["schema"]
            self.assertEqual(schema["required"], ["winner_bucket_id"])
            self.assertFalse(schema["additionalProperties"])
            result = calls.validate_response(
                {"winner_bucket_id": calls.bucket_id("a")}, comparison,
                allow_ties=False, single_comparison=True,
            )
            self.assertEqual(result[0]["winner_key"], "a")
            with self.assertRaisesRegex(ValueError, "only winner_bucket_id"):
                calls.validate_response(
                    {"winner_bucket_id": calls.bucket_id("a"), "reason": "extra"},
                    comparison, allow_ties=False, single_comparison=True,
                )
            connection.close()

    def test_oral_degree_ten_graphs_are_regular_and_connected(self):
        for size in (11, 17, 64, 159):
            keys = [f"bucket-{index}" for index in range(size)]
            edges = oral_levels._level_edges(keys, "L2")
            degree = Counter(key for edge in edges for key in edge)
            self.assertEqual(len(edges), size * min(10, size - 1) // 2)
            self.assertTrue(all(degree[key] == min(10, size - 1) for key in keys))
            self.assertTrue(levels.bbb._connected([(a, b, a) for a, b in edges]))

    def test_rolling_slots_refill_without_waiting_for_slow_provider(self):
        calls = levels.calls
        release = threading.Event()
        lock = threading.Lock()
        active = Counter()
        peak = Counter()
        seen = []
        with tempfile.TemporaryDirectory() as temporary:
            con = levels.bbb._connect(Path(temporary) / "requests.sqlite3")
            for i in range(8):
                con.execute("INSERT INTO comparisons VALUES (?, 'L2', ?, 'a', 'b', NULL, NULL, NULL)", (f"L2-{i:03}", f"batch-{i:03}"))
            def create(**request):
                provider = request["extra_body"]["provider"]["only"][0]
                batch = request["messages"][0]["content"]
                with lock:
                    active[provider] += 1
                    peak[provider] = max(peak[provider], active[provider])
                    seen.append(batch)
                if batch == "batch-000":
                    if not release.wait(3):
                        raise ValueError("Other providers did not refill")
                elif int(batch[-3:]) >= 3:
                    release.set()
                with lock:
                    active[provider] -= 1
                return SimpleNamespace(id=batch, provider=provider, model="served", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost=0.001),
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"items": [{"id": "P"+batch[-3:], "winner_bucket_id": calls.bucket_id("a"), "reason": "More informative"}]})))])
            client = Mock(); client.chat.completions.create.side_effect = create
            with patch.object(calls, "openai_compatible_client", return_value=(client,"test")), \
                 patch.object(calls,"render_request",side_effect=lambda con,batch,**kw: batch):
                calls.run_condition(con,"L2",parallelism=3,token_budget=10000,env_file=None,
                    slot_extra_bodies=[{"provider":{"only":[name]}} for name in ("open-inference/fp8","deepinfra/fp8","relace/fp4")])
            self.assertEqual(len(seen),8)
            self.assertEqual(len(set(seen)),8)
            self.assertEqual(dict(peak),dict.fromkeys(("open-inference/fp8","deepinfra/fp8","relace/fp4"),1))
            self.assertEqual(con.execute("SELECT count(*) FROM request_attempt_receipts").fetchone()[0],8)
            self.assertTrue(release.is_set())
            con.close()

    def test_budget_reservations_and_shutdown_drain(self):
        calls = levels.calls
        for budget, shutdown in ((0.0001, False), (0.04, False), (None, True)):
            with tempfile.TemporaryDirectory() as temporary:
                con = levels.bbb._connect(Path(temporary)/"requests.sqlite3")
                for i in range(4):
                    con.execute("INSERT INTO comparisons VALUES (?, 'L2', ?, 'a', 'b', NULL, NULL, NULL)",(f"L2-{i:03}",f"batch-{i:03}"))
                event = threading.Event()
                barrier = threading.Barrier(2)
                def create(**request):
                    batch=request["messages"][0]["content"]
                    if shutdown:
                        barrier.wait(timeout=3)
                        event.set()
                    return SimpleNamespace(model="served",usage=SimpleNamespace(prompt_tokens=10,completion_tokens=5),
                        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"items":[{"id":"P"+batch[-3:],"winner_bucket_id":calls.bucket_id("a"),"reason":"Useful"}]})))])
                client=Mock(); client.chat.completions.create.side_effect=create
                with patch.object(calls,"openai_compatible_client",return_value=(client,"test")), \
                     patch.object(calls,"render_request",side_effect=lambda con,batch,**kw: batch):
                    kwargs=dict(parallelism=2,token_budget=10000,env_file=None,max_cost_usd=budget,
                        input_cost_per_million=1,output_cost_per_million=1,max_completion_tokens=10,stop_event=event)
                    if shutdown or budget==0.0001:
                        with self.assertRaisesRegex(RuntimeError,"drain"):
                            calls.run_condition(con,"L2",**kwargs)
                    else:
                        calls.run_condition(con,"L2",**kwargs)
                expected=2 if shutdown else 0 if budget==0.0001 else 4
                self.assertEqual(client.chat.completions.create.call_count,expected)
                self.assertEqual(con.execute("SELECT count(*) FROM requests WHERE status='complete'").fetchone()[0],expected)
                self.assertEqual(con.execute("SELECT sum(input_tokens+output_tokens) FROM requests").fetchone()[0],expected*15)
                con.close()

    def test_compact_payload_and_paid_prompt_migration(self):
        calls = levels.calls
        with tempfile.TemporaryDirectory() as temporary:
            con = levels.bbb._connect(Path(temporary) / "requests.sqlite3")
            for key in ("a", "b", "c"):
                con.execute("INSERT INTO buckets VALUES (?, 'direct_bbb', ?, ?, 1, 1, NULL)",
                            (key, json.dumps({"endpoint": key}), json.dumps([{"species": "rat", "assay_model": key}])))
            for i, (a, b) in enumerate((("a", "b"), ("b", "c"))):
                con.execute("INSERT INTO comparisons VALUES (?, 'L2', 'batch', ?, ?, NULL, NULL, NULL)",
                            (f"L2-{i:03}", a, b))
            old = calls.render_request(con, "batch", template_path=levels.TEMPLATE)
            new = calls.render_request(con, "batch", template_path=levels.COMPACT_TEMPLATE, compact=True)
            old_pairs = json.JSONDecoder().raw_decode(old[old.index('[\n'):])[0]
            payload = json.JSONDecoder().raw_decode(new[new.index('{"buckets":'):])[0]
            self.assertEqual(len(payload["buckets"]), 3)
            for original, pair in zip(old_pairs, payload["pairs"]):
                self.assertEqual(original, {"id": pair["id"],
                    "candidate_1": payload["buckets"][pair["candidates"][0]],
                    "candidate_2": payload["buckets"][pair["candidates"][1]]})
            calls._prepare_requests(con, "L2", template_path=levels.TEMPLATE)
            con.execute("UPDATE requests SET status='failed', attempts=2, input_tokens=100, output_tokens=40, error='test error'")
            calls._prepare_requests(con, "L2", template_path=levels.COMPACT_TEMPLATE, compact=True)
            archive = json.loads(con.execute("SELECT request_json FROM request_prompt_history").fetchone()[0])
            self.assertEqual(archive["prompt"], old)
            self.assertEqual(archive["error"], "test error")
            self.assertEqual(tuple(con.execute("SELECT attempts,input_tokens,output_tokens FROM requests").fetchone()), (2,100,40))
            con.execute("UPDATE requests SET status='complete'")
            saved = dict(con.execute("SELECT * FROM requests").fetchone())
            calls._prepare_requests(con, "L2", template_path=levels.COMPACT_TEMPLATE, compact=True)
            self.assertEqual(saved, dict(con.execute("SELECT * FROM requests").fetchone()))
            self.assertEqual(con.execute("SELECT count(*) FROM request_prompt_history").fetchone()[0], 1)
            con.close()

    def test_failed_response_preserves_model_and_resume_skips_completed(self):
        calls = levels.calls
        reason = "This assay measures brain uptake directly and therefore offers more relevant evidence than the alternative."
        with tempfile.TemporaryDirectory() as temporary:
            con = levels.bbb._connect(Path(temporary) / "requests.sqlite3")
            for i in range(4):
                con.execute("INSERT INTO comparisons VALUES (?, 'L2', ?, 'a', 'b', NULL, NULL, NULL)",
                            (f"L2-{i:03}", f"batch-{i:03}"))
            client = Mock()
            fail = True

            def create(**request):
                self.assertEqual(request["model"], "requested-model")
                self.assertNotIn("response_format", request)
                self.assertIn(request["extra_body"]["provider"]["only"], [["relace/fp4"], ["baidu/fp8"]])
                batch = request["messages"][0]["content"]
                item = {"id": "P" + batch[-3:], "winner_bucket_id": calls.bucket_id("a"),
                        "reason": reason}
                if fail and batch == "batch-001":
                    item["winner_bucket_id"] = "invalid"
                return SimpleNamespace(model="served-alias", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                                       choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"items": [item]})))])

            client.chat.completions.create.side_effect = create
            with patch.object(calls, "openai_compatible_client", return_value=(client, "test")), \
                 patch.object(calls, "render_request", side_effect=lambda con, batch, **kw: batch), \
                 patch.object(calls.time, "sleep"), \
                 patch.object(calls.concurrent.futures, "as_completed", side_effect=lambda futures: list(futures)):
                kwargs = dict(parallelism=2, token_budget=10000, env_file=None,
                              model="requested-model", max_reason_words=None, structured_output=False,
                              slot_extra_bodies=[{"provider": {"only": [name], "allow_fallbacks": False}}
                                                for name in ("relace/fp4", "baidu/fp8")])
                with self.assertRaisesRegex(RuntimeError, "1 L2 request"):
                    calls.run_condition(con, "L2", **kwargs)
                self.assertEqual(con.execute("SELECT count(*) FROM requests WHERE status='complete'").fetchone()[0], 3)
                routes = [json.loads(row[0])["provider"]["only"][0] for row in
                          con.execute("SELECT request_extra_body FROM requests ORDER BY batch_id")]
                self.assertEqual(routes, ["relace/fp4", "baidu/fp8"] * 2)
                calls.run_condition(con, "L2", allow_partial=True, **kwargs)
                self.assertEqual(con.execute("SELECT count(*) FROM requests WHERE status='failed'").fetchone()[0], 1)
                fail = False
                before = client.chat.completions.create.call_count
                with patch.object(calls, "_prepare_requests", side_effect=AssertionError("Must not rescan completed requests")):
                    calls.run_condition(con, "retry_all_levels", prepared_batch_ids=[f"batch-{i:03}" for i in range(4)], **kwargs)
                self.assertEqual(client.chat.completions.create.call_count - before, 1)
                self.assertEqual(con.execute("SELECT sum(input_tokens+output_tokens) FROM requests").fetchone()[0], 150)
                comparison = calls._comparisons(con, "batch-000")
                for field, value in [("reason", ""), ("winner_bucket_id", "invalid"), ("id", "unknown")]:
                    item = {"id": "P000", "winner_bucket_id": calls.bucket_id("a"), "reason": reason}
                    item[field] = value
                    with self.assertRaises(ValueError):
                        calls.validate_response({"items": [item]}, comparison, max_reason_words=None)
            con.close()

    def test_acquisition_join_preserves_descendants_and_unmapped(self):
        records = pd.DataFrame([
            {"canonical_record_id": str(i), "source_id": source,
             "source_record_id": "ext_1", "source_row_number": 1}
            for i, source in enumerate(["a", "a", "b"])
        ])
        ledger = pd.DataFrame([
            {"source_id": s, "source_record_id": "ext_1", "acquisition_source_row_number": 1,
             "source_row_uid": "sr_" + c * 32} for s, c in [("a", "a"), ("b", "b")]
        ])
        mapping = pd.DataFrame([{"source_row_uid": "sr_" + "a" * 32,
                                 "level": 3, "family_key": "passive", "source_group_id": "g"}])
        result = levels.attach_levels(records, ledger, mapping)
        self.assertEqual(result.level.notna().sum(), 2)
        self.assertEqual(len(result), 3)
        records.loc[0, "source_row_number"] = 2
        with self.assertRaisesRegex(ValueError, "exact acquisition"):
            levels.attach_levels(records, ledger, mapping)

    def test_graph_degree_unique_edges_and_determinism(self):
        for n in (1, 2, 7, 21, 32, 100):
            keys = [str(i) for i in range(n)]
            edges = levels.level_edges(keys, "L2")
            self.assertEqual(edges, levels.level_edges(list(reversed(keys)), "L2"))
            self.assertEqual(len(edges), len({tuple(sorted(e)) for e in edges}))
            degree = Counter(k for e in edges for k in e)
            self.assertTrue(all(degree[k] == min(20, n - 1) for k in keys))

    def test_outcomes_do_not_change_selected_samples(self):
        records = [{"source_id": "direct_bbb", "canonical_endpoint_name": "brain_exposure",
                    "canonical_assay_context": str(i), "finite_scalar_value": i,
                    "canonical_reference_scope": "absolute", "bbb_transport_label": "positive"}
                   for i in range(8)]
        original = levels.semantic_samples(records)
        for row in records:
            row["finite_scalar_value"] = 999
            row["bbb_transport_label"] = "negative"
        self.assertEqual(original, levels.semantic_samples(records))
        self.assertEqual(len(original), 5)
        self.assertTrue(all("canonical_reference_scope" not in c for c in original))

    def test_same_semantic_bucket_has_separate_level_ranks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            con = levels.bbb._connect(output / "requests.sqlite3")
            manifest = {"levels": {"L2": {"buckets": 2}, "L3": {"buckets": 2}}}
            for level in manifest["levels"]:
                keys = []
                for source in ("a", "b"):
                    bucket = json.dumps({"source_id": source})
                    key = json.dumps({"level": level, "relevance_bucket": bucket})
                    keys.append(key)
                    con.execute("INSERT INTO buckets VALUES (?, ?, ?, ?, ?, 1, NULL)",
                                (key, source, bucket, "[]", 1))
                winner = keys[0 if level == "L2" else 1]
                con.execute("INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
                            (level, level, level, *keys, winner, "test judgment"))
            levels.export_rankings(con, output, manifest)
            rankings = pd.read_parquet(output / "relevance_bucket_rankings.parquet")
            self.assertEqual(rankings[rankings.level_rank == 1].source_id.tolist(), ["a", "b"])
            con.execute("UPDATE comparisons SET winner=NULL WHERE phase='L3'")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                levels.export_rankings(con, output, manifest)
            con.close()

    def test_retrieval_eligibility_is_an_exact_within_level_percentile_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ranking_dir = root / "rankings"
            output = root / "eligibility"
            ranking_dir.mkdir()
            rankings = pd.DataFrame([
                {"node_key": f"node-{level}", "level": level,
                 "level_percentile": 20.0 if level in {"L2", "L4"} else 19.99}
                for level in ("L2", "L3", "L4", "L5")
            ])
            record_map = pd.DataFrame([
                {"canonical_record_id": f"record-{level}", "node_key": f"node-{level}",
                 "level": level}
                for level in ("L2", "L3", "L4", "L5")
            ])
            ranking_path = ranking_dir / "relevance_bucket_rankings.parquet"
            record_map_path = ranking_dir / "record_relevance_map.parquet"
            rankings.to_parquet(ranking_path, index=False)
            record_map.to_parquet(record_map_path, index=False)
            (ranking_dir / "manifest.json").write_text(json.dumps({
                "task": "bbb_martins", "status": "complete",
                "input_sha256": "stage3", "mapped_L2_plus_records": 4,
                "ranking_sha256": levels.file_sha256(ranking_path),
                "record_map_sha256": levels.file_sha256(record_map_path),
            }))

            manifest = levels.export_retrieval_eligibility(ranking_dir, output)
            result = pd.read_parquet(output / "record_relevance_eligibility.parquet")
            self.assertEqual(result.columns.tolist(), [
                "canonical_record_id", "level", "relevance_bucket_retrieval_eligible"
            ])
            self.assertEqual(dict(zip(result.level, result.relevance_bucket_retrieval_eligible)), {
                "L2": True, "L3": False, "L4": True, "L5": False,
            })
            self.assertEqual(manifest["rule"]["operator"], ">=")
            self.assertEqual(manifest["counts_by_level"]["L2"], {"eligible": 1, "ineligible": 0})
            with self.assertRaisesRegex(FileExistsError, "fresh output"):
                levels.export_retrieval_eligibility(ranking_dir, output)


if __name__ == "__main__":
    unittest.main()
