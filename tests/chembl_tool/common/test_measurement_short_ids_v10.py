"""Behavioral checks for batch-local identity and cached retry safety."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data.processing.evidence_library.versions.v10 import (
    build_measurement_resolution_mapping as m,
)


def batch(prefix="a"):
    rows = tuple({"id": prefix * 63 + str(i), "source_id": "hf_bioavailability"}
                 for i in range(2))
    return m.RequestBatch("request-" + prefix, "hf_bioavailability", rows, "extract", 1000)


def answer(ids):
    return {"content": json.dumps({"rows": [
        {"id": i, "status": "ok", "measurements": [{"measurement": "18.0", "unit": "%"}]}
        for i in ids]}), "usage": {"input_tokens": 10, "output_tokens": 10}}


class ShortIdTests(unittest.TestCase):
    def test_planned_payload_and_fingerprint(self):
        config = SimpleNamespace(module=SimpleNamespace(ALLOW_REBATCH_UNATTEMPTED=True),
                                 task_id="bioavailability_ma", BATCH_SIZE=10,
                                 PROMPT_VERSION="test", render_prompt=lambda *a, **k: "extract")
        candidates = [{**r, "canonical_endpoint_name": "bioavailability",
                       "source_row_uid": "source-" + r["id"]} for r in batch().rows]
        planned = m.plan_batches(candidates, config, attempted=set())[0]
        self.assertEqual([r["id"] for r in planned.api_rows], ["r1", "r2"])
        self.assertTrue(all("source_row_uid" not in r for r in planned.api_rows))
        with patch.object(m, "ID_TRANSPORT_VERSION", "different-transport"):
            other = m.plan_batches(candidates, config, attempted=set())[0]
        self.assertNotEqual(planned.request_id, other.request_id)
        remaining = m.plan_batches(candidates, config, attempted={candidates[0]["id"]})
        self.assertEqual(remaining[0].row_ids, (candidates[1]["id"],))

    def test_reordered_and_concurrent_batches(self):
        def run(prefix):
            b = batch(prefix)
            def llm(messages, **kwargs):
                self.assertEqual([r["id"] for r in json.loads(messages["user"])["rows"]], ["r1", "r2"])
                self.assertNotIn(b.row_ids[0], messages["user"])
                return answer(["r2", "r1"])
            rows, _, status = m.query_batch(b, llm=llm, model="model", task="bioavailability_ma")
            self.assertEqual(status, "valid")
            self.assertEqual([r["cleaned_record_id"] for r in rows], list(b.row_ids))
            self.assertEqual(json.loads(rows[0]["raw_response_json"])["id"], "r1")
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(run, ["a", "b"]))

    def test_bad_ids_fail_closed(self):
        for ids in (["r1"], ["r1", "r1"], ["r1", "r3"], ["r1", None],
                    ["r1", "r2", "r3"], list(batch().row_ids)):
            with self.subTest(ids=ids):
                rows, _, status = m.query_batch(
                    batch(),
                    llm=lambda *a, current=ids, **k: answer(current),
                    model="model",
                    task="bioavailability_ma",
                )
                self.assertEqual(status, "structural_invalid_after_structural_retry")
                self.assertTrue(all(r["quantity_count"] == 0 for r in rows))

    def test_structural_retry_keeps_wire_mapping(self):
        calls = []
        def llm(messages, **kwargs):
            calls.append(messages["user"])
            return answer(["mistyped", "r2"] if len(calls) == 1 else ["r1", "r2"])
        _, _, status = m.query_batch(
            batch(), llm=llm, model="model", task="bioavailability_ma"
        )
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(status, "valid_after_structural_retry")

    def test_cache_reuse_and_retry_cap(self):
        with tempfile.TemporaryDirectory() as root:
            cache = m.SubmissionCache(Path(root) / "requests.jsonl")
            b = batch()
            cache.submit(b, epoch="test", model="model")
            rows = [m._blank(r, method="model_single_pass") for r in b.rows]
            rows[1]["assignment_method"] = "invalid_row_response:id_mismatch"
            cache.terminal(b, assignments=rows, usage=None, response_status="partial_invalid")
            cache = m.SubmissionCache(cache.path)
            self.assertEqual(m.retryable_assignment_ids(cache, retry_model="model", max_attempts=1), set())
            retry = m.retryable_assignment_ids(cache, retry_model="model", max_attempts=0)
            self.assertEqual(retry, {b.row_ids[1]})
            cache.allow_retry(retry)
            self.assertIn(b.row_ids[0], cache.attempted)
            self.assertEqual(cache.assignments[b.row_ids[0]], rows[0])

    def test_explicit_retry_cli(self):
        args = m.parse_args(["--task", "bioavailability_ma", "--retry-failed", "--retry-max-attempts", "0"])
        self.assertEqual(args.retry_max_attempts, 0)
        with self.assertRaises(SystemExit):
            m.parse_args(["--task", "bioavailability_ma", "--retry-max-attempts", "0"])


if __name__ == "__main__":
    unittest.main()
